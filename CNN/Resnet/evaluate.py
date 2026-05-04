import os
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataloader import get_dataloader
from model import LeNet5


def parse_best_run_from_summary(summary_path):
    """
    Parse the best run index from summary.txt.

    Expected format:
        Best Run: run_3
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"Best Run:\s*run_(\d+)", content)
    if match is None:
        raise ValueError(f"Cannot find 'Best Run' in {summary_path}")

    return int(match.group(1))


def parse_best_acc_from_summary(summary_path):
    """
    Parse the best test accuracy among 5 runs from summary.txt.

    Expected format:
        Best Test Accuracy among 5 runs: 0.9917
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"Best Test Accuracy among 5 runs:\s*([0-9]*\.?[0-9]+)", content)
    if match is None:
        raise ValueError(f"Cannot find 'Best Test Accuracy among 5 runs' in {summary_path}")

    return float(match.group(1))


def parse_mean_acc_from_summary(summary_path):
    """
    Parse the mean test accuracy of 5 runs from summary.txt.

    Expected format:
        Mean Test Accuracy of 5 runs: 0.9908
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"Mean Test Accuracy of 5 runs:\s*([0-9]*\.?[0-9]+)", content)
    if match is None:
        raise ValueError(f"Cannot find 'Mean Test Accuracy of 5 runs' in {summary_path}")

    return float(match.group(1))


def parse_std_acc_from_summary(summary_path):
    """
    Parse the std test accuracy of 5 runs from summary.txt.

    Expected format:
        Std Test Accuracy of 5 runs: 0.0004
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"Std Test Accuracy of 5 runs:\s*([0-9]*\.?[0-9]+)", content)
    if match is None:
        raise ValueError(f"Cannot find 'Std Test Accuracy of 5 runs' in {summary_path}")

    return float(match.group(1))


def find_best_setting(dataset_name, setting_names, output_root="output"):
    """
    Find the best hyper-parameter setting for one dataset.

    Rule:
    1. Higher best test accuracy is better
    2. If tied, higher mean test accuracy is better
    3. If still tied, lower std is better
    """
    candidates = []

    for setting_name in setting_names:
        summary_path = os.path.join(output_root, dataset_name, setting_name, "summary.txt")

        best_acc = parse_best_acc_from_summary(summary_path)
        mean_acc = parse_mean_acc_from_summary(summary_path)
        std_acc = parse_std_acc_from_summary(summary_path)

        candidates.append({
            "setting_name": setting_name,
            "best_acc": best_acc,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        })

    # sort by:
    # best_acc descending
    # mean_acc descending
    # std_acc ascending
    candidates.sort(
        key=lambda x: (-x["best_acc"], -x["mean_acc"], x["std_acc"])
    )

    best_setting_name = candidates[0]["setting_name"]
    return best_setting_name


def load_best_model(dataset_name, setting_name, in_channels, num_classes, device, output_root="output"):
    """
    Load the best model under the best run of a given setting.

    Args:
        dataset_name: "MNIST", "FashionMNIST", or "CIFAR10"
        setting_name: best setting name under this dataset
        in_channels: 1 for MNIST/FashionMNIST, 3 for CIFAR10
        num_classes: usually 10
        device: cpu or cuda
        output_root: root folder for saved experiment results

    Returns:
        model: loaded PyTorch model
        best_run_idx: integer index of the best run
        best_model_path: full path of the loaded checkpoint
    """
    # 1. Locate the summary file of this setting
    summary_path = os.path.join(output_root, dataset_name, setting_name, "summary.txt")

    # 2. Parse the best run index from summary.txt
    best_run_idx = parse_best_run_from_summary(summary_path)

    # 3. Locate the checkpoint of the best run
    best_model_path = os.path.join(
        output_root,
        dataset_name,
        setting_name,
        f"run_{best_run_idx}",
        "best_model.pth"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")

    # 4. Rebuild the model structure
    model = LeNet5(in_channels=in_channels, num_classes=num_classes).to(device)

    # 5. Load saved parameters
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 6. Switch to evaluation mode
    model.eval()

    return model, best_run_idx, best_model_path

def collect_predictions(model, data_loader, device):
    """
    Collect labels, predicted labels, and class probabilities on a dataset.

    Args:
        model: loaded PyTorch model
        data_loader: test data loader
        device: cpu or cuda

    Returns:
        all_labels: numpy array of shape [N]
        all_preds: numpy array of shape [N]
        all_probs: numpy array of shape [N, num_classes]
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)                  # [B, num_classes]
            probs = F.softmax(outputs, dim=1)       # [B, num_classes]
            preds = torch.argmax(probs, dim=1)      # [B]

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return all_labels, all_preds, all_probs

def compute_normalized_confusion_matrix(all_labels, all_preds, num_classes):
    """
    Compute a row-normalized confusion matrix.

    Args:
        all_labels: numpy array of shape [N]
        all_preds: numpy array of shape [N]
        num_classes: number of classes

    Returns:
        conf_mat: normalized confusion matrix of shape [num_classes, num_classes]
    """
    conf_mat = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(num_classes)
    )

    conf_mat = conf_mat.astype(np.float64)

    row_sums = conf_mat.sum(axis=1, keepdims=True)

    conf_mat = conf_mat / np.clip(row_sums, a_min=1e-12, a_max=None)

    return conf_mat

def plot_confusion_matrix(conf_mat, class_names, save_path, title):
    """
    Plot and save a normalized confusion matrix.

    Args:
        conf_mat: normalized confusion matrix of shape [num_classes, num_classes]
        class_names: list of class names
        save_path: path to save the figure
        title: figure title
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    num_classes = len(class_names)
    tick_marks = np.arange(num_classes)

    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # write values into each cell
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i,
                f"{conf_mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_top_confused_pairs(conf_mat, top_k=3):
    """
    Find the top-k most confused class pairs from a normalized confusion matrix.

    Args:
        conf_mat: normalized confusion matrix of shape [num_classes, num_classes]
        top_k: number of confused pairs to return

    Returns:
        confused_pairs: list of tuples
            [(true_class, pred_class, confusion_value), ...]
    """
    num_classes = conf_mat.shape[0]
    confused_pairs = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                confused_pairs.append((i, j, conf_mat[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return confused_pairs[:top_k]  

def find_example_for_pair(dataset, model, device, true_class, pred_class):
    """
    Find one example image such that:
    - true label == true_class
    - predicted label == pred_class

    Args:
        dataset: test dataset
        model: loaded best model
        device: cpu or cuda
        true_class: integer
        pred_class: integer

    Returns:
        image_tensor: tensor of shape [C, H, W]
        true_label: integer
        predicted_label: integer

    Raises:
        ValueError: if no such example is found
    """
    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]

            # add batch dimension: [C, H, W] -> [1, C, H, W]
            image_batch = image.unsqueeze(0).to(device)

            outputs = model(image_batch)
            pred = torch.argmax(outputs, dim=1).item()

            if label == true_class and pred == pred_class:
                return image.cpu(), label, pred

    raise ValueError(
        f"No example found for true class {true_class} predicted as {pred_class}"
    )

def save_example_image(image_tensor, save_path, title=None):
    """
    Save one example image to disk.

    Args:
        image_tensor: tensor of shape [C, H, W]
        save_path: path to save the image
        title: optional figure title
    """
    image = image_tensor.cpu().numpy()

    # Convert from [C, H, W] to [H, W, C] if RGB
    if image.shape[0] == 1:
        image = image.squeeze(0)
        cmap = "gray"
    else:
        image = np.transpose(image, (1, 2, 0))
        cmap = None

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def evaluate_confusion_matrix_for_dataset(
    dataset_name,
    setting_names,
    class_names,
    in_channels,
    num_classes,
    device,
    output_root="output",
    output_c_root="output_c",
    batch_size=256,
    num_workers=0
):
    """
    Complete the confusion-matrix analysis for one dataset.

    Steps:
    1. Find the best setting from Problem 1(b)
    2. Load the best model under that setting
    3. Run prediction on the test set
    4. Compute and save normalized confusion matrix
    5. Find top-3 confused pairs
    6. Save one example image for each confused pair

    Returns:
        best_setting_name
        confused_pairs
    """
    # 1. Find best setting
    best_setting_name = find_best_setting(
        dataset_name=dataset_name,
        setting_names=setting_names,
        output_root=output_root
    )

    print(f"[{dataset_name}] Best setting: {best_setting_name}")

    # 2. Build test loader
    _, test_loader = get_dataloader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 3. Build test dataset separately for example-image search
    test_dataset = test_loader.dataset

    # 4. Load best model
    model, best_run_idx, best_model_path = load_best_model(
        dataset_name=dataset_name,
        setting_name=best_setting_name,
        in_channels=in_channels,
        num_classes=num_classes,
        device=device,
        output_root=output_root
    )

    print(f"[{dataset_name}] Best run: run_{best_run_idx}")
    print(f"[{dataset_name}] Loaded model from: {best_model_path}")

    # 5. Collect predictions
    all_labels, all_preds, all_probs = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device
    )

    # 6. Compute confusion matrix
    conf_mat = compute_normalized_confusion_matrix(
        all_labels=all_labels,
        all_preds=all_preds,
        num_classes=num_classes
    )

    # 7. Prepare output folders
    dataset_output_dir = os.path.join(output_c_root, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    example_dir = os.path.join(dataset_output_dir, "examples")
    os.makedirs(example_dir, exist_ok=True)

    # 8. Save confusion matrix figure
    conf_mat_path = os.path.join(dataset_output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        conf_mat=conf_mat,
        class_names=class_names,
        save_path=conf_mat_path,
        title=f"{dataset_name} Normalized Confusion Matrix"
    )

    # 9. Find top-3 confused pairs
    confused_pairs = find_top_confused_pairs(conf_mat, top_k=3)

    # 10. Save one example for each confused pair
    for rank, (true_class, pred_class, value) in enumerate(confused_pairs, start=1):
        image_tensor, true_label, predicted_label = find_example_for_pair(
            dataset=test_dataset,
            model=model,
            device=device,
            true_class=true_class,
            pred_class=pred_class
        )

        save_path = os.path.join(
            example_dir,
            f"pair_{rank}_true_{true_class}_pred_{pred_class}.png"
        )

        title = (
            f"Rank {rank}: true={class_names[true_class]}, "
            f"pred={class_names[pred_class]}, "
            f"value={value:.2f}"
        )

        save_example_image(
            image_tensor=image_tensor,
            save_path=save_path,
            title=title
        )

    # 11. Save text summary
    summary_path = os.path.join(dataset_output_dir, "confusion_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Best Setting: {best_setting_name}\n")
        f.write(f"Best Run: run_{best_run_idx}\n")
        f.write(f"Best Model Path: {best_model_path}\n\n")
        f.write("Top 3 Confused Pairs:\n")

        for rank, (true_class, pred_class, value) in enumerate(confused_pairs, start=1):
            f.write(
                f"{rank}. true={true_class} ({class_names[true_class]}) -> "
                f"pred={pred_class} ({class_names[pred_class]}), "
                f"value={value:.4f}\n"
            )

    return best_setting_name, confused_pairs, all_labels, all_preds, all_probs

def plot_one_vs_rest_roc(all_labels, all_probs, num_classes, class_names, save_path):
    """
    Plot one-vs-rest ROC curves for a multi-class classification problem.

    Args:
        all_labels: numpy array of shape [N]
        all_probs: numpy array of shape [N, num_classes]
        num_classes: number of classes
        class_names: list of class names
        save_path: path to save the ROC figure
    """
    # Convert integer labels into one-hot / binary format
    # Example:
    #   label 3 in a 10-class problem -> [0,0,0,1,0,0,0,0,0,0]
    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    plt.figure(figsize=(10, 8))

    for class_idx in range(num_classes):
        # For one-vs-rest:
        #   y_true_bin[:, class_idx] is 1 for this class, 0 for all others
        #   all_probs[:, class_idx] is the predicted score/probability for this class
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], all_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[class_idx]} (AUC = {roc_auc:.4f})"
        )

    # Diagonal reference line: random classifier
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def compute_weighted_auc(all_labels, all_probs, num_classes):
    """
    Compute per-class AUC and weighted average AUC for a multi-class problem.

    Args:
        all_labels: numpy array of shape [N]
        all_probs: numpy array of shape [N, num_classes]
        num_classes: number of classes

    Returns:
        per_class_auc: dict, key = class index, value = AUC
        weighted_auc: float
    """
    # Convert labels to one-vs-rest binary targets
    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    per_class_auc = {}
    class_counts = {}

    for class_idx in range(num_classes):
        # Binary ground truth for current class
        y_true_class = y_true_bin[:, class_idx]

        # Predicted probability/score for current class
        y_score_class = all_probs[:, class_idx]

        # Compute ROC curve and then AUC
        fpr, tpr, _ = roc_curve(y_true_class, y_score_class)
        roc_auc = auc(fpr, tpr)

        per_class_auc[class_idx] = roc_auc
        class_counts[class_idx] = np.sum(all_labels == class_idx)

    # Weighted average AUC
    total_samples = len(all_labels)
    weighted_auc = 0.0

    for class_idx in range(num_classes):
        weight = class_counts[class_idx] / total_samples
        weighted_auc += weight * per_class_auc[class_idx]

    return per_class_auc, weighted_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    setting_names = [
        "setting_1_sgd_lr0.01",
        "setting_2_sgdm_lr0.01_m0.9",
        "setting_3_adam_lr0.001"
    ]

    dataset_configs = [
        {
            "dataset_name": "MNIST",
            "class_names": [str(i) for i in range(10)],
            "in_channels": 1,
            "num_classes": 10,
        },
        {
            "dataset_name": "FashionMNIST",
            "class_names": [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ],
            "in_channels": 1,
            "num_classes": 10,
        },
        {
            "dataset_name": "CIFAR10",
            "class_names": [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ],
            "in_channels": 3,
            "num_classes": 10,
        },
    ]

    for cfg in dataset_configs:
        print(f"\n========== Processing {cfg['dataset_name']} ==========\n")

        best_setting_name, confused_pairs, all_labels, all_preds, all_probs = evaluate_confusion_matrix_for_dataset(
            dataset_name=cfg["dataset_name"],
            setting_names=setting_names,
            class_names=cfg["class_names"],
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            device=device,
            output_root="output",
            output_c_root="output_c",
            batch_size=256,
            num_workers=0
        )
        if cfg["dataset_name"] == "CIFAR10":
            roc_save_path = os.path.join("output_c", "CIFAR10", "roc_curve.png")
            plot_one_vs_rest_roc(
                all_labels=all_labels,
                all_probs=all_probs,
                num_classes=cfg["num_classes"],
                class_names=cfg["class_names"],
                save_path=roc_save_path
            )
            per_class_auc, weighted_auc = compute_weighted_auc(
                all_labels=all_labels,
                all_probs=all_probs,
                num_classes=cfg["num_classes"]
            )
            auc_save_path = os.path.join("output_c", "CIFAR10", "auc_summary.txt")
            with open(auc_save_path, "w", encoding="utf-8") as f:
                f.write(f"Best Setting: {best_setting_name}\n\n")
                f.write("Per-class AUC:\n")
                for class_idx, auc_value in per_class_auc.items():
                    f.write(f"{class_idx} ({cfg['class_names'][class_idx]}): {auc_value:.4f}\n")

                f.write(f"\nWeighted AUC: {weighted_auc:.4f}\n")


if __name__ == "__main__":
    main()