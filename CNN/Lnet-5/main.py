import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dataloader import get_dataloaders
from model import LeNet5
from engine import train_one_epoch, evaluate, collect_predictions
from visualization import visualize_one_setting


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_text(save_path, content):
    """
    Save text content to a txt file.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_optimizer(model, setting):
    """
    Build optimizer according to the setting dictionary.
    """
    if setting["optimizer"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=setting["learning_rate"],
            momentum=setting["momentum"]
        )

    elif setting["optimizer"] == "adam":
        return optim.Adam(
            model.parameters(),
            lr=setting["learning_rate"]
        )

    else:
        raise ValueError(f"Unsupported optimizer: {setting['optimizer']}")


def confusion_matrix_binary(y_true, y_pred):
    """
    Compute binary confusion matrix entries.

    Returns:
        tn, fp, fn, tp
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return tn, fp, fn, tp


def compute_binary_metrics(y_true, y_pred, y_score):
    """
    Compute accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
    """
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

    accuracy = np.mean(y_true == y_pred)

    precision_denom = tp + fp
    recall_denom = tp + fn

    precision = tp / precision_denom if precision_denom > 0 else 0.0
    recall = tp / recall_denom if recall_denom > 0 else 0.0

    f1_denom = precision + recall
    f1_score = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0

    fprs, tprs, thresholds = roc_curve_binary(y_true, y_score)
    roc_auc = auc_score(fprs, tprs)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics


def roc_curve_binary(y_true, y_score):
    """
    Compute FPR and TPR values for a binary ROC curve.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    sorted_indices = np.argsort(-y_score)
    y_score_sorted = y_score[sorted_indices]

    thresholds = np.unique(y_score_sorted)[::-1]

    tprs = []
    fprs = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.int32)

        tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0

        tprs.append(tpr)
        fprs.append(fpr)

    fprs = np.array([0.0] + fprs + [1.0], dtype=np.float32)
    tprs = np.array([0.0] + tprs + [1.0], dtype=np.float32)
    thresholds = np.array(list(thresholds), dtype=np.float32)

    return fprs, tprs, thresholds


def auc_score(fprs, tprs):
    """
    Compute AUC using trapezoidal integration.
    """
    order = np.argsort(fprs)
    fprs_sorted = fprs[order]
    tprs_sorted = tprs[order]

    auc = np.trapz(tprs_sorted, fprs_sorted)
    return float(auc)


def plot_confusion_matrix(metrics, class_names, save_path, title):
    """
    Plot and save a binary confusion matrix figure.
    """
    conf_mat = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ], dtype=np.int32)

    plt.figure(figsize=(6, 5))
    plt.imshow(conf_mat, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(
                j,
                i,
                str(conf_mat[i, j]),
                ha="center",
                va="center",
                fontsize=12
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_score, save_path, title):
    """
    Plot and save ROC curve figure.
    """
    fprs, tprs, thresholds = roc_curve_binary(y_true, y_score)
    auc_value = auc_score(fprs, tprs)

    plt.figure(figsize=(7, 6))
    plt.plot(fprs, tprs, label=f"ROC curve (AUC = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def format_metrics_block(split_name, metrics):
    """
    Convert metrics into a readable text block.
    """
    lines = []
    lines.append(f"{split_name} Metrics")
    lines.append("-" * 40)
    lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
    lines.append(f"Precision: {metrics['precision']:.4f}")
    lines.append(f"Recall:    {metrics['recall']:.4f}")
    lines.append(f"F1-score:  {metrics['f1_score']:.4f}")
    lines.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    lines.append(f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
    lines.append("")
    lines.append("Confusion Matrix")
    lines.append("Rows = true labels, columns = predicted labels")
    lines.append("                  Pred NORMAL    Pred PNEUMONIA")
    lines.append(f"True NORMAL       {metrics['tn']:>11}    {metrics['fp']:>14}")
    lines.append(f"True PNEUMONIA    {metrics['fn']:>11}    {metrics['tp']:>14}")
    lines.append("")

    return "\n".join(lines)


def main():
    # =========================
    # 1. basic settings
    # =========================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    data_dir = os.path.join(project_root, "data")

    dataset_name = "pneumonia"

    batch_size = 32
    num_workers = 0
    num_epochs = 25

    # Only use one setting and run one time
    setting = {
        "name": "adam_lr0.001",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "momentum": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Current file directory:", current_dir)
    print("Project root:", project_root)
    print("Data directory:", data_dir)

    # =========================
    # 2. check data folders
    # =========================
    required_folders = [
        os.path.join(data_dir, "train", "NORMAL"),
        os.path.join(data_dir, "train", "PNEUMONIA"),
        os.path.join(data_dir, "val", "NORMAL"),
        os.path.join(data_dir, "val", "PNEUMONIA"),
        os.path.join(data_dir, "test", "NORMAL"),
        os.path.join(data_dir, "test", "PNEUMONIA"),
    ]

    for folder in required_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder does not exist: {folder}")

    # =========================
    # 3. output folder
    # =========================
    output_dir = os.path.join("output", dataset_name, setting["name"])
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths
    best_model_path = os.path.join(output_dir, "best_model.pth")
    metrics_path = os.path.join(output_dir, "metrics.txt")
    final_results_path = os.path.join(output_dir, "final_results.txt")
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    history_path = os.path.join(output_dir, "training_history.txt")

    # =========================
    # 4. data
    # =========================
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print("Class names:", class_names)
    print("Class to index:", class_to_idx)

    # =========================
    # 5. model settings
    # =========================
    in_channels = 3
    num_classes = len(class_names)

    # =========================
    # 6. set seed
    # =========================
    set_seed(42)

    # =========================
    # 7. build model
    # =========================
    model = LeNet5(
        in_channels=in_channels,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, setting)

    best_val_acc = 0.0
    best_epoch = 0

    train_acc_curve = []
    val_acc_curve = []
    test_acc_curve = []

    train_loss_curve = []
    val_loss_curve = []
    test_loss_curve = []

    print(f"\n========== Start Training: {setting['name']} ==========\n")

    # =========================
    # 8. epoch loop
    # =========================
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        train_acc_curve.append(train_acc)
        val_acc_curve.append(val_acc)
        test_acc_curve.append(test_acc)

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        test_loss_curve.append(test_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        # Use validation accuracy to select the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    # =========================
    # 9. load best model and evaluate on test set
    # =========================
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_test_loss, final_test_acc = evaluate(
        model, test_loader, criterion, device
    )

    # Collect final predictions from the best validation model
    val_labels, val_preds, val_probs = collect_predictions(model, val_loader, device)
    test_labels, test_preds, test_probs = collect_predictions(model, test_loader, device)

    val_scores = val_probs[:, 1]
    test_scores = test_probs[:, 1]

    val_metrics = compute_binary_metrics(val_labels, val_preds, val_scores)
    test_metrics = compute_binary_metrics(test_labels, test_preds, test_scores)

    # =========================
    # 10. save curves
    # =========================
    np.save(os.path.join(output_dir, "train_acc_curve.npy"), np.array(train_acc_curve))
    np.save(os.path.join(output_dir, "val_acc_curve.npy"), np.array(val_acc_curve))
    np.save(os.path.join(output_dir, "test_acc_curve.npy"), np.array(test_acc_curve))

    np.save(os.path.join(output_dir, "train_loss_curve.npy"), np.array(train_loss_curve))
    np.save(os.path.join(output_dir, "val_loss_curve.npy"), np.array(val_loss_curve))
    np.save(os.path.join(output_dir, "test_loss_curve.npy"), np.array(test_loss_curve))

    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc\n")

        for i in range(num_epochs):
            f.write(
                f"{i + 1},"
                f"{train_loss_curve[i]:.6f},"
                f"{val_loss_curve[i]:.6f},"
                f"{test_loss_curve[i]:.6f},"
                f"{train_acc_curve[i]:.6f},"
                f"{val_acc_curve[i]:.6f},"
                f"{test_acc_curve[i]:.6f}\n"
            )

    # =========================
    # 11. save figures
    # =========================
    visualize_one_setting(
        dataset_name=dataset_name,
        setting_name=setting["name"],
        output_root="output"
    )

    plot_confusion_matrix(
        metrics=test_metrics,
        class_names=class_names,
        save_path=confusion_matrix_path,
        title="LeNet-5 Confusion Matrix"
    )

    plot_roc_curve(
        y_true=test_labels,
        y_score=test_scores,
        save_path=roc_curve_path,
        title="LeNet-5 ROC Curve"
    )

    # =========================
    # 12. save summary
    # =========================
    summary_text = (
        f"Dataset: {dataset_name}\n"
        f"Data Dir: {data_dir}\n"
        f"Class Names: {class_names}\n"
        f"Class to Index: {class_to_idx}\n"
        f"Model: LeNet-5 style CNN\n"
        f"Input Channels: {in_channels}\n"
        f"Input Size: 32 x 32\n"
        f"Setting: {setting['name']}\n"
        f"Optimizer: {setting['optimizer']}\n"
        f"Learning Rate: {setting['learning_rate']}\n"
        f"Momentum: {setting['momentum']}\n"
        f"Batch Size: {batch_size}\n"
        f"Num Epochs: {num_epochs}\n"
        f"Best Val Accuracy: {best_val_acc:.4f}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Final Test Loss from Best Val Model: {final_test_loss:.4f}\n"
        f"Final Test Accuracy from Best Val Model: {final_test_acc:.4f}\n"
        f"Best Model Path: {best_model_path}\n"
        f"Accuracy Curve Path: {os.path.join(output_dir, 'accuracy_curve.png')}\n"
        f"Loss Curve Path: {os.path.join(output_dir, 'loss_curve.png')}\n"
        f"Confusion Matrix Path: {confusion_matrix_path}\n"
        f"ROC Curve Path: {roc_curve_path}\n"
    )

    save_text(metrics_path, summary_text)

    final_results_text = (
        "LeNet-5 Final Results\n"
        "============================================================\n\n"
        "Model Configuration\n"
        "------------------------------------------------------------\n"
        f"Dataset: {dataset_name}\n"
        f"Model: LeNet-5 style CNN\n"
        f"Input: 3-channel grayscale images resized to 32 x 32\n"
        f"Optimizer: {setting['optimizer']}\n"
        f"Learning Rate: {setting['learning_rate']}\n"
        f"Batch Size: {batch_size}\n"
        f"Num Epochs: {num_epochs}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best Val Accuracy: {best_val_acc:.4f}\n\n"
        f"{format_metrics_block('Validation', val_metrics)}\n"
        f"{format_metrics_block('Test', test_metrics)}\n"
        "Generated Files\n"
        "------------------------------------------------------------\n"
        f"Best Model: {best_model_path}\n"
        f"Accuracy Curve: {os.path.join(output_dir, 'accuracy_curve.png')}\n"
        f"Loss Curve: {os.path.join(output_dir, 'loss_curve.png')}\n"
        f"Confusion Matrix: {confusion_matrix_path}\n"
        f"ROC Curve: {roc_curve_path}\n"
        f"Training History: {history_path}\n"
        f"Metrics Summary: {metrics_path}\n"
    )

    save_text(final_results_path, final_results_text)

    print("\n========== Training Summary ==========")
    print(summary_text)

    print("\n========== Validation Metrics ==========")
    print(format_metrics_block("Validation", val_metrics))

    print("\n========== Test Metrics ==========")
    print(format_metrics_block("Test", test_metrics))

    print("Final results saved to:", final_results_path)
    print("Confusion matrix saved to:", confusion_matrix_path)
    print("ROC curve saved to:", roc_curve_path)


if __name__ == "__main__":
    main()
