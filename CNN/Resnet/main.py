import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataloader import get_dataloaders
from model import build_resnet18
from engine import train_one_epoch, evaluate, collect_predictions


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_accuracy_curves(train_acc_history, val_acc_history, test_acc_history, save_path):
    """
    Plot train / validation / test accuracy curves.
    """
    epochs = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_history, label="Train Accuracy")
    plt.plot(epochs, val_acc_history, label="Validation Accuracy")
    plt.plot(epochs, test_acc_history, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("pneumonia - resnet18_pretrained")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(train_loss_history, val_loss_history, test_loss_history, save_path):
    """
    Plot train / validation / test loss curves.
    """
    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, label="Train Loss")
    plt.plot(epochs, val_loss_history, label="Validation Loss")
    plt.plot(epochs, test_loss_history, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("pneumonia - resnet18_pretrained")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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

    np.trapz is used for compatibility with older NumPy versions.
    """
    order = np.argsort(fprs)
    fprs_sorted = fprs[order]
    tprs_sorted = tprs[order]

    auc = np.trapz(tprs_sorted, fprs_sorted)
    return float(auc)


def compute_binary_metrics(y_true, y_pred, y_score):
    """
    Compute accuracy, precision, recall, F1-score, ROC-AUC, and confusion-matrix entries.
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


def plot_confusion_matrix(metrics, class_names, save_path, title):
    """
    Plot and save a binary confusion matrix.
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
    Plot and save ROC curve.
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
    # ==================================================
    # 1. Basic settings
    # ==================================================
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    data_dir = "../../data"
    batch_size = 64
    num_workers = 0
    num_epochs = 20
    learning_rate = 1e-3

    save_dir = "outputs"
    save_path = os.path.join(save_dir, "best_model.pth")

    acc_curve_path = os.path.join(save_dir, "resnet18_accuracy_curve.png")
    loss_curve_path = os.path.join(save_dir, "resnet18_loss_curve.png")
    confusion_matrix_path = os.path.join(save_dir, "resnet18_confusion_matrix.png")
    roc_curve_path = os.path.join(save_dir, "resnet18_roc_curve.png")
    history_path = os.path.join(save_dir, "resnet18_training_history.txt")
    metrics_path = os.path.join(save_dir, "resnet18_metrics.txt")
    final_results_path = os.path.join(save_dir, "resnet18_final_results.txt")

    os.makedirs(save_dir, exist_ok=True)

    # ==================================================
    # 2. Load data
    # ==================================================
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print("Class names:", class_names)
    print("Class to index mapping:", class_to_idx)

    # ==================================================
    # 3. Build model
    # ==================================================
    model = build_resnet18(
        num_classes=2,
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    # ==================================================
    # 4. Loss and optimizer
    # ==================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # ==================================================
    # 5. History containers
    # ==================================================
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []

    train_acc_history = []
    val_acc_history = []
    test_acc_history = []

    best_val_acc = 0.0
    best_epoch = 0

    # ==================================================
    # 6. Training loop
    # ==================================================
    for epoch in range(num_epochs):
        print(f"\n========== Epoch [{epoch + 1}/{num_epochs}] ==========")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # Evaluate test set every epoch only for plotting the test curve.
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        test_loss_history.append(test_loss)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with Val Acc: {best_val_acc:.4f}")

    print("\nTraining complete. Best Val Acc: {:.4f}".format(best_val_acc))

    # ==================================================
    # 7. Plot curves
    # ==================================================
    plot_accuracy_curves(
        train_acc_history=train_acc_history,
        val_acc_history=val_acc_history,
        test_acc_history=test_acc_history,
        save_path=acc_curve_path
    )

    plot_loss_curves(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        test_loss_history=test_loss_history,
        save_path=loss_curve_path
    )

    print("Accuracy curve saved to:", acc_curve_path)
    print("Loss curve saved to:", loss_curve_path)

    # ==================================================
    # 8. Save history as txt and numpy arrays
    # ==================================================
    np.save(os.path.join(save_dir, "train_acc_curve.npy"), np.array(train_acc_history))
    np.save(os.path.join(save_dir, "val_acc_curve.npy"), np.array(val_acc_history))
    np.save(os.path.join(save_dir, "test_acc_curve.npy"), np.array(test_acc_history))

    np.save(os.path.join(save_dir, "train_loss_curve.npy"), np.array(train_loss_history))
    np.save(os.path.join(save_dir, "val_loss_curve.npy"), np.array(val_loss_history))
    np.save(os.path.join(save_dir, "test_loss_curve.npy"), np.array(test_loss_history))

    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc\n")

        for i in range(num_epochs):
            f.write(
                f"{i + 1},"
                f"{train_loss_history[i]:.6f},"
                f"{val_loss_history[i]:.6f},"
                f"{test_loss_history[i]:.6f},"
                f"{train_acc_history[i]:.6f},"
                f"{val_acc_history[i]:.6f},"
                f"{test_acc_history[i]:.6f}\n"
            )

    print("Training history saved to:", history_path)

    # ==================================================
    # 9. Test the best model on the test set
    # ==================================================
    print("\nEvaluating the best model on the test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    final_test_loss, final_test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Final Test Loss: {final_test_loss:.4f} | Final Test Acc: {final_test_acc:.4f}")

    # ==================================================
    # 10. Compute final validation and test metrics
    # ==================================================
    val_labels, val_preds, val_probs = collect_predictions(model, val_loader, device)
    test_labels, test_preds, test_probs = collect_predictions(model, test_loader, device)

    val_scores = val_probs[:, 1]
    test_scores = test_probs[:, 1]

    val_metrics = compute_binary_metrics(val_labels, val_preds, val_scores)
    test_metrics = compute_binary_metrics(test_labels, test_preds, test_scores)

    # ==================================================
    # 11. Save confusion matrix and ROC curve
    # ==================================================
    plot_confusion_matrix(
        metrics=test_metrics,
        class_names=class_names,
        save_path=confusion_matrix_path,
        title="ResNet18 Confusion Matrix"
    )

    plot_roc_curve(
        y_true=test_labels,
        y_score=test_scores,
        save_path=roc_curve_path,
        title="ResNet18 ROC Curve"
    )

    print("Confusion matrix saved to:", confusion_matrix_path)
    print("ROC curve saved to:", roc_curve_path)

    # ==================================================
    # 12. Save final metrics and results
    # ==================================================
    summary_text = (
        f"Dataset: pneumonia\n"
        f"Data Dir: {data_dir}\n"
        f"Class Names: {class_names}\n"
        f"Class to Index Mapping: {class_to_idx}\n"
        f"Model: ResNet18\n"
        f"Pretrained: True\n"
        f"Freeze Backbone: True\n"
        f"Optimizer: Adam\n"
        f"Learning Rate: {learning_rate}\n"
        f"Batch Size: {batch_size}\n"
        f"Num Epochs: {num_epochs}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best Val Accuracy: {best_val_acc:.4f}\n"
        f"Final Test Loss from Best Val Model: {final_test_loss:.4f}\n"
        f"Final Test Accuracy from Best Val Model: {final_test_acc:.4f}\n"
        f"Best Model Path: {save_path}\n"
        f"Accuracy Curve Path: {acc_curve_path}\n"
        f"Loss Curve Path: {loss_curve_path}\n"
        f"Confusion Matrix Path: {confusion_matrix_path}\n"
        f"ROC Curve Path: {roc_curve_path}\n"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    final_results_text = (
        "ResNet18 Final Results\n"
        "============================================================\n\n"
        "Model Configuration\n"
        "------------------------------------------------------------\n"
        f"Dataset: pneumonia\n"
        f"Model: ResNet18 with ImageNet pretraining\n"
        f"Backbone frozen: True\n"
        f"Input: 3-channel grayscale images resized to 224 x 224\n"
        f"Optimizer: Adam\n"
        f"Learning Rate: {learning_rate}\n"
        f"Batch Size: {batch_size}\n"
        f"Num Epochs: {num_epochs}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best Val Accuracy: {best_val_acc:.4f}\n\n"
        f"{format_metrics_block('Validation', val_metrics)}\n"
        f"{format_metrics_block('Test', test_metrics)}\n"
        "Generated Files\n"
        "------------------------------------------------------------\n"
        f"Best Model: {save_path}\n"
        f"Accuracy Curve: {acc_curve_path}\n"
        f"Loss Curve: {loss_curve_path}\n"
        f"Confusion Matrix: {confusion_matrix_path}\n"
        f"ROC Curve: {roc_curve_path}\n"
        f"Training History: {history_path}\n"
        f"Metrics Summary: {metrics_path}\n"
    )

    with open(final_results_path, "w", encoding="utf-8") as f:
        f.write(final_results_text)

    print("\n========== Training Summary ==========")
    print(summary_text)

    print("\n========== Validation Metrics ==========")
    print(format_metrics_block("Validation", val_metrics))

    print("\n========== Test Metrics ==========")
    print(format_metrics_block("Test", test_metrics))

    print("Metrics summary saved to:", metrics_path)
    print("Final results saved to:", final_results_path)


if __name__ == "__main__":
    main()
