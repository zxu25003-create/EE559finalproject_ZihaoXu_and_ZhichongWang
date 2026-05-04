# train_rf.py

import os
import matplotlib.pyplot as plt
import numpy as np

from preprocess import load_all_data
from augment import augment_dataset
from feature_extraction import extract_features_from_dataset
from random_forest import RandomForestClassifierScratch
from metrics import (
    accuracy_score,
    confusion_matrix_binary_array,
    get_classification_metrics,
    print_classification_metrics,
    roc_curve_binary,
    roc_auc_score_binary
)


def plot_accuracy_curves(train_acc_history, val_acc_history, test_acc_history, save_path):
    """
    Plot train / validation / test accuracy against the number of trees.

    Unlike neural networks, random forests do not train over epochs.
    Therefore, the x-axis is the number of trees used in the ensemble.
    """
    num_trees = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(num_trees, train_acc_history, label="Train Accuracy")
    plt.plot(num_trees, val_acc_history, label="Validation Accuracy")
    plt.plot(num_trees, test_acc_history, label="Test Accuracy")

    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("pneumonia - random_forest")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(conf_mat, class_names, save_path, title):
    """
    Plot and save a 2 x 2 confusion matrix.
    """
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
    Plot and save the ROC curve.
    """
    fprs, tprs, thresholds = roc_curve_binary(y_true, y_score)
    auc_value = roc_auc_score_binary(y_true, y_score)

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


def save_accuracy_history(history_path, train_acc_history, val_acc_history, test_acc_history):
    """
    Save the accuracy history as a CSV-style text file.
    """
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("num_trees,train_acc,val_acc,test_acc\n")

        for i in range(len(train_acc_history)):
            f.write(
                f"{i + 1},"
                f"{train_acc_history[i]:.6f},"
                f"{val_acc_history[i]:.6f},"
                f"{test_acc_history[i]:.6f}\n"
            )


def format_metrics_block(split_name, metrics):
    """
    Convert one split's metrics into readable text.
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


def save_final_results(
    save_path,
    model_config,
    val_metrics,
    test_metrics,
    output_files
):
    """
    Save model configuration, final metrics, and output file paths to a text file.
    """
    lines = []

    lines.append("Random Forest Final Results")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Model Configuration")
    lines.append("-" * 40)
    for key, value in model_config.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append(format_metrics_block("Validation", val_metrics))
    lines.append(format_metrics_block("Test", test_metrics))

    lines.append("Generated Files")
    lines.append("-" * 40)
    for name, path in output_files.items():
        lines.append(f"{name}: {path}")
    lines.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    # ==================================================
    # 1. Basic settings
    # ==================================================
    data_root = "../data"
    output_dir = "outputs_random_forest"
    os.makedirs(output_dir, exist_ok=True)

    accuracy_curve_path = os.path.join(output_dir, "random_forest_accuracy_curve.png")
    confusion_matrix_path = os.path.join(output_dir, "random_forest_confusion_matrix.png")
    roc_curve_path = os.path.join(output_dir, "random_forest_roc_curve.png")
    history_path = os.path.join(output_dir, "random_forest_accuracy_history.txt")
    final_results_path = os.path.join(output_dir, "random_forest_final_results.txt")

    # ==================================================
    # 2. Load data
    # ==================================================
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data(data_root=data_root)

    # ==================================================
    # 3. Data augmentation
    # ==================================================
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, augment_times=1)

    # ==================================================
    # 4. Feature extraction
    # ==================================================
    print("Extracting features...")
    hist_bins = 16

    X_train_features = extract_features_from_dataset(X_train_aug, hist_bins=hist_bins)
    X_val_features = extract_features_from_dataset(X_val, hist_bins=hist_bins)
    X_test_features = extract_features_from_dataset(X_test, hist_bins=hist_bins)

    # ==================================================
    # 5. Train random forest
    # ==================================================
    print("Training random forest...")

    model_config = {
        "data_root": data_root,
        "image_size": "128 x 128 grayscale",
        "feature_type": "7 statistical features + 16-bin grayscale histogram",
        "feature_dimension": X_train_features.shape[1],
        "augmentation": "rotation, translation, brightness, contrast; augment_times=1",
        "n_estimators": 10,
        "max_depth": 8,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "random_state": 42,
    }

    forest = RandomForestClassifierScratch(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        min_samples_split=model_config["min_samples_split"],
        max_features=model_config["max_features"],
        random_state=model_config["random_state"]
    )
    forest.fit(X_train_features, y_train_aug)

    # ==================================================
    # 6. Build accuracy curve
    # ==================================================
    # Random forests do not have epochs.
    # To create a training curve, evaluate the ensemble after using
    # the first 1 tree, first 2 trees, ..., first n trees.
    train_acc_history = []
    val_acc_history = []
    test_acc_history = []

    for num_trees in range(1, forest.n_estimators + 1):
        train_preds_k = forest.predict(X_train_features, num_trees=num_trees)
        val_preds_k = forest.predict(X_val_features, num_trees=num_trees)
        test_preds_k = forest.predict(X_test_features, num_trees=num_trees)

        train_acc = accuracy_score(y_train_aug, train_preds_k)
        val_acc = accuracy_score(y_val, val_preds_k)
        test_acc = accuracy_score(y_test, test_preds_k)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)

        print(
            f"Trees [{num_trees}/{forest.n_estimators}] | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    plot_accuracy_curves(
        train_acc_history=train_acc_history,
        val_acc_history=val_acc_history,
        test_acc_history=test_acc_history,
        save_path=accuracy_curve_path
    )

    save_accuracy_history(
        history_path=history_path,
        train_acc_history=train_acc_history,
        val_acc_history=val_acc_history,
        test_acc_history=test_acc_history
    )

    print("Accuracy curve saved to:", accuracy_curve_path)
    print("Accuracy history saved to:", history_path)

    # ==================================================
    # 7. Final validation evaluation
    # ==================================================
    print("\nEvaluating on validation set...")
    val_preds = forest.predict(X_val_features)
    val_probs = forest.predict_proba(X_val_features)[:, 1]

    print_classification_metrics(y_val, val_preds, prefix="Val ", y_score=val_probs)
    val_metrics = get_classification_metrics(y_val, val_preds, y_score=val_probs)

    # ==================================================
    # 8. Final test evaluation
    # ==================================================
    print("\nEvaluating on test set...")
    test_preds = forest.predict(X_test_features)
    test_probs = forest.predict_proba(X_test_features)[:, 1]

    print_classification_metrics(y_test, test_preds, prefix="Test ", y_score=test_probs)
    test_metrics = get_classification_metrics(y_test, test_preds, y_score=test_probs)

    # ==================================================
    # 9. Save confusion matrix and ROC curve figures
    # ==================================================
    test_conf_mat = confusion_matrix_binary_array(y_test, test_preds)

    plot_confusion_matrix(
        conf_mat=test_conf_mat,
        class_names=["NORMAL", "PNEUMONIA"],
        save_path=confusion_matrix_path,
        title="Random Forest Confusion Matrix"
    )

    plot_roc_curve(
        y_true=y_test,
        y_score=test_probs,
        save_path=roc_curve_path,
        title="Random Forest ROC Curve"
    )

    print("Confusion matrix figure saved to:", confusion_matrix_path)
    print("ROC curve figure saved to:", roc_curve_path)

    # ==================================================
    # 10. Save final results to text
    # ==================================================
    output_files = {
        "accuracy_curve": accuracy_curve_path,
        "confusion_matrix": confusion_matrix_path,
        "roc_curve": roc_curve_path,
        "accuracy_history": history_path,
        "final_results": final_results_path,
    }

    save_final_results(
        save_path=final_results_path,
        model_config=model_config,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        output_files=output_files
    )

    print("Final results saved to:", final_results_path)


if __name__ == "__main__":
    main()
