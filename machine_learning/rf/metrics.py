# metrics.py

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return np.mean(y_true == y_pred)


def confusion_matrix_binary(y_true, y_pred):
    """
    Compute the confusion-matrix entries for binary classification.

    Returns:
        tn, fp, fn, tp
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp


def confusion_matrix_binary_array(y_true, y_pred):
    """
    Return the binary confusion matrix in 2 x 2 array format.

    Rows are true labels:
        row 0 = true NORMAL
        row 1 = true PNEUMONIA

    Columns are predicted labels:
        col 0 = predicted NORMAL
        col 1 = predicted PNEUMONIA
    """
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int32)


def precision_score_binary(y_true, y_pred):
    """
    Compute binary precision.
    """
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    denom = tp + fp
    if denom == 0:
        return 0.0
    return tp / denom


def recall_score_binary(y_true, y_pred):
    """
    Compute binary recall.
    """
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def f1_score_binary(y_true, y_pred):
    """
    Compute binary F1-score.
    """
    precision = precision_score_binary(y_true, y_pred)
    recall = recall_score_binary(y_true, y_pred)

    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def roc_curve_binary(y_true, y_score):
    """
    Compute FPR and TPR values for a binary ROC curve.

    Args:
        y_true: shape = (N,), binary labels with values 0 or 1.
        y_score: shape = (N,), predicted score or probability for the positive class.

    Returns:
        fprs: numpy array.
        tprs: numpy array.
        thresholds: numpy array.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort samples by score in descending order.
    sorted_indices = np.argsort(-y_score)
    y_score_sorted = y_score[sorted_indices]

    # Use all unique scores as candidate thresholds.
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

    # Add the start point and end point for AUC integration.
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


def roc_auc_score_binary(y_true, y_score):
    """
    Compute binary ROC-AUC.
    """
    fprs, tprs, thresholds = roc_curve_binary(y_true, y_score)
    return auc_score(fprs, tprs)


def get_classification_metrics(y_true, y_pred, y_score=None):
    """
    Return all main binary classification metrics as a dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score_binary(y_true, y_pred)
    recall = recall_score_binary(y_true, y_pred)
    f1 = f1_score_binary(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score_binary(y_true, y_score)

    return metrics


def print_classification_metrics(y_true, y_pred, prefix="", y_score=None):
    """
    Print common binary classification metrics.
    """
    metrics = get_classification_metrics(y_true, y_pred, y_score=y_score)

    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1-score:  {metrics['f1_score']:.4f}")

    if y_score is not None:
        print(f"{prefix}ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(
        f"{prefix}TN={metrics['tn']}, FP={metrics['fp']}, "
        f"FN={metrics['fn']}, TP={metrics['tp']}"
    )
