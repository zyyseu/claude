import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_data(data: list[str]) -> tuple[list[str], np.ndarray]:
    labels = []
    arrays = []
    for item in data:
        label_str, arr_str = item.split("\n", 1)
        labels.append(label_str.strip())
        arrays.append(json.loads(arr_str.strip()))
    return labels, np.array(arrays, dtype=float)


def _pairwise_auc(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute AUC between two 4-element arrays via pairwise ranking.
    For every pair (j, k) where true[j] > true[k], check whether
    pred[j] > pred[k] (correct), pred[j] < pred[k] (wrong), or
    pred[j] == pred[k] (tie → 0.5).  Returns NaN when no valid pair exists.
    """
    n = len(true)
    correct, total = 0.0, 0
    for j in range(n):
        for k in range(n):
            if true[j] > true[k]:
                total += 1
                if pred[j] > pred[k]:
                    correct += 1
                elif pred[j] == pred[k]:
                    correct += 0.5
    return correct / total if total > 0 else float("nan")


def evaluate_model(y_true_list: list[str], y_pred_list: list[str]) -> dict:
    """
    Evaluate model predictions against ground truth.

    Each element format:  "large\\n[0,1,2,3]"  or  "small\\n[0,1,2,3]"
    - str1: classification label ("large" / "small")
    - [num0, num1, num2, num3]: 4-element array, values in [0, 4]

    Returns:
        accuracy, precision, recall, f1  — classification metrics
        auc_per_sample                   — one AUC per sample (comparing the two 4-element arrays)
        auc_mean                         — average of auc_per_sample
    """
    true_labels, true_arrays = parse_data(y_true_list)
    pred_labels, pred_arrays = parse_data(y_pred_list)

    # --- Classification ---
    accuracy  = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, pos_label="large", zero_division=0)
    recall    = recall_score(true_labels, pred_labels, pos_label="large", zero_division=0)
    f1        = f1_score(true_labels, pred_labels, pos_label="large", zero_division=0)

    # --- AUC: per-sample pairwise ranking, then average ---
    auc_per_sample = []
    for i in range(len(true_arrays)):
        auc = _pairwise_auc(true_arrays[i], pred_arrays[i])
        auc_per_sample.append(round(auc, 4))

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc_per_sample": auc_per_sample,
        "auc_mean":  round(np.nanmean(auc_per_sample), 4),
    }
