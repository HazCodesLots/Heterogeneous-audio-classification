"""
Hierarchical F-score (hF) metric for DCASE 2026 Task 1.

Implements the modified hP / hR / hF from Kiritchenko et al. (2005),
with a tuneable λ parameter that controls the credit given to predictions
that are correct at the top level but wrong at the second level.

λ = 1.0  → full credit for top-level-correct errors (lenient)
λ = 0.0  → no credit                                 (strict)
λ = 0.75 → DCASE 2026 Task 1 default
"""

from __future__ import annotations
import numpy as np
import torch
from train.model import BST_CLASSES, SECOND_TO_TOP, TOP_CODES


def _ancestor_sets(class_idx: int) -> set[str]:
    """Return the set {second_level_code, top_level_code} for a given class index."""
    second = BST_CLASSES[class_idx]
    top    = BST_CLASSES[class_idx].split("-")[0]  # e.g. "fx"
    return {second, top}


def hierarchical_precision_recall_f(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lam: float = 0.75,
) -> dict[str, float]:
    """
    Compute macro-averaged hierarchical Precision, Recall, and F-score.

    Args:
        y_true: (N,)  integer ground truth class indices.
        y_pred: (N,)  integer predicted class indices.
        lam:    Credit weight for top-level-only correct predictions.

    Returns:
        dict with keys: hP, hR, hF, hP_per_class, hR_per_class, hF_per_class
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."

    n_classes = len(BST_CLASSES)
    hP_per      = np.zeros(n_classes)
    hR_per      = np.zeros(n_classes)
    counts_gt   = np.zeros(n_classes, dtype=int)   # samples where gt == k
    counts_pred = np.zeros(n_classes, dtype=int)   # samples where pred == k

    for gt, pred in zip(y_true, y_pred):
        gt, pred = int(gt), int(pred)
        anc_gt   = _ancestor_sets(gt)
        anc_pred = _ancestor_sets(pred)

        if gt == pred:
            overlap = len(anc_gt)
        else:

            top_gt   = BST_CLASSES[gt].split("-")[0]
            top_pred = BST_CLASSES[pred].split("-")[0]
            if top_gt == top_pred:

                overlap = lam * 1
            else:
                overlap = 0.0

        hP_per[pred]      += overlap / max(len(anc_pred), 1)
        hR_per[gt]        += overlap / max(len(anc_gt),   1)
        counts_gt[gt]     += 1
        counts_pred[pred] += 1

    valid_gt   = counts_gt   > 0
    valid_pred = counts_pred > 0
    hP_c = np.where(valid_pred, hP_per / np.maximum(counts_pred, 1), 0.0)
    hR_c = np.where(valid_gt,   hR_per / np.maximum(counts_gt,   1), 0.0)

    hP_macro = hP_c[valid_pred].mean() if valid_pred.any() else 0.0
    hR_macro = hR_c[valid_gt].mean()   if valid_gt.any()   else 0.0

    denom = hP_macro + hR_macro
    hF_macro = 2 * hP_macro * hR_macro / denom if denom > 0 else 0.0

    hF_c = np.where(
        (hP_c + hR_c) > 0,
        2 * hP_c * hR_c / np.maximum(hP_c + hR_c, 1e-9),
        0.0,
    )

    return {
        "hP":           round(float(hP_macro), 6),
        "hR":           round(float(hR_macro), 6),
        "hF":           round(float(hF_macro), 6),
        "hP_per_class": {BST_CLASSES[i]: round(float(hP_c[i]), 6) for i in range(len(BST_CLASSES))},
        "hR_per_class": {BST_CLASSES[i]: round(float(hR_c[i]), 6) for i in range(len(BST_CLASSES))},
        "hF_per_class": {BST_CLASSES[i]: round(float(hF_c[i]), 6) for i in range(len(BST_CLASSES))},
    }


def top_level_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions correct at the top (coarse) BST level."""
    top_true = [SECOND_TO_TOP[int(i)] for i in y_true]
    top_pred = [SECOND_TO_TOP[int(i)] for i in y_pred]
    return float(np.mean(np.array(top_true) == np.array(top_pred)))


def second_level_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))
