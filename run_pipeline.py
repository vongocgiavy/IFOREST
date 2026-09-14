#!/usr/bin/env python3
"""
PURE UNSUPERVISED ISOLATION FOREST PIPELINE (PURE NUMPY)
NASA Shuttle Telemetry Dataset - Quy trình Học Không Giám Sát Thuần Túy.
Zero Scikit-Learn Dependency.
"""

import argparse
import os
import sys
from typing import Optional, Union, List, Tuple, Any, overload
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Import thuật toán mô hình thuần túy từ model.py
from model import (
    c_factor,
    Node,
    IsolationTree,
    IsolationForest,
    IsolationForestScratch,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# 1. PIPELINE ĐÓNG GÓI MÔ HÌNH THUẦN TÚY
# ==============================================================================

class Pipeline:
    """Pipeline đóng gói suy luận trực tiếp thuần iForest trên dữ liệu cảm biến thô."""

    def __init__(self, model: Optional[IsolationForest] = None):
        self.model = IsolationForest() if model is None else model

    @property
    def threshold_(self):
        return self.model.threshold_

    @property
    def offset_(self):
        return self.model.offset_

    @property
    def estimators_(self):
        return self.model.estimators_

    @property
    def max_depth(self):
        return self.model.max_depth

    def fit(self, X, y=None):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        self.model.fit(X_arr, y=y)
        return self

    def anomaly_score(self, X):
        return self.model.anomaly_score(X)

    def score_samples(self, X):
        return self.model.score_samples(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict(self, X, threshold=None):
        return self.model.predict(X, threshold=threshold)

    def fit_predict(self, X, y=None, threshold=None):
        return self.fit(X, y=y).predict(X, threshold=threshold)


# ==============================================================================
# 2. CHIA TẬP DỮ LIỆU & KFOLD KHÔNG GIÁM SÁT (PURE NUMPY)
# ==============================================================================

@overload
def train_test_split(X: Any, y: None = None, test_size: float = 0.2, random_state: Any = 42, stratify: bool = False) -> Tuple[Any, Any]: ...
@overload
def train_test_split(X: Any, y: Any, test_size: float = 0.2, random_state: Any = 42, stratify: bool = False) -> Tuple[Any, Any, Any, Any]: ...
def train_test_split(X, y=None, test_size: float = 0.2, random_state: int = 42, stratify: bool = False):
    """
    Phân chia Train/Test thuần túy cho cả bài toán không nhãn (Unsupervised) và có nhãn.
    - Khi y is None: Trả về (X_train, X_test).
    - Khi y is not None: Trả về (X_train, X_test, y_train, y_test).
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if hasattr(X, "ndim") and X.ndim != 2:
        raise ValueError("X must be a 2D array")

    n_samples = len(X)
    if n_samples < 2:
        raise ValueError(f"Dataset must have at least 2 samples to split, got {n_samples}")

    if y is not None:
        y_len = len(y) if hasattr(y, "__len__") else (y.shape[0] if hasattr(y, "shape") else None)
        if y_len != n_samples:
            raise ValueError(f"X and y must have the same length, got len(X)={n_samples}, len(y)={y_len}")

    rng = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)

    if y is not None and stratify:
        y_arr = np.asarray(y)
        train_idx, test_idx = [], []
        for cls in np.unique(y_arr):
            cls_indices = np.where(y_arr == cls)[0]
            rng.shuffle(cls_indices)
            n_test = int(round(len(cls_indices) * test_size))
            test_idx.extend(cls_indices[:n_test])
            train_idx.extend(cls_indices[n_test:])
        train_indices, test_indices = np.array(train_idx), np.array(test_idx)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_test = int(round(n_samples * test_size))
        test_indices, train_indices = indices[:n_test], indices[n_test:]

    X_tr = X.iloc[train_indices].copy() if isinstance(X, pd.DataFrame) else X[train_indices].copy()
    X_te = X.iloc[test_indices].copy() if isinstance(X, pd.DataFrame) else X[test_indices].copy()

    # Kiểm tra tính phân lập mẫu (Train/Test indices disjoint)
    if isinstance(X, (pd.DataFrame, pd.Series)):
        assert set(X_tr.index).isdisjoint(set(X_te.index)), "Data overlap: train/test indices overlap!"
    else:
        assert set(train_indices).isdisjoint(set(test_indices)), "Data overlap: train/test indices overlap!"

    if y is not None:
        y_arr = np.asarray(y)
        y_tr = y.iloc[train_indices].copy() if isinstance(y, pd.Series) else y_arr[train_indices].copy()
        y_te = y.iloc[test_indices].copy() if isinstance(y, pd.Series) else y_arr[test_indices].copy()
        return X_tr, X_te, y_tr, y_te

    return X_tr, X_te


class KFold:
    """K-Fold chuẩn cho bài toán Học Không Giám Sát (Unsupervised Cross-Validation)."""

    def __init__(self, n_splits: int = 3, shuffle: bool = True, random_state: int = 42):
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        if X is None:
            raise ValueError("X cannot be None")
        n_samples = len(X) if hasattr(X, "__len__") else (X.shape[0] if hasattr(X, "shape") else None)
        if n_samples is None or n_samples < self.n_splits:
            raise ValueError(f"n_samples ({n_samples}) cannot be less than n_splits={self.n_splits}")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = self.random_state if isinstance(self.random_state, np.random.RandomState) else np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        splits = np.array_split(indices, self.n_splits)
        all_indices = np.arange(n_samples)
        for val_idx in splits:
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            yield all_indices[train_mask], val_idx


class StratifiedKFold:
    """K-Fold phân tầng cho Cross-Validation (phục vụ đối chiếu khi có nhãn)."""

    def __init__(self, n_splits: int = 3, shuffle: bool = True, random_state: int = 42):
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        X_len = len(X) if hasattr(X, "__len__") else (X.shape[0] if hasattr(X, "shape") else None)
        y_len = len(y) if hasattr(y, "__len__") else (y.shape[0] if hasattr(y, "shape") else None)

        if X_len is None or y_len is None or X_len != y_len:
            raise ValueError(f"X và y phải có cùng độ dài. Nhận được len(X)={X_len}, len(y)={y_len}.")

        if y_len < self.n_splits:
            raise ValueError(f"Số lượng mẫu ({y_len}) không thể nhỏ hơn số fold n_splits={self.n_splits}.")

        y_arr = np.asarray(y)
        unique_classes, counts = np.unique(y_arr, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError("StratifiedKFold yêu cầu ít nhất 2 lớp khác nhau để phân tầng.")

        min_class_count = int(np.min(counts))
        if min_class_count < self.n_splits:
            raise ValueError(
                f"Lớp có ít mẫu nhất chỉ có {min_class_count} phần tử, nhỏ hơn số fold n_splits={self.n_splits}."
            )

        rng = self.random_state if isinstance(self.random_state, np.random.RandomState) else np.random.RandomState(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for cls in unique_classes:
            cls_idx = np.where(y_arr == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_idx)
            splits = np.array_split(cls_idx, self.n_splits)
            for fold_i, split_indices in enumerate(splits):
                folds[fold_i].extend(split_indices)

        n_samples = len(y_arr)
        all_indices = np.arange(n_samples)
        for fold_i in range(self.n_splits):
            val_idx = np.array(folds[fold_i], dtype=int)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            yield all_indices[train_mask], val_idx


def threshold_from_contamination(scores, contamination):
    """Tính ngưỡng phân vị từ điểm số tập train dựa trên contamination (không rò rỉ tập test)."""
    if isinstance(contamination, str) and contamination.lower() == "auto":
        return 0.5
    if isinstance(contamination, bool):
        raise ValueError("contamination cannot be boolean")
    try:
        contam_val = float(contamination)
    except (ValueError, TypeError):
        raise ValueError(f"contamination không hợp lệ: {contamination}. Phải là 'auto' hoặc số thực trong (0, 0.5)")

    scores_arr = np.asarray(scores, dtype=float)
    if len(scores_arr) == 0:
        raise ValueError("scores array cannot be empty")

    if not (0 < contam_val < 0.5):
        raise ValueError(f"contamination must be in (0, 0.5), got {contam_val}")

    return float(np.percentile(scores_arr, 100.0 * (1.0 - contam_val)))


# ==============================================================================
# 3. ĐỘ QUAN TRỌNG THUỘC TÍNH & PHÂN TÍCH CĂN NGUYÊN KHÔNG GIÁM SÁT
# ==============================================================================

def robust_deviation(x_val: float, col_vals: np.ndarray) -> float:
    """
    Đo lệch chuẩn phi tham số bằng Percentile Rank (robust với phân phối cực lệch).

    Trả về khoảng cách phần trăm so với trung vị [0.0, 50.0].
    Không bị ảnh hưởng bởi outlier cực đoan hay kurtosis cao (cột zero-inflated).
    Thay thế Z-score trong RCA và Feature Importance.

    Tham số:
    ----------
    x_val    : giá trị cần đo độ lệch.
    col_vals : toàn bộ giá trị của cột đó trong tập dữ liệu (dùng để xếp hạng).

    Trả về:
    ----------
    float trong [0.0, 50.0] — 0.0 = ở đúng trung vị, 50.0 = ở cực đầu phân phối.
    """
    sorted_vals = np.sort(col_vals)
    idx = int(np.searchsorted(sorted_vals, x_val, side='right'))
    pct = idx / len(sorted_vals) * 100.0
    return abs(pct - 50.0)


def unsupervised_feature_importance(model: IsolationForest, X, feature_names=None):
    """
    Tính mức độ quan trọng không giám sát của từng đặc trưng (Vectorized NumPy):
    1. Tương quan tuyến tính Pearson giữa giá trị đặc trưng và Anomaly Score: corr(X_j, s(x)).
    2. Median Percentile Deviation (non-parametric, robust) của nhóm bất thường (s >= 0.5)
       so với toàn quần thể — thay thế Z-score bị nhiễu bởi kurtosis cực cao.
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    scores = model.anomaly_score(X_arr)
    n_samples, n_features = X_arr.shape
    names = feature_names if feature_names is not None else [f"feat_{i+1}" for i in range(n_features)]

    anom_mask = (scores >= 0.5)
    mean_pop = np.mean(X_arr, axis=0)
    std_pop = np.std(X_arr, axis=0)

    # 1. Tương quan Pearson vector hóa
    std_sc = float(np.std(scores))
    if std_sc > 0:
        centered_X = X_arr - mean_pop
        centered_sc = scores - np.mean(scores)
        cov = np.mean(centered_X * centered_sc[:, None], axis=0)
        corr_arr = np.where(std_pop > 0, cov / (std_pop * std_sc), 0.0)
    else:
        corr_arr = np.zeros(n_features, dtype=float)

    # 2. Median Percentile Deviation của nhóm bất thường (Robust, non-parametric)
    # Đo trung vị của percentile rank của nhóm anomaly so với median quần thể (50%).
    # Không bị ảnh hưởng bởi outlier cực đoan hay kurtosis cao như Z-score.
    if np.any(anom_mask):
        pct_dev_arr = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            sorted_col = np.sort(X_arr[:, j])
            anom_vals = X_arr[anom_mask, j]
            # Percentile rank của từng mẫu bất thường trong cột j
            pct_ranks = np.searchsorted(sorted_col, anom_vals, side='right') / n_samples * 100.0
            # Median khoảng cách từ trung vị (50%) -> bất biến với outlier cực đoan
            pct_dev_arr[j] = float(np.median(np.abs(pct_ranks - 50.0)))
    else:
        pct_dev_arr = np.zeros(n_features, dtype=float)

    abs_corr = np.abs(corr_arr)
    # Chuẩn hóa pct_dev_arr về [0, 1]: max có thể là 50.0
    pct_dev_norm = np.minimum(pct_dev_arr, 50.0) / 50.0
    comp_importance = abs_corr * 0.5 + pct_dev_norm * 0.5

    # Stability check: Spearman rank correlation giữa 2 sub-sample anomalies (interleaved split)
    stability_note = ""
    if np.sum(anom_mask) >= 10:
        anom_indices = np.where(anom_mask)[0]
        def _pct_dev_col(X_sub, X_ref, j):
            sorted_ref = np.sort(X_ref[:, j])
            vals = X_sub[:, j]
            ranks = np.searchsorted(sorted_ref, vals, side='right') / len(X_ref) * 100.0
            return float(np.median(np.abs(ranks - 50.0)))
        dev1 = np.array([_pct_dev_col(X_arr[anom_indices[::2]], X_arr, j) for j in range(n_features)])
        dev2 = np.array([_pct_dev_col(X_arr[anom_indices[1::2]], X_arr, j) for j in range(n_features)])
        # Spearman rho via rank correlation
        rank1 = np.argsort(np.argsort(-dev1)).astype(float)
        rank2 = np.argsort(np.argsort(-dev2)).astype(float)
        std1 = np.std(rank1)
        std2 = np.std(rank2)
        if std1 > 0 and std2 > 0:
            rho = float(np.mean((rank1 - np.mean(rank1)) * (rank2 - np.mean(rank2))) / (std1 * std2))
        else:
            rho = 1.0
        if rho < 0.6:
            stability_note = f"[CANH BAO] Stability thap: Spearman rho={rho:.2f} < 0.6 (ranking khong on dinh giua cac sub-sample anomaly)"
        else:
            stability_note = f"[OK] Stability check: Spearman rho={rho:.2f} >= 0.6"

    rows = []
    for j in range(n_features):
        rows.append({
            'Đặc trưng': names[j],
            'Tương quan Pearson |r|': float(abs_corr[j]),
            'Hệ số tương quan r': float(corr_arr[j]),
            'Độ lệch Pct-Rank (Dị biệt)': float(pct_dev_arr[j]),
            'Chỉ số quan trọng tổng hợp': float(comp_importance[j])
        })

    df_imp = pd.DataFrame(rows).sort_values(by='Chỉ số quan trọng tổng hợp', ascending=False).reset_index(drop=True)
    if stability_note:
        df_imp.attrs['stability_note'] = stability_note
    return df_imp



def explain_anomalies_root_cause(model: IsolationForest, X, top_k: int = 10, feature_names=None):
    """
    Trích xuất Top-K mẫu bất thường nhất và chẩn đoán đặc trưng lệch mạnh nhất (Root Cause).

    Sử dụng Percentile Rank Deviation (non-parametric, robust) thay cho Z-score.
    Phù hợp với dữ liệu zero-inflated, kurtosis cực cao (như NASA Shuttle col1, col3, col5).
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    scores = model.anomaly_score(X_arr)
    n_features = X_arr.shape[1]
    names = feature_names if feature_names is not None else [f"feat_{i+1}" for i in range(n_features)]

    # Pre-sort mỗi cột một lần để tính percentile rank
    sorted_cols = [np.sort(X_arr[:, j]) for j in range(n_features)]

    top_indices = np.argsort(-scores)[:top_k]
    reports = []
    for rank, idx in enumerate(top_indices, 1):
        sample = X_arr[idx]
        sc = float(scores[idx])
        # Percentile Rank Deviation: robust với zero-inflated & high-kurtosis data
        pct_devs = np.array([
            abs(np.searchsorted(sorted_cols[j], sample[j], side='right') / len(X_arr) * 100.0 - 50.0)
            for j in range(n_features)
        ])
        max_dev_feat_idx = int(np.argmax(pct_devs))
        reports.append({
            'Hạng': rank,
            'Chỉ số mẫu (Index)': int(idx),
            'Anomaly Score': f"{sc:.6f}",
            'Mức cảnh báo': 'NGUY HIỂM CAO' if sc >= 0.60 else 'CẢNH BÁO BẤT THƯỜNG',
            'Đặc trưng lệch mạnh nhất': names[max_dev_feat_idx],
            'Pct Rank Deviation': f"{pct_devs[max_dev_feat_idx]:.1f}% khỏi trung vị",
            'Giá trị thực': f"{sample[max_dev_feat_idx]:.2f}",
            'Pct rank trong quần thể': f"{np.searchsorted(sorted_cols[max_dev_feat_idx], sample[max_dev_feat_idx], side='right') / len(X_arr) * 100.0:.1f}%"
        })
    return pd.DataFrame(reports)


# ==============================================================================
# 4. CÁC HÀM ĐÁNH GIÁ CHUẨN HOÁ (HỖ TRỢ ĐỐI CHIẾU & TIỆN ÍCH)
# ==============================================================================

def confusion_matrix(y_true, y_pred):
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length")
    return np.array([
        [int(np.sum((y_t == 0) & (y_p == 0))), int(np.sum((y_t == 0) & (y_p == 1)))],
        [int(np.sum((y_t == 1) & (y_p == 0))), int(np.sum((y_t == 1) & (y_p == 1)))]
    ])

def accuracy_score(y_true, y_pred) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

def balanced_accuracy_score(y_true, y_pred) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length")
    n_0 = int(np.sum(y_t == 0))
    n_1 = int(np.sum(y_t == 1))
    rec0 = float(np.sum((y_t == 0) & (y_p == 0)) / n_0) if n_0 > 0 else 0.0
    rec1 = float(np.sum((y_t == 1) & (y_p == 1)) / n_1) if n_1 > 0 else 0.0
    return float((rec0 + rec1) / 2.0) if (n_0 > 0 and n_1 > 0) else 0.0

def precision_score(y_true, y_pred, zero_division: int = 0) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length")
    denom = np.sum(y_p == 1)
    return float(np.sum((y_t == 1) & (y_p == 1)) / denom) if denom > 0 else float(zero_division)

def recall_score(y_true, y_pred, zero_division: int = 0) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length")
    denom = np.sum(y_t == 1)
    return float(np.sum((y_t == 1) & (y_p == 1)) / denom) if denom > 0 else float(zero_division)

def f1_score(y_true, y_pred, zero_division: int = 0) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length")
    p = precision_score(y_true, y_pred, zero_division=zero_division)
    r = recall_score(y_true, y_pred, zero_division=zero_division)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)

def roc_auc_score(y_true, scores) -> float:
    y_t, sc = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(y_t) != len(sc):
        raise ValueError("y_true and scores must have the same length")
    if len(y_t) == 0:
        return 0.5
    order = np.argsort(sc)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(sc) + 1)
    unique_scores, inv_indices, counts = np.unique(sc, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(sc):
        ranks = (np.bincount(inv_indices, weights=ranks) / counts)[inv_indices]
    pos_mask = (y_t == 1)
    n1 = int(np.count_nonzero(pos_mask))
    n0 = len(y_t) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    sum_ranks_pos = float(np.sum(ranks[pos_mask]))
    u1 = sum_ranks_pos - (n1 * (n1 + 1.0)) / 2.0
    return float(u1 / (n1 * n0))

def average_precision_score(y_true, scores) -> float:
    y_t, sc = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(y_t) != len(sc):
        raise ValueError("y_true and scores must have the same length")
    order = np.argsort(-sc)
    y_sorted, sc_sorted = y_t[order], sc[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    n_pos = max(int(np.sum(y_t == 1)), 1)
    precisions = tps / (tps + fps)
    recalls = tps / n_pos
    recalls_diff = np.diff(np.concatenate(([0.0], recalls)))
    return float(np.sum(precisions * recalls_diff))

def roc_curve(y_true, scores):
    y_t, sc = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(y_t) != len(sc):
        raise ValueError("y_true and scores must have the same length")
    order = np.argsort(-sc)
    y_sorted, sc_sorted = y_t[order], sc[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    n_pos = max(int(np.sum(y_t == 1)), 1)
    n_neg = max(int(np.sum(y_t == 0)), 1)
    fpr = np.concatenate(([0.0], fps / n_neg))
    tpr = np.concatenate(([0.0], tps / n_pos))
    return fpr, tpr, np.concatenate(([sc_sorted[0] + 1e-6], sc_sorted))

def precision_recall_curve(y_true, scores):
    y_t, sc = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(y_t) != len(sc):
        raise ValueError("y_true and scores must have the same length")
    order = np.argsort(-sc)
    y_sorted, sc_sorted = y_t[order], sc[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    n_pos = max(int(np.sum(y_t == 1)), 1)
    precisions = np.concatenate(([1.0], tps / (tps + fps)))
    recalls = np.concatenate(([0.0], tps / n_pos))
    return precisions, recalls, sc_sorted

def classification_report(y_true, y_pred) -> str:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    n0 = int(np.sum(y_t == 0))
    n1 = int(np.sum(y_t == 1))
    p0 = float(np.sum((y_t == 0) & (y_p == 0)) / max(np.sum(y_p == 0), 1))
    r0 = float(np.sum((y_t == 0) & (y_p == 0)) / max(n0, 1))
    f0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0
    p1 = precision_score(y_t, y_p)
    r1 = recall_score(y_t, y_p)
    f1 = f1_score(y_t, y_p)
    acc = accuracy_score(y_t, y_p)
    lines = [
        "              precision    recall  f1-score   support",
        f"    Class 0     {p0:7.4f}   {r0:7.4f}   {f0:7.4f}     {n0:5d}",
        f"    Class 1     {p1:7.4f}   {r1:7.4f}   {f1:7.4f}     {n1:5d}",
        "",
        f"    accuracy                        {acc:7.4f}     {len(y_t):5d}",
    ]
    return "\n".join(lines)


def permutation_importance_scratch(model, X, y, scoring: str = "roc_auc", n_repeats: int = 5, random_state: int = 42):
    """
    Đo lường Feature Importance bằng Permutation Importance thuần NumPy.

    Tham số:
    ---------
    scoring : str, mặc định="roc_auc"
        - "roc_auc": Độ suy giảm ROC-AUC khi hoán vị (Threshold-independent, chuẩn khoa học, loại bỏ nhiễu biên ngưỡng).
        - "f1": Độ suy giảm F1-score khi hoán vị (Threshold-dependent, phụ thuộc vào ngưỡng phân loại).
    n_repeats : int, mặc định=5
        Số lần hoán vị ngẫu nhiên để ước lượng trung bình và độ lệch chuẩn.
    random_state : int, mặc định=42
        Seed số ngẫu nhiên đảm bảo tính tái lập.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats must be > 0")
    if scoring not in ["roc_auc", "f1"]:
        raise ValueError(f"scoring không hợp lệ: '{scoring}'. Phải là 'roc_auc' hoặc 'f1'.")

    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same length")
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(X_arr) == 0:
        return np.zeros(0), np.zeros(0)

    if scoring == "roc_auc":
        base_score = roc_auc_score(y_arr, model.anomaly_score(X_arr))
    else:
        base_pred = model.predict(X_arr)
        base_score = f1_score(y_arr, base_pred)

    n_features = X_arr.shape[1]
    importances = np.zeros((n_features, n_repeats))

    for f_idx in range(n_features):
        for r in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, f_idx] = rng.permutation(X_perm[:, f_idx])
            if scoring == "roc_auc":
                score_perm = roc_auc_score(y_arr, model.anomaly_score(X_perm))
                importances[f_idx, r] = base_score - score_perm
            else:
                pred_perm = model.predict(X_perm)
                importances[f_idx, r] = base_score - f1_score(y_arr, pred_perm)

    importances_mean = np.mean(importances, axis=1)
    importances_std = np.std(importances, axis=1)
    return importances_mean, importances_std


# Aliases tương thích ngược cho test suite và quy trình kế thừa
PipelineScratch = Pipeline
accuracy_score_scratch = accuracy_score
balanced_accuracy_score_scratch = balanced_accuracy_score
precision_score_scratch = precision_score
recall_score_scratch = recall_score
f1_score_scratch = f1_score
roc_auc_score_scratch = roc_auc_score
confusion_matrix_scratch = confusion_matrix
classification_report_scratch = classification_report



# ==============================================================================
# 5. THỰC THI PIPELINE KHÔNG GIÁM SÁT
# ==============================================================================

DEFAULT_CONFIG = {
    "data_path": "shuttle.csv",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": "auto",
}

PARAM_GRID = [
    {"n_estimators": 100, "max_samples": 128, "max_features": 1.0},
    {"n_estimators": 100, "max_samples": 256, "max_features": 1.0},
    {"n_estimators": 150, "max_samples": 256, "max_features": 0.8},
    {"n_estimators": 200, "max_samples": 256, "max_features": 1.0},
]


def parse_max_samples(val):
    if val is None or (isinstance(val, str) and val.lower() == "auto"):
        return "auto"
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return float(val)
        except (ValueError, TypeError):
            return "auto"


def parse_args():
    parser = argparse.ArgumentParser(description="Pure Unsupervised Isolation Forest Pipeline - NASA Shuttle Telemetry")
    parser.add_argument("--data_path", type=str, default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_CONFIG["n_estimators"])
    parser.add_argument("--max_samples", type=str, default=str(DEFAULT_CONFIG["max_samples"]))
    parser.add_argument("--max_features", type=float, default=DEFAULT_CONFIG["max_features"])
    parser.add_argument("--contamination", type=str, default=DEFAULT_CONFIG["contamination"])
    parser.add_argument("--test_size", type=float, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--random_state", type=int, default=DEFAULT_CONFIG["random_state"])
    parser.add_argument("--tune", action="store_true", help="Chạy tìm kiếm siêu tham số Unsupervised K-Fold CV")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    n_estimators = args.n_estimators
    max_samples = parse_max_samples(args.max_samples)
    max_features = args.max_features
    contamination_arg = str(args.contamination)
    test_size = args.test_size
    random_state = args.random_state
    tune = args.tune

    print("=" * 70)
    print("   QUY TRÌNH HỌC KHÔNG GIÁM SÁT: THUẦN ISOLATION FOREST (PURE NUMPY)")
    print("   Dataset: NASA Shuttle Telemetry (Dữ liệu không nhãn)")
    print("=" * 70)

    if not os.path.exists(data_path):
        print(f"[!] Lỗi: Không tìm thấy file dữ liệu '{data_path}'!")
        sys.exit(1)

    # 1. Nap du lieu khong tieu de (Header=None)
    df_raw = pd.read_csv(data_path, header=None)
    n_raw_cols = df_raw.shape[1]
    all_col_names = [f"feat_{i+1}" for i in range(n_raw_cols)]
    df_raw.columns = all_col_names

    # --- SUA LOI RO RI NHAN (Label Leakage Fix) ---
    # shuttle.csv co 10 cot: 9 cam bien thuc su + 1 nhan lop UCI (col[9] = feat_10).
    # Xac nhan: phan phoi feat_10 = {1:45586, 2:50, 3:171, 4:8903, 5:3267, 6:10, 7:13}
    # khop chinh xac voi UCI Statlog Shuttle benchmark (class label 1-7).
    # --> feat_10 KHONG phai cam bien; no la nhan lop nguyen ban.
    # --> Mo hinh "khong giam sat" phai TUYET DOI KHONG nhin thay nhan nay khi huan luyen.
    SENSOR_COLS = [f"feat_{i+1}" for i in range(9)]   # Chi 9 cam bien thuc su
    LABEL_COL   = "feat_10"                             # Nhan UCI lop 1-7

    # Tach nhan de su dung NGOAI: external validation sau khi train (khong dung trong fit)
    # Class 1 (Rad Flow) = binh thuong (78.6% du lieu), class 2-7 = bat thuong
    y_raw = df_raw[LABEL_COL].values.astype(int)
    X = df_raw[SENSOR_COLS].copy()                     # Chi 9 cot cam bien
    feature_cols = SENSOR_COLS

    print(f"[*] Nap thanh cong: {len(X):,} dong x {len(feature_cols)} cam bien so (sau khi tach nhan).")
    print(f"[*] Danh sach cam bien: {feature_cols}")
    print(f"[*] Nhan lop UCI (feat_10): {LABEL_COL} da duoc TACH RA khoi feature matrix.")
    print(f"[*] Nhan se chi duoc dung cho External Supervised Validation SAU khi huan luyen.")
    print(f"[*] Trang thai huan luyen: HOAN TOAN KHONG CO NHAN (100% Unsupervised Anomaly Detection).")

    # 2. Phan chia Train/Test khong giam sat + giu nhan de validation sau
    # Truyen y_raw de lay y_train, y_test tuong ung (model KHONG dung y khi fit)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=test_size, random_state=random_state
    )
    assert set(X_train.index).isdisjoint(set(X_test.index)), "Loi: Ro ri du lieu giua Train va Test!"

    # Binary label cho external validation: 1 = binh thuong (class 1), 0 = bat thuong (class 2-7)
    # Dao nguoc de ROC-AUC dung voi anomaly score (score cao = bat thuong)
    y_eval = (y_test != 1).astype(int)   # 1 = bat thuong (ground truth), 0 = binh thuong
    y_eval_rate = float(np.mean(y_eval)) * 100.0

    print(f"[*] Phan chia tap du lieu: Train = {len(X_train):,} mau (80%) | Test = {len(X_test):,} mau (20%)")
    print(f"[*] Kiem tra tinh phan lap: Disjoint = True (Chi so roi rac 100%).")
    print(f"[*] Ty le bat thuong thuc (External Label, tap Test): {y_eval_rate:.2f}% ({int(np.sum(y_eval)):,} mau)")

    model_contam = "auto" if contamination_arg.lower() == "auto" else float(contamination_arg)

    # 3. Huấn luyện hoặc Tinh chỉnh Siêu tham số không giám sát
    if not tune:
        print("\n[*] Huấn luyện mô hình Thuần Isolation Forest (Single Run)...")
        best_model = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples,
            max_features=max_features, contamination=model_contam,
            random_state=random_state
        ).fit(X_train.values)
        best_params = {"n_estimators": n_estimators, "max_samples": max_samples, "max_features": max_features}
    else:
        print("\n[*] Tinh chỉnh Siêu tham số với Unsupervised 3-Fold Cross-Validation...")
        param_configs = PARAM_GRID
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

        # Tiêu chí tối ưu: Chọn cấu hình có Score Spread (Std) cao nhất.
        # Score Spread cao = Mô hình phân biệt rõ ràng hơn giữa bất thường (score cao) và bình thường (score thấp).
        # Đây là tiêu chí không giám sát chuẩn mực (Maximizing Discrimination without labels).
        best_score_spread, best_params, best_model = float('-inf'), None, None
        for idx, p in enumerate(param_configs, 1):
            cur_n = int(p.get("n_estimators", 100))
            cur_s = parse_max_samples(p.get("max_samples", 256))
            cur_f = float(p.get("max_features", 1.0))

            fold_std_list = []
            for tr_idx, val_idx in kf.split(X_train):
                m = IsolationForest(n_estimators=cur_n, max_samples=cur_s, max_features=cur_f,
                                    contamination=model_contam, random_state=random_state)
                m.fit(X_train.iloc[tr_idx].values)
                val_sc = m.anomaly_score(X_train.iloc[val_idx].values)
                fold_std_list.append(float(np.std(val_sc)))

            mean_val_std = float(np.mean(fold_std_list))
            print(f"  - Config {idx}: n_est={cur_n:3d}, max_samples={str(cur_s):>4s}, max_feat={cur_f:.1f} -> Score Spread (Std): {mean_val_std:.4f}")
            # Chọn cấu hình có score spread lớn nhất -> phân biệt tốt nhất anomaly vs normal
            if best_params is None or mean_val_std > best_score_spread:
                best_score_spread = mean_val_std
                best_params = {"n_estimators": cur_n, "max_samples": cur_s, "max_features": cur_f}

        best_model = IsolationForest(
            n_estimators=best_params["n_estimators"], max_samples=best_params["max_samples"],
            max_features=best_params["max_features"], contamination=model_contam,
            random_state=random_state
        ).fit(X_train.values)
        print(f"[+] Cấu hình tối ưu được chọn: {best_params}")

    # 4. Tính toán Anomaly Scores trên tập Train và Test
    train_scores = best_model.anomaly_score(X_train.values)
    test_scores = best_model.anomaly_score(X_test.values)
    c_val = c_factor(best_model.max_samples_actual_)

    # 5. Thiết lập Hệ thống Ngưỡng Quyết định Không Giám Sát
    th_theoretical = 0.500000
    th_top5 = float(np.percentile(train_scores, 95.0))
    th_top1 = float(np.percentile(train_scores, 99.0))
    th_top01 = float(np.percentile(train_scores, 99.9))
    th_stat_3sigma = float(np.mean(train_scores) + 2.0 * np.std(train_scores))

    # Thống kê phân phối điểm Anomaly Score trên tập Test
    sc_mean = float(np.mean(test_scores))
    sc_std = float(np.std(test_scores))
    sc_median = float(np.median(test_scores))
    sc_min = float(np.min(test_scores))
    sc_max = float(np.max(test_scores))

    # Tỷ lệ phát hiện trên tập Test (N=11,600)
    cnt_theo = int(np.sum(test_scores >= th_theoretical))
    pct_theo = cnt_theo / len(test_scores) * 100.0

    cnt_top5 = int(np.sum(test_scores >= th_top5))
    pct_top5 = cnt_top5 / len(test_scores) * 100.0

    cnt_top1 = int(np.sum(test_scores >= th_top1))
    pct_top1 = cnt_top1 / len(test_scores) * 100.0

    print("\n" + "=" * 70)
    print("PHÂN TÍCH HIỆU NĂNG KHÔNG GIÁM SÁT (UNSUPERVISED EVALUATION)")
    print("=" * 70)
    print(f"Số lượng mẫu kiểm thử (Test Samples) : {len(X_test):,}")
    print(f"Số lượng cây cô lập (n_estimators)  : {best_params['n_estimators']}")
    print(f"Kích thước mẫu con psi (max_samples) : {best_params['max_samples']}")
    print(f"Hằng số chuẩn hóa c(psi)             : {c_val:.7f}")
    print(f"Độ sâu tối đa của cây (max_depth)    : {best_model.max_depth}")
    print("-" * 70)
    print("THỐNG KÊ PHÂN PHỐI ANOMALY SCORE (TEST SET):")
    print(f"  - Giá trị nhỏ nhất (Min)           : {sc_min:.6f}")
    print(f"  - Giá trị trung bình (Mean ± Std)  : {sc_mean:.6f} ± {sc_std:.6f}")
    print(f"  - Trung vị (Median / P50)          : {sc_median:.6f}")
    print(f"  - Giá trị lớn nhất (Max)           : {sc_max:.6f}")
    print("-" * 70)
    print("BẢNG TỔNG HỢP CÁC NGƯỠNG PHÁT HIỆN BẤT THƯỜNG (UNSUPERVISED THRESHOLDS):")
    print(f"{'Loại Ngưỡng Quyết Định':<35} | {'Giá trị':<10} | {'Số mẫu phát hiện':<18} | {'Tỷ lệ %':<10}")
    print("-" * 78)
    print(f"{'1. Ngưỡng lý thuyết (Liu et al., 2008)':<35} | {th_theoretical:<10.4f} | {cnt_theo:<18,d} | {pct_theo:<10.2f}%")
    print(f"{'2. Phân vị nhiễm bẩn ước lượng Top 5%':<35} | {th_top5:<10.4f} | {cnt_top5:<18,d} | {pct_top5:<10.2f}%")
    print(f"{'3. Phân vị nguy hiểm cao Top 1%':<35} | {th_top1:<10.4f} | {cnt_top1:<18,d} | {pct_top1:<10.2f}%")
    print(f"{'4. Phân vị cực đoan Top 0.1%':<35} | {th_top01:<10.4f} | {int(np.sum(test_scores >= th_top01)):<18,d} | {(np.sum(test_scores >= th_top01)/len(test_scores)*100):<10.2f}%")
    print(f"{'5. Ngưỡng thống kê (Mean + 2*Std)':<35} | {th_stat_3sigma:<10.4f} | {int(np.sum(test_scores >= th_stat_3sigma)):<18,d} | {(np.sum(test_scores >= th_stat_3sigma)/len(test_scores)*100):<10.2f}%")
    print("=" * 78)

    # 6. Trich xuat Top-5 mau di biet nhat (Root Cause Analysis)
    print("\n[*] TRICH XUAT TOP 5 MAU BAT THUONG NHAT (ROOT CAUSE ANALYSIS - PERCENTILE RANK):")
    top_anomalies_df = explain_anomalies_root_cause(best_model, X_test, top_k=5, feature_names=feature_cols)
    print(top_anomalies_df.to_string(index=False))

    # 7. Do quan trong thuoc tinh khong giam sat (Percentile-based)
    print("\n[*] DO QUAN TRONG THUOC TINH KHONG GIAM SAT (PERCENTILE-BASED FEATURE IMPORTANCE):")
    feat_imp_df = unsupervised_feature_importance(best_model, X_test, feature_names=feature_cols)
    print(feat_imp_df.to_string(index=False))
    if 'stability_note' in feat_imp_df.attrs:
        print(f"\n    {feat_imp_df.attrs['stability_note']}")

    top3_feats = feat_imp_df.head(3)['Đặc trưng'].tolist()
    print(f"\n[*] Top 3 dac trung anh huong manh nhat den su bat thuong: {top3_feats}")

    # 8. SUPERVISED VALIDATION (External Ground Truth — chi de kiem tra, khong dung trong train)
    print("\n" + "=" * 70)
    print("SUPERVISED VALIDATION (EXTERNAL GROUND TRUTH — KHONG DUNG TRONG HUAN LUYEN)")
    print("=" * 70)
    print("[NOTE] Mo hinh chi nhan X (9 cam bien). Nhan y chi duoc dung SAU khi predict")
    print("       de do luong kha nang phat hien thuc te, khong anh huong den huan luyen.")
    print("-" * 70)

    # Nhan binary: 1 = bat thuong (class UCI 2-7), 0 = binh thuong (class UCI 1)
    roc_auc = roc_auc_score(y_eval, test_scores)
    ap = average_precision_score(y_eval, test_scores)

    # Danh gia voi nguong ly thuyet (0.5) va nguong top 5%
    y_pred_theo = (test_scores >= th_theoretical).astype(int)
    y_pred_top5 = (test_scores >= th_top5).astype(int)

    prec_theo = precision_score(y_eval, y_pred_theo)
    rec_theo  = recall_score(y_eval, y_pred_theo)
    f1_theo   = f1_score(y_eval, y_pred_theo)

    prec_top5 = precision_score(y_eval, y_pred_top5)
    rec_top5  = recall_score(y_eval, y_pred_top5)
    f1_top5   = f1_score(y_eval, y_pred_top5)

    print(f"  ROC-AUC (Leak-Free, 9 cam bien)     : {roc_auc:.4f}")
    print(f"  Average Precision (AP)               : {ap:.4f}")
    print(f"  Ty le bat thuong thuc (y_eval=1)     : {y_eval_rate:.2f}%")
    print("-" * 70)
    print(f"  Voi nguong ly thuyet (score >= 0.5):")
    print(f"    Precision = {prec_theo:.4f} | Recall = {rec_theo:.4f} | F1 = {f1_theo:.4f}")
    print(f"  Voi nguong Top 5% (score >= {th_top5:.4f}):")
    print(f"    Precision = {prec_top5:.4f} | Recall = {rec_top5:.4f} | F1 = {f1_top5:.4f}")
    print("=" * 70)

    # 9. Kiem tra suy luan mau thoi gian thuc
    pipe = Pipeline(model=best_model)
    sample_first = X_test.iloc[0].to_dict()
    sample_score = float(pipe.anomaly_score(np.array([list(sample_first.values())]))[0])
    status = "BAT THUONG" if sample_score >= th_theoretical else "BINH THUONG"
    print(f"\n[+] Kiem tra suy luan tren mau thuc te: Score = {sample_score:.6f} -> Trang thai: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
