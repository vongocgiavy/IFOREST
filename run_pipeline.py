#!/usr/bin/env python3
"""
ISOLATION FOREST ANOMALY DETECTION PIPELINE (PURE NUMPY)
NASA Shuttle Dataset - Quy trình thực thi & Đánh giá mô hình.
"""

import argparse
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Import thuật toán mô hình đã tách riêng từ model.py
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

    def __init__(self, model: IsolationForest = None):
        self.model = model

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
# 2. CHIA TẬP DỮ LIỆU & ĐÁNH GIÁ METRICS (PURE NUMPY)
# ==============================================================================

def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    """Phân chia Train/Test phân tầng không phụ thuộc thư viện ngoài."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y)

    if stratify:
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
        n_samples = len(y_arr)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_test = int(round(n_samples * test_size))
        test_indices, train_indices = indices[:n_test], indices[n_test:]

    X_tr = X.iloc[train_indices].copy() if isinstance(X, pd.DataFrame) else X[train_indices].copy()
    X_te = X.iloc[test_indices].copy() if isinstance(X, pd.DataFrame) else X[test_indices].copy()
    y_tr = y.iloc[train_indices].copy() if isinstance(y, pd.Series) else y_arr[train_indices].copy()
    y_te = y.iloc[test_indices].copy() if isinstance(y, pd.Series) else y_arr[test_indices].copy()

    assert len(X_tr) + len(X_te) == len(X), "Sum of split sizes does not equal original size"
    assert len(y_tr) + len(y_te) == len(y), "Sum of label split sizes does not equal original size"

    return X_tr, X_te, y_tr, y_te


def threshold_from_contamination(scores, contamination):
    """Tính ngưỡng phân vị từ điểm số tập train dựa trên contamination (không rò rỉ tập test)."""
    if contamination == "auto":
        return 0.5

    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be in (0, 0.5)")

    return float(np.quantile(scores, 1.0 - contamination))



class StratifiedKFold:
    """K-Fold phân tầng cho Cross-Validation."""

    def __init__(self, n_splits: int = 3, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        rng = np.random.RandomState(self.random_state)
        y_arr = np.asarray(y)
        folds = [[] for _ in range(self.n_splits)]
        for cls in np.unique(y_arr):
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


def confusion_matrix(y_true, y_pred):
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    return np.array([
        [int(np.sum((y_t == 0) & (y_p == 0))), int(np.sum((y_t == 0) & (y_p == 1)))],
        [int(np.sum((y_t == 1) & (y_p == 0))), int(np.sum((y_t == 1) & (y_p == 1)))]
    ])


def accuracy_score(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)))


def balanced_accuracy_score(y_true, y_pred) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    rec0 = float(np.sum((y_t == 0) & (y_p == 0)) / max(np.sum(y_t == 0), 1))
    rec1 = float(np.sum((y_t == 1) & (y_p == 1)) / max(np.sum(y_t == 1), 1))
    return float((rec0 + rec1) / 2.0)


def precision_score(y_true, y_pred, zero_division: int = 0) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    denom = np.sum(y_p == 1)
    return float(np.sum((y_t == 1) & (y_p == 1)) / denom) if denom > 0 else float(zero_division)


def recall_score(y_true, y_pred, zero_division: int = 0) -> float:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    denom = np.sum(y_t == 1)
    return float(np.sum((y_t == 1) & (y_p == 1)) / denom) if denom > 0 else float(zero_division)


def f1_score(y_true, y_pred, zero_division: int = 0) -> float:
    p = precision_score(y_true, y_pred, zero_division=zero_division)
    r = recall_score(y_true, y_pred, zero_division=zero_division)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)


def roc_auc_score(y_true, scores) -> float:
    """Tính ROC-AUC bằng thống kê Wilcoxon-Mann-Whitney U: O(N log N) vector hóa hoàn toàn."""
    y_t, sc = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    order = np.argsort(sc)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(sc) + 1)

    unique_scores, inv_indices, counts = np.unique(sc, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(sc):
        mean_ranks = np.bincount(inv_indices, weights=ranks) / counts
        ranks = mean_ranks[inv_indices]

    pos_mask = (y_t == 1)
    n1 = int(np.count_nonzero(pos_mask))
    n0 = len(y_t) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    u = np.sum(ranks[pos_mask]) - (n1 * (n1 + 1.0)) / 2.0
    return float(u / (n0 * n1))


def classification_report(y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    s0, s1 = tn + fp, fn + tp
    total = s0 + s1

    p0 = float(tn / max(tn + fn, 1))
    r0 = float(tn / max(s0, 1))
    f0 = 2.0 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0

    p1 = float(tp / max(tp + fp, 1))
    r1 = float(tp / max(s1, 1))
    f1 = 2.0 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0

    acc = float((tn + tp) / max(total, 1))

    lines = [
        f"{'':<15} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}",
        "",
        f"{'Normal (0)':<15} {p0:<10.4f} {r0:<10.4f} {f0:<10.4f} {s0:<10}",
        f"{'Anomaly (1)':<15} {p1:<10.4f} {r1:<10.4f} {f1:<10.4f} {s1:<10}",
        "",
        f"{'accuracy':<15} {'':<10} {'':<10} {acc:<10.4f} {total:<10}",
        f"{'macro avg':<15} {(p0+p1)/2:<10.4f} {(r0+r1)/2:<10.4f} {(f0+f1)/2:<10.4f} {total:<10}",
        f"{'weighted avg':<15} {(p0*s0+p1*s1)/total:<10.4f} {(r0*s0+r1*s1)/total:<10.4f} {(f0*s0+f1*s1)/total:<10.4f} {total:<10}"
    ]
    return "\n".join(lines)


def roc_curve(y_true, y_score):
    """Tính đường cong ROC (FPR, TPR, thresholds) thuần túy bằng NumPy."""
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    desc_order = np.argsort(scores)[::-1]
    y_sorted = y_t[desc_order]
    scores_sorted = scores[desc_order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(scores_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    total_pos = max(int(np.sum(y_t == 1)), 1)
    total_neg = max(int(np.sum(y_t == 0)), 1)

    fpr = np.r_[0.0, fps / total_neg]
    tpr = np.r_[0.0, tps / total_pos]
    thresholds = np.r_[scores_sorted[threshold_idxs[0]] + 1e-5, scores_sorted[threshold_idxs]]
    return fpr, tpr, thresholds


def precision_recall_curve(y_true, y_score):
    """Tính đường cong Precision-Recall thuần túy bằng NumPy."""
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    desc_order = np.argsort(scores)[::-1]
    y_sorted = y_t[desc_order]
    scores_sorted = scores[desc_order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(scores_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    total_pos = max(int(np.sum(y_t == 1)), 1)
    recall = np.r_[0.0, tps / total_pos]
    precision = np.r_[1.0, tps / np.maximum(tps + fps, 1)]
    thresholds = scores_sorted[threshold_idxs]
    return precision, recall, thresholds


def average_precision_score(y_true, y_score) -> float:
    """Tính Average Precision (AP) thuần túy theo diện tích dưới đường cong PR."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Area under PR curve: tổng delta(Recall) * Precision
    return float(np.sum(np.diff(recall) * precision[1:]))


def permutation_importance_scratch(model, X, y, n_repeats: int = 3, random_state: int = 42):
    """Đo lường Feature Importance bằng Permutation Importance thuần NumPy dựa trên độ suy giảm F1."""
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    base_pred = model.predict(X_arr)
    base_f1 = f1_score(y_arr, base_pred)

    n_features = X_arr.shape[1]
    importances = np.zeros((n_features, n_repeats))

    for f_idx in range(n_features):
        for r in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, f_idx] = rng.permutation(X_perm[:, f_idx])
            pred_perm = model.predict(X_perm)
            importances[f_idx, r] = base_f1 - f1_score(y_arr, pred_perm)

    importances_mean = np.mean(importances, axis=1)
    importances_std = np.std(importances, axis=1)
    return importances_mean, importances_std


# Aliases tương thích ngược cho test suite
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
# 3. HÀM HỖ TRỢ & THỰC THI PIPELINE
# ==============================================================================

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


# ==============================================================================
# CẤU HÌNH MẶC ĐỊNH & LƯỚI SIÊU THAM SỐ (KHÔNG CẦN FILE NGOÀI)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Pure Isolation Forest Pipeline - NASA Shuttle Dataset")
    parser.add_argument("--data_path", type=str, default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_CONFIG["n_estimators"])
    parser.add_argument("--max_samples", type=str, default=str(DEFAULT_CONFIG["max_samples"]))
    parser.add_argument("--max_features", type=float, default=DEFAULT_CONFIG["max_features"])
    parser.add_argument("--contamination", type=str, default=DEFAULT_CONFIG["contamination"])
    parser.add_argument("--test_size", type=float, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--random_state", type=int, default=DEFAULT_CONFIG["random_state"])
    parser.add_argument("--tune", action="store_true", help="Chạy tìm kiếm siêu tham số Stratified 3-Fold CV")
    parser.add_argument("--odds", action="store_true", help="Chế độ ODDS Benchmark quốc tế (loại bỏ Class 4, tỷ lệ anomaly ~7.15%%)")
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

    print("=" * 60)
    print("   THUẦN ISOLATION FOREST PIPELINE (PURE NUMPY)")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"[!] Lỗi: Không tìm thấy file dữ liệu '{data_path}'!")
        sys.exit(1)

    df = pd.read_csv(data_path)
    if args.odds:
        df = df[df["class"] != 4].copy()
        print("[*] Cấu hình ODDS Benchmark: Loại bỏ Class 4 (Chế độ van phụ lớn)")
    df["label"] = (df["class"] != 1).astype(int)
    feature_cols = [c for c in df.columns if c.startswith("att_")]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in ["class", "label"]]

    X, y = df[feature_cols].copy(), df["label"].copy()
    if not args.odds and not os.path.exists("shuttle_preprocessed.csv"):
        df.to_csv("shuttle_preprocessed.csv", index=False)

    print(f"[*] Dữ liệu: {len(df):,} mẫu | {len(feature_cols)} cảm biến | Tỷ lệ bất thường: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=True
    )
    if isinstance(X_train, (pd.DataFrame, pd.Series)):
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert train_indices.isdisjoint(test_indices), "Data leakage: train/test indices overlap!"

    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples : {len(X_test):,}")

    model_contam = "auto" if contamination_arg.lower() == "auto" else float(contamination_arg)
    train_contam = float(y_train.mean())

    if not tune:
        print("\n[*] Huấn luyện mô hình Thuần Isolation Forest (Single Run)...")
        best_model = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples,
            max_features=max_features, contamination=model_contam,
            random_state=random_state
        ).fit(X_train.values)
        best_params = {"n_estimators": n_estimators, "max_samples": max_samples, "max_features": max_features}
    else:
        print("\n[*] Hyperparameter Tuning với Stratified 3-Fold CV...")
        param_configs = PARAM_GRID
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        best_cv_f1, best_params, best_model = -1.0, None, None
        for idx, p in enumerate(param_configs, 1):
            cur_n = int(p.get("n_estimators", 100))
            cur_s = parse_max_samples(p.get("max_samples", 256))
            cur_f = float(p.get("max_features", 1.0))

            cv_f1s = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                fold_contam = "auto" if contamination_arg.lower() == "auto" else float(contamination_arg)
                m = IsolationForest(n_estimators=cur_n, max_samples=cur_s, max_features=cur_f,
                                    contamination=fold_contam, random_state=random_state)
                m.fit(X_train.iloc[tr_idx].values)
                cv_f1s.append(f1_score(y_train.iloc[val_idx].values, m.predict(X_train.iloc[val_idx].values)))

            mean_f1 = float(np.mean(cv_f1s))
            print(f"  - Config {idx}: n_est={cur_n}, max_samples={cur_s}, max_feat={cur_f} -> CV F1: {mean_f1:.4f}")
            if mean_f1 > best_cv_f1:
                best_cv_f1 = mean_f1
                best_params = {"n_estimators": cur_n, "max_samples": cur_s, "max_features": cur_f}

        best_model = IsolationForest(
            n_estimators=best_params["n_estimators"], max_samples=best_params["max_samples"],
            max_features=best_params["max_features"], contamination=model_contam,
            random_state=random_state
        ).fit(X_train.values)
        print(f"[+] Cấu hình tối ưu: {best_params} (CV F1 = {best_cv_f1:.4f})")

    # Điểm Anomaly Score trên tập Train và tập Test (Không rò rỉ nhãn giám sát tập Test)
    train_scores = best_model.anomaly_score(X_train.values)
    test_scores = best_model.anomaly_score(X_test.values)
    c_val = c_factor(best_model.max_samples_actual_)

    # 1. Ngưỡng lý thuyết (Theoretical threshold) từ bài báo gốc Liu et al. (2008)
    th_theoretical = 0.500000

    # 2. Ngưỡng thực nghiệm (Empirical threshold tính từ tập Train)
    th_empirical = float(np.percentile(train_scores, 100.0 * (1.0 - train_contam)))

    # Đánh giá tập Test theo Ngưỡng Lý thuyết
    y_pred_theo = (test_scores >= th_theoretical).astype(int)
    acc_t = accuracy_score(y_test, y_pred_theo)
    bal_t = balanced_accuracy_score(y_test, y_pred_theo)
    prec_t = precision_score(y_test, y_pred_theo)
    rec_t = recall_score(y_test, y_pred_theo)
    f1_t = f1_score(y_test, y_pred_theo)

    # Đánh giá tập Test theo Ngưỡng Thực nghiệm
    y_pred_emp = (test_scores >= th_empirical).astype(int)
    acc_e = accuracy_score(y_test, y_pred_emp)
    bal_e = balanced_accuracy_score(y_test, y_pred_emp)
    prec_e = precision_score(y_test, y_pred_emp)
    rec_e = recall_score(y_test, y_pred_emp)
    f1_e = f1_score(y_test, y_pred_emp)

    # Chỉ số độc lập ngưỡng (Ranking metrics)
    auc_val = roc_auc_score(y_test, test_scores)
    ap_val = average_precision_score(y_test, test_scores)

    dataset_title = "Statlog Shuttle (ODDS Benchmark)" if args.odds else "Statlog Shuttle"
    anomaly_classes_str = "2, 3, 5, 6, 7" if args.odds else "2, 3, 4, 5, 6, 7"
    contam_str = "auto" if contamination_arg.lower() == "auto" else str(contamination_arg)

    print("\n" + "=" * 60)
    print("ISOLATION FOREST - MAIN EXPERIMENT")
    print("=" * 60)
    print(f"Dataset             : {dataset_title}")
    print(f"Samples             : {len(df):,}")
    print(f"Features            : {len(feature_cols)}")
    print(f"\nNormal class        : 1")
    print(f"Anomaly classes     : {anomaly_classes_str}")
    print(f"Anomaly rate        : {y.mean():.2%}")
    print(f"\nTrain samples       : {len(X_train):,}")
    print(f"Test samples        : {len(X_test):,}")
    print(f"\nRandom state        : {random_state}")
    print(f"n_estimators        : {best_params['n_estimators']}")
    print(f"max_samples         : {best_params['max_samples']}")
    print(f"max_features        : {best_params['max_features']}")
    print(f"max_depth           : {best_model.max_depth}")
    print(f"contamination       : {contam_str}")
    print(f"\nc({best_model.max_samples_actual_})              : {c_val:.7f}")
    print(f"\nThreshold type      : theoretical")
    print(f"Threshold           : {th_theoretical:.6f}")
    print("-" * 60)
    print(f"Accuracy            : {acc_t:.4f}")
    print(f"Balanced Accuracy   : {bal_t:.4f}")
    print(f"Precision           : {prec_t:.4f}")
    print(f"Recall              : {rec_t:.4f}")
    print(f"F1                  : {f1_t:.4f}")
    print(f"ROC-AUC             : {auc_val:.4f}")
    print(f"Average Precision   : {ap_val:.4f}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS - EMPIRICAL THRESHOLD")
    print("=" * 60)
    print(f"Threshold type       : empirical (train quantile: 1 - contam)")
    print(f"Threshold            : {th_empirical:.6f}")
    print(f"Theoretical threshold: {th_theoretical:.6f}")
    print("-" * 60)
    print(f"Accuracy             : {acc_e:.4f}")
    print(f"Balanced Accuracy    : {bal_e:.4f}")
    print(f"Precision            : {prec_e:.4f}")
    print(f"Recall               : {rec_e:.4f}")
    print(f"F1                   : {f1_e:.4f}")
    print(f"ROC-AUC              : {auc_val:.4f}")
    print(f"Average Precision    : {ap_val:.4f}")
    print("-" * 60)

    print("\n" + "=" * 68)
    print(f"      SO SÁNH 2 NGƯỠNG TRÊN TẬP TEST (N={len(X_test):,})")
    print("=" * 68)
    print(f"{'Chỉ số':<22} | {'Ngưỡng Lý thuyết (0.5000)':<25} | {'Ngưỡng Thực nghiệm (' + f'{th_empirical:.4f}' + ')':<25}")
    print("-" * 68)
    print(f"{'Overall Accuracy':<22} | {acc_t:<25.4f} | {acc_e:<25.4f}")
    print(f"{'Balanced Accuracy':<22} | {bal_t:<25.4f} | {bal_e:<25.4f}")
    print(f"{'Precision (Anomaly)':<22} | {prec_t:<25.4f} | {prec_e:<25.4f}")
    print(f"{'Recall (Anomaly)':<22} | {rec_t:<25.4f} | {rec_e:<25.4f}")
    print(f"{'F1-Score (Anomaly)':<22} | {f1_t:<25.4f} | {f1_e:<25.4f}")
    print(f"{'ROC-AUC':<22} | {auc_val:<25.4f} | {auc_val:<25.4f}")
    print(f"{'Average Precision':<22} | {ap_val:<25.4f} | {ap_val:<25.4f}")
    print("-" * 68)

    print("\n[*] Báo cáo Phân loại Chi tiết (Ngưỡng Lý thuyết 0.5000):")
    print(classification_report(y_test, y_pred_theo))
    cm_t = confusion_matrix(y_test, y_pred_theo)
    print(f"[*] Confusion Matrix: TN={cm_t[0,0]}  FP={cm_t[0,1]}  FN={cm_t[1,0]}  TP={cm_t[1,1]}")

    print("\n[*] Báo cáo Phân loại Chi tiết (Ngưỡng Thực nghiệm):")
    print(classification_report(y_test, y_pred_emp))
    cm_e = confusion_matrix(y_test, y_pred_emp)
    print(f"[*] Confusion Matrix: TN={cm_e[0,0]}  FP={cm_e[0,1]}  FN={cm_e[1,0]}  TP={cm_e[1,1]}")

    # Ghi nhận kết quả chuẩn hóa vào metrics.csv (23 trường chuẩn xác)
    dataset_name = "Statlog_Shuttle_ODDS" if args.odds else "Statlog_Shuttle"
    mode_name = "ODDS_Benchmark" if args.odds else ("Full_58k_Tuned" if tune else "Full_58k_Base")
    anomaly_classes_metric = "2,3,5,6,7" if args.odds else "2,3,4,5,6,7"

    row_theo = {
        "mode": mode_name,
        "dataset": dataset_name,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "normal_classes": "1",
        "anomaly_classes": anomaly_classes_metric,
        "random_state": random_state,
        "n_estimators": best_params["n_estimators"],
        "max_samples": best_params["max_samples"],
        "max_features": best_params["max_features"],
        "max_depth": best_model.max_depth,
        "contamination": contam_str,
        "threshold_type": "theoretical",
        "threshold": round(th_theoretical, 6),
        "accuracy": round(acc_t, 4),
        "balanced_accuracy": round(bal_t, 4),
        "precision": round(prec_t, 4),
        "recall": round(rec_t, 4),
        "f1": round(f1_t, 4),
        "roc_auc": round(auc_val, 4),
        "average_precision": round(ap_val, 4),
    }

    row_emp = {
        "mode": mode_name,
        "dataset": dataset_name,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "normal_classes": "1",
        "anomaly_classes": anomaly_classes_metric,
        "random_state": random_state,
        "n_estimators": best_params["n_estimators"],
        "max_samples": best_params["max_samples"],
        "max_features": best_params["max_features"],
        "max_depth": best_model.max_depth,
        "contamination": contam_str,
        "threshold_type": "empirical",
        "threshold": round(th_empirical, 6),
        "accuracy": round(acc_e, 4),
        "balanced_accuracy": round(bal_e, 4),
        "precision": round(prec_e, 4),
        "recall": round(rec_e, 4),
        "f1": round(f1_e, 4),
        "roc_auc": round(auc_val, 4),
        "average_precision": round(ap_val, 4),
    }

    metrics_df = pd.DataFrame([row_theo, row_emp])
    metrics_path = "metrics.csv"
    if not os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path, index=False)
    else:
        metrics_df.to_csv(metrics_path, mode="a", header=False, index=False)

    # Suy luận thử nghiệm qua Pipeline
    pipe = Pipeline(model=best_model)
    sample_s = float(pipe.anomaly_score(X_test.iloc[[0]].values)[0])
    print(f"\n[+] Suy luận mẫu đầu: Score = {sample_s:.4f} -> {'BẤT THƯỜNG (1)' if sample_s >= th_theoretical else 'BÌNH THƯỜNG (0)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

