"""
WEIGHTS & CONFIGURATION — ISOLATION FOREST PROJECT
====================================================
File tập trung toàn bộ hằng số, trọng số, và cấu hình siêu tham số.
Tách ra để dễ điều chỉnh mà không cần đụng vào logic chính.

Cấu trúc:
  1. DEFAULT_CONFIG       — Cấu hình chạy mặc định (single-run pipeline)
  2. PARAM_GRID           — Lưới siêu tham số cho Unsupervised K-Fold CV tuning
  3. THRESHOLD_CONFIG     — Các mức ngưỡng phát hiện bất thường
  4. FEATURE_IMPORTANCE   — Trọng số tổng hợp Feature Importance
  5. DATA_CONFIG          — Cấu hình dataset (cột cảm biến, cột nhãn, v.v.)
  6. MODEL_MATH           — Hằng số toán học cốt lõi của thuật toán
  7. EVALUATION           — Ngưỡng đánh giá mức độ cảnh báo anomaly
  8. WEIGHTS_IO           — Hàm lưu/tải trọng số mô hình ra/vào thư mục weights/
"""

# ==============================================================================
# 1. CẤU HÌNH CHẠY MẶC ĐỊNH (SINGLE-RUN PIPELINE)
# ==============================================================================

DEFAULT_CONFIG = {
    # --- Dữ liệu ---
    "data_path":     "shuttle.csv",     # Đường dẫn dataset
    "test_size":     0.2,               # 20% tập kiểm thử

    # --- Mô hình ---
    "n_estimators":  100,               # Số cây iTree trong rừng
    "max_samples":   256,               # Kích thước mẫu con psi (Liu et al., 2008 mặc định 256)
    "max_features":  1.0,               # Tỷ lệ đặc trưng Feature Bagging (1.0 = dùng tất cả)
    "contamination": "auto",            # "auto" → ngưỡng lý thuyết 0.5; float ∈ (0,0.5) → phân vị

    # --- Tái lập ---
    "random_state":  42,                # Seed ngẫu nhiên
}


# ==============================================================================
# 2. LƯỚI SIÊU THAM SỐ — UNSUPERVISED K-FOLD CV TUNING  (--tune)
# ==============================================================================
# Tiêu chí lựa chọn: Cấu hình có Score Spread (Std anomaly score) LỚN NHẤT
# trên tập validation → phân biệt tốt nhất normal vs anomaly mà không cần nhãn.

PARAM_GRID = [
    {"n_estimators": 100, "max_samples": 128, "max_features": 1.0},   # Baseline nhỏ
    {"n_estimators": 100, "max_samples": 256, "max_features": 1.0},   # Mặc định Liu et al. (2008)
    {"n_estimators": 150, "max_samples": 256, "max_features": 0.8},   # Feature Bagging 80%
    {"n_estimators": 200, "max_samples": 256, "max_features": 1.0},   # Nhiều cây hơn
]

# Số fold trong cross-validation không giám sát
KFOLD_N_SPLITS: int = 3


# ==============================================================================
# 3. CẤU HÌNH NGƯỠNG PHÁT HIỆN BẤT THƯỜNG (UNSUPERVISED THRESHOLDS)
# ==============================================================================

THRESHOLD_CONFIG = {
    # Ngưỡng lý thuyết từ bài báo gốc Liu et al. (2008): s >= 0.5 → anomaly
    "theoretical":   0.5,

    # Các phân vị trên tập TRAIN (ước lượng contamination không giám sát)
    "top5_pct":      95.0,   # Phân vị 95 → ~5%  mẫu được coi là bất thường
    "top1_pct":      99.0,   # Phân vị 99 → ~1%  mẫu cực kỳ bất thường
    "top01_pct":     99.9,   # Phân vị 99.9 → 0.1% cực đoan

    # Ngưỡng thống kê: Mean + k*Std (Gaussian approximation)
    "stat_sigma_k":   2.0,   # k = 2 → ~2.275% tail dưới phân phối chuẩn
}

# Nhãn mức cảnh báo hiển thị trong báo cáo RCA
ALERT_LEVELS = {
    "high_danger":  0.60,    # score >= 0.60 → "NGUY HIỂM CAO"
    "warning":      0.50,    # score >= 0.50 → "CẢNH BÁO BẤT THƯỜNG"
}


# ==============================================================================
# 4. TRỌNG SỐ TỔNG HỢP — FEATURE IMPORTANCE
# ==============================================================================
# Chỉ số quan trọng tổng hợp = w_corr * |Pearson r| + w_pct * PctRankDev_norm
# Tổng trọng số phải bằng 1.0.

FEATURE_IMPORTANCE_WEIGHTS = {
    "pearson_abs_corr":  0.5,   # Trọng số tương quan Pearson tuyệt đối |r|
    "pct_rank_dev_norm": 0.5,   # Trọng số Median Percentile Rank Deviation chuẩn hóa
}

# Ngưỡng Stability Check (Spearman rho giữa 2 sub-sample anomaly):
STABILITY_RHO_THRESHOLD: float = 0.6   # rho < 0.6 → cảnh báo ranking không ổn định

# Phạm vi cực đoan dùng trong Co-dominant Feature Detection (RCA)
CO_DOMINANT_PCT_MIN:  float = 49.0     # Pct deviation >= 49.0% được coi là cực đoan
CO_DOMINANT_MARGIN:   float = 0.15     # Sai lệch tối đa so với max để vẫn là "đồng cực đoan"

# Hệ số pha trộn IQR tie-breaker (không để IQR át pct_dev chính)
IQR_TIEBREAK_SCALE:   float = 1e-4    # composite = pct_dev + min(iqr_dev, 1000) * 1e-4
IQR_TIEBREAK_CAP:     float = 1000.0  # Giới hạn trên cho iqr_dev trước khi nhân scale


# ==============================================================================
# 5. CẤU HÌNH DỮ LIỆU — NASA SHUTTLE TELEMETRY
# ==============================================================================
# UCI Statlog Shuttle: 10 cột = 9 cảm biến thực sự + 1 nhãn lớp (1-7).
# feat_10 là nhãn UCI — KHÔNG được đưa vào model khi huấn luyện.

N_SENSOR_COLS:  int = 9                                              # Số cột cảm biến thực sự
SENSOR_COLS         = [f"feat_{i+1}" for i in range(N_SENSOR_COLS)] # feat_1 → feat_9
LABEL_COL:  str = "feat_10"                                          # Cột nhãn UCI lớp 1-7

# Nhãn nhị phân cho External Validation:
# Class 1 (Rad Flow) = bình thường (78.6% dữ liệu); Class 2-7 = bất thường
NORMAL_CLASS: int = 1   # Giá trị nhãn UCI ứng với trạng thái bình thường


# ==============================================================================
# 6. HẰNG SỐ TOÁN HỌC CỐT LÕI — ISOLATION FOREST
# ==============================================================================

# Euler-Mascheroni constant gamma — dùng trong c_factor(n) (Liu et al., 2008)
EULER_MASCHERONI: float = 0.5772156649015329

# Giá trị max_samples mặc định khi "auto" (theo bài báo gốc Liu et al., 2008)
DEFAULT_MAX_SAMPLES_AUTO: int = 256

# Quy ước nhãn dự đoán (khác scikit-learn dùng -1/1)
LABEL_ANOMALY: int = 1    # Bất thường
LABEL_NORMAL:  int = 0    # Bình thường

# Ngưỡng offset mặc định (khi contamination="auto")
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_OFFSET:    float = -0.5


# ==============================================================================
# 7. CẤU HÌNH ĐÁNH GIÁ (EVALUATION)
# ==============================================================================

# Permutation Importance
PERMUTATION_N_REPEATS: int = 5          # Số lần hoán vị để ước lượng trung bình
PERMUTATION_SCORING:   str = "roc_auc"  # "roc_auc" hoặc "f1"

# Số mẫu bất thường tối thiểu để chạy Stability Check
STABILITY_MIN_ANOMALIES: int = 10

# Top-K mẫu bất thường nhất xuất trong báo cáo RCA
RCA_TOP_K: int = 10

# Khoảng chuẩn hóa PctRankDev: max có thể là 50.0 → [0, 50]
PCT_DEV_MAX_RANGE: float = 50.0


# ==============================================================================
# 8. WEIGHTS I/O — LƯU / TẢI TRỌNG SỐ MÔ HÌNH
# ==============================================================================
# Không có code nào tự chạy khi import module này.
# Gọi save_weights(model, params, name) sau khi train xong để lưu.
# Gọi load_weights(name) để tải lại trọng số từ file.
#
# Định dạng file (mỗi name sinh ra 3 file):
#   weights/<name>.json  — siêu tham số + threshold + toàn bộ cây (JSON)
#   weights/<name>.npz   — split_feat/split_val mỗi node (NumPy compressed)
#   weights/<name>.txt   — tóm tắt dạng văn bản (human-readable)
#
# Tên quy ước:
#   baseline_weights     → DEFAULT_CONFIG, không tune
#   best_model_weights   → cấu hình tốt nhất từ 3-Fold CV
# ==============================================================================

import os
import json


# Thư mục chứa tất cả file trọng số (tuyệt đối, tính từ vị trí file này)
WEIGHTS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


# ------------------------------------------------------------------------------
# Helpers nội bộ — serialize / deserialize cây đệ quy (không dùng pickle)
# ------------------------------------------------------------------------------

def _serialize_node(node):
    """Chuyển đổi một Node thành dict (đệ quy, không phụ thuộc pickle)."""
    if node is None:
        return None
    return {
        "is_leaf":     node.is_leaf,
        "size":        node.size,
        "split_feat":  node.split_feat,
        "split_val":   node.split_val,
        "leaf_adj":    node.leaf_adj,
        "left":        _serialize_node(node.left),
        "right":       _serialize_node(node.right),
    }


def _deserialize_node(d, Node):
    """Tái tạo Node từ dict (đệ quy)."""
    if d is None:
        return None
    node = Node.__new__(Node)
    node.is_leaf    = d["is_leaf"]
    node.size       = d["size"]
    node.split_feat = d["split_feat"]
    node.split_val  = d["split_val"]
    node.leaf_adj   = d["leaf_adj"]
    node.left       = _deserialize_node(d["left"],  Node)
    node.right      = _deserialize_node(d["right"], Node)
    return node


import numpy as _np


def _collect_splits(node, feats: list, vals: list) -> None:
    """Thu thập (split_feat, split_val) từ mọi nút trong — đệ quy."""
    if node is None or node.is_leaf:
        return
    feats.append(node.split_feat if node.split_feat is not None else -1)
    vals.append(node.split_val   if node.split_val  is not None else 0.0)
    _collect_splits(node.left,  feats, vals)
    _collect_splits(node.right, feats, vals)


def _build_payload(model, params: dict) -> dict:
    """Đóng gói toàn bộ model thành dict thuần Python (JSON-serializable)."""
    trees_json = [_serialize_node(t.root) for t in model.trees]

    # Lấy random_state dạng int (xử lý cả np.RandomState object)
    rs = model.random_state
    rs_int = int(rs) if isinstance(rs, (int, float)) else 42

    return {
        "params": {
            "n_estimators": int(params.get("n_estimators", model.n_estimators)),
            "max_samples":  params.get("max_samples",       model.max_samples),
            "max_features": float(params.get("max_features", model.max_features)),
            "contamination": str(model.contamination),
            "random_state":  rs_int,
        },
        "model_state": {
            "n_features_in_":      model.n_features_in_,
            "max_samples_actual_": model.max_samples_actual_,
            "max_depth":           model.max_depth,
            "c_psi_":              model.c_psi_,
            "threshold_":          model.threshold_,
            "offset_":             model.offset_,
        },
        "trees": trees_json,
    }


def save_weights(model, params: dict, name: str = "best_model_weights",
                 output_dir: str = WEIGHTS_DIR) -> None:
    """
    Lưu trọng số mô hình ra thư mục weights/ với tên tùy chọn.

    Tham số:
    ---------
    model      : IsolationForest đã được fit().
    params     : dict siêu tham số (n_estimators, max_samples, max_features).
    name       : Prefix tên file, ví dụ 'best_model_weights' hoặc 'baseline_weights'.
    output_dir : Thư mục đích, mặc định là weights/ trong thư mục dự án.

    File được tạo:
    ---------------
    <name>.json  — metadata + cấu trúc toàn bộ cây (JSON, human-readable).
    <name>.npz   — split arrays cho inference nhanh (NumPy compressed).
    <name>.txt   — tóm tắt dạng văn bản.
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = _build_payload(model, params)
    p  = payload["params"]
    ms = payload["model_state"]
    trees_json = payload["trees"]

    # 1. JSON — cấu trúc đầy đủ
    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 2. NPZ — split arrays tất cả cây (inference nhanh)
    npz_path = os.path.join(output_dir, f"{name}.npz")
    npz_dict: dict = {}
    for i, tree in enumerate(model.trees):
        feats: list = []; vals: list = []
        _collect_splits(tree.root, feats, vals)
        npz_dict[f"tree_{i}_feats"] = _np.array(feats, dtype=_np.int32)
        npz_dict[f"tree_{i}_vals"]  = _np.array(vals,  dtype=_np.float64)
    _np.savez_compressed(npz_path, **npz_dict)

    # 3. TXT — tóm tắt
    txt_path = os.path.join(output_dir, f"{name}.txt")
    lines = [
        "=" * 60,
        f"WEIGHTS: {name.upper()}",
        "=" * 60, "",
        "[Sieu tham so]",
        f"  n_estimators  : {p['n_estimators']}",
        f"  max_samples   : {p['max_samples']}",
        f"  max_features  : {p['max_features']}",
        f"  contamination : {p['contamination']}",
        f"  random_state  : {p['random_state']}", "",
        "[Trang thai mo hinh]",
        f"  n_features_in_      : {ms['n_features_in_']}",
        f"  max_samples_actual_ : {ms['max_samples_actual_']}",
        f"  max_depth           : {ms['max_depth']}",
        f"  c_psi_              : {ms['c_psi_']:.6f}",
        f"  threshold_          : {ms['threshold_']:.6f}",
        f"  offset_             : {ms['offset_']:.6f}", "",
        f"[So cay da luu] : {len(trees_json)}", "",
        "Tao boi: save_weights.py",
        "=" * 60,
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[weights] Da luu '{name}':")
    print(f"  JSON : {json_path}")
    print(f"  NPZ  : {npz_path}")
    print(f"  TXT  : {txt_path}")


def load_weights(name: str = "best_model_weights", input_dir: str = WEIGHTS_DIR):
    """
    Tải trọng số từ <name>.json và tái tạo IsolationForest (không fit lại).

    Tham số:
    ---------
    name      : Prefix tên file ('best_model_weights' hoặc 'baseline_weights').
    input_dir : Thư mục chứa file, mặc định là weights/.

    Trả về:
    --------
    (model, params) : IsolationForest phục hồi đầy đủ + dict siêu tham số.

    Ném:
    -----
    FileNotFoundError nếu file JSON chưa tồn tại.
    """
    from model import IsolationForest, IsolationTree, Node

    json_path = os.path.join(input_dir, f"{name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"[weights] Chua tim thay: {json_path}\n"
            "  -> Hay chay save_weights.py mot lan de sinh ra file nay."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    p  = payload["params"]
    ms = payload["model_state"]

    # Tái tạo IsolationForest (không gọi fit)
    model = IsolationForest(
        n_estimators  = int(p["n_estimators"]),
        max_samples   = p["max_samples"],
        max_features  = float(p["max_features"]),
        contamination = str(p["contamination"]),
        random_state  = int(p["random_state"]) if isinstance(p["random_state"], (int, float)) else 42,
    )
    model.n_features_in_      = ms["n_features_in_"]
    model.max_samples_actual_ = ms["max_samples_actual_"]
    model.max_depth           = ms["max_depth"]
    model.c_psi_              = ms["c_psi_"]
    model.threshold_          = ms["threshold_"]
    model.offset_             = ms["offset_"]

    # Tái tạo từng IsolationTree từ dict đệ quy
    trees = []
    for tree_dict in payload["trees"]:
        tree = IsolationTree.__new__(IsolationTree)
        tree.max_depth       = ms["max_depth"]
        tree.features_subset = None
        tree.rng             = None  # type: ignore[assignment]
        tree.root            = _deserialize_node(tree_dict, Node)
        trees.append(tree)
    model.trees = trees

    print(f"[weights] Da tai '{name}' tu: {json_path}")
    print(f"          ({len(trees)} cay | threshold={model.threshold_:.4f} | c_psi={model.c_psi_:.4f})")
    return model, dict(p)


# Backward-compatible aliases
def save_best_model_weights(model, params: dict, output_dir: str = WEIGHTS_DIR) -> None:
    """Alias tương thích ngược. Dùng save_weights(model, params, name) thay thế."""
    save_weights(model, params, name="best_model_weights", output_dir=output_dir)


def load_best_model_weights(input_dir: str = WEIGHTS_DIR):
    """Alias tương thích ngược. Dùng load_weights(name) thay thế."""
    return load_weights(name="best_model_weights", input_dir=input_dir)
