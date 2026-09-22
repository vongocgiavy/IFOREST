# AI Prompting Guide — Isolation Forest Project
### Cách tôi dùng AI để tạo ra toàn bộ project này từ đầu

> Đây là bản ghi lại **từng bước prompting theo đúng thứ tự** mà tôi đã dùng để xây dựng project.
> Bất kỳ ai dùng một tài khoản AI mới hoàn toàn, không có lịch sử hội thoại, làm theo đúng trình tự này đều có thể tái tạo lại project **100% y hệt**.

---

## Phần 1 — Xác lập bối cảnh: bài toán, dataset, ràng buộc kỹ thuật

Đây là bước đầu tiên và quan trọng nhất. Toàn bộ project được xây trên nền tảng của hai prompt này — nếu bỏ qua hoặc viết mơ hồ thì AI sẽ tự chọn thư viện tuỳ ý, tự đặt tên hàm tuỳ ý, và project sẽ ra khác hoàn toàn.

### Prompt 1.1 — Giới thiệu bài toán

Gửi nguyên đoạn này vào chat AI, không thêm không bớt:

```
Tôi muốn xây dựng một project phát hiện bất thường (Anomaly Detection)
trên dataset NASA Shuttle Telemetry (UCI Statlog Shuttle, 58,000 mẫu, 9 cảm biến).

Yêu cầu kỹ thuật bắt buộc:
- Thuật toán: Isolation Forest theo đúng bài báo gốc Liu, Ting & Zhou (2008) — ICDM 2008, pp. 413-422
- Ngôn ngữ: Python 3
- Thư viện: CHỈ DÙNG NumPy và Pandas. KHÔNG ĐƯỢC dùng scikit-learn hay bất kỳ thư viện ML
  bậc cao nào. Mọi thứ phải tự implement từ đầu bằng Pure NumPy.
- Mô hình hoàn toàn không giám sát (Unsupervised) — không được dùng nhãn khi huấn luyện.
- Dataset có 10 cột: feat_1 đến feat_9 là cảm biến (input X), feat_10 là nhãn UCI
  (chỉ dùng để đánh giá sau khi đã predict xong — label leakage phải bằng 0%).

Hãy xác nhận bạn hiểu các ràng buộc trên trước khi chúng ta bắt đầu.
```

### Prompt 1.2 — Đặt cấu trúc project

Ngay sau khi AI xác nhận, gửi tiếp prompt này để khóa cấu trúc file:

```
Tốt. Project sẽ có cấu trúc file sau — hãy ghi nhớ và giữ nguyên trong suốt quá trình:

iforest/
├── model.py              # Class Node, IsolationTree, IsolationForest — thuần NumPy
├── weights.py            # Hằng số cấu hình toàn cục + hàm save_weights() / load_weights()
├── run_pipeline.py       # Pipeline điều phối: load data, train, evaluate, RCA, CLI
├── save_weights.py       # Script một lần: train -> lưu trọng số ra thư mục weights/
├── shuttle.csv           # Dataset đã có sẵn
├── requirements.txt
├── .gitignore
├── weights/              # Thư mục chứa model đã train (tự tạo khi chạy save_weights.py)
│   ├── baseline_weights.json / .npz / .txt
│   └── best_model_weights.json / .npz / .txt
└── tests/
    └── test_pipeline.py  # 21 unit tests bằng pytest

Xác nhận cấu trúc này và chúng ta sẽ bắt đầu viết từng file một.
```

---

## Phần 2 — `model.py`: Implement thuật toán từ đầu

Đây là file cốt lõi. Cần 2 prompt — prompt đầu tạo khung, prompt sau thêm validation đầy đủ.

### Prompt 2.1 — Viết toàn bộ model.py

```
Hãy viết file model.py — implement Isolation Forest hoàn toàn từ đầu bằng Pure NumPy.
Không được import sklearn hay bất kỳ thư viện ML nào.

Cần 4 thành phần, theo đúng thứ tự này:

1. Hàm c_factor(n: int) -> float
   Tính độ dài đường đi trung bình tìm kiếm không thành công trong BST, theo công thức
   Liu et al. (2008): 2*(ln(n-1) + gamma) - 2*(n-1)/n
   Trong đó gamma = 0.5772156649015329 (hằng số Euler-Mascheroni)
   Edge case: c_factor(0) = 0.0, c_factor(1) = 0.0, c_factor(2) = 1.0

2. Class Node
   - Dùng __slots__ = ['left', 'right', 'split_feat', 'split_val', 'size', 'is_leaf', 'leaf_adj']
   - __init__(self, left=None, right=None, split_feat=None, split_val=None, size=0, is_leaf=False)
   - leaf_adj = c_factor(size) nếu is_leaf=True, = 0.0 nếu is_leaf=False

3. Class IsolationTree
   - __init__(self, max_depth, features_subset=None, rng=None, random_state=42)
     Nếu rng không None thì dùng rng đó. Nếu random_state không None thì tạo RandomState(random_state).
   - fit(self, X: np.ndarray) -> self
     Gọi _build_tree(X, current_depth=0), lưu vào self.root
   - _build_tree(self, X, current_depth) -> Node
     Trả về Node lá nếu current_depth >= max_depth hoặc n_samples <= 1.
     Nếu features_subset không None thì chỉ xét các cột đó.
     Chọn ngẫu nhiên 1 feature có min < max (valid feature), chọn split_val ngẫu nhiên đều
     trong [feat_min, feat_max]. Trả về Node lá nếu split chia không được (n_left=0 hoặc =n_samples).
   - compute_path_lengths(self, X: np.ndarray) -> np.ndarray
     KHÔNG dùng đệ quy. Dùng stack lặp: stack chứa tuple (indices, node, depth).
     Khi node là lá: lengths[indices] = depth + node.leaf_adj
     Khi node là nhánh: tính mask, push nhánh trái và nhánh phải vào stack.

4. Class IsolationForest
   - __init__(self, n_estimators=100, max_samples="auto", max_features=1.0,
               contamination="auto", random_state=42)
   - Attribute sau khi fit: trees (list IsolationTree), max_samples_actual_,
     c_psi_, threshold_, offset_, n_features_in_, max_depth
   - fit(self, X, y=None) -> self
     Khi max_samples="auto": psi = max(2, min(256, n_samples))
     Khi max_samples là int: psi = max(2, min(max_samples, n_samples))
     Khi max_samples là float (0,1]: psi = max(2, int(round(max_samples * n_samples)))
     max_depth = ceil(log2(psi))
     c_psi_ = c_factor(psi)
     Feature bagging: k_feat = max(1, round(max_features * n_features))
     Mỗi cây: subsample ngẫu nhiên không hoàn lại psi mẫu, chọn k_feat features ngẫu nhiên
     Khi contamination="auto": threshold_=0.5, offset_=-0.5
     Khi contamination là float: tính np.percentile(train_scores, 100*(1-contamination))
   - anomaly_score(self, X) -> np.ndarray
     s(x) = 2^(-E[h(x)] / c_psi_), khoảng [0,1], càng gần 1 càng bất thường
   - score_samples(self, X) -> np.ndarray: trả về -anomaly_score(X)
   - decision_function(self, X) -> np.ndarray: trả về threshold_ - anomaly_score(X)
   - predict(self, X, threshold=None) -> np.ndarray
     Trả về 1 (bất thường) hoặc 0 (bình thường). KHÔNG dùng -1/1 như sklearn.
   - fit_predict(self, X, y=None, threshold=None) -> np.ndarray
   - property estimators_: trả về self.trees (tương thích sklearn convention)

Thêm alias ở cuối file: IsolationForestScratch = IsolationForest
Docstring tiếng Việt cho mọi class và method.
```

### Prompt 2.2 — Thêm _validate_X() và validation tham số

```
Thêm method _validate_X(self, X, check_features=False) -> np.ndarray vào class IsolationForest.

Method này làm:
- Nếu X có attribute .values (pandas) thì lấy .values, sau đó asarray dtype=float64
- Kiểm tra ndim == 2, nếu không raise ValueError
- Kiểm tra shape[0] == 0 -> raise ValueError "n_samples == 0"
- Kiểm tra shape[1] == 0 -> raise ValueError "n_features == 0"
- Kiểm tra np.all(np.isfinite(X_arr)) -> nếu False raise ValueError "chứa NaN hoặc Inf"
- Nếu check_features=True và n_features_in_ is None -> raise ValueError "chưa fit"
- Nếu check_features=True và shape[1] != n_features_in_ -> raise ValueError kèm số cột mismatch
- Trả về X_arr

Đồng thời thêm validation vào __init__ của IsolationForest:
- n_estimators: isinstance bool -> raise, không phải int -> raise, <= 0 -> raise
  Message: "n_estimators must be > 0"
- max_samples != "auto":
  isinstance bool -> raise "cannot be boolean"
  không phải int/float hoặc <= 0 -> raise "must be > 0"
  là int và < 2 -> raise "as integer must be >= 2"
  là float và > 1.0 -> raise "as float must be in (0, 1]"
- max_features: isinstance bool -> raise, không phải số hoặc không trong (0,1] -> raise
  Message: "max_features must be in (0, 1]"
- contamination != "auto":
  isinstance bool -> raise, không phải số hoặc không trong (0, 0.5) -> raise
  Message: "contamination must be in (0, 0.5)"

Gọi _validate_X(X, check_features=False) trong fit() và _validate_X(X, check_features=True)
trong anomaly_score().
```

---

## Phần 3 — `weights.py`: Cấu hình tập trung và I/O model

### Prompt 3.1 — Viết toàn bộ weights.py

```
Viết file weights.py — module tập trung toàn bộ hằng số cấu hình và logic lưu/tải model.
Không import sklearn. Import: json, os, math, numpy as np, và from model import các class cần thiết.

Phần 1 — DEFAULT_CONFIG (dict):
{
    "data_path":     "shuttle.csv",
    "test_size":     0.2,
    "n_estimators":  100,
    "max_samples":   256,
    "max_features":  1.0,
    "contamination": "auto",
    "random_state":  42,
}

Phần 2 — PARAM_GRID (list of dict), 4 cấu hình theo thứ tự:
[
    {"n_estimators": 100, "max_samples": 128, "max_features": 1.0},
    {"n_estimators": 100, "max_samples": 256, "max_features": 1.0},
    {"n_estimators": 150, "max_samples": 256, "max_features": 0.8},
    {"n_estimators": 200, "max_samples": 256, "max_features": 1.0},
]

KFOLD_N_SPLITS: int = 3

Phần 3 — THRESHOLD_CONFIG (dict):
{
    "theoretical":  0.5,
    "top5_pct":     95.0,
    "top1_pct":     99.0,
    "top01_pct":    99.9,
    "stat_sigma_k": 2.0,
}

ALERT_LEVELS (dict):
{
    "high_danger": 0.60,
    "warning":     0.50,
}

Phần 4 — FEATURE_IMPORTANCE (dict): 9 key feat_1 đến feat_9, giá trị float 0.0 (placeholder)

Phần 5 — DATA_CONFIG (dict):
{
    "feature_cols": ["feat_1","feat_2","feat_3","feat_4","feat_5","feat_6","feat_7","feat_8","feat_9"],
    "label_col":    "feat_10",
    "n_features":   9,
    "n_samples":    58000,
}

Phần 6 — MODEL_MATH (dict):
{
    "euler_mascheroni": 0.5772156649015329,
    "default_psi":      256,
    "anomaly_threshold": 0.5,
}

Phần 7 — EVALUATION (dict):
{
    "high_danger": 0.60,
    "warning":     0.50,
}

Phần 8 — Hàm save_weights(model, filepath_prefix: str):
  Lưu ra 3 file: {filepath_prefix}.json, {filepath_prefix}.npz, {filepath_prefix}.txt

  File JSON: dict với keys:
    "params": {n_estimators, max_samples_actual_, max_features, contamination, random_state,
               max_depth, c_psi_, threshold_, offset_, n_features_in_}
    "trees": list, mỗi phần tử là list of dict (pre-order DFS traversal của cây)
    Mỗi dict node: {"is_leaf": bool, "size": int, "split_feat": int|null, "split_val": float|null}

  File NPZ: lưu params dưới dạng numpy arrays để inference nhanh
    Mỗi cây i lưu: "tree_{i}_feat" (array split_feat), "tree_{i}_val" (array split_val),
    "tree_{i}_size" (array size), "tree_{i}_leaf" (array is_leaf, bool)
    Thứ tự node theo pre-order DFS

  File TXT: report con người đọc được, ghi:
    - Tên model, ngày giờ lưu
    - Params: n_estimators, max_samples_actual_, max_depth, c_psi_, threshold_
    - Thống kê: n_features_in_, tổng số node

Phần 9 — Hàm load_weights(filepath_prefix: str) -> tuple(model, params_dict):
  Đọc file JSON, phục hồi lại đầy đủ IsolationForest:
  - Tạo IsolationForest với params từ JSON
  - Phục hồi từng cây từ list of dict bằng cách xây lại cây đệ quy (dùng iterator qua list)
  - Gán lại max_samples_actual_, c_psi_, threshold_, offset_, n_features_in_, max_depth
  - Trả về (model, params_dict)

Comment phân chia rõ từng section bằng dấu ==. Docstring tiếng Việt.
```

### Prompt 3.2 — Đảm bảo save/load chính xác

```
Kiểm tra lại hàm save_weights và load_weights:

Yêu cầu quan trọng về serialization cây:
- Pre-order DFS: mỗi node được thêm vào list TRƯỚC khi đệ quy vào con trái và con phải
- Node lá: {"is_leaf": true, "size": n, "split_feat": null, "split_val": null}
- Node trong: {"is_leaf": false, "size": n, "split_feat": i, "split_val": v}

Yêu cầu về load_weights:
- Dùng iterator (iter + next) để duyệt list node theo đúng thứ tự pre-order
- Hàm phục hồi đệ quy: lấy next(it) làm node hiện tại, nếu không phải lá thì
  đệ quy xây nhánh trái trước rồi nhánh phải sau
- Sau khi load xong, model.anomaly_score(X) phải cho kết quả sai lệch < 1e-10
  so với model gốc trước khi save

Nếu file .json không tồn tại thì raise FileNotFoundError với message rõ ràng.
```

---

## Phần 4 — `run_pipeline.py`: Pipeline điều phối chính

File này lớn nhất. Tôi chia thành 4 prompt, mỗi prompt tương ứng một nhóm chức năng.
Mỗi prompt build tiếp lên file đang có, KHÔNG viết lại từ đầu.

### Prompt 4.1 — Khung file và các utility Split/KFold

```
Viết phần đầu của run_pipeline.py. Đây là file Pipeline điều phối chính.

Imports cần có:
  import argparse, os, sys
  from typing import Optional, Union, List, Tuple, Any, overload
  import numpy as np
  import pandas as pd
  from model import c_factor, Node, IsolationTree, IsolationForest, IsolationForestScratch

Ngay sau import, thêm xử lý encoding UTF-8 cho Windows:
  if sys.platform == "win32":
      try:
          sys.stdout.reconfigure(encoding="utf-8")
          sys.stderr.reconfigure(encoding="utf-8")
      except Exception:
          pass

Viết 5 thành phần theo thứ tự:

1. Class Pipeline
   - __init__(self, model=None): tạo IsolationForest() mặc định nếu model là None
   - Properties: threshold_, offset_, estimators_, max_depth (đều delegate sang self.model)
   - fit(self, X, y=None): convert X sang numpy nếu là DataFrame rồi gọi model.fit
   - Các method: anomaly_score, score_samples, decision_function, predict, fit_predict
     — tất cả đều delegate thẳng sang self.model

2. Hàm train_test_split với @overload signature:
   - Overload 1: (X, y=None) -> Tuple[X_train, X_test]
   - Overload 2: (X, y) -> Tuple[X_train, X_test, y_train, y_test]
   - Params: test_size=0.2, random_state=42, stratify=False
   - Validate: 0 < test_size < 1, X phải 2D, n_samples >= 2, len(X)==len(y) nếu y không None
   - Khi stratify=True và y không None: shuffle từng class riêng rồi lấy n_test đều
   - Sau split, assert set(train_idx).isdisjoint(set(test_idx))
   - Hỗ trợ cả pandas (dùng .iloc) và numpy (dùng indexing trực tiếp)
   - Trả về copies (dùng .copy())

3. Class KFold(n_splits=3, shuffle=True, random_state=42)
   - Validate: n_splits phải là int >= 2, không phải bool
   - split(self, X, y=None): yield (train_indices, val_indices)
     Dùng np.array_split(indices, n_splits) để tạo n_splits fold đều nhau

4. Class StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
   - Validate: n_splits int >= 2, X và y cùng độ dài, n classes >= 2, min class count >= n_splits
   - split(self, X, y): phân tầng theo class, chia đều mỗi class ra n_splits fold
     yield (train_indices, val_indices) theo từng fold

5. Hàm threshold_from_contamination(scores, contamination) -> float
   - contamination="auto" hoặc "AUTO" -> return 0.5
   - contamination là bool -> raise ValueError
   - contamination là số không trong (0, 0.5) -> raise ValueError
   - Trả về np.percentile(scores, 100 * (1 - contamination))
   - Nếu scores rỗng -> raise ValueError

Docstring tiếng Việt. Comment phân section bằng dấu ==.
```

### Prompt 4.2 — Feature Importance và Root Cause Analysis

```
Tiếp tục thêm vào run_pipeline.py (không xoá phần trước).

Thêm 3 hàm phân tích, theo đúng thứ tự:

1. Hàm robust_deviation(x_val: float, col_vals: np.ndarray) -> float
   - Sort col_vals
   - Dùng np.searchsorted để tìm vị trí x_val, tính percentile = idx/len * 100
   - Trả về abs(pct - 50.0) — khoảng cách từ trung vị, trong [0.0, 50.0]
   - Đây là thay thế Z-score, hoàn toàn robust với kurtosis cực cao

2. Hàm unsupervised_feature_importance(model, X, feature_names=None) -> pd.DataFrame
   Đầu vào: model đã fit, X (numpy hoặc DataFrame), feature_names (list string hoặc None)
   Nếu feature_names là None thì tự đặt tên ["feat_1", ..., "feat_9"]

   Bước 1 — Tính Pearson Correlation vector hóa:
     scores = model.anomaly_score(X_arr)
     std_sc = np.std(scores)
     Nếu std_sc > 0:
       centered_X = X_arr - mean_pop
       centered_sc = scores - mean(scores)
       cov = np.mean(centered_X * centered_sc[:, None], axis=0)
       corr_arr = np.where(std_pop > 0, cov / (std_pop * std_sc), 0.0)
     Nếu std_sc == 0: corr_arr = zeros

   Bước 2 — Tính Median Percentile Rank Deviation của nhóm bất thường (score >= 0.5):
     anom_mask = (scores >= 0.5)
     Với mỗi feature j:
       sorted_col = np.sort(X_arr[:, j])
       anom_vals = X_arr[anom_mask, j]
       pct_ranks = np.searchsorted(sorted_col, anom_vals, side='right') / n_samples * 100
       pct_dev_arr[j] = np.median(np.abs(pct_ranks - 50.0))
     Nếu không có anomaly: pct_dev_arr = zeros

   Bước 3 — Composite score:
     abs_corr = np.abs(corr_arr)
     pct_dev_norm = np.minimum(pct_dev_arr, 50.0) / 50.0
     comp_importance = abs_corr * 0.5 + pct_dev_norm * 0.5

   Bước 4 — Spearman stability check (nếu có >= 10 anomaly):
     Tách anomaly indices thành 2 sub-sample bằng interleaved split (::2 và 1::2)
     Tính pct_dev cho từng sub-sample trên từng feature
     Tính Spearman rho qua rank correlation thủ công (không dùng scipy)
     Nếu rho < 0.6: stability_note = "[CANH BAO] Stability thap: Spearman rho=..."
     Nếu rho >= 0.6: stability_note = "[OK] Stability check: Spearman rho=..."
     Lưu note vào df_imp.attrs['stability_note']

   Trả về DataFrame sắp xếp giảm dần theo 'Chỉ số quan trọng tổng hợp', với các cột:
   ['Đặc trưng', 'Tương quan Pearson |r|', 'Hệ số tương quan r',
    'Độ lệch Pct-Rank (Dị biệt)', 'Chỉ số quan trọng tổng hợp']

3. Hàm explain_anomalies_root_cause(model, X, top_k=10, feature_names=None) -> pd.DataFrame
   Đầu vào: model đã fit, X, top_k số mẫu cần trích, feature_names

   Pre-compute cho toàn bộ X_arr:
     sorted_cols = [np.sort(X_arr[:, j]) for j in range(n_features)]
     medians = np.median(X_arr, axis=0)
     q25 = np.percentile(X_arr, 25, axis=0)
     q75 = np.percentile(X_arr, 75, axis=0)
     iqrs = np.maximum(q75 - q25, 1.0)

   Với mỗi sample trong top_k (sắp xếp giảm dần theo score):
     Tính pct_ranks và pct_devs cho tất cả features
     Tính iqr_devs = abs(sample - medians) / iqrs
     composite_dev = pct_devs + np.minimum(iqr_devs, 1000.0) * 1e-4
     max_dev_feat_idx = argmax(composite_dev)
     Co-extreme features: j khác với max_dev_feat_idx mà pct_devs[j] >= max_pct - 0.15 và >= 49.0
     Mức cảnh báo: score >= 0.60 -> "NGUY HIỂM CAO", score >= 0.50 -> "CẢNH BÁO BẤT THƯỜNG"

   Trả về DataFrame với các cột:
   ['Hạng', 'Chỉ số mẫu (Index)', 'Anomaly Score', 'Mức cảnh báo',
    'Đặc trưng lệch mạnh nhất', 'Đặc trưng đồng cực đoan',
    'Pct Rank Deviation', 'Giá trị thực', 'Pct rank trong quần thể']
```

### Prompt 4.3 — Các hàm metric đánh giá thuần NumPy

```
Tiếp tục thêm vào run_pipeline.py (không xoá phần trước).
Implement tất cả metric đánh giá, KHÔNG dùng sklearn:

1. confusion_matrix(y_true, y_pred) -> np.ndarray shape (2,2)
   [[TN, FP], [FN, TP]] — validate len(y_true)==len(y_pred)

2. accuracy_score(y_true, y_pred) -> float
   Trả về 0.0 nếu array rỗng

3. balanced_accuracy_score(y_true, y_pred) -> float
   (recall_class_0 + recall_class_1) / 2
   Trả về 0.0 nếu một class không có mẫu nào

4. precision_score(y_true, y_pred, zero_division=0) -> float
   TP / (TP + FP), trả về zero_division nếu mẫu thực tế dự đoán là 1 bằng 0

5. recall_score(y_true, y_pred, zero_division=0) -> float
   TP / (TP + FN), trả về zero_division nếu không có mẫu dương thực

6. f1_score(y_true, y_pred, zero_division=0) -> float
   2*P*R / (P+R), trả về zero_division nếu P+R==0

7. roc_auc_score(y_true, scores) -> float
   Dùng công thức Mann-Whitney U:
   - np.argsort(scores) để lấy thứ tự
   - ranks[order] = arange(1, n+1)
   - Xử lý tied scores: np.unique(scores, return_inverse=True, return_counts=True)
     ranks bị tie -> thay bằng average rank qua np.bincount
   - n1 = sum(y_true==1), n0 = n - n1
   - Nếu n1==0 hoặc n0==0 -> return 0.5
   - auc = (sum_ranks_positive - n1*(n1+1)/2) / (n1 * n0)

8. average_precision_score(y_true, scores) -> float
   - Sắp xếp giảm dần theo score
   - Tính precision và recall tại mỗi ngưỡng (duyệt qua sorted thresholds)
   - Xử lý tied scores: tại mỗi unique threshold, tính precision/recall của tập đó
   - Tích phân hình thang: np.sum(np.diff(recall_pts) * precision_pts[1:]) hoặc tương đương
   - Nếu không có mẫu dương thực -> return 0.0

Mỗi hàm: validate len(y_true)==len(y_pred), ném ValueError nếu sai.
Docstring tiếng Việt.
```

### Prompt 4.4 — Hàm main() và CLI

```
Tiếp tục thêm vào run_pipeline.py hàm main() điều phối toàn bộ pipeline.
Import thêm từ weights.py: DEFAULT_CONFIG, PARAM_GRID, KFOLD_N_SPLITS, THRESHOLD_CONFIG.

CLI bằng argparse: 1 flag duy nhất là --tune (store_true, default False).

Bên trong main(), thực hiện tuần tự 9 bước:

BƯỚC 1 — Load và kiểm tra dữ liệu:
  Đọc shuttle.csv bằng pd.read_csv, đặt tên cột feat_1..feat_10
  Kiểm tra: không có NaN (assert df.isna().sum().sum() == 0)
  Kiểm tra: không có Inf (assert np.isfinite(df.values).all())
  feature_cols = ["feat_1","feat_2","feat_3","feat_4","feat_5","feat_6","feat_7","feat_8","feat_9"]
  X = df[feature_cols]  # chỉ 9 cảm biến, KHÔNG bao gồm feat_10
  y_uci = df["feat_10"]
  y_eval = (y_uci != 1).astype(int)  # 1=bất thường (class 2-7), 0=bình thường (class 1)
  In: tổng mẫu, tỉ lệ bất thường thực

BƯỚC 2 — Train/Test split:
  X_train, X_test, y_train_dummy, y_test_dummy = train_test_split(
      X, y_eval, test_size=DEFAULT_CONFIG["test_size"],
      random_state=DEFAULT_CONFIG["random_state"], stratify=True)
  y_eval = y_eval.iloc[y_test_dummy.index] nếu cần — hoặc lấy y_eval tương ứng với X_test
  (QUAN TRỌNG: y_eval chỉ dùng ở bước 8, không truyền vào model)

BƯỚC 3 — K-Fold Unsupervised CV tuning (chỉ khi --tune):
  Nếu args.tune:
    kf = KFold(n_splits=KFOLD_N_SPLITS)
    Với mỗi config trong PARAM_GRID:
      Tính trung bình Score Spread = std(anomaly_score trên fold val) qua 3 fold
    best_config = config có Score Spread trung bình lớn nhất
    In bảng kết quả từng config
  Nếu không --tune:
    best_config = DEFAULT_CONFIG

BƯỚC 4 — Train model:
  best_model = IsolationForest(
      n_estimators=best_config["n_estimators"],
      max_samples=best_config["max_samples"],
      max_features=best_config["max_features"],
      contamination="auto",
      random_state=DEFAULT_CONFIG["random_state"])
  best_model.fit(X_train)
  train_scores = best_model.anomaly_score(X_train)
  test_scores  = best_model.anomaly_score(X_test)

BƯỚC 5 — Tính 5 mức ngưỡng từ train_scores:
  th_theoretical = THRESHOLD_CONFIG["theoretical"]  # 0.5
  th_top5  = np.percentile(train_scores, THRESHOLD_CONFIG["top5_pct"])   # P95
  th_top1  = np.percentile(train_scores, THRESHOLD_CONFIG["top1_pct"])   # P99
  th_top01 = np.percentile(train_scores, THRESHOLD_CONFIG["top01_pct"])  # P99.9
  th_stat  = np.mean(train_scores) + THRESHOLD_CONFIG["stat_sigma_k"] * np.std(train_scores)
  In bảng 5 ngưỡng với: tên ngưỡng, giá trị, % mẫu test vượt ngưỡng

BƯỚC 6 — Root Cause Analysis:
  top_anomalies_df = explain_anomalies_root_cause(best_model, X_test, top_k=10, feature_names=feature_cols)
  In bảng top_anomalies_df

BƯỚC 7 — Feature Importance:
  feat_imp_df = unsupervised_feature_importance(best_model, X_test, feature_names=feature_cols)
  In bảng feat_imp_df
  Nếu 'stability_note' trong feat_imp_df.attrs thì in thêm note đó
  In top 3 feature quan trọng nhất

BƯỚC 8 — Supervised Validation (external ground truth):
  In disclaimer: "Nhãn y chỉ dùng SAU khi predict xong, không ảnh hưởng huấn luyện"
  y_eval_arr = np.asarray(y_eval)
  roc_auc = roc_auc_score(y_eval_arr, test_scores)
  ap = average_precision_score(y_eval_arr, test_scores)
  y_pred_theo = (test_scores >= th_theoretical).astype(int)
  y_pred_top5 = (test_scores >= th_top5).astype(int)
  In ROC-AUC, AP, tỉ lệ bất thường thực
  In Precision/Recall/F1 với ngưỡng lý thuyết và top 5%

  So sánh Baseline Distance-to-Centroid:
    Normalize X_test bằng mean/std của X_train
    baseline_dists = L2 distance từ centroid (trung tâm X_train đã normalize)
    Normalize baseline_dists về [0,1] bằng min-max
    baseline_roc = roc_auc_score(y_eval_arr, baseline_scores)
    baseline_ap  = average_precision_score(y_eval_arr, baseline_scores)
    In bảng so sánh: Baseline vs Isolation Forest (ROC-AUC, AP, AP/Baseline)

BƯỚC 9 — Test inference thực tế:
  pipe = Pipeline(model=best_model)
  sample_first = X_test.iloc[0]
  sample_score = float(pipe.anomaly_score(np.array([list(sample_first.values())]))[0])
  In: Score và trạng thái BAT THUONG / BINH THUONG so với th_theoretical

Cuối file:
if __name__ == "__main__":
    main()
```

---

## Phần 5 — `save_weights.py`: Script lưu trọng số

### Prompt 5.1 — Viết toàn bộ save_weights.py

```
Viết file save_weights.py — script chạy một lần để train model và lưu weights.

Import: argparse, os, sys, numpy as np, pandas as pd
Import từ model.py: IsolationForest
Import từ weights.py: DEFAULT_CONFIG, PARAM_GRID, KFOLD_N_SPLITS, save_weights, load_weights
Import từ run_pipeline.py: train_test_split, KFold

Xử lý encoding Windows ngay sau import (giống run_pipeline.py).

CLI: argparse với 1 flag --tune (store_true).

Logic chạy:
1. Đọc shuttle.csv, tách X (feat_1..feat_9), bỏ feat_10
2. Nếu --tune:
   - Chạy 3-Fold CV trên X với từng config trong PARAM_GRID
   - Tiêu chí: Score Spread = std(anomaly_score trên val), chọn config có trung bình lớn nhất
   - Train lại trên toàn bộ X với best_config
   - Lưu ra weights/best_model_weights.json / .npz / .txt
3. Nếu không --tune:
   - Train với DEFAULT_CONFIG trên toàn bộ X
   - Lưu ra weights/baseline_weights.json / .npz / .txt

Sau khi lưu:
- In đường dẫn 3 file đã lưu
- Load lại bằng load_weights() để verify
- Tính anomaly_score trên 1 mẫu test (hàng đầu tiên của X)
- In score để xác nhận weights load đúng

Tạo thư mục weights/ nếu chưa tồn tại (os.makedirs).
```

---

## Phần 6 — `tests/test_pipeline.py`: 21 Unit Tests

### Prompt 6.1 — Viết toàn bộ test file

```
Viết file tests/test_pipeline.py — bộ unit tests bằng pytest. 
KHÔNG import sklearn hay bất kỳ thư viện ML nào.

Import: pytest, numpy as np, pandas as pd
Import từ model: c_factor, Node, IsolationTree, IsolationForest
Import từ run_pipeline: train_test_split, KFold, StratifiedKFold,
    threshold_from_contamination, confusion_matrix, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    unsupervised_feature_importance, explain_anomalies_root_cause, Pipeline
Import từ weights: save_weights, load_weights

Tạo fixture @pytest.fixture tên X_small:
  rng = np.random.RandomState(0)
  Trả về rng.randn(200, 9).astype(np.float64)

--- Group 1: Model core (7 tests) ---

test_c_factor_edge_cases:
  assert c_factor(0) == 0.0
  assert c_factor(1) == 0.0
  assert c_factor(2) == 1.0
  assert c_factor(256) > 0

test_isolation_tree_fit(X_small):
  tree = IsolationTree(max_depth=8)
  tree.fit(X_small)
  assert tree.root is not None

test_iforest_basic_fit(X_small):
  model = IsolationForest(n_estimators=10, max_samples=64, random_state=0)
  model.fit(X_small)
  assert len(model.trees) == 10
  assert model.n_features_in_ == 9

test_anomaly_score_range(X_small):
  model = IsolationForest(n_estimators=10, random_state=0).fit(X_small)
  scores = model.anomaly_score(X_small)
  assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

test_anomaly_score_shape(X_small):
  model = IsolationForest(n_estimators=10, random_state=0).fit(X_small)
  scores = model.anomaly_score(X_small)
  assert scores.shape == (200,)

test_predict_binary(X_small):
  model = IsolationForest(n_estimators=10, random_state=0).fit(X_small)
  preds = model.predict(X_small)
  assert set(np.unique(preds)).issubset({0, 1})

test_reproducibility(X_small):
  m1 = IsolationForest(n_estimators=10, random_state=42).fit(X_small)
  m2 = IsolationForest(n_estimators=10, random_state=42).fit(X_small)
  np.testing.assert_array_equal(m1.anomaly_score(X_small), m2.anomaly_score(X_small))

--- Group 2: Validation & Error Handling (6 tests) ---

test_invalid_n_estimators:
  with pytest.raises(ValueError): IsolationForest(n_estimators=0)
  with pytest.raises(ValueError): IsolationForest(n_estimators=-1)
  with pytest.raises(ValueError): IsolationForest(n_estimators=True)

test_invalid_max_samples:
  with pytest.raises(ValueError): IsolationForest(max_samples=1)
  with pytest.raises(ValueError): IsolationForest(max_samples=1.5)

test_invalid_max_features:
  with pytest.raises(ValueError): IsolationForest(max_features=1.5)
  with pytest.raises(ValueError): IsolationForest(max_features=0.0)

test_nan_input(X_small):
  model = IsolationForest(n_estimators=5, random_state=0).fit(X_small)
  X_bad = X_small.copy(); X_bad[0, 0] = np.nan
  with pytest.raises(ValueError): model.anomaly_score(X_bad)

test_inf_input(X_small):
  model = IsolationForest(n_estimators=5, random_state=0).fit(X_small)
  X_bad = X_small.copy(); X_bad[0, 0] = np.inf
  with pytest.raises(ValueError): model.anomaly_score(X_bad)

test_wrong_feature_count(X_small):
  model = IsolationForest(n_estimators=5, random_state=0).fit(X_small)
  X_bad = X_small[:, :5]  # chỉ 5 features thay vì 9
  with pytest.raises(ValueError): model.anomaly_score(X_bad)

--- Group 3: Pipeline utilities (5 tests) ---

test_train_test_split_no_overlap(X_small):
  X_tr, X_te = train_test_split(X_small, test_size=0.2, random_state=42)
  assert X_tr.shape[0] + X_te.shape[0] == 200
  # Kiểm tra không overlap bằng cách so sánh nội dung

test_kfold_coverage(X_small):
  kf = KFold(n_splits=3, random_state=42)
  val_indices = []
  for _, val_idx in kf.split(X_small):
      val_indices.extend(val_idx.tolist())
  assert sorted(val_indices) == list(range(200))  # mọi mẫu xuất hiện đúng 1 lần

test_threshold_from_contamination_auto:
  scores = np.random.RandomState(0).rand(100)
  assert threshold_from_contamination(scores, "auto") == 0.5
  assert threshold_from_contamination(scores, "AUTO") == 0.5

test_metrics_roc_auc_perfect:
  y_true = np.array([0, 0, 1, 1])
  scores = np.array([0.1, 0.2, 0.8, 0.9])
  assert roc_auc_score(y_true, scores) == 1.0

test_metrics_ap_random:
  rng = np.random.RandomState(42)
  y_true = (rng.rand(1000) > 0.8).astype(int)
  scores = rng.rand(1000)
  ap = average_precision_score(y_true, scores)
  assert 0.05 < ap < 0.40  # xấp xỉ tỉ lệ dương (khoảng 20%)

--- Group 4: Integration (3 tests) ---

test_full_pipeline_runs(X_small):
  pipe = Pipeline()
  pipe.fit(X_small)
  scores = pipe.anomaly_score(X_small)
  preds = pipe.predict(X_small)
  assert scores.shape == (200,)
  assert set(np.unique(preds)).issubset({0, 1})

test_save_load_weights(X_small, tmp_path):
  model = IsolationForest(n_estimators=5, random_state=0).fit(X_small)
  prefix = str(tmp_path / "test_weights")
  save_weights(model, prefix)
  loaded_model, _ = load_weights(prefix)
  orig_scores = model.anomaly_score(X_small)
  load_scores = loaded_model.anomaly_score(X_small)
  np.testing.assert_allclose(orig_scores, load_scores, atol=1e-10)

test_unsupervised_feature_importance(X_small):
  model = IsolationForest(n_estimators=10, random_state=0).fit(X_small)
  df = unsupervised_feature_importance(model, X_small)
  assert df.shape[0] == 9  # 9 features
  assert 'Chỉ số quan trọng tổng hợp' in df.columns
  assert 'Đặc trưng' in df.columns
```

---

## Phần 7 — Jupyter Notebook

### Prompt 7.1 — Yêu cầu notebook

```
Tôi cần một Jupyter Notebook phân tích đầy đủ cho project này.
Tên file: shuttle_anomaly_detection_iforest.ipynb

Notebook phải có đúng 21 bước ML chuẩn theo thứ tự:
1.  Problem Definition
2.  Data Collection & Loading
3.  Exploratory Data Analysis (EDA)
4.  Data Quality Assessment
5.  Feature Analysis — phân phối từng cảm biến
6.  Correlation Analysis — heatmap
7.  Data Preprocessing — tách X (feat_1..feat_9) và y_uci (feat_10)
8.  Train/Test Split — dùng hàm train_test_split từ run_pipeline.py
9.  Model Architecture Overview — giải thích thuật toán
10. Hyperparameter Tuning — 3-Fold CV với PARAM_GRID từ weights.py
11. Model Training — train IsolationForest với best config
12. Anomaly Score Distribution Analysis
13. Threshold Analysis — 5 mức ngưỡng
14. Root Cause Analysis — top 10 mẫu bất thường
15. Feature Importance Analysis
16. Supervised Validation — ROC-AUC, AP với nhãn UCI (external)
17. ROC Curve & PR Curve
18. Baseline Comparison — so sánh với Distance-to-Centroid
19. Model Serialization — save/load weights
20. Real-time Inference Demo
21. Conclusions & Deployment Notes

Cấu trúc: mỗi bước = 1 markdown cell (header H2) + ít nhất 1 code cell.
Tổng khoảng 50 cells. Không import sklearn trong bất kỳ cell nào.
Import từ model.py, run_pipeline.py, weights.py.
```

### Prompt 7.2 — Yêu cầu visualization

```
Trong notebook, thêm 18 biểu đồ phân bổ đều qua các bước:

Bước 3 (EDA): 1 histogram tổng quan phân phối các cảm biến (subplots 3x3)
Bước 5 (Feature Analysis): 1 boxplot từng cảm biến
Bước 6 (Correlation): 1 heatmap correlation matrix 9x9
Bước 12 (Score Distribution): 2 histogram score (train màu xanh, test màu cam, vẽ chung 1 figure)
Bước 13 (Threshold): 1 histogram score với 5 đường threshold dọc khác màu, có legend
Bước 14 (RCA): scatter plot anomaly score vs feature quan trọng nhất
Bước 15 (Feature Importance): 1 horizontal bar chart, sắp xếp theo composite score
Bước 16 (Supervised): in confusion matrix dưới dạng bảng (không cần plot)
Bước 17 (ROC & PR): 2 plot riêng — ROC curve với diagonal baseline và AUC annotation,
  PR curve với AP annotation và horizontal baseline (tỉ lệ dương thực)
Bước 18 (Baseline): 1 grouped bar chart so sánh ROC-AUC và AP của Baseline vs Isolation Forest
Bước 20 (Inference): in kết quả dạng text, không cần plot

Toàn bộ plot dùng:
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.style.use("seaborn-v0_8-whitegrid")
Mọi plot có title, xlabel, ylabel. figsize tối thiểu (8,5). dpi=100. plt.tight_layout().
```

---

## Phần 8 — `requirements.txt` và `.gitignore`

### Prompt 8.1 — requirements.txt

```
Viết requirements.txt cho project Isolation Forest này.

Nội dung chính xác:
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.0.0
jupyter>=1.0.0
nbconvert>=7.0.0

KHÔNG thêm scikit-learn hay bất kỳ thư viện ML nào.
Thêm comment tiếng Việt giải thích từng nhóm (xử lý dữ liệu, visualization, testing, notebook).
```

### Prompt 8.2 — .gitignore

```
Viết .gitignore chuẩn cho Python project. Gồm:
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
env/
.ipynb_checkpoints/
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db

KHÔNG ignore các file sau vì đây là model artifacts cần commit:
weights/*.json
weights/*.npz
weights/*.txt
```

---

## Phần 9 — `README.md` chính

### Prompt 9.1 — Viết README

```
Viết README.md cho project Isolation Forest phát hiện bất thường trên NASA Shuttle.
Viết tiếng Việt không dấu để tránh encoding issues trên mọi hệ điều hành.

Cấu trúc bắt buộc, theo thứ tự:

1. Tiêu đề H1: "Isolation Forest — NASA Shuttle Telemetry Anomaly Detection"
   Subtitle: "Pure NumPy · Zero Scikit-Learn · 100% Unsupervised"

2. Mô tả 1 đoạn: phát hiện bất thường trên dữ liệu cảm biến tàu con thoi NASA
   (UCI Statlog Shuttle, 58,000 mẫu x 9 cảm biến), thuật toán Liu, Ting & Zhou (2008),
   không phụ thuộc thư viện ML bậc cao.

3. Cây thư mục (code block), ghi chú bên cạnh mỗi file.

4. Bảng thông tin dataset: So mau | So cam bien | Cot nhan UCI | Ti le bat thuong | Chat luong
   Kèm ghi chú: Label Leakage = 0%, ROC-AUC thực tế = 0.8474

5. Công thức anomaly score (text, không LaTeX):
   s(x, psi) = 2^(-E[h(x)] / c(psi))
   c(psi) = 2*ln(psi-1) + 2*gamma - 2*(psi-1)/psi
   Giải thích: s -> 1 = bat thuong, s ~= 0.5 = binh thuong, gamma = 0.5772...

6. Bảng 5 ngưỡng quyết định (các giá trị số đã chạy thực tế):
   Ly thuyet 0.5, Top 5% P95=0.5559, Top 1% P99=0.6308, Top 0.1% P99.9=0.6625, mu+2sigma=0.5498

7. Bảng kết quả:
   ROC-AUC (leak-free) = 0.8474, Average Precision = 0.6461, AP/Baseline = 3.0x,
   Precision@Top5% = 93.1%, Spearman stability rho >= 0.98, Unit tests 21/21 passed

8. Hướng dẫn cài và chạy (code block powershell):
   py -3 -m pip install -r requirements.txt
   py -3 run_pipeline.py
   py -3 run_pipeline.py --tune
   py -3 save_weights.py
   py -3 save_weights.py --tune
   py -3 -m pytest tests/test_pipeline.py -v
   jupyter notebook shuttle_anomaly_detection_iforest.ipynb

9. Code ví dụ load weights và inference (code block python):
   from weights import load_weights
   import numpy as np
   model, params = load_weights("best_model_weights")
   packet = np.array([[50, 21, 77, 0, 28, 0, 27, 48, 22]])
   score = float(model.anomaly_score(packet)[0])
   status = "BAT THUONG" if score >= 0.5 else "BINH THUONG"
   print(f"Score: {score:.4f} -> {status}")

10. Đặc điểm kỹ thuật nổi bật (bullet list):
    - Zero Scikit-Learn
    - RCA phi tham so: Percentile Rank Deviation, mien nhiem voi kurtosis > 2000
    - Weights I/O: JSON (khong pickle) + NPZ cho inference nhanh
    - 21 buoc ML chuan: Problem Definition -> Deployment

11. Tai lieu tham khao:
    Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. ICDM 2008, pp. 413-422.
```

---

## Phần 10 — Các prompt fix lỗi khi chạy thực tế

Các vấn đề này xảy ra khi chạy lần đầu. Nếu gặp thì gửi đúng prompt tương ứng:

### Prompt 10.1 — Fix RecursionError khi duyệt cây

Gặp khi: `RecursionError: maximum recursion depth exceeded`

```
Method compute_path_lengths trong IsolationTree đang dùng đệ quy, gây RecursionError
với dataset 58,000 mẫu và max_depth = ceil(log2(256)) = 8.

Viết lại compute_path_lengths theo pattern Batch Traversal với explicit stack:
  stack = [(np.arange(n_samples), self.root, 0)]
  while stack:
      indices, curr_node, depth = stack.pop()
      if curr_node.is_leaf:
          lengths[indices] = depth + curr_node.leaf_adj
          continue
      col_vals = X[indices, curr_node.split_feat]
      mask = col_vals < curr_node.split_val
      stack.append((indices[~mask], curr_node.right, depth + 1))  # nhánh phải
      stack.append((indices[mask],  curr_node.left,  depth + 1))  # nhánh trái
Kiểm tra len(indices) > 0 và node không None trước mỗi bước.
```

### Prompt 10.2 — Fix UnicodeEncodeError trên Windows

Gặp khi: `UnicodeEncodeError: 'charmap' codec can't encode character`

```
Thêm đoạn code này vào đầu run_pipeline.py ngay sau tất cả các import:

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
```

### Prompt 10.3 — Fix Z-score vô nghĩa do kurtosis cực cao

Gặp khi: Feature Importance cho kết quả bất hợp lý, các feature có kurtosis cao (feat_2: 2647, feat_4: 7698) bị xếp hạng sai.

```
Trong unsupervised_feature_importance và explain_anomalies_root_cause,
thay thế Z-score bằng Percentile Rank Deviation:

Thay vì tính (x - mean) / std, hãy:
1. Sort toàn bộ giá trị của cột: sorted_col = np.sort(X_arr[:, j])
2. Tìm vị trí của x: idx = np.searchsorted(sorted_col, x, side='right')
3. Tính percentile rank: pct = idx / len(sorted_col) * 100
4. Độ lệch = abs(pct - 50.0) — khoảng cách từ trung vị, trong [0.0, 50.0]

Metric này hoàn toàn bất biến với phân phối lệch hay kurtosis bất kỳ.
Cập nhật cả hai hàm unsupervised_feature_importance() và explain_anomalies_root_cause().
```

### Prompt 10.4 — Fix roc_auc_score sai khi có tied scores

Gặp khi: roc_auc_score trả về kết quả nhỏ hơn mong đợi, hoặc > 1.0.

```
Hàm roc_auc_score bị sai khi nhiều mẫu có cùng score (tied scores).
Sửa lại phần tính rank:

# Bước 1: argsort để lấy vị trí
order = np.argsort(sc)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(sc) + 1)

# Bước 2: xử lý tied scores bằng average rank (vectorized, không dùng loop)
unique_scores, inv_indices, counts = np.unique(sc, return_inverse=True, return_counts=True)
if len(unique_scores) < len(sc):
    ranks = (np.bincount(inv_indices, weights=ranks) / counts)[inv_indices]

# Bước 3: tính AUC bằng công thức Mann-Whitney
sum_ranks_pos = float(np.sum(ranks[pos_mask]))
auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)
```

### Prompt 10.5 — Fix label leakage

Gặp khi: ROC-AUC trả về 1.0 hoặc quá cao một cách khả nghi.

```
Kiểm tra ngay trong hàm main() của run_pipeline.py:
- X phải chỉ gồm feat_1 đến feat_9, tuyệt đối không có feat_10
- Thêm assertion: assert "feat_10" not in X.columns
- y_eval (từ feat_10) chỉ được dùng ở bước 8 Supervised Validation
- y_eval không được truyền vào model.fit(), model.anomaly_score(), hay bất kỳ
  hàm nào trước bước 8

Thêm comment rõ ràng tại dòng tạo X:
# X chỉ gồm 9 cảm biến — KHÔNG bao gồm feat_10 (label UCI)
# Label Leakage = 0%: feat_10 chỉ dùng ở bước external validation
```

---

## Tổng kết — Thứ tự build project từ đầu

```
Bước 1:  Prompt 1.1 → Prompt 1.2                          (Xác lập bối cảnh & cấu trúc)
Bước 2:  Prompt 2.1 → Prompt 2.2                          (Viết model.py)
Bước 3:  Prompt 3.1 → Prompt 3.2                          (Viết weights.py)
Bước 4:  Prompt 4.1 → 4.2 → 4.3 → 4.4                    (Viết run_pipeline.py)
Bước 5:  Prompt 5.1                                        (Viết save_weights.py)
Bước 6:  Prompt 6.1                                        (Viết tests/test_pipeline.py)
Bước 7:  Prompt 7.1 → Prompt 7.2                          (Viết Jupyter Notebook)
Bước 8:  Prompt 8.1 → Prompt 8.2                          (requirements.txt, .gitignore)
Bước 9:  Prompt 9.1                                        (Viết README.md)
Bước 10: Prompt 10.x (chỉ khi gặp lỗi tương ứng)          (Debug & Fix)
```

Tổng cộng **~20 prompts** từ ý tưởng đến project hoàn chỉnh: 5 file Python, 21 unit tests pass, 1 notebook 21 bước, weights lưu được và load lại y hệt.

---

*Ghi lại ngày 22/09/2026 — Isolation Forest · Liu, Ting & Zhou, ICDM 2008 · NASA Shuttle Telemetry*
