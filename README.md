# Phát hiện Bất thường Cảm biến Tàu Con thoi (NASA Shuttle) bằng Isolation Forest (Build Tay - Zero Sklearn)

Dự án Machine Learning chuẩn nghiên cứu học thuật: Tự xây dựng từ số 0 (**From Scratch - Pure NumPy**) toàn bộ thuật toán **Isolation Forest** (theo bài báo gốc Liu et al., 2008) thuần túy trên dữ liệu cảm biến gốc (không lồng ghép StandardScaler, không PCA), phân chia `StratifiedKFold`, và toàn bộ hệ thống đánh giá Metrics không giám sát (ROC-AUC Wilcoxon U, F1, Balanced Accuracy, Confusion Matrix) mà **không dùng thư viện `sklearn`**.

---

## 1. Cấu trúc Dự án

```
d:/May_Hoc/iforest/
│
├── requirements.txt                         # Thư viện phụ thuộc (chỉ cần numpy, pandas, matplotlib, seaborn)
├── README.md                                # Hướng dẫn sử dụng dự án
│
├── shuttle.csv                              # Dữ liệu gốc 58,000 mẫu (UCI Statlog Shuttle)
├── shuttle_preprocessed.csv                 # Dữ liệu sau tiền xử lý (9 cảm biến và nhãn nhị phân)
│
├── model.py                                 # Core Mô hình THUẦN ISOLATION FOREST (100% Pure NumPy: Node, iTree, iForest)
├── run_pipeline.py                          # Pipeline điều phối, tiền xử lý, K-Fold CV, đánh giá metrics & CLI
├── shuttle_anomaly_detection_iforest.ipynb  # Báo cáo Jupyter Notebook phân tích chi tiết (20 phần kèm đồ họa)
│
└── tests/                                   # Bộ kiểm thử tự động
    └── test_pipeline.py                     # 8 bài Unit Tests độc lập (toán học, pipeline, tính toàn vẹn)
```

---

## 2. Cài đặt Môi trường

```powershell
py -m pip install -r requirements.txt
```

---

## 3. Thực thi Pipeline (`run_pipeline.py`)

### 3.1. Chạy mặc định (Single Run)
```powershell
py run_pipeline.py
```
*Huấn luyện mô hình và đánh giá đồng thời trên 2 ngưỡng tự nhiên: Ngưỡng Contamination Train & Ngưỡng Lý thuyết 0.5000.*

### 3.2. Chạy Tinh chỉnh Siêu tham số (Stratified 3-Fold CV)
```powershell
py run_pipeline.py --tune
```

### 3.3. Chạy theo Chuẩn Benchmark Quốc Tế (ODDS Benchmark - ROC-AUC ~0.998)
```powershell
py run_pipeline.py --odds
```
*Loại bỏ Class 4 theo đúng quy chuẩn benchmark của ODDS (Outlier Detection DataSets, Rayana 2016). Tỷ lệ bất thường 7.15%, đạt **ROC-AUC = 0.9981** và **F1 = 97.34%**.*

### 3.4. Chạy Kiểm thử Tự động (Unit Tests)
```powershell
py -m unittest tests/test_pipeline.py
```

---

## 4. Dự đoán Điểm Bất thường Mới (Pure iForest Pipeline Inference)

Quy trình suy luận thuần túy được đóng gói thống nhất vào `Pipeline` (Pure NumPy):

```python
import pandas as pd
from model import IsolationForest
from run_pipeline import Pipeline

# 1. Khởi tạo Pipeline thuần Isolation Forest
iforest = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
pipe = Pipeline(model=iforest)
pipe.fit(X_train)

# 2. Dữ liệu cảm biến mới cần kiểm tra
new_sample = pd.DataFrame([{
    'att_1': 55, 'att_2': 0, 'att_3': 92, 'att_4': 0,
    'att_5': 0,  'att_6': 26, 'att_7': 36, 'att_8': 92, 'att_9': 56
}])

# 3. Tính Anomaly Score trực tiếp từ công thức toán học Liu et al. (2008)
score = pipe.anomaly_score(new_sample)[0]

# 4. Phân loại theo ngưỡng tự nhiên nguyên bản
is_anomaly = score >= iforest.threshold_

print(f"Anomaly Score: {score:.4f} -> Trạng thái: {'BẤT THƯỜNG' if is_anomaly else 'BÌNH THƯỜNG'}")
```

---

## 5. Điểm nổi bật về Kỹ thuật & Học thuật

1. **Pure NumPy 100% (Zero Scikit-Learn):** Tự lập trình toàn bộ thuật toán Isolation Forest (`Node`, `IsolationTree`, `IsolationForest`, `c_factor`), cấu trúc Pipeline và toàn bộ hệ thống Metrics.
2. **Chuẩn xác theo bài báo gốc Liu et al. (2008):**
   - Không hardcode tỷ lệ bất thường: `contamination="auto"` sử dụng ngưỡng lý thuyết $s \ge 0.5$.
   - Feature bagging thực hiện độc lập cho từng cây (per-tree).
   - Tại mỗi node, chọn ngẫu nhiên đều 1 thuộc tính từ tập thuộc tính có thể phân hoạch ($x_{\min} < x_{\max}$).
3. **Bộ tiện ích và kiểm định hoàn chỉnh:** Hỗ trợ đầy đủ `score_samples`, `decision_function` (âm = bất thường), `fit_predict`, xác thực số chiều `n_features_in_`, bẫy lỗi `NaN/Inf`.
