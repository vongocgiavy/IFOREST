# Phát hiện Bất thường Cảm biến Tàu Con thoi (NASA Shuttle) bằng Isolation Forest (Build Tay - Zero Sklearn)

Dự án Machine Learning chuẩn nghiên cứu học thuật: Tự xây dựng từ số 0 (**From Scratch - Pure NumPy**) toàn bộ thuật toán **Isolation Forest** (theo bài báo gốc Liu et al., 2008) thuần túy trên dữ liệu cảm biến gốc (không lồng ghép StandardScaler, không PCA), phân chia Stratified Train/Test 80/20 (46,400 train / 11,600 test) chống rò rỉ dữ liệu, và toàn bộ hệ thống đánh giá Metrics không giám sát (ROC-AUC Wilcoxon U, F1, Balanced Accuracy, Precision, Recall, Average Precision, Confusion Matrix) mà **không dùng thư viện `sklearn`**.

---

## 1. Cấu trúc Dự án

```
d:/May_Hoc/iforest/
│
├── requirements.txt                         # Thư viện phụ thuộc (numpy, pandas, matplotlib, seaborn, pytest)
├── README.md                                # Hướng dẫn và báo cáo học thuật chuẩn hóa
├── metrics.csv                              # Bảng kết quả thực nghiệm với metadata đầy đủ
│
├── shuttle.csv                              # Dữ liệu gốc 58,000 mẫu (UCI Statlog Shuttle)
├── shuttle_preprocessed.csv                 # Dữ liệu sau tiền xử lý (9 cảm biến và nhãn nhị phân)
│
├── model.py                                 # Core Mô hình THUẦN ISOLATION FOREST (100% Pure NumPy: Node, iTree, iForest)
├── run_pipeline.py                          # Pipeline điều phối, tiền xử lý, K-Fold CV, đánh giá metrics & CLI
├── shuttle_anomaly_detection_iforest.ipynb  # Báo cáo Jupyter Notebook phân tích chi tiết (20 phần kèm đồ họa)
│
└── tests/                                   # Bộ kiểm thử tự động
    └── test_pipeline.py                     # 21 bài Unit Tests độc lập (toán học, pipeline, validation, consistency)
```

---

## 2. Bảng Kết quả Thực nghiệm Chuẩn hóa

| Thí nghiệm (Experiment) | Số mẫu (Samples) | Tỷ lệ Bất thường (Anomaly Rate) | ROC-AUC | F1-Score | Loại Ngưỡng (Threshold Type) | Ngưỡng (Threshold) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Main – Full Shuttle (Chính)** | 58,000 | 21.40% | **0.8432** | **0.5097** | Theoretical_Auto | 0.500000 |
| **Main – Full Shuttle (Phân tích)** | 58,000 | 21.40% | **0.8432** | **0.5424** | Train_Contamination | 0.464012 |
| **ODDS Benchmark (Biến thể)** | 49,097 | 7.15% | **0.9981** | **0.9734** | Train_Contamination | 0.558067 |

> [!IMPORTANT]
> **Quy chuẩn Học thuật:** Kết quả chính thức của bài toán trên toàn bộ 58,000 mẫu là **ROC-AUC = 0.8432**. Kết quả **ROC-AUC = 0.9981** là kết quả của biến thể benchmark quốc tế ODDS (Rayana, 2016 - đã loại bỏ Class 4), không được lấy làm kết quả đại diện cho bài toán gốc 58,000 mẫu.

---

## 3. Cài đặt Môi trường

```powershell
py -m pip install -r requirements.txt
```

---

## 4. Thực thi Pipeline (`run_pipeline.py`)

### 4.1. Thí nghiệm Chính (Main Experiment - Single Run)
```powershell
py run_pipeline.py
```
- Phân chia tập dữ liệu theo tỷ lệ Stratified 80/20: **46,400 mẫu Huấn luyện (Train)** và **11,600 mẫu Kiểm thử (Test)**, kiểm tra tính toàn vẹn và đảm bảo không có rò rỉ dữ liệu (`train_indices.isdisjoint(test_indices)`).
- Đánh giá độc lập trên cả 2 ngưỡng:
  - **Theoretical_Auto**: Ngưỡng lý thuyết $0.500000$ theo bài báo gốc Liu et al. (2008).
  - **Train_Contamination**: Ngưỡng thực nghiệm trích xuất từ phân vị $(1 - \text{contamination})$ trên tập Train (không sử dụng nhãn test để tránh test leakage).

### 4.2. Tinh chỉnh Siêu tham số (Stratified 3-Fold CV)
```powershell
py run_pipeline.py --tune
```
- Thuật toán `StratifiedKFold` được sử dụng **chuyên biệt cho giai đoạn Hyperparameter Tuning** trên tập huấn luyện (Train set) nhằm tìm kiếm cấu hình tối ưu mà không làm rò rỉ dữ liệu tập Test.

### 4.3. Chạy theo Chuẩn Benchmark Quốc Tế (ODDS Benchmark - ROC-AUC ~0.998)
```powershell
py run_pipeline.py --odds
```
- Loại bỏ Class 4 theo quy chuẩn benchmark của ODDS (Outlier Detection DataSets, Rayana 2016). Tỷ lệ bất thường $7.15\%$, đạt **ROC-AUC = 0.9981** và **F1 = 97.34%**.

### 4.4. Chạy Toàn bộ Kiểm thử Tự động (Unit Tests)
```powershell
py -m pytest -v
```
Bộ kiểm thử gồm **21 bài test** bao phủ toàn bộ các khía cạnh:
- Tính hợp lệ dữ liệu và không trùng lặp Train/Test.
- Công thức toán học $c(256) \approx 10.2447709201$ và $s(x, \psi)$.
- Tính toán $max\_depth = \lceil \log_2(max\_samples) \rceil = 8$ với $max\_samples=256$.
- Bẫy lỗi các siêu tham số không hợp lệ (`n_estimators <= 0`, `max_features > 1`, `contamination >= 0.5`, dữ liệu sai kích thước hoặc $< 2$ mẫu).
- Tính nhất quán giữa `anomaly_score`, `score_samples` (bằng $-s$) và `decision_function` (bằng $0.5 - s$).

---

## 5. Dự đoán Điểm Bất thường Mới (Pure iForest Pipeline Inference)

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

# 4. Phân loại theo ngưỡng lý thuyết (Theoretical threshold)
is_anomaly = score >= iforest.threshold_

print(f"Anomaly Score: {score:.4f} -> Trạng thái: {'BẤT THƯỜNG' if is_anomaly else 'BÌNH THƯỜNG'}")
```

---

## 6. Điểm nổi bật về Kỹ thuật & Học thuật

1. **Pure NumPy 100% (Zero Scikit-Learn):** Tự lập trình toàn bộ thuật toán Isolation Forest (`Node`, `IsolationTree`, `IsolationForest`, `c_factor`), cấu trúc Pipeline và toàn bộ hệ thống Metrics.
2. **Chuẩn xác theo bài báo gốc Liu et al. (2008):**
   - Không hardcode tỷ lệ bất thường: `contamination="auto"` sử dụng ngưỡng quyết định lý thuyết $s \ge 0.5$.
   - **Feature subsampling:** Mô hình hỗ trợ feature subsampling theo từng cây thông qua `max_features`. Trong cấu hình thí nghiệm chính, `max_features=1.0`, vì vậy mỗi cây sử dụng toàn bộ 9 thuộc tính.
   - Tại mỗi node, chọn ngẫu nhiên đều 1 thuộc tính từ tập thuộc tính có thể phân hoạch ($x_{\min} < x_{\max}$).
3. **Bộ tiện ích và kiểm định hoàn chỉnh:** Hỗ trợ đầy đủ `score_samples`, `decision_function` (âm = bất thường), `fit_predict`, xác thực số chiều `n_features_in_`, bẫy lỗi `NaN/Inf`, và xác thực đầu vào nghiêm ngặt.
4. **Tính Tái lập & Minh bạch (Reproducibility):** Ghi nhận đầy đủ siêu tham số, kích thước mẫu, tỷ lệ bất thường và phân định rõ loại ngưỡng trong `metrics.csv`.
