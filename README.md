# Phát hiện Bất thường Cảm biến Tàu Con thoi (NASA Shuttle) bằng Isolation Forest (Build Tay - Zero Sklearn)

Dự án Machine Learning chuẩn nghiên cứu học thuật: Tự xây dựng từ số 0 (**From Scratch - Pure NumPy**) toàn bộ thuật toán **Isolation Forest** (theo bài báo gốc Liu et al., 2008) thuần túy trên dữ liệu cảm biến gốc (không lồng ghép StandardScaler, không PCA), phân chia Stratified Train/Test 80/20 (46,400 train / 11,600 test) chống rò rỉ dữ liệu.

Thuật toán Isolation Forest cốt lõi được huấn luyện hoàn toàn theo nguyên lý **không giám sát (unsupervised)**: các cây cô lập (iTree) được dựng hoàn toàn ngẫu nhiên trên không gian đặc trưng $X$ mà không sử dụng bất kỳ thông tin nhãn $y$ nào. Quy trình thực nghiệm tuân thủ giao thức chuẩn mực:
- **Tập Train (Semi-supervised / Label-guided validation):** Nhãn tập Train được sử dụng để tinh chỉnh siêu tham số (Hyperparameter Tuning qua Stratified 3-Fold CV tối ưu F1-score) và tính toán ngưỡng thực nghiệm (Empirical Quantile Threshold theo tỷ lệ contamination của tập train).
- **Tập Test (Unbiased Evaluation):** Tập kiểm thử hoàn toàn cô lập, độc lập và không bị rò rỉ dữ liệu (`train_indices.isdisjoint(test_indices)`), dùng để đánh giá khách quan hiệu năng mô hình với các chỉ số ROC-AUC (Wilcoxon Mann-Whitney U), F1-Score, Balanced Accuracy, Precision, Recall, Average Precision, Confusion Matrix mà **không dùng thư viện `sklearn`**.

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
    └── test_pipeline.py                     # 16 bài Unit Tests toàn diện (toán học, edge cases, validation, consistency)
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
Bộ kiểm thử gồm **16 bài test toàn diện** (được tinh gọn không trùng lặp) bao phủ toàn bộ các khía cạnh:
- Tính hợp lệ dữ liệu và không trùng lặp Train/Test (`train_indices.isdisjoint(test_indices)`).
- Công thức toán học $c(256) \approx 10.2447709201$, $c(1)=0$, $c(2)=1$ và $s(x, \psi)$.
- Tính toán $max\_depth = \lceil \log_2(max\_samples\_actual\_) \rceil$, tự co giãn theo kích thước mẫu thực tế $\psi$.
- Xử lý các edge case nghiêm ngặt: $max\_samples < 2$, dữ liệu $n < 2$, ma trận rỗng hoặc chứa NaN/Inf.
- Xác thực toàn diện cho `StratifiedKFold` (kiểm tra lệch kích thước, $n < n\_splits$, số lớp $< 2$, và cỡ lớp thiểu số $< n\_splits$).
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

1. **Pure NumPy 100% (Zero Scikit-Learn):** Tự lập trình từ số 0 toàn bộ cấu trúc thuật toán Isolation Forest (`Node`, `IsolationTree`, `IsolationForest`, `c_factor`), cấu trúc Pipeline và toàn bộ hệ thống Metrics chuẩn xác.
2. **Chuẩn xác theo bài báo gốc Liu et al. (2008):**
   - **Học không giám sát (Unsupervised):** Quá trình dựng cây hoàn toàn không sử dụng nhãn. Phân hoạch không gian ngẫu nhiên dựa trên phân phối dữ liệu thuộc tính.
   - **Ngưỡng lý thuyết tự nhiên:** Cung cấp `contamination="auto"` với ngưỡng quyết định lý thuyết $s \ge 0.5$ (điểm bất thường có chiều dài đường đi ngắn hơn $c(\psi)$).
   - **Độ sâu cây tự co giãn:** $max\_depth = \lceil \log_2(\psi) \rceil$ tính trực tiếp từ kích thước mẫu thực tế $\psi = max\_samples\_actual\_$ (đảm bảo $\psi \ge 2$).
   - **Feature subsampling:** Mô hình hỗ trợ `max_features` theo từng cây; trong cấu hình chính `max_features = 1.0` dùng toàn bộ 9 cảm biến.
3. **Phân chia dữ liệu & Giao thức đánh giá chuẩn mực:**
   - Phân chia Stratified 80/20 train-test split (`random_state=42`) cách ly dữ liệu triệt để, kiểm tra `train_indices.isdisjoint(test_indices)`.
   - **Tập Train (Semi-supervised / Label-guided validation):** `StratifiedKFold` 3-fold dùng để tinh chỉnh siêu tham số và xác định ngưỡng thực nghiệm $(1 - \text{train\_contam})$.
   - **Tập Test (Khách quan):** Đánh giá độc lập trên cả 2 loại ngưỡng mà không rò rỉ thông tin tập test.
4. **Phân định rõ ràng kết quả Benchmark:**
   - **Bài toán thực tế chính (Full Shuttle 58k mẫu):** Đạt **ROC-AUC = 0.8432**, F1 = 0.5097 (ngưỡng lý thuyết 0.5) và F1 = 0.5424 (ngưỡng thực nghiệm).
   - **Benchmark quốc tế ODDS (49k mẫu, Rayana 2016):** Đạt **ROC-AUC = 0.9981**, F1 = 0.9734. Đây là benchmark đối sánh kinh điển với các nghiên cứu học thuật quốc tế.
5. **Đầy đủ tiện ích và kiểm thử tự động:** Hỗ trợ `score_samples`, `decision_function` (âm = bất thường), `fit_predict`, xác thực số chiều `n_features_in_`, bẫy lỗi `NaN/Inf`, và bộ test tự động 16 bài pass 100%.
