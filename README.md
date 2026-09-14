# Phát hiện Bất thường Cảm biến Tàu Con thoi (NASA Shuttle) bằng Isolation Forest (Build Tay - Zero Sklearn)

Dự án Machine Learning chuẩn nghiên cứu học thuật: Tự xây dựng từ số 0 (**From Scratch - Pure NumPy**) toàn bộ thuật toán **Isolation Forest** (theo bài báo gốc Liu et al., 2008) thuần túy trên dữ liệu cảm biến telemetry gốc của NASA Shuttle. 

Dự án vận hành theo chuẩn mực **100% Học Không Giám Sát (Pure Unsupervised Anomaly Detection)** trên toàn bộ **10 đặc trưng cảm biến thô** (`feat_1` đến `feat_10`), không sử dụng bất kỳ nhãn mục tiêu ($y$) nào trong toàn bộ quy trình huấn luyện, tinh chỉnh và đánh giá.

---

## 1. Cấu trúc Dự án

```
d:/May_Hoc/iforest/
│
├── requirements.txt                         # Thư viện phụ thuộc (numpy, pandas, matplotlib, seaborn, pytest)
├── README.md                                # Hướng dẫn và báo cáo khoa học chuẩn hóa
├── metrics_unsupervised.csv                 # Bảng ghi nhật ký thực nghiệm không giám sát (Pure Unsupervised Logs)
│
├── shuttle.csv                              # Dữ liệu telemetry gốc 58,000 mẫu x 10 đặc trưng (không nhãn, không tiêu đề)
│
├── model.py                                 # Core Mô hình THUẦN ISOLATION FOREST (100% Pure NumPy: Node, iTree, iForest)
├── run_pipeline.py                          # Pipeline điều phối không giám sát, Unsupervised K-Fold CV, Root Cause Analysis & CLI
├── shuttle_anomaly_detection_iforest.ipynb  # Jupyter Notebook phân tích học thuật trực quan (15 phần, 37 cells hoàn chỉnh)
│
└── tests/                                   # Bộ kiểm thử tự động
    └── test_pipeline.py                     # 19 bài Unit Tests toàn diện (toán học, unsupervised K-Fold, edge cases, RCA)
```

---

## 2. Đặc tả Dữ liệu Cảm biến NASA Shuttle (Unsupervised Telemetry)

- **Số lượng mẫu:** 58,000 quan sát từ các hệ thống cảm biến tàu con thoi không gian NASA.
- **Số chiều đặc trưng:** 10 đặc trưng số nguyên thuần túy (`feat_1` đến `feat_10`).
- **Trạng thái nhãn:** **Hoàn toàn không có nhãn (Pure Unsupervised)**.
- **Giá trị khuyết thiếu (Missing Values):** 0 giá trị NaN/Inf (dữ liệu hoàn chỉnh 100%).
- **Tính bất biến tỷ lệ (Scale Invariance):** Thuật toán Isolation Forest chỉ thực hiện phép so sánh thứ tự trên từng trục $X_j < v$ với $v \sim \text{Uniform}(\min(X_j), \max(X_j))$, do đó **bất biến với mọi phép biến đổi đơn điệu (Monotonic Transformations)** như Min-Max Scaling, Z-score Standardization. Vì vậy, mô hình được huấn luyện trực tiếp trên dữ liệu telemetry thô mà không làm méo mó phân phối gốc.
- **Phân chia không giám sát:** Train/Test 80/20 (46,400 mẫu Train / 11,600 mẫu Test) bảo đảm tập chỉ số hoàn toàn rời rạc (`set(X_train.index).isdisjoint(set(X_test.index))`), loại bỏ triệt để hiện tượng rò rỉ dữ liệu (Data Leakage).

---

## 3. Hệ thống Ngưỡng Quyết định & Đánh giá Không Giám Sát

Do bài toán không có nhãn ground truth $y$, hệ thống áp dụng giao thức đánh giá khoa học dựa trên phân phối điểm bất thường (Anomaly Score $s(x, \psi) \in [0, 1]$):

$$s(x, \psi) = 2^{-\frac{\mathbb{E}(h(x))}{c(\psi)}}$$

với $c(\psi) = 2 \ln(\psi - 1) + 2\gamma - \frac{2(\psi - 1)}{\psi}$ là hằng số Euler-Mascheroni ($\gamma \approx 0.5772156649$).

### Bảng 5 Ngưỡng Quyết định Đa Tầng trên Tập Test ($N = 11,600$):

| Loại Ngưỡng Quyết Định | Căn Cứ Lý Thuyết | Giá trị Ngưỡng | Số Mẫu Phát Hiện | Tỷ Lệ Phát Hiện (%) | Mức Độ Can Thiệp |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Ngưỡng Lý Thuyết (Liu et al., 2008)** | $\mathbb{E}(h(x)) < c(\psi) \iff s \ge 0.50$ | `0.5000` | 1,690 | **14.57%** | Cảnh báo giám sát tự động |
| **2. Phân vị Ô nhiễm Top 5%** | $P_{95}$ từ phân phối điểm tập Train | `0.5725` | 612 | **5.28%** | Cảnh báo bất thường telemetry |
| **3. Phân vị Nguy hiểm Cao Top 1%** | $P_{99}$ từ phân phối điểm tập Train | `0.6436` | 119 | **1.03%** | Báo động rủi ro hệ thống |
| **4. Phân vị Cực Đoan Top 0.1%** | $P_{99.9}$ từ phân phối điểm tập Train | `0.6687` | 16 | **0.14%** | Dừng khẩn cấp / Cách ly cảm biến |
| **5. Ngưỡng Thống Kê ($\mu + 2\sigma$)** | Độ lệch chuẩn phân phối Gaussian | `0.5612` | 710 | **6.12%** | Đánh giá trôi dạt dữ liệu |

---

## 4. Chẩn đoán Căn nguyên (Root Cause Analysis) & Độ quan trọng Thuộc tính

1. **Unsupervised Feature Importance:**
   - Kết hợp hệ số tương quan tuyến tính Pearson $|r(X_j, s(x))|$ giữa từng cảm biến với Anomaly Score và độ lệch chuẩn $Z$-score của nhóm bất thường so với quần thể:
   - **Top 3 đặc trưng ảnh hưởng mạnh nhất đến sự bất thường:** `feat_9`, `feat_8`, `feat_10`.
2. **Root Cause Analysis (RCA):**
   - Với mỗi mẫu bất thường phát hiện được, hệ thống tự động tính vector $Z\text{-score} = \frac{x_j - \mu_j}{\sigma_j}$ trên toàn bộ các cảm biến và định danh chính xác thuộc tính có độ lệch chuẩn cực đoan nhất (ví dụ: `feat_2` lệch $+61.89\sigma$ hoặc `feat_9` lệch $+3.87\sigma$), giúp kỹ sư định vị sự cố phần cứng ngay lập tức.

---

## 5. Cài đặt & Hướng dẫn Thực thi

### 5.1. Cài đặt Môi trường

```powershell
py -3.13 -m pip install -r requirements.txt
```

### 5.2. Chạy Pipeline Không Giám Sát (Single Run)

```powershell
py -3.13 run_pipeline.py
```
- Tự động nạp `shuttle.csv` (10 cột, không nhãn).
- Phân chia Train (46,400) / Test (11,600) rời rạc 100%.
- Huấn luyện 100 cây cô lập $iTree$ ($\psi=256$, $max\_depth=8$).
- Xuất bảng 5 ngưỡng quyết định không giám sát, trích xuất Top 5 mẫu bất thường nhất kèm Root Cause Analysis và ghi nhật ký vào `metrics_unsupervised.csv`.

### 5.3. Tinh chỉnh Siêu tham số với Unsupervised 3-Fold CV

```powershell
py -3.13 run_pipeline.py --tune
```
- Sử dụng thuật toán `KFold` thuần túy (không nhãn) chia tập Train thành 3 fold độc lập.
- Đánh giá độ ổn định của phân phối Anomaly Score qua độ lệch chuẩn trên các fold validation.
- Lựa chọn cấu hình siêu tham số có độ ổn định cao nhất.

### 5.4. Chạy Toàn bộ Bộ Kiểm thử Tự động (Unit Tests)

```powershell
py -3.13 -m pytest tests/test_pipeline.py -v
```
Bộ kiểm thử gồm **19 bài test toàn diện**, pass **100% (19/19 tests)**:
- `test_01` & `test_02`: Kiểm tra tồn tại file dữ liệu và tính toàn vẹn.
- `test_03`: Kiểm tra pipeline suy luận mẫu cảm biến mới.
- `test_04`: Kiểm tra tính hợp lệ của cấu hình mặc định và tham số.
- `test_05` & `test_12`: Kiểm tra công thức toán học $c(n)$, $c(1)=0$, $c(2)=1$, $c(256) \approx 10.2447709$.
- `test_06` & `test_08`: Kiểm tra tính chính xác các hàm metric, đường cong ROC/PR và permutation importance kế thừa.
- `test_07`: Kiểm tra tính tổng quát của mô hình, số chiều và bẫy lỗi NaN/Inf.
- `test_09`: Kiểm tra tính phân lập Train/Test 80/20 và không trùng lặp index.
- `test_10` & `test_11`: Kiểm tra cấu hình iForest và công thức $max\_depth = \lceil \log_2(\psi) \rceil$.
- `test_13` & `test_14`: Kiểm tra miền giá trị $s(x) \in [0, 1]$ và tính nhất quán toán học với `score_samples`, `decision_function`.
- `test_15`: Kiểm tra bẫy lỗi các siêu tham số không hợp lệ.
- `test_16`: Kiểm tra các trường hợp biên, ma trận rỗng và bẫy lỗi lệch kích thước.
- `test_17`: Kiểm tra Unsupervised `KFold` (chia 3-fold không nhãn, phủ kín dữ liệu, fold rời rạc).
- `test_18`: Kiểm tra `unsupervised_feature_importance` và `explain_anomalies_root_cause`.
- `test_19`: Kiểm tra tính toàn vẹn của dữ liệu telemetry NASA Shuttle thô không nhãn (58,000 dòng x 10 cột số nguyên, 0 NaN).

---

## 6. Suy luận Thời gian thực (Real-time Inference Pipeline)

```python
import numpy as np
import pandas as pd
from model import IsolationForest
from run_pipeline import Pipeline

# 1. Khởi tạo Pipeline thuần Isolation Forest
iforest = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
pipe = Pipeline(model=iforest)
pipe.fit(X_train.values)

# 2. Vector telemetry cảm biến mới từ tàu con thoi (10 đặc trưng)
raw_telemetry = [50, 21, 77, 0, 28, 0, 27, 48, 22, 2]

# 3. Tính Anomaly Score trực tiếp
score = float(pipe.anomaly_score(np.array([raw_telemetry]))[0])

# 4. Phân loại theo hệ thống 3 mức cảnh báo
if score >= 0.6436:
    level = "NGUY HIỂM CAO (BÁO ĐỘNG)"
elif score >= 0.5000:
    level = "CẢNH BÁO BẤT THƯỜNG"
else:
    level = "BÌNH THƯỜNG (AN TOÀN)"

print(f"Anomaly Score: {score:.6f} -> Trạng thái: {level}")
```

---

## 7. Cam kết Kỹ thuật & Học thuật

1. **Zero Scikit-Learn Dependency:** 100% cấu trúc thuật toán và quy trình xử lý được xây dựng từ số 0 bằng Pure Python & NumPy.
2. **Tuân thủ Tuyệt đối Bản chất Không Giám Sát:** Mô hình không cần nhãn $y$ để hoạt động, phù hợp với các ứng dụng thực tiễn trong công nghiệp hàng không vũ trụ và giám sát thiết bị IoT thời gian thực.
3. **Phòng chống Rò rỉ Dữ liệu (No Data Leakage):** Mọi ngưỡng phân vị và tham số đều được ước lượng nghiêm ngặt trên tập Train và kiểm thử độc lập trên tập Test.
4. **Giải thích Minh bạch (White-box Interpretability):** Kết hợp định lượng toán học giữa Anomaly Score và độ lệch chuẩn $Z$-score để xác định nguyên nhân gốc rễ cho từng cảnh báo sự cố.
