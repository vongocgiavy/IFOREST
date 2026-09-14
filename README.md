# Phát Hiện Bất Thường Cảm Biến Tàu Con Thoi NASA (Statlog Shuttle Telemetry)
## Mô Hình Học Máy Không Giám Sát Isolation Forest (Triển Khai Thuần NumPy - Zero Scikit-Learn)

Dự án phát triển hệ thống phát hiện dị thường trên dữ liệu đo đạc cảm biến thời gian thực (telemetry) của tàu con thoi không gian NASA (Statlog Shuttle Dataset). Toàn bộ thuật toán **Isolation Forest** (theo công bố gốc của Liu, Ting & Zhou, 2008) được tự xây dựng từ số 0 (**From Scratch - Pure NumPy**) mà không sử dụng bất kỳ thư viện học máy bậc cao nào (Zero Scikit-Learn).

Quy trình vận hành theo chuẩn mực **100% Học Không Giám Sát (Pure Unsupervised Anomaly Detection)** trên toàn bộ **10 kênh cảm biến vật lý thô** (`feat_1` đến `feat_10`), không sử dụng nhãn mục tiêu ($y$) trong toàn bộ các pha huấn luyện, kiểm định chéo và suy luận. Toàn bộ tiến trình nghiên cứu được tổ chức theo khung phương pháp luận **21 bước chuẩn hóa của vòng đời phát triển mô hình Machine Learning**.

---

## 1. Cấu Trúc Dự Án

Thư mục dự án được tinh giản tối đa, chỉ lưu trữ mã nguồn cốt lõi, bộ kiểm thử, tài liệu báo cáo và dữ liệu quan trắc gốc:

```
d:/May_Hoc/iforest/
│
├── requirements.txt                         # Danh mục thư viện phụ thuộc (numpy, pandas, matplotlib, seaborn, pytest)
├── README.md                                # Báo cáo khoa học kỹ thuật và hướng dẫn vận hành toàn diện
│
├── shuttle.csv                              # Dữ liệu telemetry cảm biến NASA gốc (58,000 mẫu x 10 đặc trưng, không nhãn)
│
├── model.py                                 # Thuật toán thuần Isolation Forest (100% Pure NumPy: Node, IsolationTree, IsolationForest)
├── run_pipeline.py                          # Pipeline điều phối không giám sát, Unsupervised 3-Fold CV, Root Cause Analysis & CLI
├── shuttle_anomaly_detection_iforest.ipynb  # Jupyter Notebook báo cáo khoa học trực quan (21 bước chuẩn mực, 42 cells)
│
└── tests/
    └── test_pipeline.py                     # Bộ 19 bài kiểm thử tự động toàn diện (Toán học, CV, Edge Cases, Reproducibility)
```

---

## 2. Đặc Tả Dữ Liệu Cảm Biến NASA Shuttle (Statlog Telemetry)

- **Quy mô mẫu:** 58,000 quan sát từ hệ thống cảm biến tàu con thoi không gian NASA.
- **Không gian thuộc tính:** 10 biến số nguyên định lượng (`feat_1` đến `feat_10`) đo đạc áp suất van thủy lực, lưu lượng chất tải nhiệt, nhiệt độ buồng trao đổi và chu kỳ bơm.
- **Trạng thái nhãn:** **Thuần không giám sát (100% Unlabelled)**; không sử dụng nhãn trong quá trình xây dựng mô hình.
- **Tính toàn vẹn dữ liệu:** 0 giá trị khuyết thiếu (NaN), 0 giá trị vô hạn (Inf) trên toàn bộ 580,000 điểm đo.
- **Bảo toàn giá trị ngoại lai (Outlier Retention):** Trong bài toán phát hiện bất thường, các điểm ngoại lai phản ánh trạng thái sự cố phần cứng, do đó được giữ nguyên vẹn thay vì áp dụng các kỹ thuật lọc nhiễu thông thường.
- **Tính bất biến tỷ lệ (Scale Invariance):** Cây cô lập chỉ thực hiện phép so sánh thứ tự nhị phân $X_j < v$ với $v \sim \text{Uniform}(\min(X_j), \max(X_j))$, bảo toàn tuyệt đối thứ tự phân hoạch dưới mọi phép biến đổi đơn điệu tăng ngặt. Mô hình được huấn luyện trực tiếp trên thang đo thô của cảm biến mà không cần chuẩn hóa (Feature Scaling).
- **Phân tách kiểm soát rò rỉ (Leakage Control):** Phân chia tập Huấn luyện (Train - 46,400 mẫu / 80%) và tập Kiểm thử (Test - 11,600 mẫu / 20%) với tập chỉ mục rời rạc tuyệt đối:
  $$\text{Index}_{\text{train}} \cap \text{Index}_{\text{test}} = \emptyset$$

---

## 3. Cơ Sở Lý Thuyết & Kiến Trúc Giải Thuật

### 3.1. Thuật toán Isolation Forest (Liu et al., 2008)
Khác với các phương pháp dựa trên mật độ hoặc khoảng cách (DBSCAN, LOF, k-Means) có độ phức tạp $O(N^2)$, Isolation Forest tận dụng hai đặc tính tự nhiên của dị thường: **số lượng thiểu số** và **tọa độ tách biệt**. Bằng cách phân hoạch đệ quy ngẫu nhiên không gian, các điểm bất thường bị cô lập ở độ sâu rất nông của cây.

### 3.2. Hàm Tính Điểm Bất Thường (Anomaly Score)
Độ dài đường đi kỳ vọng $\mathbb{E}(h(\mathbf{x}))$ qua $t$ cây cô lập được chuẩn hóa bằng hệ số $c(\psi)$ (độ dài tìm kiếm trung bình không thành công trong cây tìm kiếm nhị phân BST với kích thước mẫu con $\psi$):

$$c(\psi) = 2 \ln(\psi - 1) + 2\gamma - \frac{2(\psi - 1)}{\psi}$$

với $\gamma \approx 0.5772156649$ là hằng số Euler-Mascheroni. Điểm bất thường $s(\mathbf{x}, \psi) \in [0, 1]$ được xác định bởi:

$$s(\mathbf{x}, \psi) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(\psi)}}$$

- Khi $\mathbb{E}(h(\mathbf{x})) \to 0 \implies s \to 1$: Mẫu có tính dị biệt cực cao (cô lập rất sớm).
- Khi $\mathbb{E}(h(\mathbf{x})) \to c(\psi) \implies s \to 0.5$: Mẫu có trạng thái bình thường danh định.
- Khi $\mathbb{E}(h(\mathbf{x})) \to \psi - 1 \implies s \to 0$: Mẫu nằm sâu trong vùng cụm mật độ dày đặc.

---

## 4. Hệ Thống 5 Ngưỡng Quyết Định Không Giám Sát

Do bài toán không có nhãn ground truth, hệ thống thiết lập cơ chế đánh giá đa tầng dựa trên phân phối xác suất và độ dài đường đi trên tập kiểm thử Test ($N = 11,600$):

| Cấp Độ Ngưỡng | Căn Cứ Khoa Học | Giá Trị Ngưỡng | Số Mẫu Phát Hiện | Tỷ Lệ Phát Hiện (%) | Mức Độ Can Thiệp Kỹ Thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Ngưỡng Lý Thuyết (Liu et al., 2008)** | $\mathbb{E}(h(\mathbf{x})) < c(\psi) \iff s \ge 0.50$ | `0.5000` | 1,690 | **14.57%** | Cảnh báo giám sát thường quy |
| **2. Phân Vị Ô Nhiễm Ước Lượng (Top 5%)** | Phân vị $P_{95}$ từ phân phối điểm tập Train | `0.5725` | 612 | **5.28%** | Cảnh báo bất thường telemetry |
| **3. Phân Vị Nguy Hiểm Cao (Top 1%)** | Phân vị $P_{99}$ từ phân phối điểm tập Train | `0.6436` | 119 | **1.03%** | Báo động nguy cơ sự cố cấp 2 |
| **4. Phân Vị Cực Đoan Thảm Họa (Top 0.1%)** | Phân vị $P_{99.9}$ từ phân phối điểm tập Train | `0.6687` | 16 | **0.14%** | Báo động đỏ / Ngắt hệ thống khẩn cấp |
| **5. Giới Hạn Thống Kê ($\mu + 2\sigma$)** | Độ lệch chuẩn phân phối Gaussian | `0.5612` | 710 | **6.12%** | Theo dõi trôi dạt dữ liệu (Data Drift) |

---

## 5. Chẩn Đoán Nguyên Nhân Gốc Rễ (Root Cause Analysis - RCA)

Hệ thống cung cấp cơ chế giải thích mô hình hộp trắng (White-box Interpretability) phục vụ trực tiếp kỹ sư buồng lái:

1. **Độ quan trọng đặc trưng không giám sát (Unsupervised Feature Importance):**
   - Kết hợp hệ số tương quan tuyến tính Pearson $|r(X_j, s)|$ với độ lệch chuẩn hóa trung bình của các mẫu dị biệt so với phân phối danh định.
   - **Top 3 cảm biến chi phối mạnh nhất đến sự bất thường:** `feat_9`, `feat_8`, `feat_10`.
2. **Định danh nguyên nhân gốc rễ (Root Cause Analysis):**
   - Với mỗi mẫu bất thường phát hiện được, hệ thống tính toán vector độ lệch chuẩn hóa đa biến:
     $$Z_j = \frac{x_j - \mu_{j, \text{normal}}}{\sigma_{j, \text{normal}}}$$
   - Định danh chính xác kênh cảm biến phát sinh xung lệch cực đoan (ví dụ: `feat_2` lệch $+61.89\sigma$ hoặc `feat_9` lệch $+3.87\sigma$), giúp kỹ sư định vị hư hỏng van hoặc rò rỉ áp suất ngay lập tức.

---

## 6. Khung 21 Bước Phát Triển Mô Hình Machine Learning

Toàn bộ quy trình trong dự án và Jupyter Notebook [shuttle_anomaly_detection_iforest.ipynb](file:///d:/May_Hoc/iforest/shuttle_anomaly_detection_iforest.ipynb) được liên kết chặt chẽ theo 21 bước kinh điển của vòng đời học máy:

| STT | Bước Chuẩn Hóa | Mô Tả Thực Hiện Trong Dự Án |
| :---: | :--- | :--- |
| **1** | **Problem Definition** | Nhận diện sớm sai lệch telemetry cảm biến tàu con thoi NASA; 10 kênh đầu vào thô, đầu ra là chỉ số $s \in [0, 1]$ và cảnh báo phân tầng. |
| **2** | **Nature of ML Problem** | Xác định bản chất Thuần Không Giám Sát (Unsupervised Learning); dị thường là thiểu số tách biệt, không thể dùng hệ luật if-else cố định. |
| **3** | **Domain Understanding** | Khảo sát 10 kênh đo đạc áp suất, lưu lượng, nhiệt độ buồng trao đổi; phân tích thống kê mô tả bậc 1 đến 4 và ma trận tương quan Pearson. |
| **4** | **Data Cleaning** | Thẩm định tính toàn vẹn (0 NaN, 0 Inf); bảo toàn 100% giá trị ngoại lai vật lý phục vụ phát hiện dị thường. |
| **5** | **Feature Scaling** | Chứng minh toán học và thực nghiệm tính bất biến tỷ lệ của $iTree$ trước phép biến đổi đơn điệu; giữ nguyên thang đo nguyên gốc `int64`. |
| **6** | **Categorical Encoding** | Thẩm định 10 kênh đều là biến định lượng liên tục/rời rạc, không phát sinh chi phí mã hóa One-Hot. |
| **7** | **Algorithm & Loss Selection** | Lựa chọn Isolation Forest (Liu et al., 2008); thiết lập hàm mục tiêu điểm dị biệt $s(\mathbf{x}, \psi) = 2^{-\mathbb{E}(h(\mathbf{x}))/c(\psi)}$ qua phân hoạch không gian ngẫu nhiên. |
| **8** | **Feature Engineering** | Khảo sát lời nguyền số chiều; tận dụng cơ chế phân hoạch 1D ngẫu nhiên ở mỗi nút để kháng hiện tượng đồng nhất khoảng cách. |
| **9** | **Data Splitting & Leakage** | Phân chia Train/Test 80/20 ($46,400$ / $11,600$) với tập chỉ số rời rạc tuyệt đối (`isdisjoint == True`), loại trừ triệt để rò rỉ thông tin. |
| **10** | **Splitting Strategy** | Luận giải sự phù hợp của phân chia ngẫu nhiên đồng đều có cố định hạt giống (`random_state=42`) trên dữ liệu không nhãn. |
| **11** | **Baseline Model** | Xây dựng mô hình chuẩn đối sánh thống kê đa biến (Distance-to-Centroid / Normalized Euclidean Baseline). |
| **12** | **No Free Lunch Theorem** | Phân tích không gian giả thuyết phân hoạch trực giao của $iTree$ so với giả định phân phối lồi của Baseline; phân tích giới hạn với dị thường phân bố xiên góc. |
| **13** | **Bias-Variance Tradeoff** | Khảo sát số lượng cây $t$ (kiểm soát Variance, hội tụ tại $t \ge 100$) và kích thước mẫu con $\psi=256$ (kiểm soát Bias, chống swamping và masking). |
| **14** | **Hyperparameter Tuning** | Thiết lập lưới tìm kiếm không giám sát $\psi \in \{128, 256\}$, $t \in \{50, 100\}$ theo tiêu chí tối đa hóa độ trải rộng điểm số (Score Spread $\sigma_s$). |
| **15** | **Evaluation Metrics** | Xây dựng hệ thống 5 cấp độ ngưỡng phân tầng (Lý thuyết $0.50$, Top 5%, Top 1%, Top 0.1%, Gaussian $\mu + 2\sigma$). |
| **16** | **Cross-Validation** | Thực hiện Unsupervised 3-Fold Cross-Validation trên tập Train, chứng minh tính ổn định cao của phân phối điểm số trên từng fold kiểm định. |
| **17** | **Model Training & Inference** | Huấn luyện mô hình sản xuất tối ưu ($n=100, \psi=256$) trên $46,400$ mẫu Train và thực hiện suy luận trên $11,600$ mẫu Test. |
| **18** | **Statistical Significance** | Tính khoảng tin cậy 95% ($\text{CI}_{95\%} = [0.3957, 0.3986]$) và kiểm định giả thuyết hai mẫu độc lập $Z$-test ($Z = -34.81, p < 10^{-50}$). |
| **19** | **Error Analysis** | Phân tích độ bất định tại vùng biên quyết định $s \in [0.48, 0.52]$; thiết lập cơ chế vùng đệm cảnh báo vàng (Yellow Buffer Zone) kết hợp chuyên gia. |
| **20** | **Model Interpretability & RCA** | Trích xuất Top 5 mẫu bất thường nhất kèm chẩn đoán căn nguyên (RCA) và phân tích mật độ phân bố KDE của Top 3 cảm biến chủ đạo. |
| **21** | **Iterative Deployment** | Thiết lập chu trình theo dõi trôi dạt dữ liệu (Drift Detection) và đóng gói lớp `TelemetryInferencePipeline` tối ưu thời gian suy luận dưới 1ms/mẫu. |

---

## 7. Cài Đặt & Hướng Dẫn Vận Hành

### 7.1. Cài Đặt Môi Trường
Yêu cầu Python $\ge$ 3.10 (khuyến nghị Python 3.13). Cài đặt các thư viện phụ thuộc:

```powershell
py -3.13 -m pip install -r requirements.txt
```

### 7.2. Thực Thi Pipeline Không Giám Sát (Single Run)
Chạy quy trình huấn luyện, phân tích 5 ngưỡng quyết định và trích xuất Root Cause Analysis:

```powershell
py -3.13 run_pipeline.py
```

### 7.3. Tinh Chỉnh Siêu Tham Số với Unsupervised 3-Fold CV
Chạy khảo sát lưới tham số và kiểm định chéo K-Fold không giám sát:

```powershell
py -3.13 run_pipeline.py --tune
```

### 7.4. Chạy Toàn Bộ Bộ Kiểm Thử Tự Động (Unit Tests)
Thực thi 19 bài kiểm thử nghiêm ngặt bao quát toán học, thuật toán và tính tái lập:

```powershell
py -3.13 -m pytest tests/test_pipeline.py -v
```

*Kết quả kiểm thử thực tế:* **19 passed in 1.04s (100% Pass Rate)**.

### 7.5. Khởi Chạy Jupyter Notebook
Mở notebook phân tích trực quan toàn diện:

```powershell
jupyter notebook shuttle_anomaly_detection_iforest.ipynb
```

---

## 8. Mô Phỏng Suy Luận Thời Gian Thực (Inference Simulation)

Đoạn mã mẫu thể hiện cách tích hợp mô hình vào luồng dữ liệu telemetry thời gian thực:

```python
import numpy as np
import pandas as pd
from model import IsolationForest
from run_pipeline import Pipeline

# 1. Nạp dữ liệu và huấn luyện mô hình sản xuất
df = pd.read_csv("shuttle.csv", header=None)
X_train = df.iloc[:46400].values

iforest = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
pipe = Pipeline(model=iforest)
pipe.fit(X_train)

# 2. Gói tin telemetry cảm biến mới từ tàu con thoi (10 thuộc tính vật lý)
test_packet = np.array([[50, 21, 77, 0, 28, 0, 27, 48, 22, 2]])

# 3. Tính Anomaly Score trực tiếp
score = float(pipe.anomaly_score(test_packet)[0])

# 4. Phân loại theo hệ thống ngưỡng quyết định
if score >= 0.6436:
    status = "NGUY HIỂM CAO (BÁO ĐỘNG ĐỎ)"
elif score >= 0.5000:
    status = "CẢNH BÁO BẤT THƯỜNG (MỨC VÀNG)"
else:
    status = "DANH ĐỊNH (BÌNH THƯỜNG)"

print(f"Anomaly Score: {score:.6f} -> Trạng thái: {status}")
```

---

## 9. Cam Kết Tiêu Chuẩn Kỹ Thuật & Học Thuật

1. **Zero Scikit-Learn Dependency:** 100% cấu trúc cây nhị phân, thuật toán phân hoạch không gian và tính toán đường đi được hiện thực thuần túy bằng Python và NumPy.
2. **Chuẩn Mực Học Không Giám Sát:** Tuyệt đối không sử dụng nhãn mục tiêu ($y$) trong toàn bộ chu trình phát triển.
3. **Phòng Chống Rò Rỉ Dữ Liệu Tuyệt Đối:** Mọi tham số phân vị và thống kê đều được ước lượng độc quyền trên tập Train trước khi áp dụng cho tập Test.
4. **Giải Thích Minh Bạch (White-Box AI):** Định vị trực tiếp kênh cảm biến sự cố qua độ lệch $Z$-score đa biến, đáp ứng yêu cầu an toàn khắt khe trong hàng không vũ trụ.
5. **Độ Tin Cậy & Tinh Gọn Mã Nguồn:** 0 lỗi/cảnh báo linter (Pyrefly 0 diagnostics), 100% không chứa icon/emoji, 19/19 unit tests tự động vượt qua hoàn hảo.
