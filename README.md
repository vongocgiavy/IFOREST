# Isolation Forest — NASA Shuttle Telemetry Anomaly Detection
### Pure NumPy · Zero Scikit-Learn · 100% Unsupervised

Phat hien bat thuong tren du lieu cam bien tau con thoi NASA (UCI Statlog Shuttle, 58,000 mau x 9 cam bien) bang thuat toan **Isolation Forest** tu xay dung hoan toan tu dau (Liu, Ting & Zhou, 2008) — khong phu thuoc bat ky thu vien ML bac cao nao.

---

## Cau Truc Du An

```
iforest/
├── model.py              # Isolation Forest thuan NumPy (Node, IsolationTree, IsolationForest)
├── run_pipeline.py       # Pipeline dieu phoi: CV, RCA, Baseline so sanh, CLI
├── weights.py            # Hang so cau hinh + ham save_weights() / load_weights()
├── save_weights.py       # Script mot lan: train -> luu trong so ra weights/
│
├── weights/
│   ├── baseline_weights.{json,npz,txt}    # DEFAULT_CONFIG (n_est=100, ms=256)
│   └── best_model_weights.{json,npz,txt}  # Tot nhat qua 3-Fold CV (n_est=100, ms=128)
│
├── shuttle.csv           # Dataset NASA (58,000 x 10: 9 cam bien + 1 nhan UCI)
├── shuttle_anomaly_detection_iforest.ipynb  # Notebook 21 buoc, 50 cells, 18 do thi
├── shuttle_anomaly_detection_iforest.html   # Xuat HTML day du
├── requirements.txt
└── tests/
    └── test_pipeline.py  # 21 unit tests (100% pass)
```

---

## Du Lieu

| Thuoc tinh | Gia tri |
|:-----------|:--------|
| So mau | 58,000 |
| So cam bien | 9 (feat_1 -> feat_9) |
| Cot nhan UCI | feat_10 — **boc tach hoan toan**, chi dung external validation |
| Ti le bat thuong (test) | 21.48% (class 2-7 theo UCI Statlog) |
| Chat luong du lieu | 0 NaN, 0 Inf |

> **Label Leakage = 0%**: feat_10 khong duoc dua vao ma tran X khi huan luyen. ROC-AUC thuc te (leak-free) = **0.8474**.

---

## Ly Thuyet Cot Loi

Anomaly Score theo Liu et al. (2008):

s(\mathbf{x}, \psi) = 2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(\psi)}}, \quad c(\psi) = 2\ln(\psi-1) + 2\gamma - \frac{2(\psi-1)}{\psi}

- s → 1: bat thuong (co lap som, duong di ngan)
- s ≈ 0.5: binh thuong
- γ ≈ 0.5772 (hang so Euler-Mascheroni)

**Nguong quyet dinh (5 cap, khong giam sat):**

| Nguong | Gia tri | Phat hien | Y nghia |
|:-------|:-------:|:---------:|:--------|
| Ly thuyet s >= 0.50 | 0.5000 | 13.73% | Giam sat thuong quy |
| Top 5% (P95) | 0.5559 | 5.22% | Canh bao telemetry |
| Top 1% (P99) | 0.6308 | 1.17% | Bao dong cap 2 |
| Top 0.1% (P99.9) | 0.6625 | 0.10% | Bao dong do / Ngat khan |
| mu + 2*sigma | 0.5498 | 5.77% | Theo doi data drift |

---

## Ket Qua

| Chi so | Gia tri |
|:-------|:-------:|
| ROC-AUC (leak-free, 9 cam bien) | **0.8474** |
| Average Precision (AP) | **0.6461** |
| AP / Baseline | **3.0x** |
| Precision @ Top 5% | **93.1%** |
| Spearman stability (RCA) | rho >= 0.98 |
| Unit tests | **21/21 passed** |

---

## Cai Dat & Chay

```powershell
# Cai thu vien
py -3 -m pip install -r requirements.txt

# Chay pipeline day du
py -3 run_pipeline.py

# Tim sieu tham so tot nhat (3-Fold CV)
py -3 run_pipeline.py --tune

# Luu trong so (chay mot lan)
py -3 save_weights.py              # -> weights/baseline_weights.*
py -3 save_weights.py --tune       # -> weights/best_model_weights.*

# Unit tests
py -3 -m pytest tests/test_pipeline.py -v

# Notebook
jupyter notebook shuttle_anomaly_detection_iforest.ipynb
```

---

## Tai Lai Trong So Da Luu

```python
from weights import load_weights

# Tai model tot nhat (khong can train lai)
model, params = load_weights("best_model_weights")

# Suy luan truc tiep
import numpy as np
packet = np.array([[50, 21, 77, 0, 28, 0, 27, 48, 22]])
score  = float(model.anomaly_score(packet)[0])
status = "BAT THUONG" if score >= 0.5 else "BINH THUONG"
print(f"Score: {score:.4f} -> {status}")
```

---

## Dac Diem Ky Thuat

- **Zero Scikit-Learn**: toan bo cay nhi phan, phan hoach khong gian va tinh duong di bang Python + NumPy thuan.
- **RCA phi tham so**: thay Z-score bang Percentile Rank Deviation — mien nhiem voi kurtosis > 2,000 (feat_2: k=2647; feat_4: k=7698).
- **Weights I/O**: serialize/deserialize toan bo cay sang JSON (khong dung pickle), NPZ cho inference nhanh.
- **21 buoc ML chuan**: Problem Definition -> Deployment, to chuc day du trong notebook.

---

## Tai Lieu Tham Khao

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. ICDM 2008, pp. 413-422.
