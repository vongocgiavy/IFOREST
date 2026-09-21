#!/usr/bin/env python3
"""
SAVE WEIGHTS — ISOLATION FOREST PROJECT
=========================================
Script chay MOT LAN de:
  1. Nap du lieu shuttle.csv (9 cam bien, tach nhan feat_10)
  2. Phan chia train/test 80/20
  3. [Neu --tune] Tim cau hinh tot nhat qua Unsupervised 3-Fold CV (score spread)
  4. Train mo hinh tot nhat tren toan bo X_train
  5. Luu weights ra weights/ (JSON + NPZ + TXT)

Dung:
  py -3 save_weights.py              # Luu baseline_weights (DEFAULT_CONFIG)
  py -3 save_weights.py --tune       # Tim best + luu best_model_weights
  py -3 save_weights.py --tune --name my_exp_weights  # Ten file tuy chinh

KHONG chay trong vong lap hay len lich — chi chay khi can cap nhat trong so moi.
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import numpy as np
import pandas as pd

from model import IsolationForest
from weights import DEFAULT_CONFIG, PARAM_GRID, KFOLD_N_SPLITS, WEIGHTS_DIR, save_weights


# ---------------------------------------------------------------------------
# Helpers noi bo (khong phu thuoc run_pipeline.py)
# ---------------------------------------------------------------------------

def _train_test_split(X: pd.DataFrame, y: np.ndarray,
                      test_size: float = 0.2, random_state: int = 42):
    """Phan chia train/test thuan NumPy. Tra ve (X_train, X_test, y_train, y_test)."""
    rng = np.random.RandomState(random_state)
    n   = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    train_idx, test_idx = idx[n_test:], idx[:n_test]
    return (X.iloc[train_idx].copy(), X.iloc[test_idx].copy(),
            y[train_idx], y[test_idx])


def _kfold(n: int, k: int = KFOLD_N_SPLITS, random_state: int = 42):
    """K-Fold thuan NumPy, yield (train_idx, val_idx)."""
    rng   = np.random.RandomState(random_state)
    idx   = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Nap du lieu
# ---------------------------------------------------------------------------

def load_data(data_path: str):
    """Nap shuttle.csv, tach nhan feat_10. Tra ve (X, y_raw)."""
    if not os.path.exists(data_path):
        print(f"[!] Khong tim thay file: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, header=None)
    df.columns = [f"feat_{i+1}" for i in range(df.shape[1])]

    sensor_cols = [f"feat_{i+1}" for i in range(9)]   # 9 cam bien thuc su
    X     = df[sensor_cols].copy()
    y_raw = df["feat_10"].values.astype(int)           # nhan UCI 1-7

    print(f"[data] Nap thanh cong: {len(X):,} mau x {len(sensor_cols)} cam bien")
    return X, y_raw


# ---------------------------------------------------------------------------
# Tim sieu tham so tot nhat (Unsupervised 3-Fold CV, score spread)
# ---------------------------------------------------------------------------

def find_best_params(X_train: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Duyet PARAM_GRID voi K-Fold CV khong giam sat.
    Tieu chi: Score Spread (std anomaly score tren val) LON NHAT
    -> phan biet tot nhat normal vs anomaly ma khong can nhan.
    """
    k = KFOLD_N_SPLITS
    print(f"\n[tune] Unsupervised {k}-Fold CV tren PARAM_GRID ({len(PARAM_GRID)} configs)...")

    best_spread: float = float("-inf")
    best_params: dict  = {}

    for idx, p in enumerate(PARAM_GRID, 1):
        n_est = int(p["n_estimators"])
        m_samp = p["max_samples"]
        m_feat = float(p["max_features"])

        fold_stds = []
        for tri, vi in _kfold(len(X_train), k=k, random_state=random_state):
            m = IsolationForest(
                n_estimators=n_est, max_samples=m_samp,
                max_features=m_feat, contamination="auto",
                random_state=random_state,
            )
            m.fit(X_train.iloc[tri].values)
            val_scores = m.anomaly_score(X_train.iloc[vi].values)
            fold_stds.append(float(np.std(val_scores)))

        mean_spread = float(np.mean(fold_stds))
        print(f"  Config {idx}: n_est={n_est:3d} | max_samples={str(m_samp):>4s} | "
              f"max_feat={m_feat:.1f}  ->  Score Spread: {mean_spread:.4f}")

        if mean_spread > best_spread:
            best_spread = mean_spread
            best_params = {"n_estimators": n_est, "max_samples": m_samp, "max_features": m_feat}

    print(f"[tune] Cau hinh tot nhat: {best_params}  (spread={best_spread:.4f})")
    return best_params


# ---------------------------------------------------------------------------
# Script chinh — chay mot lan, khong vong lap
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Luu trong so IsolationForest ra weights/ (chay mot lan)."
    )
    ap.add_argument("--data_path",    type=str,   default=DEFAULT_CONFIG["data_path"],
                    help="Duong dan dataset (mac dinh: shuttle.csv)")
    ap.add_argument("--test_size",    type=float, default=DEFAULT_CONFIG["test_size"],
                    help="Ti le tap test (mac dinh: 0.2)")
    ap.add_argument("--random_state", type=int,   default=DEFAULT_CONFIG["random_state"],
                    help="Seed ngau nhien (mac dinh: 42)")
    ap.add_argument("--tune",         action="store_true",
                    help="Tim sieu tham so tot nhat qua Unsupervised K-Fold CV")
    ap.add_argument("--name",         type=str,   default="",
                    help="Prefix ten file (tu dong: baseline_weights | best_model_weights)")
    args = ap.parse_args()

    print("=" * 60)
    print("  SAVE WEIGHTS — ISOLATION FOREST")
    print("=" * 60)

    # 1. Nap du lieu
    X, y_raw = load_data(args.data_path)

    # 2. Phan chia train/test (chi dung X_train de fit, y khong dung trong fit)
    X_train, X_test, _, _ = _train_test_split(
        X, y_raw, test_size=args.test_size, random_state=args.random_state
    )
    print(f"[data] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Chon sieu tham so
    if args.tune:
        best_params = find_best_params(X_train, random_state=args.random_state)
        file_name   = args.name or "best_model_weights"
    else:
        best_params = {
            "n_estimators": DEFAULT_CONFIG["n_estimators"],
            "max_samples":  DEFAULT_CONFIG["max_samples"],
            "max_features": DEFAULT_CONFIG["max_features"],
        }
        file_name = args.name or "baseline_weights"
        print(f"\n[info] Dung DEFAULT_CONFIG (khong --tune): {best_params}")

    # 4. Train mo hinh tren toan bo X_train
    print(f"\n[train] Huan luyen '{file_name}': {best_params} ...")
    best_model = IsolationForest(
        n_estimators  = best_params["n_estimators"],
        max_samples   = best_params["max_samples"],
        max_features  = best_params["max_features"],
        contamination = "auto",
        random_state  = args.random_state,
    ).fit(X_train.values)
    print(f"[train] Xong. threshold={best_model.threshold_:.4f} | "
          f"c_psi={best_model.c_psi_:.4f} | n_cay={len(best_model.trees)}")

    # 5. Luu weights (mot lan, khong vong lap)
    save_weights(best_model, best_params, name=file_name, output_dir=WEIGHTS_DIR)

    print(f"\n[done] Hoan tat. Chi chay lai file nay khi muon cap nhat trong so moi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
