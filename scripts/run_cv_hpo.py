#!/usr/bin/env python3
"""Run randomized HPO + K-fold CV for the three MLP architectures.

Outputs (saved to `saved_models/` by default):
- `cv_hpo_results.json` : per-config CV metrics
- `cv_hpo_summary.json` : best config per architecture (by mean AUC)
- optional: best model files `best_v1.keras`, `best_v2.keras`, `best_v3.keras` and `scaler_{arch}.pkl`

Usage example:
    python scripts/run_cv_hpo.py --n-configs 6 --folds 5 --epochs 80 --save-models
"""
from pathlib import Path
import argparse
import json
import time
import math

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import joblib


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["glucose_insulin_ratio"] = f["Glucose"] / (f["Insulin"] + 1.0)
    f["glucose_bmi_index"]     = (f["Glucose"] / 100.0) * (f["BMI"] / 30.0)
    f["age_risk_index"]        = f["Age"] / 50.0
    f["family_risk_index"]     = f["DiabetesPedigreeFunction"] * 1.5
    return f


def build_v1(input_dim, params):
    units1 = params.get("units1", 32)
    units2 = params.get("units2", 16)
    lr = params.get("lr", 1e-3)
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units1, activation="relu"),
        keras.layers.Dense(units2, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return m


def build_v2(input_dim, params):
    units1 = params.get("units1", 64)
    units2 = params.get("units2", 32)
    d1 = params.get("dropout1", 0.3)
    d2 = params.get("dropout2", 0.2)
    lr = params.get("lr", 1e-3)
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units1, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(d1),
        keras.layers.Dense(units2, activation="relu"),
        keras.layers.Dropout(d2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return m


def build_v3(input_dim, params):
    units1 = params.get("units1", 128)
    units2 = params.get("units2", 64)
    units3 = params.get("units3", 32)
    d1 = params.get("dropout1", 0.4)
    d2 = params.get("dropout2", 0.3)
    lr = params.get("lr", 1e-3)
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units1, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(d1),
        keras.layers.Dense(units2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(d2),
        keras.layers.Dense(units3, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return m


def sample_configs(rng, arch, n):
    configs = []
    for _ in range(n):
        lr = float(10 ** rng.uniform(-4, -2))
        batch = int(rng.choice([16, 32, 64]))
        if arch == "v1":
            configs.append({
                "lr": lr,
                "batch": batch,
                "units1": int(rng.choice([32, 48, 64])),
                "units2": int(rng.choice([16, 24, 32])),
            })
        elif arch == "v2":
            configs.append({
                "lr": lr,
                "batch": batch,
                "units1": int(rng.choice([48, 64, 96])),
                "units2": int(rng.choice([24, 32, 48])),
                "dropout1": float(rng.choice([0.2, 0.3, 0.4])),
                "dropout2": float(rng.choice([0.1, 0.2, 0.3])),
            })
        else:
            configs.append({
                "lr": lr,
                "batch": batch,
                "units1": int(rng.choice([64, 96, 128])),
                "units2": int(rng.choice([48, 64])),
                "units3": int(rng.choice([16, 32])),
                "dropout1": float(rng.choice([0.3, 0.4])),
                "dropout2": float(rng.choice([0.2, 0.3])),
            })
    return configs


def run_cv_hpo(
    data_path: Path,
    out_dir: Path,
    n_configs_per_arch: int = 8,
    folds: int = 5,
    epochs: int = 80,
    seed: int = 42,
    save_models: bool = False,
):
    rng = np.random.default_rng(seed)

    df = pd.read_csv(data_path)
    df = add_features(df)
    feature_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "glucose_insulin_ratio",
        "glucose_bmi_index",
        "age_risk_index",
        "family_risk_index",
    ]
    X = df[feature_cols].values
    y = df["Outcome"].values

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    search_space = {
        "v1": sample_configs(rng, "v1", n_configs_per_arch),
        "v2": sample_configs(rng, "v2", n_configs_per_arch),
        "v3": sample_configs(rng, "v3", n_configs_per_arch),
    }

    results = []

    start_time = time.time()
    for arch, configs in search_space.items():
        print(f"Running arch={arch} configs={len(configs)}")
        for i, cfg in enumerate(configs, start=1):
            fold_aucs = []
            fold_f1s = []
            fold_precs = []
            fold_recs = []
            t0 = time.time()
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                scaler = StandardScaler().fit(X_train)
                Xt = scaler.transform(X_train)
                Xv = scaler.transform(X_val)

                cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
                class_weights = {0: float(cw[0]), 1: float(cw[1])}

                input_dim = Xt.shape[1]
                if arch == "v1":
                    model = build_v1(input_dim, cfg)
                elif arch == "v2":
                    model = build_v2(input_dim, cfg)
                else:
                    model = build_v3(input_dim, cfg)

                cb = [
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6, verbose=0),
                ]

                history = model.fit(
                    Xt,
                    y_train,
                    validation_data=(Xv, y_val),
                    epochs=epochs,
                    batch_size=int(cfg["batch"]),
                    callbacks=cb,
                    verbose=0,
                    class_weight=class_weights,
                )

                probs = model.predict(Xv, verbose=0).ravel()
                auc = float(roc_auc_score(y_val, probs))
                preds = (probs >= 0.5).astype(int)
                f1 = float(f1_score(y_val, preds, zero_division=0))
                prec = float(precision_score(y_val, preds, zero_division=0))
                rec = float(recall_score(y_val, preds, zero_division=0))

                fold_aucs.append(auc)
                fold_f1s.append(f1)
                fold_precs.append(prec)
                fold_recs.append(rec)

                tf.keras.backend.clear_session()

            elapsed = time.time() - t0
            res = {
                "arch": arch,
                "config": cfg,
                "mean_auc": float(np.mean(fold_aucs)),
                "std_auc": float(np.std(fold_aucs)),
                "mean_f1": float(np.mean(fold_f1s)),
                "std_f1": float(np.std(fold_f1s)),
                "mean_precision": float(np.mean(fold_precs)),
                "mean_recall": float(np.mean(fold_recs)),
                "time_s": float(elapsed),
            }
            results.append(res)
            print(f"[{arch}] cfg {i}/{len(configs)} — AUC {res['mean_auc']:.4f} F1 {res['mean_f1']:.4f} (t={elapsed:.1f}s)")

    out_path = out_dir / "cv_hpo_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Summarize best per arch by mean_auc
    best_per_arch = {}
    for arch in ["v1", "v2", "v3"]:
        arch_results = [r for r in results if r["arch"] == arch]
        if arch_results:
            best = max(arch_results, key=lambda r: r["mean_auc"])
            best_per_arch[arch] = best

    summary_path = out_dir / "cv_hpo_summary.json"
    summary_path.write_text(json.dumps(best_per_arch, indent=2))

    # Optionally retrain best configs on full dataset and save
    if save_models:
        print("Retraining best configs on full dataset and saving models...")
        for arch, info in best_per_arch.items():
            cfg = info["config"]
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            if arch == "v1":
                model = build_v1(Xs.shape[1], cfg)
            elif arch == "v2":
                model = build_v2(Xs.shape[1], cfg)
            else:
                model = build_v3(Xs.shape[1], cfg)

            cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
            class_weights = {0: float(cw[0]), 1: float(cw[1])}

            cb = [
                keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=4, factor=0.5, min_lr=1e-6, verbose=0),
            ]
            model.fit(Xs, y, epochs=epochs, batch_size=int(cfg["batch"]), callbacks=cb, verbose=0, class_weight=class_weights)

            model_file = out_dir / f"best_{arch}.keras"
            scaler_file = out_dir / f"scaler_{arch}.pkl"
            model.save(str(model_file))
            joblib.dump(scaler, scaler_file)
            print(f"Saved {model_file} and {scaler_file}")

    total = time.time() - start_time
    print(f"\nDone — results: {out_path}  summary: {summary_path}  elapsed: {total:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/diabetes_clean.csv", help="Path to CSV data (default: ../data/diabetes_clean.csv)")
    p.add_argument("--out-dir", default="../saved_models", help="Output directory for results/models")
    p.add_argument("--n-configs", type=int, default=8, help="Number of random configs per architecture")
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--epochs", type=int, default=80, help="Max epochs per training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-models", action="store_true", help="Retrain best config on full data and save model+scaler")
    args = p.parse_args()

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cv_hpo(
        data_path=data_path,
        out_dir=out_dir,
        n_configs_per_arch=args.n_configs,
        folds=args.folds,
        epochs=args.epochs,
        seed=args.seed,
        save_models=args.save_models,
    )
