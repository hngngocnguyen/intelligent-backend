#!/usr/bin/env python3
"""Post-process HuggingFace zero-shot batch results to correct likely mislabels.

Saves a corrected CSV and a distribution plot in `saved_models/`.
Heuristic rules are intentionally simple and interpretable; tweak thresholds as needed.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "saved_models"
IN_FILE = MODELS_DIR / "hf_batch_results.csv"
OUT_CSV = MODELS_DIR / "hf_batch_results_corrected.csv"
OUT_PNG = MODELS_DIR / "hf_batch_distribution_corrected.png"


def correct_row(row: pd.Series) -> str:
    """Return a corrected label for a single row using simple heuristics.

    Current heuristics:
    - If `score_diabetes` >= 0.20 => `diabetes symptoms` (promotes diabetic signal)
    - If top label is `fatigue and general weakness` but `score_diabetes` >= 0.08 => promote to diabetes
    - Otherwise keep `top_label`
    Tweak thresholds for your dataset or replace with an ML re-eval.
    """
    try:
        s = float(row.get("score_diabetes", 0.0))
    except Exception:
        s = 0.0
    top = str(row.get("top_label", "")).strip()

    if s >= 0.20:
        return "diabetes symptoms"
    if top == "fatigue and general weakness" and s >= 0.08:
        return "diabetes symptoms"
    return top


def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    df = pd.read_csv(IN_FILE)

    df["corrected_label"] = df.apply(correct_row, axis=1)

    print("Label counts — before")
    print(df["top_label"].value_counts())
    print("\nLabel counts — after (corrected)")
    print(df["corrected_label"].value_counts())

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote corrected CSV: {OUT_CSV}")

    # Plot corrected distribution and confidences
    counts = df["corrected_label"].value_counts()

    COLORS = {
        "diabetes symptoms": "#ef4444",
        "hypertension symptoms": "#f97316",
        "cardiovascular symptoms": "#eab308",
        "neurological symptoms": "#8b5cf6",
        "digestive symptoms": "#06b6d4",
        "fatigue and general weakness": "#10b981",
        "no significant symptoms": "#6b7280",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(counts.values, labels=counts.index, colors=[COLORS.get(l, "#94a3b8") for l in counts.index],
                autopct="%1.0f%%", startangle=140, pctdistance=0.8)
    axes[0].set_title("Repartition des categories (corrected)")

    bar_colors = [COLORS.get(l, "#94a3b8") for l in df["corrected_label"]]
    axes[1].barh(df["case_id"], df["confidence"], color=bar_colors, edgecolor="white")
    axes[1].axvline(0.5, color="red", ls="--", lw=1, alpha=0.6)
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("Score de confiance")
    axes[1].set_title("Confiance par cas patient (corrected)")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote corrected plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
