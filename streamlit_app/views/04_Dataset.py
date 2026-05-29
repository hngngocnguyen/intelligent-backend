from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st  # type: ignore[reportMissingImports]


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
    }
    if "Outcome" in df.columns:
        value_counts = df["Outcome"].value_counts(dropna=False).to_dict()
        summary["target_balance"] = {
            str(key): int(value) for key, value in value_counts.items()
        }
    return summary


def render_dataset_page() -> None:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <h2 style="font-family:'Fraunces',serif;font-size:1.5rem;margin-bottom:0.3rem">
            📊 Exploration du Dataset
        </h2>
        <p style="color:#6b7280;font-size:0.95rem">
            Aperçu des données cliniques historiques (Pima Indians Diabetes) 
            utilisées pour entraîner le <strong>modèle Deep Learning tabulaire</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    available_files = [
        DATA_DIR / "diabetes_clean.csv",
        DATA_DIR / "diabetes.csv",
    ]
    available_files = [path for path in available_files if path.exists()]

    if not available_files:
        st.warning(
            "Aucun fichier dataset trouvé dans le dossier data/."
        )
        return

    labels = [path.name for path in available_files]
    
    st.markdown('<div class="glass-card" style="padding:1.5rem;margin-bottom:1.5rem">', unsafe_allow_html=True)
    selected = st.selectbox("Sélectionner le fichier source", labels, index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    selected_path = next(
        path for path in available_files if path.name == selected
    )

    df = load_dataset(selected_path)
    summary = dataset_summary(df)

    st.markdown(
        '<div class="section-title">Synthèse globale</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric("Lignes", summary["rows"])
    cols[1].metric("Colonnes", summary["columns"])
    cols[2].metric("Valeurs manquantes", summary["missing_values"])

    if "target_balance" in summary:
        st.markdown(
            '<div class="section-title">Équilibre de la cible (Outcome)</div>',
            unsafe_allow_html=True,
        )
        
        balance_cols = st.columns(len(summary["target_balance"]))
        for i, (key, value) in enumerate(summary["target_balance"].items()):
            label = "Diabétique (1)" if key == "1" else "Non Diabétique (0)" if key == "0" else f"Classe {key}"
            color = "#ea580c" if key == "1" else "#059669" if key == "0" else "#2563eb"
            bg_color = "rgba(234, 88, 12, 0.12)" if key == "1" else "rgba(5, 150, 105, 0.12)" if key == "0" else "rgba(37,99,235,0.1)"
            
            with balance_cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="background:{bg_color};border:1px solid {color}44;padding:1.2rem;text-align:center">
                    <p style="margin:0;font-size:0.8rem;text-transform:uppercase;font-weight:700;color:{color}">{label}</p>
                    <p style="margin:0.5rem 0 0;font-size:1.8rem;font-weight:800;color:{color}">{value}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Aperçu des données</div>',
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="glass-card" style="padding:1rem">', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
