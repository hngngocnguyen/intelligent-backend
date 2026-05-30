from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st  # type: ignore[reportMissingImports]

from utils.api_client import ApiClient


# ── colour helpers ────────────────────────────────────────────────────────────
_RISK_COLORS = {"low": "#059669", "moderate": "#ea580c", "high": "#dc2626"}
_RISK_BG     = {"low": "rgba(5, 150, 105, 0.12)", "moderate": "rgba(234, 88, 12, 0.12)", "high": "rgba(220, 38, 38, 0.12)"}
_RISK_EMOJI  = {"low": "🟢", "moderate": "🟠", "high": "🔴"}


def _gauge_figure(probability: float, risk_level: str) -> go.Figure:
    bar_color = _RISK_COLORS.get(risk_level, "#2563eb")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 42, "color": "var(--ink)", "family": "Manrope"}},
            delta={"reference": 50, "increasing": {"color": "#dc2626"}, "decreasing": {"color": "#059669"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "var(--muted)"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": "rgba(5, 150, 105, 0.1)"},
                    {"range": [35, 70], "color": "rgba(234, 88, 12, 0.1)"},
                    {"range": [70,100], "color": "rgba(220, 38, 38, 0.1)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 3},
                    "thickness": 0.82,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="var(--ink)", family="Manrope"),
    )
    return fig


def _feature_bar_figure(feature_summary: dict) -> go.Figure:
    numeric = {k: float(v) for k, v in feature_summary.items() if isinstance(v, (int, float))}
    if not numeric:
        return None
    labels = list(numeric.keys())
    values = list(numeric.values())
    colors = ["#2563eb" if v >= 1 else "#ea580c" for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=20, b=20, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(17,24,39,0.06)"),
        xaxis=dict(showgrid=False),
        font=dict(color="var(--ink)", size=12, family="Manrope"),
    )
    return fig


def render_tabular_page(client: ApiClient) -> None:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <h2 style="font-family:'Fraunces',serif;font-size:1.5rem;margin-bottom:0.3rem">
            🧬 Prédiction Tabulaire — Modèle Deep Learning
        </h2>
        <p style="color:inherit;opacity:0.75;font-size:0.92rem">
            Saisissez les valeurs cliniques du patient. Le modèle MLP entraîné sur le
            dataset <strong>Pima Indians Diabetes</strong> estime le risque en temps réel.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        model_version = st.session_state.get("tabular_model_version", "mlp_v3")
        st.markdown(
            f'<p style="font-size:0.8rem;color:inherit;opacity:0.75;margin-bottom:0.6rem">'
            f'Modèle actif : <strong style="color:#0f6b6f">{model_version}</strong> — changeable dans le panneau latéral</p>',
            unsafe_allow_html=True,
        )
        with st.form("tabular_form", border=False):
            c1, c2 = st.columns(2)
            with c1:
                pregnancies = st.number_input("🤰 Grossesses", 0, 25,
                    key="tabular_pregnancies")
                blood_pressure = st.slider("💉 Pression artérielle (mmHg)", 0.0, 140.0,
                    key="tabular_blood_pressure")
                insulin = st.slider("🧪 Insuline (μU/mL)", 0.0, 900.0,
                    key="tabular_insulin")
                pedigree = st.slider("🧬 Indice familial (DPF)", 0.0, 3.0,
                    key="tabular_pedigree")
            with c2:
                glucose = st.slider("🍬 Glucose (mg/dL)", 0.0, 250.0,
                    key="tabular_glucose")
                skin_thickness = st.slider("📏 Épaisseur cutanée (mm)", 0.0, 99.0,
                    key="tabular_skin_thickness")
                bmi = st.slider("⚖️ IMC (BMI)", 0.0, 60.0,
                    key="tabular_bmi")
                age = st.slider("🎂 Âge", 1, 100,
                    key="tabular_age")

            submitted = st.form_submit_button("⚡ Lancer la prédiction", use_container_width=True)

    if submitted:
        payload = {
            "pregnancies": pregnancies, "glucose": glucose,
            "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
            "insulin": insulin, "bmi": bmi,
            "diabetes_pedigree_function": pedigree, "age": age,
        }
        with st.spinner("Analyse en cours…"):
            result = client.predict_tabular(payload, model_version)

        prob       = result.get("probability", 0.0)
        prediction = result.get("prediction", "-")
        risk       = result.get("risk_level", "low")
        latency    = result.get("inference_time_ms", 0)

        # ── right column: gauge ──────────────────────────────────────────────
        with col_right:
            st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.1rem;font-weight:600;margin-bottom:0">Score de risque estimé</p>', unsafe_allow_html=True)
            st.plotly_chart(_gauge_figure(prob, risk), use_container_width=True)

        # ── risk badge ───────────────────────────────────────────────────────
        emoji = _RISK_EMOJI.get(risk, "⚪")
        color = _RISK_COLORS.get(risk, "#2563eb")
        bg    = _RISK_BG.get(risk, "rgba(37,99,235,0.1)")
        st.markdown(f"""
        <div class="glass-card" style="background:{bg};border:1px solid {color}33;
                    padding:1.4rem;margin:1rem 0;display:flex;align-items:center;gap:1.2rem">
            <span style="font-size:2.5rem">{emoji}</span>
            <div>
                <p style="margin:0;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.12em;
                           font-weight:800;color:{color}">Résultat de la prédiction</p>
                <p style="margin:0;font-size:1.6rem;font-weight:800;color:{color}">{prediction.upper()}</p>
                <p style="margin:0;font-size:0.9rem;color:inherit;opacity:0.75;font-weight:500;">
                    Probabilité : <strong>{prob:.1%}</strong> &nbsp;·&nbsp; Risque : <strong>{risk}</strong>
                    &nbsp;·&nbsp; Latence : <strong>{latency} ms</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── engineered feature bars ──────────────────────────────────────────
        feature_summary = result.get("feature_summary", {})
        if feature_summary:
            st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.05rem;font-weight:600;margin:1rem 0 0.4rem">Signaux cliniques dérivés</p>', unsafe_allow_html=True)
            # Text features
            text_feats = {k: v for k, v in feature_summary.items() if isinstance(v, str)}
            if text_feats:
                cols_tf = st.columns(len(text_feats))
                for col, (k, v) in zip(cols_tf, text_feats.items()):
                    col.metric(k.replace("_", " ").title(), v)
            # Numeric feature bar chart
            fig_feat = _feature_bar_figure(feature_summary)
            if fig_feat:
                st.plotly_chart(fig_feat, use_container_width=True)

        with st.expander("📄 Réponse JSON brute"):
            st.json(result)
    else:
        with col_right:
            st.markdown("""
            <div class="glass-card" style="padding:3rem 2rem;text-align:center;margin-top:1.5rem">
                <p style="font-size:2.5rem;margin:0">🩺</p>
                <p style="color:inherit;opacity:0.75;margin:1rem 0 0;font-size:1.05rem;line-height:1.6">
                    Remplissez le formulaire et cliquez sur<br>
                    <strong>Lancer la prédiction</strong> pour voir le score.
                </p>
            </div>
            """, unsafe_allow_html=True)
