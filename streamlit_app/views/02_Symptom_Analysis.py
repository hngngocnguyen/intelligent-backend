from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st  # type: ignore[reportMissingImports]

from utils.api_client import ApiClient


def _scores_figure(scores: dict[str, float]) -> go.Figure:
    # Sort scores for better visualization
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    labels = [k.capitalize() for k, _ in sorted_scores]
    values = [v * 100 for _, v in sorted_scores]
    
    # Highlight the top category in a distinct color
    colors = ["#ea580c" if i < len(values) - 1 else "#2563eb" for i in range(len(values))]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(t=20, b=20, l=10, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(17,24,39,0.06)", range=[0, 105]),
        yaxis=dict(showgrid=False),
        font=dict(color="#111827", size=13, family="Manrope"),
    )
    return fig


def render_symptom_page(client: ApiClient) -> None:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <h2 style="font-family:'Fraunces',serif;font-size:1.5rem;margin-bottom:0.3rem">
            🗣️ Analyse des Symptômes — Zero-Shot NLP
        </h2>
        <p style="color:#5b6b74;font-size:0.92rem">
            Décrivez les symptômes ressentis par le patient en anglais. Le modèle 
            <strong>DeBERTa (Hugging Face)</strong> classifie ces symptômes sans aucun entraînement spécifique préalable.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1], gap="large")

    examples = {
        "Soif et fatigue": "I feel very thirsty, tired, and have blurry vision.",
        "Soif intense et urines fréquentes": "I feel extremely thirsty, I urinate frequently, and my vision is blurry.",
        "Fatigue et perte de poids": "I'm very tired all day, I've lost some weight, and I feel constantly hungry.",
        "Maux de tête et vertiges": "I have headaches, dizziness, and a pulsing sensation in my head.",
        "Douleurs thoraciques": "I feel chest tightness, shortness of breath, and occasional palpitations.",
        "Digestif": "I feel nauseous, have abdominal pain, and my digestion is upset.",
        "Engourdissements": "My hands feel numb and I have tingling in my feet.",
        "Aucun symptôme": "I feel fine and do not notice any symptoms.",
    }

    with col_left:
        st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.1rem;font-weight:600;margin-bottom:0.5rem">Entrée du patient</p>', unsafe_allow_html=True)
        
        selected_example = st.selectbox(
            "Exemples rapides",
            list(examples.keys()),
            index=0,
            key="symptom_example_select",
            on_change=lambda: st.session_state.__setitem__(
                "symptom_text",
                examples[st.session_state["symptom_example_select"]],
            ),
        )

        text = st.text_area(
            "Description des symptômes (en anglais)",
            height=150,
            key="symptom_text",
        )

        submitted = st.button("🧠 Lancer l'analyse NLP", use_container_width=True)

    if submitted:
        with st.spinner("Analyse sémantique en cours…"):
            result = client.analyze_symptoms(text)
            
        top_category = result.get("top_category", "-").title()
        confidence = result.get("confidence", 0.0)
        mode = result.get("mode", "-")
        model_name = result.get("model", "-")

        with col_right:
            st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.1rem;font-weight:600;margin-bottom:0.5rem">Résultat de la classification</p>', unsafe_allow_html=True)
            
            # ── prediction badge ────────────────────────────────────────────────
            bg = "rgba(37,99,235,0.1)"
            color = "#2563eb"
            if "diabetes" in top_category.lower():
                bg = "rgba(234, 88, 12, 0.12)"
                color = "#ea580c"
            elif "no significant" in top_category.lower():
                bg = "rgba(5, 150, 105, 0.12)"
                color = "#059669"
                
            st.markdown(f"""
            <div class="glass-card" style="background:{bg};border:1px solid {color}33;
                        padding:1.5rem;margin-bottom:1rem;text-align:center">
                <p style="margin:0;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.12em;
                           font-weight:800;color:{color}">Catégorie dominante</p>
                <p style="margin:0.5rem 0;font-size:1.8rem;font-weight:800;color:{color}">{top_category}</p>
                <div style="display:flex;justify-content:center;gap:1.5rem;margin-top:0.8rem">
                    <div>
                        <span style="font-size:0.85rem;color:#6b7280;font-weight:600;">Confiance</span><br>
                        <strong style="font-size:1.2rem;color:{color}">{confidence:.1%}</strong>
                    </div>
                    <div>
                        <span style="font-size:0.85rem;color:#6b7280;font-weight:600;">Mode</span><br>
                        <strong style="font-size:1.2rem;color:{color}">{mode.capitalize()}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(
                f'<p style="font-size:0.85rem;color:#6b7280;text-align:center">'
                f'Modèle : <strong>{model_name}</strong></p>',
                unsafe_allow_html=True,
            )

        st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.1rem;font-weight:600;margin-top:1.5rem">Distribution des probabilités sémantiques</p>', unsafe_allow_html=True)
        chart = _scores_figure(result.get("all_scores", {}))
        st.plotly_chart(chart, use_container_width=True)
        
        with st.expander("📄 Réponse JSON brute"):
            st.json(result)
    else:
        with col_right:
            st.markdown("""
            <div class="glass-card" style="padding:3rem 2rem;text-align:center;margin-top:1.5rem">
                <p style="font-size:2.5rem;margin:0">🧠</p>
                <p style="color:#6b7280;margin:1rem 0 0;font-size:1.05rem;line-height:1.6">
                    Sélectionnez un exemple ou décrivez les symptômes, puis cliquez sur<br>
                    <strong>Lancer l'analyse NLP</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
