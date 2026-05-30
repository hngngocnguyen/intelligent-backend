from __future__ import annotations

import streamlit as st  # type: ignore[reportMissingImports]

from utils.api_client import ApiClient

_RISK_COLORS = {"low": "#059669", "moderate": "#ea580c", "high": "#dc2626"}
_RISK_BG     = {"low": "rgba(5, 150, 105, 0.12)", "moderate": "rgba(234, 88, 12, 0.12)", "high": "rgba(220, 38, 38, 0.12)"}

def render_pipeline_page(client: ApiClient) -> None:
    st.markdown("""
    <div style="margin-bottom:1rem">
        <h2 style="font-family:'Fraunces',serif;font-size:1.5rem;margin-bottom:0.3rem">
            ⚙️ Orchestration Multimodale (Consensus)
        </h2>
        <p style="color:inherit;opacity:0.75;font-size:0.92rem">
            Combine le <strong>modèle Deep Learning tabulaire</strong>, le <strong>modèle NLP Zero-Shot</strong>, et 
            l'<strong>API LLM</strong> pour générer un diagnostic final robuste par consensus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    pipeline_examples = {
        "Profil diabète": "I feel very thirsty, urinate frequently, and I am always tired.",
        "Fatigue et vision floue": "I feel extremely tired, my vision is blurry, and I'm losing weight.",
        "Soif intense et faim constante": "I feel extremely thirsty and hungry, and I go to the bathroom very often.",
        "Maux de tête et vertiges": "I have headaches, dizziness, and a pulsing sensation in my head.",
        "Douleurs thoraciques": "I feel chest tightness, shortness of breath, and palpitations.",
        "Engourdissements": "My hands feel numb and I have tingling in my feet.",
        "Digestif": "I feel nauseous, have abdominal pain, and my digestion is upset.",
        "Aucun symptôme": "I feel fine and do not notice any symptoms.",
    }

    with st.expander("🛠️ Paramètres d'entrée cliniques & symptômes", expanded=True):
        st.markdown('<p style="font-family:\'Fraunces\',serif;font-weight:600;margin-bottom:0.5rem">Données textuelles (Symptômes)</p>', unsafe_allow_html=True)
        col_text, col_sel = st.columns([2, 1], gap="medium")
        with col_sel:
            pipeline_choice = st.selectbox(
                "Exemples",
                list(pipeline_examples.keys()),
                index=0,
                key="pipeline_example_select",
                on_change=lambda: st.session_state.__setitem__(
                    "pipeline_symptoms",
                    pipeline_examples[st.session_state["pipeline_example_select"]],
                ),
            )
        with col_text:
            symptoms = st.text_area(
                "Description en anglais",
                height=90,
                key="pipeline_symptoms",
                label_visibility="collapsed",
            )

        st.markdown('<hr style="margin:1rem 0;opacity:0.3">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'Fraunces\',serif;font-weight:600;margin-bottom:0.5rem">Données tabulaires cliniques</p>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        pregnancies = c1.number_input("Grossesses", key="pipeline_pregnancies")
        glucose = c2.number_input("Glucose", key="pipeline_glucose")
        blood_pressure = c3.number_input("Tension", key="pipeline_blood_pressure")
        skin_thickness = c4.number_input("Peau (mm)", key="pipeline_skin_thickness")
        
        c5, c6, c7, c8 = st.columns(4)
        insulin = c5.number_input("Insuline", key="pipeline_insulin")
        bmi = c6.number_input("IMC", key="pipeline_bmi")
        pedigree = c7.number_input("Pedigree", key="pipeline_diabetes_pedigree_function")
        age = c8.number_input("Âge", key="pipeline_age")
        
        clinical_data = {
            "pregnancies": pregnancies, "glucose": glucose, "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness, "insulin": insulin, "bmi": bmi,
            "diabetes_pedigree_function": pedigree, "age": age,
        }

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Exécuter le Pipeline Complet", use_container_width=True, type="primary"):
        model_version = st.session_state.get("pipeline_model_version", "mlp_v3")
        with st.spinner("Exécution des modèles et synthèse LLM en cours…"):
            result = client.full_pipeline(clinical_data, symptoms, model_version=model_version)
            
        classic_level = result.get("results", {}).get("classic_model", {})
        hf_result = result.get("results", {}).get("huggingface", {})
        llm_result = result.get("results", {}).get("llm", {})
        consensus = result.get("consensus", {})

        # ── Dashboard layer display ──────────────────────────────────────────
        st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.2rem;font-weight:600;margin:1rem 0">Résultats des trois piliers</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        # Classic Model
        r_classic = classic_level.get('risk_level', 'low')
        c_classic = _RISK_COLORS.get(r_classic, "#2563eb")
        b_classic = _RISK_BG.get(r_classic, "rgba(37,99,235,0.1)")
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="background:{b_classic};border:1px solid {c_classic}44;padding:1.5rem;height:100%">
                <div style="color:inherit;opacity:0.75;font-size:0.75rem;font-weight:700;text-transform:uppercase;margin-bottom:0.5rem">1. Modèle Classique (MLP)</div>
                <div style="font-size:2.2rem;font-weight:800;color:{c_classic};margin-bottom:0.2rem">{r_classic.upper()}</div>
                <div style="font-size:0.9rem;color:inherit;opacity:0.75;font-weight:500;">Probabilité: <strong>{classic_level.get('probability', 0):.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        # HuggingFace Model
        h_cat = hf_result.get('top_category', '-')
        is_diab = "diabetes" in h_cat.lower()
        c_hf = "#ea580c" if is_diab else "#059669" if "no significant" in h_cat.lower() else "#2563eb"
        b_hf = "rgba(234, 88, 12, 0.12)" if is_diab else "rgba(5, 150, 105, 0.12)" if "no significant" in h_cat.lower() else "rgba(37,99,235,0.1)"
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="background:{b_hf};border:1px solid {c_hf}44;padding:1.5rem;height:100%">
                <div style="color:inherit;opacity:0.75;font-size:0.75rem;font-weight:700;text-transform:uppercase;margin-bottom:0.5rem">2. Zero-Shot NLP</div>
                <div style="font-size:1.6rem;line-height:1.2;font-weight:800;color:{c_hf};margin-bottom:0.2rem">{h_cat.title()}</div>
                <div style="font-size:0.9rem;color:inherit;opacity:0.75;margin-top:0.4rem;font-weight:500;">Confiance: <strong>{hf_result.get('confidence', 0):.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # LLM Model
        r_llm = llm_result.get('risk_level', 'low')
        c_llm = _RISK_COLORS.get(r_llm, "#2563eb")
        b_llm = _RISK_BG.get(r_llm, "rgba(37,99,235,0.1)")
        with col3:
            st.markdown(f"""
            <div class="glass-card" style="background:{b_llm};border:1px solid {c_llm}44;padding:1.5rem;height:100%">
                <div style="color:inherit;opacity:0.75;font-size:0.75rem;font-weight:700;text-transform:uppercase;margin-bottom:0.5rem">3. Synthèse LLM</div>
                <div style="font-size:2.2rem;font-weight:800;color:{c_llm};margin-bottom:0.2rem">{r_llm.upper()}</div>
                <div style="font-size:0.9rem;color:inherit;opacity:0.75;font-weight:500;">Urgence: <strong>{llm_result.get('urgency', '-').upper()}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Consensus & Report ───────────────────────────────────────────────
        st.markdown('<hr style="margin:2rem 0;opacity:0.2">', unsafe_allow_html=True)
        
        c_final = _RISK_COLORS.get(consensus.get('final_risk_level', 'low'), "#2563eb")
        b_final = _RISK_BG.get(consensus.get('final_risk_level', 'low'), "rgba(37,99,235,0.1)")
        
        st.markdown(f"""
        <div class="glass-card" style="background:{b_final};border:2px solid {c_final};padding:2rem;display:flex;align-items:center;gap:2rem">
            <div style="text-align:center;min-width:160px;border-right:1px solid {c_final}44;padding-right:2rem">
                <p style="margin:0;font-size:0.85rem;text-transform:uppercase;font-weight:800;color:inherit;opacity:0.75">Consensus Final</p>
                <p style="margin:0;font-size:2.5rem;font-weight:800;color:{c_final}">{consensus.get('final_risk_level', '-').upper()}</p>
                <p style="margin:0;font-size:0.9rem;color:inherit;opacity:0.75;font-weight:600;">Accord: <strong>{consensus.get('agreement', '-').replace('_', ' ')}</strong></p>
            </div>
            <div style="flex-grow:1">
                <p style="margin:0;font-family:'Fraunces',serif;font-size:1.3rem;font-weight:700;color:inherit">Justification Croisée</p>
                <p style="margin:0.5rem 0 0;font-size:1.05rem;color:inherit;line-height:1.6">{consensus.get('cross_validation', '-')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Detailed Report ──────────────────────────────────────────────────
        st.markdown('<p style="font-family:\'Fraunces\',serif;font-size:1.2rem;font-weight:600;margin:2rem 0 0.5rem">Rapport Médical Détaillé (Généré par LLM)</p>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="background:var(--panel-solid);padding:2rem">
            <p style="color:inherit;font-size:1.1rem;line-height:1.7;margin-bottom:1.5rem">{llm_result.get('summary', '')}</p>
            <p style="font-weight:800;color:inherit;opacity:0.75;margin-bottom:0.5rem;text-transform:uppercase;font-size:0.85rem;letter-spacing:0.05em;">Recommandations</p>
            <ul style="color:inherit;margin:0;padding-left:1.5rem;font-size:1.05rem;line-height:1.6">
                {''.join(f'<li style="margin-bottom:0.5rem">{r}</li>' for r in llm_result.get('recommendations', []))}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📄 Réponse JSON brute", expanded=False):
            st.json(result)
    else:
        st.markdown("""
        <div class="glass-card" style="padding:4rem 2rem;text-align:center;margin-top:2.5rem">
            <p style="font-size:3rem;margin:0">⚖️</p>
            <p style="color:inherit;opacity:0.75;margin:1rem 0 0;font-size:1.15rem;line-height:1.6">
                Remplissez les données et cliquez sur<br>
                <strong>Exécuter le Pipeline Complet</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
