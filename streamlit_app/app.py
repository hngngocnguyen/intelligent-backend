from __future__ import annotations

from importlib import util
import os
from pathlib import Path
from typing import Any

import streamlit as st  # type: ignore[reportMissingImports]

from utils.api_client import ApiClient


BASE_DIR = Path(__file__).resolve().parent


def load_renderer(filename: str, attribute_name: str):
    page_path = BASE_DIR / "views" / filename
    spec = util.spec_from_file_location(filename.replace(".", "_"), page_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load page module: {filename}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute_name)


render_tabular_page = load_renderer(
    "01_Tabular_Prediction.py",
    "render_tabular_page",
)
render_symptom_page = load_renderer(
    "02_Symptom_Analysis.py",
    "render_symptom_page",
)
render_pipeline_page = load_renderer(
    "03_Full_Pipeline.py",
    "render_pipeline_page",
)
render_dataset_page = load_renderer(
    "04_Dataset.py",
    "render_dataset_page",
)


st.set_page_config(
    page_title="Observatoire Clinique du Risque Diabétique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            /* Map our custom variables directly to Streamlit's native theme engine */
            --bg: var(--background-color, #f4f6f9);
            --panel: var(--secondary-background-color, rgba(255, 255, 255, 0.75));
            --panel-solid: var(--secondary-background-color, #ffffff);
            --ink: var(--text-color, #111827);
            --muted: var(--text-color, #6b7280); /* Ensure high contrast */
            --accent: var(--primary-color, #2563eb);
            --accent-hover: #1d4ed8;
            --accent-2: #ea580c;
            --accent-3: #059669;
            --border: rgba(128, 128, 128, 0.2);
            --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
            --glass-blur: blur(16px);
        }

        /* Animated gradient background */
        .stApp {
            background: linear-gradient(-45deg, var(--bg), var(--panel), var(--bg));
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: var(--ink);
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        html, body, [class*="css"] {
            font-family: "Manrope", sans-serif;
        }

        h1, h2, h3, h4, h5 {
            font-family: "Fraunces", serif;
            letter-spacing: -0.01em;
            color: var(--ink);
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Glassmorphic generic class */
        .glass-card {
            background: var(--panel);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 20px;
            box-shadow: var(--shadow-md);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            background: rgba(255, 255, 255, 0.85);
        }

        .hero {
            background: var(--panel);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 24px;
            padding: 2.5rem 2.5rem;
            box-shadow: var(--shadow-md);
            animation: fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
        }
        
        .hero:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .hero h1 {
            margin-bottom: 0.5rem;
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(90deg, #1e3a8a, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            color: var(--muted);
            margin-bottom: 0;
            font-size: 1.1rem;
            line-height: 1.6;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 1rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.1);
            color: var(--accent);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 1rem;
            border: 1px solid rgba(37, 99, 235, 0.2);
            transition: all 0.2s ease;
        }
        
        .pill:hover {
            background: rgba(37, 99, 235, 0.15);
            transform: scale(1.02);
        }

        .section-title {
            margin: 2rem 0 1rem;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .story-card {
            background: var(--panel);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            animation: fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .story-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-md);
            border-color: rgba(37, 99, 235, 0.2);
        }

        .story-card h3 {
            margin: 0.5rem 0 0.3rem;
            font-size: 1.2rem;
            color: #1f2937;
        }

        .status-label {
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.7rem;
            font-weight: 800;
            color: var(--accent);
            background: rgba(37, 99, 235, 0.1);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            display: inline-block;
        }

        .story-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Streamlit overrides */
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px;
            border: 1px solid transparent;
            padding: 0.5rem 1.2rem;
            margin-right: 0.5rem;
            color: var(--muted);
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.5);
            color: var(--ink);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            color: var(--accent);
            border: 1px solid rgba(0,0,0,0.05);
            box-shadow: var(--shadow-sm);
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--accent), var(--accent-hover));
            color: #fff;
            border-radius: 14px;
            padding: 0.6rem 1.8rem;
            border: none;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
            background: linear-gradient(135deg, #3b82f6, var(--accent));
            border-color: transparent;
            color: white;
        }
        
        .stButton > button:active {
            transform: translateY(1px);
        }

        /* Input styling overrides */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stSlider div[data-baseweb="slider"] {
            border-radius: 12px;
            border: 1px solid rgba(17, 24, 39, 0.1);
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            transition: all 0.2s ease;
        }
        
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus,
        .stSelectbox div[data-baseweb="select"] > div:focus-within {
            background: #ffffff;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
        }

        /* Custom metric card generic styling */
        div[data-testid="stMetric"] {
            background: var(--panel);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border-radius: 16px;
            padding: 1.2rem;
            border: 1px solid rgba(255,255,255,0.4);
            box-shadow: var(--shadow-sm);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="pill">Santé · IA hybride · FastAPI + Streamlit</span>
            <h1>Observatoire Clinique du Risque Diabétique</h1>
            <p>
                Un cockpit clinique qui combine un modèle tabulaire entraîné,
                une analyse open source des symptômes et un LLM externe pour
                générer un rapport exploitable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_storyline() -> None:
    st.markdown(
        '<div class="section-title">Parcours de décision</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    steps = [
        (
            "1. Recueillir",
            "Renseignez les valeurs cliniques clés pour estimer le risque.",
        ),
        (
            "2. Interpréter",
            "Analysez les symptômes et leurs signaux faibles associés.",
        ),
        (
            "3. Arbitrer",
            "Obtenez un résumé multi-modèle clair et exploitable.",
        ),
    ]
    for col, (title, description) in zip(cols, steps, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="story-card">
                    <p class="status-label">Étape</p>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_default_state(default_url: str) -> dict[str, Any]:
    return {
        "backend_url": default_url,
        "default_model_version": "mlp_v3",
        "tabular_model_version": "mlp_v3",
        "pipeline_model_version": "mlp_v3",
        "show_latency": True,
        "tabular_pregnancies": 2,
        "tabular_glucose": 120.0,
        "tabular_blood_pressure": 72.0,
        "tabular_skin_thickness": 25.0,
        "tabular_insulin": 90.0,
        "tabular_bmi": 32.0,
        "tabular_pedigree": 0.47,
        "tabular_age": 45,
        "symptom_text": "I feel very thirsty, tired, and have blurry vision.",
        "pipeline_symptoms": (
            "I feel very thirsty, urinate frequently, and I am always tired."
        ),
        "pipeline_pregnancies": 2,
        "pipeline_glucose": 148.0,
        "pipeline_blood_pressure": 72.0,
        "pipeline_skin_thickness": 25.0,
        "pipeline_insulin": 90.0,
        "pipeline_bmi": 33.6,
        "pipeline_pedigree": 0.47,
        "pipeline_age": 50,
    }


def init_session_defaults(default_url: str) -> None:
    defaults = build_default_state(default_url)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_demo_state(default_url: str) -> None:
    defaults = build_default_state(default_url)
    # Keys bound to widgets that should not be modified after instantiation
    skip_keys = ["backend_url", "tabular_model_version", "show_latency"]
    for key, value in defaults.items():
        if key not in skip_keys:
            st.session_state[key] = value


def render_sidebar(client: ApiClient, default_url: str) -> None:
    st.sidebar.title("Pilotage")
    st.sidebar.caption("Pilotez les paramètres clés de l'expérience.")

    backend_url = st.sidebar.text_input(
        "URL du backend",
        key="backend_url",
    )
    client.base_url = backend_url

    health = client.safe_health()
    status = health.get("status", "offline")
    status_label = "En ligne" if status == "ok" else "Hors ligne"
    st.sidebar.metric("Statut API", status_label)

    if st.sidebar.button("Rafraîchir l'état"):
        st.rerun()

    st.sidebar.markdown("#### Modèle tabulaire")
    model_options = ["mlp_v1", "mlp_v2", "mlp_v3"]
    selected_version = st.sidebar.selectbox(
        "Version MLP",
        model_options,
        key="tabular_model_version",
    )
    st.session_state["pipeline_model_version"] = selected_version

    st.sidebar.markdown("#### Affichage")
    st.sidebar.checkbox("Afficher la latence", key="show_latency")

    if st.sidebar.button("Réinitialiser les démos"):
        reset_demo_state(default_url)
        st.rerun()

    models = client.safe_models_info()
    with st.sidebar.expander("Modèles détectés"):
        st.sidebar.json(models)



def main() -> None:
    inject_styles()
    default_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    init_session_defaults(default_url)
    client = ApiClient(base_url=st.session_state.get("backend_url", default_url))
    render_sidebar(client, default_url)
    render_hero()
    render_storyline()

    tab_one, tab_two, tab_three, tab_four = st.tabs(
        [
            "Prédiction tabulaire",
            "Analyse des symptômes",
            "Pipeline complet",
            "Dataset",
        ]
    )

    with tab_one:
        render_tabular_page(client)

    with tab_two:
        render_symptom_page(client)

    with tab_three:
        render_pipeline_page(client)

    with tab_four:
        render_dataset_page()


if __name__ == "__main__":
    main()
