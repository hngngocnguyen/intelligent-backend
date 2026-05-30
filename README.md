---
title: Diabete Risk Predictor
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Intelligent Backend — Diabetes Risk Predictor

Une application full-stack intelligente (FastAPI + Streamlit) combinant trois couches d'Intelligence Artificielle pour analyser le risque de diabète chez un patient. Ce projet a été développé en respectant la méthodologie **CRISP-DM**.

## 🧠 L'Architecture Multi-IA (Hybride)

Ce projet ne se limite pas à un simple modèle de classification. Il intègre un pipeline robuste à trois niveaux :

1. **Couche Tabulaire (TensorFlow / Keras)** : 
   - Entraîné sur le dataset Pima Indians Diabetes.
   - Fournit une prédiction de risque ultra-rapide (< 15ms) basée sur les données cliniques (Glucose, BMI, Age, etc.).
2. **Couche NLP Zero-Shot (HuggingFace)** :
   - Utilise le modèle open-source `typeform/distilbert-base-uncased-mnli` (version allégée et rapide).
   - Analyse les textes et plaintes du patient pour en extraire des catégories de symptômes.
3. **Couche Générative LLM (OpenRouter)** :
   - Interroge un modèle LLM gratuit (`:free`) avec failover (ex: `openai/gpt-oss-20b:free`, `mistralai/mistral-7b-instruct:free`).
   - Combine les résultats médicaux tabulaires et les symptômes NLP pour générer un compte-rendu compréhensible, empathique et renvoyer un JSON structuré exploitable par l'UI.

## 🚀 Liens du Projet

- **Démonstration en ligne (HuggingFace Spaces)** : [https://huggingface.co/spaces/theamazingruby/diabete-risk-predictor](https://huggingface.co/spaces/theamazingruby/diabete-risk-predictor)
- **Dépôt du code source (GitHub)** : [https://github.com/hngngocnguyen/intelligent-backend](https://github.com/hngngocnguyen/intelligent-backend)

## 📂 Guide de navigation pour l'évaluation

Pour faciliter la lecture du projet, voici où trouver les éléments clés :
- **Interface Utilisateur (Frontend)** : Le point d'entrée principal est `streamlit_app/app.py`.
- **API & Logique Métier (Backend)** : Le serveur FastAPI se trouve dans `backend/main.py` et les différents modèles IA sont gérés dans `backend/models/` (`classic_model.py`, `hf_model.py`, `llm_model.py`).
- **Analyse Comparative** : L'analyse des performances, avantages et limites des 3 approches (ainsi que toute l'exploration Data Science) se trouve dans le dossier `notebooks/`, particulièrement dans **`notebooks/05_comparative_analysis.ipynb`**.

## 📂 Structure du Projet

- `data/` : Contient le dataset brut et nettoyé.
- `notebooks/` : Toute la phase de Data Science (EDA, entraînement MLP, tests NLP HuggingFace, requêtes LLM OpenRouter, Synthèse comparative).
- `saved_models/` : Fichiers `.keras` (modèles), `scaler.pkl`, et `.csv` des métriques.
- `backend/` : Code source de l'API FastAPI et de l'orchestrateur.
- `streamlit_app/` : Code source de l'interface graphique utilisateur avec Streamlit, qui offre les onglets suivants :
  - **Prédiction tabulaire** : Interface de test pour le modèle de Machine Learning (MLP).
  - **Analyse des symptômes** : Interface pour extraire les symptômes depuis du texte médical (NLP Zero-Shot).
  - **Pipeline complet** : Interface combinant la prédiction tabulaire, l'analyse NLP, et l'interprétation par LLM pour générer un rapport clinique.
  - **Dataset** : Exploration visuelle et statistiques du dataset d'entraînement.

## 📸 Aperçu de l'Application

### 1. Pipeline Complet
![Pipeline Complet](assets/pipeline_complet.png)

### 2. Prédiction Tabulaire
![Prédiction Tabulaire](assets/prediction_tabulaire.png)

### 3. Analyse des Symptômes NLP
![Analyse NLP](assets/analyse_nlp.png)

### 4. Exploration du Dataset
![Exploration Dataset](assets/dataset_eda.png)

## 🚀 Comment lancer l'application en local

Vous devez ouvrir **deux terminaux distincts** à l'intérieur du dossier du projet.

### 1. Démarrer le Backend (API FastAPI)
Dans le 1er terminal :
```bash
# S'assurer d'être dans le dossier du projet (ex: diabete-risk-predictor)
# cd diabete-risk-predictor

# Lancer le serveur (il tournera sur http://127.0.0.1:8000)
python -m uvicorn backend.main:app --reload --port 8000
```
Le backend va charger les modèles TensorFlow et HuggingFace en mémoire. Une fois que vous voyez `Application startup complete`, le backend est prêt.
👉 *Documentation de l'API (Swagger) : http://127.0.0.1:8000/docs*

### 2. Démarrer le Frontend (Interface Streamlit)
Dans le 2ème terminal :
```bash
# S'assurer d'être dans le dossier du projet
# cd diabete-risk-predictor

# Lancer l'interface web (elle s'ouvrira sur http://127.0.0.1:8501)
streamlit run streamlit_app/app.py
```

## 🐋 Lancement via Docker (Optionnel)

Si Docker est installé sur votre machine, l'application est entièrement conteneurisée :
```bash
docker-compose up --build
```
- Frontend Streamlit : http://localhost:8501
- Backend FastAPI : http://localhost:8000

## 🔑 Configuration (.env)
Pour que la 3ème couche (LLM) fonctionne, vous devez posséder un fichier `.env` à la racine contenant votre clé OpenRouter :
```env
OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxx"
OPENROUTER_MODEL="openai/gpt-oss-20b:free"
OPENROUTER_MODEL_CANDIDATES="openai/gpt-oss-20b:free,mistralai/mistral-7b-instruct:free,google/gemma-4-31b-it:free,deepseek/deepseek-v4-flash:free"
```
Si l'API n'est pas disponible, le backend dispose d'un mécanisme de *failover* (repli) local pour éviter les crashs en production.
