#!/bin/bash
# Lancer FastAPI en arrière-plan
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Attendre quelques secondes pour s'assurer que le backend a démarré
sleep 5

# Lancer Streamlit au premier plan (HuggingFace expose le port 7860)
python -m streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 7860
