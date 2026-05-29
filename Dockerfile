FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create user for HuggingFace space (required by HF)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app

# Ensure script is executable
RUN chmod +x run.sh

# HF Spaces run on port 7860
EXPOSE 7860

CMD ["./run.sh"]
