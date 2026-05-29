FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install supervisor and basic system dependencies just in case
RUN apt-get update && apt-get install -y supervisor libgomp1 && rm -rf /var/lib/apt/lists/*

# Create user for HuggingFace space
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app

# HF Spaces run on port 7860
EXPOSE 7860

CMD ["supervisord", "-c", "supervisord.conf"]
