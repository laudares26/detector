FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV YOLO_CONFIG_DIR=/app/.ultralytics \
    HF_HOME=/app/.hf_cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN mkdir -p /app/.ultralytics /app/.hf_cache /app/outputs

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Pré-baixa e cacheia os modelos na imagem, para evitar downloads
# a cada cold start do Fly.io (machines com auto_stop_machines).
RUN python -c "\
from app.pipeline import detect, translate; \
detect.prepare('person'); \
print('YOLO-World ok'); \
translate._load_translator(); \
print('Tradutor ok'); \
"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
