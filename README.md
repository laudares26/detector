# DETECTOR — Localize objetos em vídeos do YouTube

MVP rápido de localização de objetos por texto livre em vídeos do YouTube,
com resultado em **PDF** (frames onde o objeto aparece, número de aparições,
quadros e tempos). Hospedado no Fly.io: https://detector-objetos.fly.dev/

## Arquitetura

```
URL do YouTube + consulta (PT)
    → yt-dlp -g (resolve a URL do stream — o vídeo NÃO é baixado inteiro)
    → ffmpeg (amostra até 60 frames dos primeiros 60s, direto do stream)
    → tradução PT→EN (Marian/Opus)
    → YOLO-World (detecção por texto livre + tracking nativo)
    → PDF (frames anotados com ocorrência) + JSON (todas as detecções)
```

## Estágios (`app/pipeline/`)

| Arquivo | Papel |
|---|---|
| `ingest.py` | Resolve a URL do stream (`yt-dlp -g`) e extrai frames amostrados via `ffmpeg`, sem baixar o vídeo inteiro |
| `translate.py` | Traduz a consulta PT→EN (o encoder de texto do YOLO-World é treinado em inglês) |
| `detect.py` | **YOLO-World** (`yolov8s-worldv2`): detecção por vocabulário aberto + tracking nativo em um único modelo |
| `export.py` | Desenha caixas + IDs nos frames com ocorrência, gera o relatório PDF (fpdf2) e o JSON |
| `orchestrator.py` | Executa os estágios em sequência, reportando progresso |

`app/main.py` expõe tudo via FastAPI:

- `POST /process` → inicia o job em background e retorna `job_id`
- `GET /jobs/{id}/status` → progresso ao vivo (polling do frontend)
- `GET /jobs/{id}/report` → relatório PDF
- `GET /jobs/{id}/json` → detecções em JSON

Frontend minimalista em `app/static/index.html` (título DETECTOR, dois
campos: link do YouTube + o que localizar, barra de progresso com etapas).

## Como rodar localmente

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export YOLO_CONFIG_DIR=$(pwd)/tmp/ultralytics   # evita erro de permissão do Ultralytics
uvicorn app.main:app --reload --port 8000
```

Abra `http://localhost:8000` ou chame a API direto:

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://youtube.com/watch?v=XXXX","query":"cachorro"}'
```

## Deploy no Fly.io

`Dockerfile` e `fly.toml` já configurados para o app `detector-objetos`
(região `gru`, máquina `performance-2x` com 4 GB — mínimo exigido pelo Fly
para esse tipo de máquina):

```bash
fly deploy --remote-only --ha=false
```

O Dockerfile pré-baixa e cacheia os modelos (YOLO-World e tradutor)
durante o build, evitando downloads a cada cold start.

## Modelos usados

- Detecção + tracking: `yolov8s-worldv2.pt` ([Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/))
- Tradução PT→EN: [`geralt/Opus-mt-pt-en`](https://huggingface.co/geralt/Opus-mt-pt-en)

## Desempenho

Um vídeo real (60 frames amostrados dos primeiros 60s) leva ~2–3 min
ponta a ponta em CPU no Fly — a maior parte do tempo é a resolução do
stream e a amostragem via rede; a detecção em si roda em segundos.
