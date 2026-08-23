# Protótipo — Pipeline Híbrido VLM → Detecção → SAM 2

Implementação funcional da arquitetura desenhada anteriormente:

```
Entrada (YouTube + consulta) → VLM (roteamento) → Detecção (YOLO ou Grounding DINO) → SAM 2 (máscara + tracking) → Exportação (MP4 + JSON)
```

Pensado para plugar na mesma linha do seu app **localizeobjetos.fly.dev**
(FastAPI + OpenCV + Ultralytics + yt-dlp), adicionando o estágio de VLM
para roteamento inteligente e SAM 2 para segmentação em nível de pixel.

## Estágios (`app/pipeline/`)

| Arquivo | Papel |
|---|---|
| `ingest.py` | Baixa o vídeo (yt-dlp) e extrai frames (OpenCV) |
| `vlm.py` | **Florence-2-base** descreve a cena e decide o roteamento: a consulta bate com uma classe do YOLO (COCO) → rota `yolo`; senão → rota `open_vocab` |
| `translate.py` | Traduz a consulta PT→EN antes de enviar ao Grounding DINO (o encoder de texto dele é treinado em inglês) |
| `detect.py` | `YOLO.track()` (rota rápida, com tracking nativo) **ou** `Grounding DINO tiny` (rota de texto livre, via `transformers`) |
| `segment.py` | **SAM 2 tiny** (via Ultralytics `SAM`) gera a máscara de cada caixa; resolve o ID de rastreio por IoU quando a origem não tem ID nativo (Grounding DINO) |
| `export.py` | Desenha máscara + caixa + ID sobre os frames, grava MP4 (`cv2.VideoWriter`) e um JSON com todas as detecções |
| `orchestrator.py` | Executa os 5 estágios em sequência |

`app/main.py` expõe tudo via FastAPI (`POST /process`, `GET /jobs/{id}/video`, `GET /jobs/{id}/json`).

## Como rodar localmente

```bash
cd video_pipeline_prototype
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export YOLO_CONFIG_DIR=$(pwd)/tmp/ultralytics   # evita erro de permissão do Ultralytics
uvicorn app.main:app --reload --port 8000
```

Abra `http://localhost:8000` para o frontend simples (`app/static/index.html`)
ou chame `/process` direto.

## Deploy no Fly.io (`localizeobjetos.fly.dev`)

Este diretório já inclui `Dockerfile` e `fly.toml` configurados para o app
`localizeobjetos` existente (região `gru`):

```bash
cd video_pipeline_prototype
fly deploy
```

O Dockerfile pré-baixa e cacheia todos os modelos (Florence-2, YOLO,
Grounding DINO, SAM 2, tradutor) **durante o build**, para não depender de
downloads de vários GB a cada cold start do Fly.

**Mudança importante em relação à config anterior**: aumentei a memória da
VM de 2gb para **4gb** (`fly.toml`) — com Florence-2 + Grounding DINO + SAM 2
+ tradutor carregados simultaneamente em memória, 2gb tende a estourar
(OOM). Isso aumenta o custo da máquina no Fly; ajuste se preferir outro
trade-off (ex: carregar os modelos sob demanda e descarregá-los depois).

Chamada de exemplo:

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://youtube.com/watch?v=XXXX","query":"person","max_frames":60}'
```

A resposta traz `job_id`, o roteamento decidido pelo VLM e os links do
MP4 anotado e do JSON de detecções.

## Rotas testadas nesta sessão

Testei o pipeline ponta a ponta com vídeos públicos de exemplo (o
YouTube bloqueou downloads automatizados neste sandbox — "Sign in to
confirm you're not a bot" —, então usei arquivos de teste locais para
validar toda a lógica; a função `ingest.download_video` com yt-dlp
está intacta e é a mesma usada no seu app em produção):

1. **Rota YOLO** (`query="person"`) — 2-3 pessoas detectadas e
   rastreadas com IDs estáveis, máscaras do SAM 2 bem ajustadas ao
   contorno de cada pessoa.
2. **Rota Grounding DINO** (`query="pessoa andando de bicicleta"` e
   `"pessoa vestindo roupa preta"`) — tradução automática para inglês,
   detecção por texto livre funcionando, tracking por IoU entre frames.

### Limitações observadas (importantes para você saber)

- **Grounding DINO tiny** tem dificuldade em cenas aéreas/vistas de
  cima com objetos pequenos (testei com um vídeo de drone e ele
  devolveu uma caixa cobrindo quase o frame todo, com confiança baixa
  ~0.36). Em vídeos em ângulo mais normal funcionou bem. Vale filtrar
  por `score` mais alto ou trocar para `grounding-dino-base` em
  produção se precisar de mais robustez.
- Atributos finos (cor de roupa, por exemplo) às vezes saem
  imprecisos — é a mesma fraqueza de VLMs pequenos em grounding
  espacial/atributos que apareceu na análise comparativa anterior.
- Sem GPU neste sandbox: ~4-6s/frame no caminho SAM 2 + Grounding DINO
  combinados. Em produção (Fly.io com GPU, se optar) isso cai bastante;
  mesmo em CPU dedicado deve rodar mais rápido que este sandbox de 2 vCPU.
- `YOLO_CONFIG_DIR` precisa ser exportado antes de rodar (o Ultralytics
  tenta escrever em `~/.config` por padrão).

## Modelos usados

- VLM: [`microsoft/Florence-2-base`](https://huggingface.co/microsoft/Florence-2-base)
- Detecção fechada: `yolo11n.pt` ([Ultralytics](https://docs.ultralytics.com/models/yolo11/))
- Detecção aberta: [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- Segmentação: `sam2_t.pt` ([Ultralytics SAM 2](https://docs.ultralytics.com/models/sam-2/))
- Tradução PT→EN: [`geralt/Opus-mt-pt-en`](https://huggingface.co/geralt/Opus-mt-pt-en) (espelho comunitário do Marian oficial, que foi descontinuado no Hub)

## Próximos passos sugeridos

- Trocar `Florence-2-base` por uma chamada a um VLM maior (GPT-4o /
  Gemini) via API se quiser roteamento mais confiável em consultas
  ambíguas — o Florence-2 é ótimo para rodar local/CPU mas é mais
  fraco que um VLM grande em raciocínio sobre a consulta.
- Cache de frame-chave: hoje o Grounding DINO roda a cada N frames
  (`open_vocab_detect_every`, padrão 5) e reaproveita a caixa entre
  eles — dá pra melhorar isso rodando um KLT/optical-flow leve entre
  detecções em vez de reusar a caixa fixa.
- Adicionar fila assíncrona (Celery/RQ) no FastAPI para não bloquear
  a requisição durante o processamento, igual você provavelmente já
  faz no `localizeobjetos.fly.dev`.
