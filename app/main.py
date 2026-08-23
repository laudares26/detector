"""
API FastAPI do protótipo — mesma convenção do seu app existente
(localizeobjetos.fly.dev): POST com URL do YouTube + consulta,
retorna MP4 anotado + JSON.

Rodar localmente:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /process        {"youtube_url": "...", "query": "...", "max_frames": 60}
    GET  /jobs/{job_id}/video   -> MP4 anotado
    GET  /jobs/{job_id}/json    -> detecções em JSON
"""
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline.orchestrator import run_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Video Object Search — Pipeline Híbrido (VLM + YOLO + Grounding DINO + SAM 2)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ProcessRequest(BaseModel):
    youtube_url: str
    query: str
    max_frames: int = 60
    frame_stride: int = 2


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/process")
def process(req: ProcessRequest):
    job_id = uuid.uuid4().hex[:10]
    work_dir = os.path.join(JOBS_DIR, job_id)
    try:
        result = run_pipeline(
            youtube_url=req.youtube_url,
            query=req.query,
            work_dir=work_dir,
            max_frames=req.max_frames,
            frame_stride=req.frame_stride,
        )
    except Exception as exc:  # pragma: no cover - protótipo
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "job_id": job_id,
        "meta": result["meta"],
        "video_url": f"/jobs/{job_id}/video",
        "json_url": f"/jobs/{job_id}/json",
    }


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str):
    path = os.path.join(JOBS_DIR, job_id, "output_annotated.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="job não encontrado")
    return FileResponse(path, media_type="video/mp4")


@app.get("/jobs/{job_id}/json")
def get_json(job_id: str):
    path = os.path.join(JOBS_DIR, job_id, "output_detections.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="job não encontrado")
    return FileResponse(path, media_type="application/json")
