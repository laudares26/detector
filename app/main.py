"""
API FastAPI do DETECTOR — localiza objetos em vídeos do YouTube.

Endpoints:
    GET  /health
    POST /process               {"youtube_url": "...", "query": "..."} -> {"job_id": ...}
    GET  /jobs/{job_id}/status  -> progresso em tempo real
    GET  /jobs/{job_id}/video   -> MP4 anotado
    GET  /jobs/{job_id}/json    -> detecções em JSON
"""
import os
import threading
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

app = FastAPI(title="DETECTOR")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MAX_FRAMES = 40
FRAME_STRIDE = 3
MAX_DURATION_S = 60

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class ProcessRequest(BaseModel):
    youtube_url: str
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _run_job(job_id: str, req: ProcessRequest):
    work_dir = os.path.join(JOBS_DIR, job_id)

    def progress(msg: str):
        with jobs_lock:
            jobs[job_id]["progress"].append(msg)

    try:
        result = run_pipeline(
            youtube_url=req.youtube_url,
            query=req.query,
            work_dir=work_dir,
            max_frames=MAX_FRAMES,
            frame_stride=FRAME_STRIDE,
            max_duration_s=MAX_DURATION_S,
            progress_cb=progress,
        )
        with jobs_lock:
            jobs[job_id].update(
                status="done",
                meta=result["meta"],
                video_url=f"/jobs/{job_id}/video",
                json_url=f"/jobs/{job_id}/json",
            )
    except Exception as exc:  # pragma: no cover - protótipo
        with jobs_lock:
            jobs[job_id].update(status="error", error=str(exc))


@app.post("/process")
def process(req: ProcessRequest):
    job_id = uuid.uuid4().hex[:10]
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": []}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}/status"}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job não encontrado")
        return dict(job)


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
