"""
Estágio 1 — Entrada
Extrai apenas os frames necessários direto do stream do YouTube
(yt-dlp resolve a URL do stream; ffmpeg amostra os frames), sem
baixar o arquivo de vídeo inteiro.
"""
import os
import subprocess

import cv2


def _stream_url(youtube_url: str) -> str:
    cmd = [
        "yt-dlp", "-g",
        "-f", "best[height<=480]/bestvideo[height<=480]/best",
        "--no-playlist",
        youtube_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        err = proc.stderr.strip().splitlines()[-1] if proc.stderr else "erro desconhecido"
        raise RuntimeError(f"Falha ao resolver o vídeo: {err}")
    return proc.stdout.strip().splitlines()[0]


def extract_frames(youtube_url: str, out_dir: str, max_frames: int = 60,
                   max_duration_s: int = 60) -> tuple[list[tuple[int, float, "cv2.Mat"]], dict]:
    """Amostra até `max_frames` frames dos primeiros `max_duration_s` segundos
    do vídeo, direto do stream. Retorna [(nº do frame no vídeo, timestamp_s,
    frame_bgr), ...] e infos do vídeo."""
    os.makedirs(out_dir, exist_ok=True)
    url = _stream_url(youtube_url)

    sample_fps = max_frames / max_duration_s
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-t", str(max_duration_s),
        "-i", url,
        "-vf", f"fps={sample_fps},scale='min(640,iw)':-2",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    if not files:
        err = proc.stderr.strip().splitlines()[-1] if proc.stderr else ""
        raise RuntimeError(f"Nenhum frame extraído do vídeo. {err}")

    # fps real do vídeo (para converter timestamp em nº de frame do vídeo original)
    video_fps = _probe_fps(url) or 25.0

    frames = []
    for i, fname in enumerate(files):
        frame = cv2.imread(os.path.join(out_dir, fname))
        if frame is None:
            continue
        timestamp_s = i / sample_fps
        original_frame_number = int(round(timestamp_s * video_fps))
        frames.append((original_frame_number, timestamp_s, frame))

    info = {
        "fps": video_fps,
        "sample_fps": sample_fps,
        "frames_sampled": len(frames),
        "duration_analyzed_s": max_duration_s,
    }
    return frames, info


def _probe_fps(url: str) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", url],
        capture_output=True, text=True,
    )
    raw = proc.stdout.strip()
    if proc.returncode != 0 or not raw or "/" not in raw:
        return None
    num, den = raw.split("/")[:2]
    try:
        return float(num) / float(den) if float(den) else None
    except ValueError:
        return None
