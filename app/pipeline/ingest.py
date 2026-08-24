"""
Estágio 1 — Entrada
Faz o download do vídeo (YouTube via yt-dlp) e extrai frames com OpenCV.
"""
import os
import subprocess
import cv2

def download_video(youtube_url: str, out_dir: str, max_duration_s: int | None = None) -> str:
    """Baixa o vídeo do YouTube com yt-dlp e retorna o caminho do arquivo mp4.

    max_duration_s: se informado, baixa apenas os primeiros N segundos
    (útil para testes rápidos do protótipo).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "best[height<=480]/bestvideo[height<=480]/best",
        "--no-playlist",
        "-o", out_template,
        youtube_url,
    ]
    if max_duration_s is not None:
        cmd += ["--download-sections", f"*0-{max_duration_s}", "--force-keyframes-at-cuts"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha ao baixar o vídeo: {proc.stderr.strip().splitlines()[-1] if proc.stderr else 'erro desconhecido'}")
    source = None
    for f in os.listdir(out_dir):
        if f.startswith("source."):
            source = os.path.join(out_dir, f)
            break
    if source is None:
        raise RuntimeError("Download falhou: nenhum arquivo 'source.*' encontrado.")
    return _normalize_video(source, out_dir, max_duration_s=max_duration_s)


def _normalize_video(source: str, out_dir: str, max_duration_s: int | None = None) -> str:
    """Reencoda o vídeo para H.264 (MP4), garantindo que o OpenCV consiga
    decodificá-lo (YouTube frequentemente entrega VP9/AV1, que o build do
    OpenCV pode não abrir, resultando em zero frames extraídos)."""
    normalized = os.path.join(out_dir, "normalized.mp4")
    cmd = ["ffmpeg", "-y", "-i", source]
    if max_duration_s is not None:
        cmd += ["-t", str(max_duration_s)]
    cmd += [
        "-vf", "scale='min(640,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-an", normalized,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(normalized):
        raise RuntimeError("Falha ao converter o vídeo para um formato legível.")
    return normalized


def extract_frames(video_path: str, max_frames: int | None = None, stride: int = 1):
    """Generator que produz (frame_idx, frame_bgr) do vídeo."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    produced = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
            produced += 1
            if max_frames is not None and produced >= max_frames:
                break
        idx += 1
    cap.release()


def probe_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info
