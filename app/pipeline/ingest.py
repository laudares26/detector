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
        "-f", "mp4[height<=480]/mp4/best",
        "-o", out_template,
        youtube_url,
    ]
    if max_duration_s is not None:
        cmd += ["--download-sections", f"*0-{max_duration_s}", "--force-keyframes-at-cuts"]
    subprocess.run(cmd, check=True, capture_output=True)
    for f in os.listdir(out_dir):
        if f.startswith("source."):
            return os.path.join(out_dir, f)
    raise RuntimeError("Download falhou: nenhum arquivo 'source.*' encontrado.")


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
