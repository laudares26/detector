"""
Pipeline rápido:
Entrada -> Tradução (PT->EN) -> YOLO-World (detecção por texto + tracking) -> Exportação
"""
import os
import time

from . import ingest, translate, detect as detect_mod, export as export_mod


def run_pipeline(
    youtube_url: str,
    query: str,
    work_dir: str,
    max_frames: int = 60,
    frame_stride: int = 2,
    max_duration_s: int | None = None,
    progress_cb=None,
) -> dict:
    def log(msg):
        if progress_cb:
            progress_cb(msg)
        print(msg, flush=True)

    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()

    # 1) Entrada -----------------------------------------------------------
    log("[1/4] Ingestão: baixando vídeo com yt-dlp…")
    video_path = ingest.download_video(youtube_url, work_dir, max_duration_s=max_duration_s)
    info = ingest.probe_video(video_path)
    log(f"      vídeo: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
        f"{info['frame_count']} frames totais")

    frames_iter = list(ingest.extract_frames(video_path, max_frames=max_frames, stride=frame_stride))
    if not frames_iter:
        raise RuntimeError("Nenhum frame extraído do vídeo.")

    # 2) Consulta ------------------------------------------------------------
    log("[2/4] Interpretando a consulta…")
    query_en = translate.pt_to_en(query)
    log(f"      consulta: '{query}' -> '{query_en}'")
    model = detect_mod.prepare(query_en)

    # 3) Detecção + tracking, frame a frame ----------------------------------
    log("[3/4] Detectando e rastreando, frame a frame…")
    out_mp4 = os.path.join(work_dir, "output_annotated.mp4")
    out_json = os.path.join(work_dir, "output_detections.json")
    exporter = export_mod.VideoExporter(out_mp4, fps=info["fps"] / frame_stride,
                                         width=info["width"], height=info["height"])

    total_detections = 0
    for i, (frame_idx, frame) in enumerate(frames_iter):
        timestamp_s = frame_idx / info["fps"]
        detections = detect_mod.detect(model, frame)
        total_detections += len(detections)

        annotated = export_mod.draw_frame(frame, detections)
        exporter.write(frame_idx, timestamp_s, annotated, detections)

        if i % 5 == 0 or i == len(frames_iter) - 1:
            log(f"      frame {i + 1}/{len(frames_iter)} — {len(detections)} objeto(s) localizado(s)")

    meta = {
        "youtube_url": youtube_url,
        "query": query,
        "query_en": query_en,
        "video_info": info,
        "frames_processed": len(frames_iter),
        "frame_stride": frame_stride,
        "total_mask_instances": total_detections,
        "elapsed_seconds": round(time.time() - t0, 1),
        "pipeline": ["ingest", "translate", "detect:yolo_world", "export"],
    }
    exporter.close(out_json, meta)
    log(f"[4/4] Exportação concluída em {meta['elapsed_seconds']}s — "
        f"{out_mp4}, {out_json}")

    return {"mp4": out_mp4, "json": out_json, "meta": meta}
