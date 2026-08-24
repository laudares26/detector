"""
Pipeline rápido:
Frames direto do stream -> Tradução (PT->EN) -> YOLO-World (texto + tracking) -> Relatório PDF
"""
import os
import time

import cv2

from . import ingest, translate, detect as detect_mod, export as export_mod


def run_pipeline(
    youtube_url: str,
    query: str,
    work_dir: str,
    max_frames: int = 60,
    max_duration_s: int = 60,
    progress_cb=None,
) -> dict:
    def log(msg):
        if progress_cb:
            progress_cb(msg)
        print(msg, flush=True)

    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()

    # 1) Frames direto do stream (sem baixar o vídeo inteiro) ----------------
    log("[1/4] Extraindo frames do vídeo (streaming)…")
    frames, info = ingest.extract_frames(
        youtube_url, os.path.join(work_dir, "frames"),
        max_frames=max_frames, max_duration_s=max_duration_s,
    )
    log(f"      {len(frames)} frames amostrados dos primeiros {max_duration_s}s")

    # 2) Consulta -------------------------------------------------------------
    log("[2/4] Interpretando a consulta…")
    query_en = translate.pt_to_en(query)
    log(f"      consulta: '{query}' -> '{query_en}'")
    model = detect_mod.prepare(query_en)

    # 3) Detecção + tracking, frame a frame -----------------------------------
    log("[3/4] Detectando e rastreando, frame a frame…")
    hits: list[dict] = []
    frames_json: list[dict] = []
    total_detections = 0

    annotated_dir = os.path.join(work_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    for i, (frame_number, timestamp_s, frame) in enumerate(frames):
        detections = detect_mod.detect(model, frame)
        total_detections += len(detections)
        frames_json.append({
            "frame": frame_number,
            "timestamp_s": round(timestamp_s, 3),
            "detections": detections,
        })
        if detections:
            annotated = export_mod.draw_frame(frame, detections)
            image_path = os.path.join(annotated_dir, f"hit_{frame_number:06d}.jpg")
            cv2.imwrite(image_path, annotated)
            hits.append({
                "frame_number": frame_number,
                "timestamp_s": timestamp_s,
                "image_path": image_path,
                "detections": detections,
            })

        if i % 5 == 0 or i == len(frames) - 1:
            log(f"      frame {i + 1}/{len(frames)} — {len(detections)} objeto(s) localizado(s)")

    # 4) Relatório -------------------------------------------------------------
    track_ids = {d["track_id"] for h in hits for d in h["detections"] if d["track_id"] >= 0}
    meta = {
        "youtube_url": youtube_url,
        "query": query,
        "query_en": query_en,
        "video_info": info,
        "frames_processed": len(frames),
        "frames_with_object": len(hits),
        "distinct_objects": len(track_ids) if track_ids else (1 if hits else 0),
        "total_detections": total_detections,
        "elapsed_seconds": round(time.time() - t0, 1),
        "pipeline": ["ingest:stream", "translate", "detect:yolo_world", "report:pdf"],
    }

    out_pdf = os.path.join(work_dir, "report.pdf")
    out_json = os.path.join(work_dir, "output_detections.json")
    export_mod.build_pdf(out_pdf, meta, hits)
    export_mod.write_json(out_json, meta, frames_json)
    log(f"[4/4] Relatório PDF gerado em {meta['elapsed_seconds']}s")

    return {"pdf": out_pdf, "json": out_json, "meta": meta}
