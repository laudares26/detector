"""
Orquestra os 5 estágios na sequência desenhada:
Entrada -> VLM -> Detecção (YOLO ou Grounding DINO) -> SAM 2 -> Exportação
"""
import os
import time

from . import ingest, vlm, detect as detect_mod, segment as segment_mod, export as export_mod


def run_pipeline(
    youtube_url: str,
    query: str,
    work_dir: str,
    max_frames: int = 60,
    frame_stride: int = 2,
    open_vocab_detect_every: int = 5,
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
    log("[1/5] Ingestão: baixando vídeo com yt-dlp…")
    video_path = ingest.download_video(youtube_url, work_dir, max_duration_s=max_duration_s)
    info = ingest.probe_video(video_path)
    log(f"      vídeo: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
        f"{info['frame_count']} frames totais")

    frames_iter = list(ingest.extract_frames(video_path, max_frames=max_frames, stride=frame_stride))
    if not frames_iter:
        raise RuntimeError("Nenhum frame extraído do vídeo.")

    # 2) VLM -----------------------------------------------------------
    log("[2/5] VLM: interpretando a consulta e a cena…")
    _, first_frame = frames_iter[0]
    interpretation = vlm.interpret_query(query, first_frame)
    log(f"      cena: {interpretation['scene_caption']}")
    log(f"      roteamento: {interpretation['route']} — {interpretation['reasoning']}")

    # 3+4) Detecção + SAM 2, frame a frame ---------------------------------
    log(f"[3/5]+[4/5] Detecção ({interpretation['route']}) + SAM 2, frame a frame…")
    out_mp4 = os.path.join(work_dir, "output_annotated.mp4")
    out_json = os.path.join(work_dir, "output_detections.json")
    exporter = export_mod.VideoExporter(out_mp4, fps=info["fps"] / frame_stride,
                                         width=info["width"], height=info["height"])

    previous_detections: list[dict] = []
    last_raw_detections: list[dict] = []
    total_masks = 0

    for i, (frame_idx, frame) in enumerate(frames_iter):
        timestamp_s = frame_idx / info["fps"]

        run_detector_now = (
            interpretation["route"] == "yolo"
            or i % open_vocab_detect_every == 0
            or not last_raw_detections
        )
        if run_detector_now:
            raw_detections = detect_mod.detect(frame, interpretation["route"], interpretation["target_phrase"])
            last_raw_detections = raw_detections
        else:
            # reaproveita as últimas caixas conhecidas (economiza chamadas ao Grounding DINO)
            raw_detections = [dict(d) for d in last_raw_detections]
            for d in raw_detections:
                d.pop("_mask", None)

        detections = segment_mod.segment(frame, raw_detections, previous_detections)
        total_masks += len(detections)
        previous_detections = detections

        annotated = export_mod.draw_frame(frame, detections)
        exporter.write(frame_idx, timestamp_s, annotated, detections)

        if i % 10 == 0:
            log(f"      frame {i + 1}/{len(frames_iter)} — {len(detections)} objeto(s) segmentado(s)")

    meta = {
        "youtube_url": youtube_url,
        "query": query,
        "vlm_interpretation": interpretation,
        "video_info": info,
        "frames_processed": len(frames_iter),
        "frame_stride": frame_stride,
        "total_mask_instances": total_masks,
        "elapsed_seconds": round(time.time() - t0, 1),
        "pipeline": ["ingest", "vlm", f"detect:{interpretation['route']}", "sam2", "export"],
    }
    exporter.close(out_json, meta)
    log(f"[5/5] Exportação concluída em {meta['elapsed_seconds']}s — "
        f"{out_mp4}, {out_json}")

    return {"mp4": out_mp4, "json": out_json, "meta": meta}
