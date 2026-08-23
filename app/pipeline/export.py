"""
Estágio 5 — Exportação
Desenha máscaras + caixas + IDs de tracking sobre os frames e grava:
  - um MP4 anotado
  - um JSON com todas as detecções por frame (pronto para o frontend)
"""
import json
import colorsys

import cv2
import numpy as np


def _color_for_id(track_id: int):
    hue = (track_id * 0.17) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_frame(frame_bgr, detections: list[dict]):
    out = frame_bgr.copy()
    overlay = out.copy()
    for det in detections:
        color = _color_for_id(det["track_id"])
        mask = det.get("_mask")
        if mask is not None:
            colored_mask = np.zeros_like(out)
            colored_mask[mask.astype(bool)] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.45, 0)
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} #{det['track_id']} ({det['score']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


class VideoExporter:
    def __init__(self, out_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        self.frames_json = []

    def write(self, frame_idx: int, timestamp_s: float, annotated_frame, detections: list[dict]):
        self.writer.write(annotated_frame)
        clean = []
        for det in detections:
            d = {k: v for k, v in det.items() if not k.startswith("_")}
            clean.append(d)
        self.frames_json.append({
            "frame": frame_idx,
            "timestamp_s": round(timestamp_s, 3),
            "detections": clean,
        })

    def close(self, json_path: str, meta: dict):
        self.writer.release()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "frames": self.frames_json}, f, ensure_ascii=False, indent=2)
