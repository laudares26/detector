"""
Estágio 4 — SAM 2
Recebe as caixas do estágio de Detecção (YOLO ou Grounding DINO) e produz
máscaras em nível de pixel. Também resolve o tracking entre frames quando
a origem foi o Grounding DINO (que não tem ID nativo), por sobreposição
(IoU) com a máscara do frame anterior.
"""
from functools import lru_cache

import numpy as np
from ultralytics import SAM

SAM_WEIGHTS = "sam2_t.pt"  # variante tiny — mais rápida em CPU

_next_track_id = [1000]  # namespace separado dos IDs nativos do YOLO


@lru_cache(maxsize=1)
def _load_sam():
    return SAM(SAM_WEIGHTS)


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _assign_track_ids(detections: list[dict], previous: list[dict]) -> None:
    """Para detecções sem ID nativo (Grounding DINO, ou YOLO ainda não confirmado),
    casa por IoU com o frame anterior; senão abre um novo ID."""
    used_prev = set()
    for det in detections:
        if det["track_id"] != -1:
            continue
        if det["source"] == "yolo":
            # tracker nativo ainda inicializando neste frame — mantém -1 (pendente)
            continue
        best_iou, best_prev = 0.0, None
        for prev in previous:
            if id(prev) in used_prev:
                continue
            iou = _iou(det["bbox"], prev["bbox"])
            if iou > best_iou:
                best_iou, best_prev = iou, prev
        if best_prev is not None and best_iou > 0.3:
            det["track_id"] = best_prev["track_id"]
            used_prev.add(id(best_prev))
        else:
            det["track_id"] = _next_track_id[0]
            _next_track_id[0] += 1


def segment(frame_bgr, detections: list[dict], previous_detections: list[dict] | None = None) -> list[dict]:
    """Roda o SAM 2 prompt-a-prompt (uma caixa por vez) e anexa a máscara (RLE simplificado)."""
    if not detections:
        return []

    _assign_track_ids(detections, previous_detections or [])

    model = _load_sam()
    boxes = [d["bbox"] for d in detections]
    results = model(frame_bgr, bboxes=boxes, verbose=False)[0]

    if results.masks is None:
        for det in detections:
            det["mask_area_px"] = 0
        return detections

    masks = results.masks.data.cpu().numpy()  # (N, H, W) float 0/1
    for det, mask in zip(detections, masks):
        det["mask_area_px"] = int(mask.sum())
        det["_mask"] = mask  # usado internamente pela exportação (não serializado)
    return detections
