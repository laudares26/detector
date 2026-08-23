"""
Estágio 3 — Detecção
Caminho rápido: YOLO (Ultralytics) para classes conhecidas, com tracking nativo.
Caminho aberto: Grounding DINO (via transformers) para busca por texto livre.
Ambos retornam uma lista de caixas no mesmo formato, consumida pelo SAM 2.
"""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

YOLO_WEIGHTS = "yolo11n.pt"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_BOX_THRESHOLD = 0.30
GROUNDING_DINO_TEXT_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def _load_yolo():
    return YOLO(YOLO_WEIGHTS)


@lru_cache(maxsize=1)
def _load_grounding_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID)
    model.eval()
    return processor, model


def detect_yolo(frame_bgr, target_class: str, persist: bool = True) -> list[dict]:
    """Detecta + rastreia usando YOLO.track(), filtrando pela classe do alvo."""
    model = _load_yolo()
    results = model.track(frame_bgr, persist=persist, verbose=False)[0]
    detections = []
    if results.boxes is None:
        return detections
    names = results.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        if cls_name != target_class:
            continue
        xyxy = box.xyxy[0].tolist()
        track_id = int(box.id[0]) if box.id is not None else -1
        detections.append({
            "bbox": [round(v, 1) for v in xyxy],
            "label": cls_name,
            "score": round(float(box.conf[0]), 3),
            "track_id": track_id,
            "source": "yolo",
        })
    return detections


def detect_grounding_dino(frame_bgr, text_query: str) -> list[dict]:
    """Detecta objetos a partir de uma frase em texto livre (sem tracking nativo)."""
    processor, model = _load_grounding_dino()
    rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(rgb)
    # Grounding DINO espera frases terminadas em ponto, em minúsculas.
    prompt = text_query.strip().lower()
    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        box_threshold=GROUNDING_DINO_BOX_THRESHOLD,
        text_threshold=GROUNDING_DINO_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results.get("labels", results.get("text_labels", [text_query] * len(boxes)))
    for bbox, score, label in zip(boxes, scores, labels):
        detections.append({
            "bbox": [round(v, 1) for v in bbox],
            "label": str(label),
            "score": round(float(score), 3),
            "track_id": -1,  # associação de ID acontece no estágio SAM 2
            "source": "grounding_dino",
        })
    return detections


def detect(frame_bgr, route: str, target_phrase: str) -> list[dict]:
    if route == "yolo":
        return detect_yolo(frame_bgr, target_phrase)
    return detect_grounding_dino(frame_bgr, target_phrase)
