"""
Estágio de Detecção — YOLO-World (Ultralytics).
Um único modelo rápido que aceita consultas em texto livre (vocabulário aberto)
e roda com tracking nativo (`model.track`), dispensando VLM de roteamento,
Grounding DINO e SAM 2 — ordens de magnitude mais rápido em CPU.
"""
from functools import lru_cache

YOLO_WORLD_WEIGHTS = "yolov8s-worldv2.pt"
CONF_THRESHOLD = 0.15


@lru_cache(maxsize=1)
def _load_yolo_world():
    from ultralytics import YOLOWorld
    return YOLOWorld(YOLO_WORLD_WEIGHTS)


def prepare(target_phrase_en: str):
    """Define a classe-alvo (texto livre, em inglês) no modelo."""
    model = _load_yolo_world()
    model.set_classes([target_phrase_en.strip().lower()])
    return model


def detect(model, frame_bgr, persist: bool = True) -> list[dict]:
    """Detecta + rastreia a classe-alvo num frame."""
    results = model.track(frame_bgr, persist=persist, conf=CONF_THRESHOLD, verbose=False)[0]
    detections = []
    if results.boxes is None:
        return detections
    names = results.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        detections.append({
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            "label": names.get(cls_id, str(cls_id)),
            "score": round(float(box.conf[0]), 3),
            "track_id": track_id,
            "source": "yolo_world",
        })
    return detections
