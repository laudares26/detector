"""
Estágio 2 — VLM (Vision-Language Model)
Usa o Florence-2 (Microsoft) para:
  1) descrever a cena (interpretação semântica real, não regra fixa)
  2) decidir o roteamento: a consulta do usuário bate com uma classe fixa
     do YOLO (caminho rápido) ou é uma descrição livre (Grounding DINO)?

Florence-2-base é pequeno o suficiente (~230M parâmetros) para rodar em CPU
num protótipo, mantendo o papel de "raciocínio visual em linguagem" que um
VLM maior (GPT-4o, Gemini, Qwen-VL) desempenharia em produção.
"""
import re
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

FLORENCE_MODEL_ID = "microsoft/Florence-2-base"

# Vocabulário fechado do YOLO (classes COCO) usado para decidir o roteamento.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@lru_cache(maxsize=1)
def _load_florence():
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    return processor, model


def _run_florence_task(image: Image.Image, task_prompt: str, text_input: str | None = None) -> str:
    processor, model = _load_florence()
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed.get(task_prompt, "")


def describe_scene(frame_bgr) -> str:
    """Gera uma legenda da cena a partir de um frame (contexto para o log/JSON)."""
    import cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    caption = _run_florence_task(image, "<MORE_DETAILED_CAPTION>")
    return caption.strip()


def _match_yolo_class(query: str) -> str | None:
    q = query.lower()
    # normaliza plural simples e remove pontuação
    q_norm = re.sub(r"[^a-z0-9 ]", " ", q)
    for cls in COCO_CLASSES:
        if cls in q_norm:
            return cls
        singular = cls[:-1] if cls.endswith("s") else cls
        if singular and singular in q_norm:
            return cls
    return None


def interpret_query(query: str, sample_frame_bgr) -> dict:
    """
    Papel do VLM no pipeline: entender a consulta em linguagem natural do
    usuário, olhar a cena e decidir o roteamento para o estágio de detecção.
    """
    scene_caption = describe_scene(sample_frame_bgr)
    matched_class = _match_yolo_class(query)

    if matched_class is not None:
        route = "yolo"
        target_phrase = matched_class
        reasoning = (
            f"A consulta menciona '{matched_class}', que já é uma classe treinada do "
            f"YOLO — roteando para o caminho rápido (vocabulário fechado)."
        )
    else:
        from . import translate
        route = "open_vocab"
        target_phrase_original = query.strip().lower()
        target_phrase = translate.pt_to_en(target_phrase_original)
        reasoning = (
            f"A consulta '{query}' não corresponde a nenhuma classe do YOLO — "
            f"roteando para o Grounding DINO com busca por texto livre "
            f"(traduzida para '{target_phrase}' para o encoder de texto do modelo)."
        )
        return {
            "scene_caption": scene_caption,
            "route": route,
            "target_phrase": target_phrase,
            "target_phrase_original": target_phrase_original,
            "reasoning": reasoning,
        }

    return {
        "scene_caption": scene_caption,
        "route": route,
        "target_phrase": target_phrase,
        "target_phrase_original": target_phrase,
        "reasoning": reasoning,
    }
