"""
Tradução PT -> EN usada apenas na rota de vocabulário aberto.
O Grounding DINO usa um encoder de texto (BERT) treinado em inglês —
frases em português produzem "grounding" ruim (tokens quebrados,
baixa confiança). Um modelo de tradução local pequeno resolve isso
sem depender de APIs externas (que se mostraram instáveis/limitadas
no ambiente de teste).
"""
from functools import lru_cache

# Repo oficial "Helsinki-NLP/opus-mt-pt-en" foi descontinuado no Hub; usamos
# um espelho comunitário do mesmo modelo Marian PT->EN.
TRANSLATION_MODEL_ID = "geralt/Opus-mt-pt-en"


@lru_cache(maxsize=1)
def _load_translator():
    from transformers import pipeline
    return pipeline("translation", model=TRANSLATION_MODEL_ID)


def pt_to_en(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    translator = _load_translator()
    result = translator(text, max_length=64)
    return result[0]["translation_text"].strip().lower()
