"""
Estágio final — Exportação
Desenha caixas + IDs sobre os frames com detecção e gera:
  - um PDF-relatório (frames onde o objeto aparece, nº de aparições, quadros e tempos)
  - um JSON com todas as detecções por frame
"""
import json
import colorsys
import textwrap

import cv2
from fpdf import FPDF


def _color_for_id(track_id: int):
    hue = (track_id * 0.17) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_frame(frame_bgr, detections: list[dict]):
    out = frame_bgr.copy()
    for det in detections:
        color = _color_for_id(det["track_id"])
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} #{det['track_id']} ({det['score']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _txt(s: str) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def build_pdf(pdf_path: str, meta: dict, hits: list[dict], max_images: int = 24):
    """hits: [{"frame_number", "timestamp_s", "image_path", "detections"}, ...]
    apenas frames onde o objeto foi encontrado."""
    track_ids = {d["track_id"] for h in hits for d in h["detections"] if d["track_id"] >= 0}
    n_objects = len(track_ids) if track_ids else (1 if hits else 0)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---- página de resumo ----
    pdf.add_page()
    pdf.set_font("helvetica", "B", 22)
    pdf.cell(0, 12, "DETECTOR - Relatorio de localizacao", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("helvetica", "", 12)
    for line in textwrap.wrap(f"Video: {meta['youtube_url']}", width=80):
        pdf.cell(0, 8, _txt(line), new_x="LMARGIN", new_y="NEXT")
    for line in textwrap.wrap(f"Objeto procurado: {meta['query']} (en: {meta['query_en']})", width=80):
        pdf.cell(0, 8, _txt(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, _txt(f"Objetos distintos encontrados: {n_objects}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, _txt(f"Frames com o objeto: {len(hits)} de {meta['frames_processed']} analisados"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    if hits:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Onde o objeto aparece:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        for h in hits:
            labels = ", ".join(sorted({f"#{d['track_id']}" for d in h["detections"]}))
            pdf.cell(0, 7, _txt(
                f"  quadro {h['frame_number']}  ·  tempo {_fmt_time(h['timestamp_s'])}  ·  "
                f"{len(h['detections'])} deteccao(oes) {labels}"
            ), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, "O objeto nao foi encontrado no trecho analisado.",
                 new_x="LMARGIN", new_y="NEXT")

    # ---- páginas com os frames anotados ----
    for h in hits[:max_images]:
        pdf.add_page()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, _txt(
            f"Quadro {h['frame_number']}  ·  tempo {_fmt_time(h['timestamp_s'])}  ·  "
            f"{len(h['detections'])} deteccao(oes)"
        ), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.image(h["image_path"], w=pdf.epw)

    pdf.output(pdf_path)


def write_json(json_path: str, meta: dict, frames_json: list[dict]):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames_json}, f, ensure_ascii=False, indent=2)
