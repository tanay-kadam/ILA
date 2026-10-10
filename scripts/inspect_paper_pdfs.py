"""Read-only manuscript extraction and page rendering for review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp/pdf_deps"))
from pypdf import PdfReader
import pymupdf

files = {
    "workshop": ROOT / "A_Large_Token_Cost_Signal_Can_Be_a_Failure_Signal__Training_Trace_Composition_Under_Degraded_Reasoning_Adaptation.pdf",
    "tmlr": ROOT / "When_Training_Trace_Composition_Changes_Capped_Reasoning_Cost__A_Boundary_Analysis_of_LoRA_Adaptation.pdf",
}
for label, path in files.items():
    dest = ROOT / "tmp/pdfs" / label
    dest.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(path)
    texts = [page.extract_text() or "" for page in reader.pages]
    (dest / "extracted.txt").write_text("\n\n".join(f"=== PAGE {i+1} ===\n{text}" for i, text in enumerate(texts)), encoding="utf-8")
    doc = pymupdf.open(path)
    overflow = []
    for i, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if line["bbox"][2] > page.rect.width - 35:
                    overflow.append({"page": i+1, "right_edge_pt": round(line["bbox"][2], 1),
                                     "beyond_page": line["bbox"][2] > page.rect.width,
                                     "text": "".join(s["text"] for s in line["spans"])[:180]})
        page.get_pixmap(matrix=pymupdf.Matrix(1400 / page.rect.width, 1400 / page.rect.width)).save(dest / f"page-{i+1:02d}.png")
    print(json.dumps({"label": label, "pages": len(texts), "characters": sum(map(len, texts)), "text": str(dest / "extracted.txt"), "metadata": dict(reader.metadata or {}), "overflow": overflow}))
