"""Document text extraction for the travel-card ID check (no AI).

Tiers, cheapest first:
  1. PDF text layer (pymupdf)      instant; digital Aadhaar / e-PAN / DigiLocker PDFs have one
  2. OCR (RapidOCR, optional)      scanned PDFs and photos; runs on this server, free
  3. nothing                       the caller (the verify-document edge function) may then spend ONE AI call

Only text comes back. Name / number / phone matching is done by the app (utils/docVerify.js), so editing the typed name
never re-runs extraction. Nothing is stored or logged here: the text can contain an Aadhaar number.
"""
import base64
import io
import threading

import pymupdf

MIN_USEFUL_CHARS = 30
MAX_OCR_DIM = 1600

_ocr = None
_ocr_lock = threading.Lock()
_ocr_failed = False


def _get_ocr():
    """Lazy + optional: the service works (PDF text only) without the OCR packages installed."""
    global _ocr, _ocr_failed
    if _ocr is not None or _ocr_failed:
        return _ocr
    with _ocr_lock:
        if _ocr is None and not _ocr_failed:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _ocr = RapidOCR()
            except Exception:  # noqa: BLE001  not installed / cannot load
                _ocr_failed = True
    return _ocr


def ocr_available():
    return _get_ocr() is not None


def _ocr_image_bytes(data: bytes) -> str:
    engine = _get_ocr()
    if engine is None:
        return ""
    result, _ = engine(data)
    if not result:
        return ""
    # result: [[box, text, score], ...] -> keep reading order (top to bottom, left to right)
    def key(r):
        ys = [p[1] for p in r[0]]
        xs = [p[0] for p in r[0]]
        return (round(min(ys) / 12), min(xs))
    return "\n".join(r[1] for r in sorted(result, key=key) if len(r) > 1 and r[1])


def extract_text(file_b64: str, mime_type: str):
    """Returns {"text": str, "source": "pdf_text" | "ocr" | "none", "pages": int}."""
    raw = base64.b64decode(file_b64)
    if mime_type == "application/pdf" or raw[:5] == b"%PDF-":
        doc = pymupdf.open(stream=raw, filetype="pdf")
        try:
            if doc.page_count == 0:
                return {"text": "", "source": "none", "pages": 0}
            if doc.needs_pass:
                return {"text": "", "source": "none", "pages": doc.page_count, "password_protected": True}
            text = "\n".join(doc[i].get_text() for i in range(min(doc.page_count, 3))).strip()
            if len(text) >= MIN_USEFUL_CHARS:
                return {"text": text[:6000], "source": "pdf_text", "pages": doc.page_count}
            # Scanned PDF: render page 1 and OCR it.
            page = doc[0]
            scale = min(MAX_OCR_DIM / max(page.rect.width, page.rect.height), 2.0)
            jpeg = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("jpeg", jpg_quality=80)
            text = _ocr_image_bytes(jpeg)
            return {"text": text[:6000], "source": "ocr" if text else "none", "pages": doc.page_count}
        finally:
            doc.close()
    text = _ocr_image_bytes(raw)
    return {"text": text[:6000], "source": "ocr" if text else "none", "pages": 1}
