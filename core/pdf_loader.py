"""PDF text extraction — pypdf first, Mistral OCR fallback for scanned PDFs."""

import base64
import os
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MIN_TEXT_CHARS = 100


class PDFLoadError(Exception):
    """Raised when a PDF cannot be read (corrupt, encrypted, or invalid)."""


def _try_pypdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    if not pdf_bytes:
        raise PDFLoadError("PDF file is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    except PdfReadError as exc:
        raise PDFLoadError(str(exc)) from exc
    except Exception as exc:
        raise PDFLoadError(f"Could not open PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise PDFLoadError("PDF is password-protected.")
        except PDFLoadError:
            raise
        except Exception as exc:
            raise PDFLoadError("PDF is password-protected.") from exc

    parts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            parts.append(page_text)

    return "\n\n".join(parts).strip()


def _pages_to_text(response: object) -> str:
    pages = getattr(response, "pages", None) or []
    parts: list[str] = []
    for page in sorted(pages, key=lambda p: getattr(p, "index", 0)):
        markdown = getattr(page, "markdown", None) or ""
        if markdown.strip():
            parts.append(markdown.strip())
    return "\n\n".join(parts).strip()


def _mistral_ocr_extract(pdf_bytes: bytes, filename: str) -> str:
    """OCR scanned PDF via Mistral Document AI."""
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise PDFLoadError("MISTRAL_API_KEY is not set — cannot OCR scanned PDFs.")

    try:
        from mistralai.client import Mistral
        from mistralai.client.models import DocumentURLChunk, File
    except ImportError as exc:
        raise PDFLoadError("mistralai package is not installed.") from exc

    client = Mistral(api_key=api_key)
    model = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    errors: list[str] = []

    # 1) Base64 data URL (fast, no file upload)
    try:
        encoded = base64.b64encode(pdf_bytes).decode("ascii")
        data_url = f"data:application/pdf;base64,{encoded}"
        response = client.ocr.process(
            model=model,
            document=DocumentURLChunk(
                document_url=data_url,
                document_name=filename,
            ),
        )
        text = _pages_to_text(response)
        if text:
            return text
        errors.append("Base64 OCR returned no text.")
    except Exception as exc:
        errors.append(f"Base64 OCR: {exc}")

    # 2) Upload + signed URL (reliable for larger PDFs)
    try:
        uploaded = client.files.upload(
            file=File(
                file_name=filename,
                content=pdf_bytes,
                content_type="application/pdf",
            ),
            purpose="ocr",
        )
        signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
        response = client.ocr.process(
            model=model,
            document=DocumentURLChunk(
                document_url=signed.url,
                document_name=filename,
            ),
        )
        text = _pages_to_text(response)
        if text:
            return text
        errors.append("Upload OCR returned no text.")
    except Exception as exc:
        errors.append(f"Upload OCR: {exc}")

    detail = errors[0] if len(errors) == 1 else "; ".join(errors[:2])
    raise PDFLoadError(f"Mistral OCR failed for {filename}. {detail}")


def extract_text(pdf_bytes: bytes, filename: str) -> tuple[str, bool]:
    """
    Returns (extracted_text, ocr_used).
    Uses pypdf first; falls back to Mistral OCR when text < MIN_TEXT_CHARS.
    """
    text = _try_pypdf(pdf_bytes)
    cleaned = text.strip()

    if len(cleaned) >= MIN_TEXT_CHARS:
        return cleaned, False

    if not os.getenv("MISTRAL_API_KEY", "").strip():
        return cleaned, False

    try:
        ocr_text = _mistral_ocr_extract(pdf_bytes, filename).strip()
        if ocr_text:
            return ocr_text, True
    except (PDFLoadError, Exception):
        # Keep app running — caller may surface a limited-text warning
        pass

    return cleaned, False
