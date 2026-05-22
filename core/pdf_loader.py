"""PDF text extraction — pypdf first; Mistral OCR fallback added in Step 10."""

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


def extract_text(pdf_bytes: bytes, filename: str) -> tuple[str, bool]:
    """
    Returns (extracted_text, ocr_used).

    Step 4: pypdf only — ocr_used is always False.
    Step 10 will call Mistral OCR when pypdf yields fewer than MIN_TEXT_CHARS.
    """
    _ = filename  # used by OCR fallback in Step 10
    text = _try_pypdf(pdf_bytes)
    cleaned = text.strip()

    if len(cleaned) >= MIN_TEXT_CHARS:
        return cleaned, False

    # Short or empty text (likely scanned) — return what we have until OCR is wired
    return cleaned, False
