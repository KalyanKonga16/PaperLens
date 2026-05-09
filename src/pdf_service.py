import re
import io
from dataclasses import dataclass

import fitz  # PyMuPDF for PDF
from docx import Document  # python-docx for DOCX


@dataclass
class PageText:
    page_number: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages_from_pdf(pdf_bytes: bytes, max_pages: int = 60) -> list[PageText]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    total_pages = min(len(doc), max_pages)

    for i in range(total_pages):
        page = doc[i]
        text = page.get_text("text")
        text = clean_text(text)
        if len(text) >= 50:
            pages.append(PageText(page_number=i + 1, text=text))

    doc.close()
    return pages


def extract_pages_from_docx(file_bytes: bytes, max_pages: int = 60) -> list[PageText]:
    """
    Word documents do not have real page breaks until rendered.
    We split paragraphs into pseudo-pages of ~500 words each.
    """
    file_stream = io.BytesIO(file_bytes)
    doc = Document(file_stream)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = " ".join(paragraphs)
    full_text = clean_text(full_text)

    if not full_text:
        return []

    words = full_text.split()
    words_per_page = 500
    pages = []

    for idx in range(0, len(words), words_per_page):
        page_number = (idx // words_per_page) + 1
        if page_number > max_pages:
            break
        page_text = " ".join(words[idx:idx + words_per_page])
        if len(page_text) >= 50:
            pages.append(PageText(page_number=page_number, text=page_text))

    return pages


def extract_pages_from_text(file_bytes: bytes, max_pages: int = 60) -> list[PageText]:
    """
    For TXT and MD files, split into pseudo-pages of ~500 words each.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = file_bytes.decode("latin-1", errors="replace")

    text = clean_text(text)
    if not text:
        return []

    words = text.split()
    words_per_page = 500
    pages = []

    for idx in range(0, len(words), words_per_page):
        page_number = (idx // words_per_page) + 1
        if page_number > max_pages:
            break
        page_text = " ".join(words[idx:idx + words_per_page])
        if len(page_text) >= 50:
            pages.append(PageText(page_number=page_number, text=page_text))

    return pages


def extract_pages_from_file(file_bytes: bytes, filename: str, max_pages: int = 60) -> list[PageText]:
    """
    Universal entry point. Detects file type by extension and routes
    to the correct extractor.
    """
    filename_lower = filename.lower().strip()

    if filename_lower.endswith(".pdf"):
        return extract_pages_from_pdf(file_bytes, max_pages=max_pages)

    if filename_lower.endswith(".docx"):
        return extract_pages_from_docx(file_bytes, max_pages=max_pages)

    if filename_lower.endswith(".txt"):
        return extract_pages_from_text(file_bytes, max_pages=max_pages)

    if filename_lower.endswith(".md"):
        return extract_pages_from_text(file_bytes, max_pages=max_pages)

    raise ValueError(
        f"Unsupported file format: {filename}. "
        f"Please upload PDF, DOCX, TXT, or MD files only."
    )
