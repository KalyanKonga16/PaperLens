import re
import fitz
from dataclasses import dataclass


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
