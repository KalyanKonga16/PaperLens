import re
import yake


GENERIC_BAD = {
    "paper", "study", "result", "results", "introduction", "conclusion",
    "analysis", "approach", "method", "methods", "proposed method",
    "proposed approach", "experiment", "experiments", "research",
    "data", "model", "models", "problem", "work"
}


def normalize_keyword(keyword: str) -> str:
    keyword = keyword.replace("\n", " ").strip(" ,.;:[](){}<>\"'")
    keyword = re.sub(r"\s+", " ", keyword)
    return keyword.strip()


def is_valid_keyword(keyword: str) -> bool:
    low = keyword.lower()

    if len(keyword) < 3:
        return False
    if len(keyword.split()) > 5:
        return False
    if low in GENERIC_BAD:
        return False
    if not re.search(r"[A-Za-z]", keyword):
        return False

    return True


def postprocess_keywords(keywords: list[str], max_keywords: int = 40) -> list[str]:
    cleaned = []
    seen = set()

    for kw in keywords:
        kw = normalize_keyword(kw)
        if not is_valid_keyword(kw):
            continue

        key = kw.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(kw)

        if len(cleaned) >= max_keywords:
            break

    return cleaned


def extract_candidate_keywords(text: str, max_keywords: int = 40, max_ngram_size: int = 3) -> list[str]:
    extractor = yake.KeywordExtractor(
        lan="en",
        n=max_ngram_size,
        dedupLim=0.9,
        top=max_keywords * 2,
        features=None
    )

    raw_keywords = extractor.extract_keywords(text)
    ordered_keywords = [kw for kw, _score in sorted(raw_keywords, key=lambda x: x[1])]

    return postprocess_keywords(ordered_keywords, max_keywords=max_keywords)
