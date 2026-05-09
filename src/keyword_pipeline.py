import re
from .config import settings
from .cache import JsonCache, make_cache_key
from .pdf_service import extract_pages_from_file
from .page_index_service import chunk_pages, SimplePageIndex
from .candidate_extractor import extract_candidate_keywords, postprocess_keywords
from .hf_client import HFKeywordRefiner


def categorize_keyword(keyword: str) -> str:
    kw = keyword.lower()

    if any(x in kw for x in [
        "ai", "ml", "machine learning", "deep learning", "neural", "model", "llm", "prompt", "token", "generation"
    ]):
        return "AI / Machine Learning"

    if any(x in kw for x in [
        "system", "infrastructure", "server", "network", "api", "platform", "service"
    ]):
        return "Systems / Infrastructure"

    if any(x in kw for x in [
        "data", "dataset", "database", "storage", "cache", "index", "vector", "memory", "retrieval"
    ]):
        return "Data / Storage"

    if any(x in kw for x in [
        "performance", "latency", "throughput", "speed", "optimization", "efficiency"
    ]):
        return "Performance / Optimization"

    if any(x in kw for x in [
        "user", "customer", "experience", "interface", "ux", "ui", "design"
    ]):
        return "User / Experience"

    if any(x in kw for x in [
        "business", "market", "revenue", "sales", "growth", "strategy", "industry"
    ]):
        return "Business / Strategy"

    if any(x in kw for x in [
        "security", "privacy", "encryption", "authentication", "compliance", "risk"
    ]):
        return "Security / Compliance"

    if any(x in kw for x in [
        "metric", "evaluation", "benchmark", "measurement", "analysis", "report"
    ]):
        return "Metrics / Evaluation"

    return "General Concepts"


def count_keyword_occurrences(text: str, keyword: str) -> int:
    text = text.lower()
    keyword = keyword.lower().strip()
    if not keyword:
        return 0
    pattern = r"\b" + re.escape(keyword) + r"\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches)


def build_keyword_metrics(keywords: list[str], pages, retrieved_chunks: list[dict]) -> list[dict]:
    metrics = []
    for kw in keywords:
        pages_found = []
        total_occurrences = 0

        for page in pages:
            count = count_keyword_occurrences(page.text, kw)
            if count > 0:
                pages_found.append(page.page_number)
                total_occurrences += count

        chunk_support = 0
        retrieval_score_sum = 0.0

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "").lower()
            if kw.lower() in chunk_text:
                chunk_support += 1
                retrieval_score_sum += float(chunk.get("score", 0))

        phrase_length = len(kw.split())

        raw_score = (
            total_occurrences * 2.0 +
            len(pages_found) * 4.0 +
            chunk_support * 5.0 +
            phrase_length * 1.5 +
            retrieval_score_sum * 10.0
        )

        metrics.append({
            "keyword": kw,
            "occurrences": total_occurrences,
            "page_coverage": len(pages_found),
            "pages": pages_found,
            "chunk_support": chunk_support,
            "phrase_length": phrase_length,
            "category": categorize_keyword(kw),
            "raw_score": raw_score,
        })

    max_score = max([m["raw_score"] for m in metrics], default=1)

    for m in metrics:
        if max_score > 0:
            m["evidence_score"] = round((m["raw_score"] / max_score) * 100, 2)
        else:
            m["evidence_score"] = 0.0

    metrics.sort(key=lambda x: x["evidence_score"], reverse=True)
    return metrics


def build_retrieval_query(candidates: list[str]) -> str:
    anchor_terms = [
        "important keywords",
        "main concepts",
        "key entities",
        "core ideas",
        "primary topics",
        "technical terms",
        "domain terms",
        "methods",
        "products",
        "organizations",
        "tools",
        "metrics",
        "applications"
    ]
    return " ".join(anchor_terms + candidates[:18])


def local_rank_keywords(candidates: list[str], pages, max_keywords: int) -> list[str]:
    if not candidates:
        return []
    page_texts = [p.text.lower() for p in pages]
    ranked = []
    for kw in candidates:
        key = kw.lower()
        freq = sum(text.count(key) for text in page_texts)
        spread = sum(1 for text in page_texts if key in text)
        multiword_bonus = 1.25 if len(kw.split()) > 1 else 1.0
        score = (spread * 3 + freq) * multiword_bonus
        ranked.append((kw, score))
    ranked.sort(key=lambda x: (-x[1], len(x[0])))
    return postprocess_keywords([kw for kw, _ in ranked], max_keywords=max_keywords)


def process_pdf(pdf_bytes: bytes, filename: str, use_cache: bool = True, max_keywords: int | None = None) -> dict:
    if max_keywords is None or max_keywords < 1:
        max_keywords = settings.max_keywords

    cache = JsonCache(settings.cache_dir)
    cache_key = make_cache_key(
        pdf_bytes=pdf_bytes, filename=filename, models=settings.hf_models, max_keywords=max_keywords
    )

    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            cached["cache_hit"] = True
            return cached

    pages = extract_pages_from_file(pdf_bytes, filename=filename, max_pages=settings.max_pdf_pages)
    if not pages:
        raise ValueError("No readable text found in the PDF. If it is a scanned image PDF, OCR is needed.")

    full_text = "\n".join(p.text for p in pages)

    candidate_pool = max(max_keywords * 3, 60)
    candidates = extract_candidate_keywords(text=full_text, max_keywords=candidate_pool, max_ngram_size=3)

    chunks = chunk_pages(pages=pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    index = SimplePageIndex().build(chunks)
    retrieval_query = build_retrieval_query(candidates)
    retrieved_chunks = index.search(retrieval_query, top_k=settings.retrieval_top_k)

    display_chunks = []
    for ch in retrieved_chunks:
        display_chunks.append({
            "chunk_id": ch["chunk_id"], "page_number": ch["page_number"],
            "score": ch["score"], "text": ch["text"][:900]
        })

    llm_model = None
    llm_error = None
    summary = ""
    pages_used = sorted(set(ch["page_number"] for ch in retrieved_chunks)) if retrieved_chunks else []

    if settings.hf_token and retrieved_chunks:
        try:
            refiner = HFKeywordRefiner(token=settings.hf_token, models=settings.hf_models, max_retries=2)
            llm_result = refiner.refine_keywords(
                candidates=candidates, retrieved_chunks=retrieved_chunks, max_keywords=max_keywords
            )
            keywords = llm_result["keywords"]
            summary = llm_result.get("summary", "")
            llm_model = llm_result.get("model")
            if llm_result.get("pages_used"):
                pages_used = llm_result["pages_used"]
            method = "HF LLM refinement + local page-aware retrieval"
        except Exception as e:
            llm_error = str(e)
            keywords = local_rank_keywords(candidates, pages, max_keywords)
            summary = "Used local fallback because HF inference failed, quota was reached, or model unavailable."
            method = "Local YAKE fallback + local page-aware retrieval"
    else:
        keywords = local_rank_keywords(candidates, pages, max_keywords)
        summary = "Used local extraction because HF_TOKEN was missing or no relevant chunks were retrieved."
        method = "Local YAKE fallback + local page-aware retrieval"

    keyword_metrics = build_keyword_metrics(keywords=keywords, pages=pages, retrieved_chunks=retrieved_chunks)

    result = {
        "file_name": filename,
        "page_count": len(pages),
        "extracted_characters": len(full_text),
        "keywords": keywords,
        "keyword_metrics": keyword_metrics,
        "summary": summary,
        "candidate_keywords": candidates,
        "retrieved_chunks": display_chunks,
        "pages_used": pages_used,
        "method": method,
        "llm_model": llm_model,
        "llm_error": llm_error,
        "max_keywords_requested": max_keywords,
        "cache_hit": False,
    }

    cache.set(cache_key, result)
    return result
