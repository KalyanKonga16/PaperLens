import json
import time
from huggingface_hub import InferenceClient
from .candidate_extractor import postprocess_keywords


class HFKeywordRefiner:
    def __init__(self, token: str, models: list[str], max_retries: int = 2):
        self.token = token
        self.models = [m for m in models if m.strip()]
        self.max_retries = max_retries

    def _build_prompt(self, candidates: list[str], retrieved_chunks: list[dict], max_keywords: int) -> str:
        candidate_text = ", ".join(candidates[:35]) if candidates else "None"

        context_blocks = []
        for chunk in retrieved_chunks[:6]:
            snippet = " ".join(chunk["text"].split())[:700]
            context_blocks.append(f"[Page {chunk['page_number']}] {snippet}")

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are an expert keyword and keyphrase extraction system designed to work on any type of document.

Task:
Extract up to {max_keywords} high-quality, meaningful keywords or short keyphrases from the document.

Rules:
- Use only information supported by the context.
- Prefer specific, domain-meaningful terms relevant to the document type.
- Capture concepts, entities, methods, products, processes, technologies, organizations, metrics, places, or any subject the document focuses on.
- Avoid generic filler words like document, report, study, paper, content, information, section, table.
- Do not include author names, page numbers, dates, or boilerplate.
- Remove duplicates and near-duplicates.
- Prefer multi-word phrases when they carry stronger meaning.
- Return concise phrases only.
- Return ONLY valid JSON.

Candidate keywords from local extraction:
{candidate_text}

Relevant page-aware context:
{context}

Return exactly this JSON:
{{"keywords":["term1","term2","term3"],"pages_used":[1,2],"summary":"one sentence summary of the document"}}
"""
        return prompt.strip()

    def _extract_json_text(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response did not contain valid JSON.")
        return text[start:end + 1]

    def _call_model(self, model: str, prompt: str) -> str:
        client = InferenceClient(model=model, token=self.token)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=300,
                    do_sample=False,
                    return_full_text=False,
                )
                return response
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise last_error

    def refine_keywords(self, candidates: list[str], retrieved_chunks: list[dict], max_keywords: int = 15) -> dict:
        if not self.token:
            raise RuntimeError("HF_TOKEN is missing.")

        prompt = self._build_prompt(candidates, retrieved_chunks, max_keywords)
        last_error = None

        for model in self.models:
            try:
                raw_response = self._call_model(model, prompt)
                json_text = self._extract_json_text(raw_response)
                parsed = json.loads(json_text)

                keywords = parsed.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [keywords]

                keywords = postprocess_keywords([str(k) for k in keywords], max_keywords=max_keywords)
                if not keywords:
                    raise ValueError("No valid keywords returned by model.")

                pages_used = []
                for p in parsed.get("pages_used", []):
                    try:
                        pages_used.append(int(p))
                    except Exception:
                        pass

                return {
                    "keywords": keywords,
                    "pages_used": sorted(set(pages_used)),
                    "summary": str(parsed.get("summary", "")).strip(),
                    "model": model,
                    "raw_response": raw_response,
                }

            except Exception as e:
                last_error = f"{model}: {e}"
                continue

        raise RuntimeError(last_error or "All HF models failed.")
