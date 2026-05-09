import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    hf_token: str = os.getenv("HF_TOKEN", "").strip()
    hf_models: list[str] = None
    cache_dir: str = os.getenv("CACHE_DIR", ".cache")
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "60"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    max_keywords: int = int(os.getenv("MAX_KEYWORDS", "15"))

    def __post_init__(self):
        models_env = os.getenv(
            "HF_MODELS",
            "mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceH4/zephyr-7b-beta"
        )
        self.hf_models = [m.strip() for m in models_env.split(",") if m.strip()]


settings = Settings()
