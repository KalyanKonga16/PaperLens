import os
import json
import hashlib


class JsonCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str):
        path = self._path(key)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, data: dict):
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def make_cache_key(pdf_bytes: bytes, filename: str, models: list[str], max_keywords: int) -> str:
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    config_hash = hashlib.sha256(
        f"{filename}|{'|'.join(models)}|{max_keywords}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{pdf_hash}_{config_hash}"
