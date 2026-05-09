```markdown
# 🔬 Scientific Keyword Extractor

> AI-powered keyword extraction from scientific papers using page-aware retrieval and LLM refinement.

Upload any research paper (PDF) and instantly extract the most relevant scientific keywords, ranked by real document evidence — not just word frequency.

---

## 🎯 What Problem Does This Solve?

Extracting meaningful keywords from scientific papers is harder than it looks:

- **Simple word counters** return generic terms like "method", "result", "system"
- **Basic NLP tools** miss domain-specific phrases like "cold start latency" or "agent coordination overhead"
- **Naive RAG systems** chunk documents randomly, mixing unrelated sections

This tool solves all three problems using a **page-aware retrieval architecture** that preserves document structure and uses LLM intelligence to identify truly important scientific concepts.

---

## 🏗️ Architecture

```
┌──────────────┐
│  PDF Upload  │
└──────┬───────┘
       ▼
┌──────────────────────┐
│  Page-Aware Indexing  │  ← Splits by page, not random chunks
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│  Semantic Retrieval   │  ← Finds most relevant sections
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│  Local Candidate      │  ← YAKE extracts initial keywords
│  Extraction           │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│  LLM Refinement       │  ← Cloud LLM selects best keywords
│  (with auto fallback) │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│  Evidence Scoring     │  ← Real metrics: frequency, spread,
│                       │     context support
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│  Visual Analytics     │  ← Word Cloud, Donut, Radar, Bar
└──────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Page-Aware Retrieval** | Preserves document structure instead of naive chunking |
| **LLM Refinement** | Uses cloud LLM to select scientifically meaningful keywords |
| **Automatic Fallback** | If LLM is unavailable, local extraction still works |
| **Evidence Scoring** | Keywords ranked by real document metrics, not synthetic values |
| **Category Analysis** | Keywords automatically grouped into technical domains |
| **4 Business Visualizations** | Word Cloud, Topic Distribution, Evidence Radar, Top Keywords |
| **Adjustable Extraction** | Slider to control exact number of keywords (5–50) |
| **Export** | Download results as CSV or JSON |

---

## 🆚 How Is This Different From Basic RAG?

| Aspect | Basic RAG | This Project |
|--------|-----------|--------------|
| Chunking | Fixed-size, arbitrary splits | Page-aware, structure-preserving |
| Retrieval | Embedding similarity only | TF-IDF + semantic + page context |
| Output | Raw LLM text | Structured JSON with evidence scores |
| Fallback | Fails if LLM unavailable | Automatic local fallback |
| Ranking | No ranking or synthetic | Evidence-based (frequency + spread + context) |
| Visualization | None | 4 business-focused charts |

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit** — Interactive web UI
- **PyMuPDF (fitz)** — PDF text extraction
- **YAKE** — Local keyword candidate generation
- **scikit-learn** — TF-IDF vectorization and retrieval
- **Hugging Face Inference API** — Cloud LLM refinement
- **Plotly + Matplotlib** — Business visualizations
- **WordCloud** — Visual keyword representation

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/KalyanKonga16/ScientificKeywordExtractor.git
cd ScientificKeywordExtractor
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env
```

Open `.env` and add your Hugging Face token:

```env
HF_TOKEN=hf_your_token_here
```

Get your free token at: https://huggingface.co/settings/tokens

### 5. Run the application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📊 Sample Output

### Keyword Table

| # | Keyword | Category | Evidence Score | Occurrences |
|---|---------|----------|---------------|-------------|
| 1 | Agent Coordination Latency | Latency / Performance | 100.0 | 12 |
| 2 | Cold Start Latency | Latency / Performance | 85.3 | 8 |
| 3 | Retrieval Latency | Retrieval / Cache | 72.1 | 6 |
| 4 | Multi-Agent System | AI Agents | 68.4 | 9 |

### Visual Insights

The tool generates 4 evidence-based visualizations:

1. **Word Cloud** — Keyword prominence based on evidence score
2. **Topic Distribution** — Donut chart showing technical domain breakdown
3. **Evidence Radar** — Circular view of all keyword strengths
4. **Top Keywords** — Bar chart of strongest keywords

---

## 📁 Project Structure

```
ScientificKeywordExtractor/
├── app.py                         # Streamlit UI
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
├── .gitignore
├── README.md
│
└── src/
    ├── __init__.py
    ├── config.py                  # Configuration management
    ├── cache.py                   # Result caching layer
    ├── pdf_service.py             # PDF text extraction
    ├── page_index_service.py      # Page-aware indexing and retrieval
    ├── candidate_extractor.py     # Local keyword extraction (YAKE)
    ├── hf_client.py               # Hugging Face LLM integration
    ├── keyword_pipeline.py        # Main orchestration pipeline
    └── visuals.py                 # Business visualizations
```

---

## 🔮 Future Enhancements

- [ ] Multi-document comparative analysis
- [ ] API endpoint (FastAPI) for programmatic access
- [ ] OCR support for scanned PDFs
- [ ] Citation-aware keyword weighting
- [ ] Domain-specific category models

---

## 📜 License

This project is for educational and portfolio purposes.

---

## 🙏 Acknowledgements

- [PageIndex by VectifyAI](https://github.com/VectifyAI/PageIndex) — Inspiration for page-aware retrieval
- [Hugging Face](https://huggingface.co) — LLM inference API
- [YAKE](https://github.com/LIAAD/yake) — Keyword extraction
- [Streamlit](https://streamlit.io) — Web framework
```
