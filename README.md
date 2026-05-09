# 🔬 PaperLens

> Look deeper into any Document — AI-powered keyword extraction using page-aware retrieval and LLM refinement.

Upload any Document (PDF, DOCX, TXT, MD be it research papers, technical reports, business documents, whitepapers, contracts) and instantly extract the most relevant keywords, ranked by real document evidence — not just word frequency.

🌐 **Live Demo:** [https://paperlensforu.streamlit.app/](https://paperlensforu.streamlit.app/)

👨‍💻 **Built By:** [Kalyan Konga](https://github.com/KalyanKonga16)

---

## 🎯 What Problem Does PaperLens Solve?

Extracting meaningful keywords from documents is harder than it looks:

- **Simple word counters** return generic terms like "method", "result", "system"
- **Basic NLP tools** miss domain-specific phrases like "cold start latency" or "agent coordination overhead"
- **Naive RAG systems** chunk documents randomly, mixing unrelated sections

PaperLens solves all three problems using a **page-aware retrieval architecture** that preserves document structure and uses LLM intelligence to identify truly important concepts.

---

## 🏗️ Architecture

PaperLens follows a page-aware retrieval and refinement pipeline designed for accurate keyword extraction across any document type.

| Step | Module | Purpose |
|------|--------|---------|
| 1 | 📄 **PDF Upload** | User uploads any PDF document |
| 2 | 📑 **Page-Aware Indexing** | The PDF is processed page-by-page to preserve document structure |
| 3 | 🔍 **Relevant Context Retrieval** | The most important page sections are retrieved for keyword extraction |
| 4 | 🧠 **Local Candidate Extraction** | YAKE generates initial keyword candidates from the document |
| 5 | 🤖 **LLM Refinement** | A Hugging Face LLM refines and selects meaningful keywords |
| 6 | 📊 **Evidence Scoring** | Keywords are ranked using real document signals such as frequency and relevance |
| 7 | 📈 **Visual Analytics Dashboard** | Results are displayed using tables, word cloud, topic distribution, evidence radar, and top keyword charts |

### Pipeline Flow

**PDF Upload**  
⬇️  
**Page-Aware Indexing**  
⬇️  
**Relevant Context Retrieval**  
⬇️  
**Local Candidate Keyword Extraction**  
⬇️  
**LLM-Based Keyword Refinement**  
⬇️  
**Evidence Scoring**  
⬇️  
**Visual Analytics Dashboard**

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Format Support** | Works with PDF, DOCX, TXT, and Markdown files |
| **Search & Filter** | Quickly filter the keyword table by typing |
| **File Size Validation** | Warnings for large files to prevent failures |
| **Page-Aware Retrieval** | Preserves document structure instead of naive chunking |
| **LLM Refinement** | Uses cloud LLM to select meaningful keywords |
| **Automatic Fallback** | If LLM is unavailable, local extraction still works |
| **Evidence Scoring** | Keywords ranked by real document metrics, not synthetic values |
| **Category Analysis** | Keywords automatically grouped into business and technical domains |
| **4 Business Visualizations** | Word Cloud, Topic Distribution, Evidence Radar, Top Keywords |
| **Adjustable Extraction** | Slider to control exact number of keywords (5–50) |
| **Export** | Download results as CSV or JSON |

---

## 🆚 How Is PaperLens Different From Basic RAG?

| Aspect | Basic RAG | PaperLens |
|--------|-----------|-----------|
| Chunking | Fixed-size, arbitrary splits | Page-aware, structure-preserving |
| Retrieval | Embedding similarity only | TF-IDF + semantic + page context |
| Output | Raw LLM text | Structured JSON with evidence scores |
| Fallback | Fails if LLM unavailable | Automatic local fallback |
| Ranking | No ranking or synthetic | Evidence-based (frequency + spread + context) |
| Visualization | None | 4 business-focused charts |
| Document Scope | Limited | Universal PDF support |

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

### Try It Online (No Setup Required)

🌐 **[Open PaperLens Live Demo →](https://paperlensforu.streamlit.app/)**

### Or Run Locally

#### 1. Clone the repository

```bash
git clone https://github.com/KalyanKonga16/PaperLens.git
cd PaperLens
```

#### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure environment

```bash
copy .env.example .env
```

Open `.env` and add your Hugging Face token:

```env
HF_TOKEN=hf_your_token_here
```

Get your free token at: https://huggingface.co/settings/tokens

#### 5. Run PaperLens

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📊 Sample Output

### Keyword Table

| # | Keyword | Category | Evidence Score | Occurrences |
|---|---------|----------|---------------|-------------|
| 1 | Agent Coordination Latency | AI / Machine Learning | 100.0 | 12 |
| 2 | Cold Start Latency | Performance / Optimization | 85.3 | 8 |
| 3 | Retrieval Latency | Data / Storage | 72.1 | 6 |
| 4 | Multi-Agent System | AI / Machine Learning | 68.4 | 9 |

### Visual Insights

PaperLens generates 4 evidence-based visualizations:

1. **Word Cloud** — Keyword prominence based on evidence score
2. **Topic Distribution** — Donut chart showing technical domain breakdown
3. **Evidence Radar** — Circular view of all keyword strengths
4. **Top Keywords** — Bar chart of strongest keywords

---

## 📁 Project Structure

```
PaperLens/
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

---

## 👨‍💻 About the Developer

**Kalyan Konga**  
GitHub: [@KalyanKonga16](https://github.com/KalyanKonga16)

---

**Built with ❤️ by Kalyan Konga**
