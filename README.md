# 🔬 PaperLens

> Look deeper into research papers — AI-powered keyword extraction using page-aware retrieval and LLM refinement.

Upload any research paper (PDF) and instantly extract the most relevant scientific keywords, ranked by real document evidence — not just word frequency.

---

## 🎯 What Problem Does PaperLens Solve?

Extracting meaningful keywords from scientific papers is harder than it looks:

- **Simple word counters** return generic terms like "method", "result", "system"
- **Basic NLP tools** miss domain-specific phrases like "cold start latency" or "agent coordination overhead"
- **Naive RAG systems** chunk documents randomly, mixing unrelated sections

PaperLens solves all three problems using a **page-aware retrieval architecture** that preserves document structure and uses LLM intelligence to identify truly important scientific concepts.

---

## 🏗️ Architecture

PaperLens follows a page-aware retrieval and refinement pipeline designed for accurate scientific keyword extraction.

| Step | Module | Purpose |
|------|--------|---------|
| 1 | 📄 **PDF Upload** | User uploads a scientific research paper in PDF format |
| 2 | 📑 **Page-Aware Indexing** | The PDF is processed page-by-page to preserve document structure |
| 3 | 🔍 **Relevant Context Retrieval** | The most important page sections are retrieved for keyword extraction |
| 4 | 🧠 **Local Candidate Extraction** | YAKE generates initial keyword candidates from the document |
| 5 | 🤖 **LLM Refinement** | A Hugging Face LLM refines and selects scientifically meaningful keywords |
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
| **Page-Aware Retrieval** | Preserves document structure instead of naive chunking |
| **LLM Refinement** | Uses cloud LLM to select scientifically meaningful keywords |
| **Automatic Fallback** | If LLM is unavailable, local extraction still works |
| **Evidence Scoring** | Keywords ranked by real document metrics, not synthetic values |
| **Category Analysis** | Keywords automatically grouped into technical domains |
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
git clone https://github.com/KalyanKonga16/PaperLens.git
cd PaperLens
