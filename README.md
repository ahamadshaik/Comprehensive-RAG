# 🧠 Comprehensive-RAG — Hybrid, Rerank, Query Expansion, Compression

**Comprehensive-RAG** is a flexible, research-grade Retrieval-Augmented Generation (RAG) framework that runs over your local `./docs` folder or `gs://bucket/prefix`.  
It combines **hybrid retrieval**, **reranking**, **query expansion**, and **contextual compression** to deliver high-precision, explainable answers.

---

## 🚀 Features

- **Hybrid Retrieval** — FAISS (MiniLM) + BM25 with score fusion  
- **Cross-Encoder Reranking** — `BAAI/bge-reranker-base`  
- **Query Expansion** — Multi-Query, HyDE (Hypothetical Document Expansion), HyPE (Prompt Enrichment)  
- **Contextual Compression** — LLM-based sentence filtering  
- **Adaptive Retrieval** — Dynamically expands top-K and abstains on low confidence  
- **Explainability** — Returns BM25, dense cosine, and rerank scores per chunk  
- **Minimal Evaluation Harness** — Precision@K and MRR-like metrics  
- **Cloud-Friendly** — Works locally or with `gs://` paths (Vertex AI compatible)  
- **Config-Driven** — Switch between OpenAI and Vertex backends with `.env`  

---

## 📦 Quickstart

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# → fill in values (see Configuration section below)

# 4) Ingest docs → build FAISS index
python -m app.ingest

# 5) Ask programmatically (no API needed)
python -c "from app.rag import answer; print(answer('What is in the architecture doc?'))"
```

Optional convenience targets:

```bash
make setup
make ingest
make eval
make test    # unit tests (pytest)
make serve   # start FastAPI server
```

> **Vertex AI Workbench:**  
> Uses your Application Default Credentials.  
> Set `DOCS_PATH=gs://your-bucket/prefix` in `.env`.

---

## 🧭 Project Layout

```
Comprehensive-RAG/
├─ app/
│  ├─ config.py        # pydantic settings (.env)
│  ├─ utils.py         # GCS helpers, timers
│  ├─ ingest.py        # load docs -> chunk -> embed -> FAISS
│  ├─ retriever.py     # hybrid (BM25 + dense) + rerank + expansion
│  ├─ query.py         # Multi-Query, HyDE, HyPE
│  ├─ compress.py      # contextual compression (LLM filter)
│  ├─ llm.py           # OpenAI/Vertex adapters
│  ├─ rag.py           # orchestrate → answer()
│  └─ api.py           # optional FastAPI endpoint
├─ notebooks/
│  ├─ 00_quickstart.ipynb
│  └─ 01_eval_demo.ipynb
├─ eval/
│  ├─ qa.jsonl
│  └─ evaluate.py
├─ docs/               # your source files (or GCS source)
├─ data/               # built artifacts (ignored)
├─ requirements.txt
├─ .env.example
├─ Makefile
└─ README.md
```

---

## ⚙️ Configuration

Copy `.env.example` → `.env` and edit as needed:

| Key | Example | Notes |
|-----|----------|-------|
| `PROVIDER` | `openai` or `vertex` | Switches backend |
| `MODEL` | `gpt-4o-mini` | For OpenAI |
| `OPENAI_API_KEY` | `sk-...` | Required if OpenAI |
| `VERTEX_MODEL` | `gemini-1.5-pro` | For Vertex |
| `VERTEX_LOCATION` | `us-central1` | Vertex region |
| `VERTEX_PROJECT_ID` | `my-project` | GCP project ID |
| `DOCS_PATH` | `docs` or `gs://bucket/prefix` | Ingestion source |
| `DATA_DIR` | `data` | Artifacts (FAISS, metadata) |
| `TOP_K_*` | `50 / 20 / 8` | Dense, fused, final |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | `0.4 / 0.6` | Score fusion |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | CrossEncoder model |
| `MIN_CTX_SCORE` | `0.35` | Abstain/confidence threshold |
| `ENABLE_QUERY_EXPANSION` | `true` | Enable Multi-Query / HyDE / HyPE |
| `ENABLE_COMPRESSION` | `true` | Contextual compression |
| `ENABLE_TRACING` | `true` | Print step timings |

> **Vertex users:** Ensure ADC is active (`gcloud auth application-default login`)  
> **OpenAI users:** Set `OPENAI_API_KEY` in `.env` or environment variables.

---

## 💻 Usage

### In Notebooks (Recommended)

```python
from app.ingest import build_index
from app.rag import answer

build_index()  # uses DOCS_PATH/DATA_DIR from .env
resp = answer("Summarize the main architectural decisions.")
print(resp)
```

Output example:

```python
{
  "answer": "...",
  "confidence": 0.87,
  "sources": ["docs/architecture.md"]
}
```

---

### As an API (Optional)

```bash
uvicorn app.api:app --reload --port 8000
```

Example request:

```bash
curl -s http://localhost:8000/ask -H "Content-Type: application/json"   -d '{"question":"What’s in the architecture doc?"}' | jq .
```

---

## 📊 Evaluation

Put a few ground-truth prompts in `eval/qa.jsonl`:

```jsonl
{"q": "What are the main modules?", "must_contain": "docs/architecture.md"}
{"q": "How is auth handled?", "must_contain": "docs/security.md"}
```

Then run:

```bash
python eval/evaluate.py
# prints P@retrieval and latency
```

**Tips:**
- Compare runs **with/without reranking or compression** to see gains.  
- Add an LLM “faithfulness” judge for sampled outputs if needed.

---

## 🧩 Design Notes

### Flow

```
Query
 ├─ Query Expansion (Multi-Query, HyDE, HyPE)
 ├─ Retrieval: Dense (FAISS) + BM25  ──► Score Fusion
 ├─ Cross-Encoder Rerank (top-K)
 ├─ Adaptive K (expand if low confidence)
 ├─ Contextual Compression (LLM sentence filter)
 └─ Generation with Citations + Abstain on low-confidence
```

### Why these choices?

| Technique | Purpose |
|------------|----------|
| Hybrid + Rerank | Strong relevance with modest compute |
| Query Expansion | Better recall on vague queries |
| Compression | Lower token cost, fewer hallucinations |
| Abstention | Honest answers when confidence is low |
| Config-driven | Simple switch between OpenAI and Vertex |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| **FAISS not found** | Install `faiss-cpu` matching your Python version |
| **OpenAI auth error** | Set `OPENAI_API_KEY` in `.env` |
| **Vertex auth error** | Run `gcloud auth application-default login` and check `VERTEX_PROJECT_ID` |
| **Slow reranking** | Reduce `TOP_K_FUSED` or use a smaller reranker |
| **Too many tokens** | Enable compression or lower `TOP_K_FINAL` |
| **Weak answers** | Increase `TOP_K_*`, tune `BM25_WEIGHT/DENSE_WEIGHT`, enable expansion |

---

## 🧭 Roadmap

- Lightweight Graph-RAG hop for entity queries  
- RAPTOR hierarchical summaries for long docs  
- Multimodal captions/OCR mode for PDFs  
- Optional Streamlit UI  
- Better offline “faithfulness” metrics  

---

## 📄 License

MIT — see [LICENSE](./LICENSE)
