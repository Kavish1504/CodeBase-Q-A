# 🧠 Codebase Q&A Tool

> Ask natural-language questions about any GitHub repository and get grounded answers with exact file paths and line numbers — powered by LangChain, Groq, and ChromaDB.

---

## 🎯 What It Does

Point it at any GitHub repo → it indexes the entire codebase → ask questions in plain English:

- *"Where is the authentication logic?"*
- *"How does the payment flow work?"*
- *"Which files handle database connections?"*
- *"What does the `process_order()` function do?"*

Returns **file name + line numbers + explanation** grounded only in your actual code — no hallucinations.

---


## 🏗️ Architecture

```
GitHub Repo URL
      │
      ▼
┌─────────────────┐
│  repo_loader.py │  ← Git clone + file walking
└────────┬────────┘
         │ raw source files
         ▼
┌─────────────────┐
│ code_chunker.py │  ← AST-aware chunking by function/class
└────────┬────────┘
         │ LangChain Documents + metadata
         ▼
┌─────────────────┐
│   embedder.py   │  ← HuggingFace Embeddings(BAAI/bge-base-en-v1.5)
└────────┬────────┘
         │ vectors + metadata
         ▼
┌─────────────────┐
│    ChromaDB     │  ← Persistent vector store
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
Semantic     BM25
Search      Search
    └────┬─────┘
         │ EnsembleRetriever (RRF)
         ▼
┌─────────────────┐
│  Cohere Rerank  │  ← (optional) cross-encoder reranking
└────────┬────────┘
         │ top-k docs
         ▼
┌─────────────────┐
│  Groq LLaMA-3.3 │  ← LLM generation
│ 70b-versatile   │  ← ConversationalRetrievalChain
└────────┬────────┘
         │
         ▼
  Answer + File + Line References
```

---

## 📁 Project Structure

```
codebase-qa/
├── ingestion/
│   ├── repo_loader.py      # Clone repo, walk file tree
│   ├── code_chunker.py     # AST-aware chunking (function/class boundaries)
│   └── embedder.py         # Embed chunks + persist in ChromaDB
├── retrieval/
│   └── retriever.py        # Hybrid search (semantic + BM25) + Cohere rerank
├── generation/
│   ├── prompt_templates.py # Code-specific system + user prompts
│   └── qa_chain.py         # ConversationalRetrievalChain with memory
├── evaluation/
│   └── ragas_eval.py       # RAGAS metrics: faithfulness, relevancy, precision
├── app/
│   ├── streamlit_app.py    # Chat UI
│   └── api.py              # FastAPI REST backend
├── config.py
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Kavish1504/codebase-qa
cd codebase-qa
conda create -n codebase-qa python=3.11
conda activate codebase-qa
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```



**Streamlit UI:**
```bash
streamlit run app/streamlit_app.py
```

**FastAPI backend:**
```bash
uvicorn app.api:app --reload
# Docs at http://localhost:8000/docs
```

---

## 🔌 API Reference

| Method | Endpoint  | Description                             |
|--------|-----------|-----------------------------------------|
| `POST` | `/ingest` | Index a repo (runs in background)       |
| `GET`  | `/status` | Check indexing status                   |
| `POST` | `/ask`    | Ask a question                          |
|`DELETE`| `/reset`  | Clear conversation memory               |
| `GET`  | `/health` | Liveness probe                          |

**Example:**
```bash
# Index
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/tiangolo/fastapi"}'

# Ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/tiangolo/fastapi", "question": "How are routes registered?"}'
```

---

## 📊 Evaluation (RAGAS)

```bash
python -m evaluation.ragas_eval --repo-url https://github.com/tiangolo/fastapi
# Results saved to evaluation/results.json
```

Metrics tracked:
- **Faithfulness** — answer is grounded in retrieved code
- **Answer Relevancy** — answer addresses the question
- **Context Precision** — retrieved chunks are relevant
- **Context Recall** — all relevant chunks retrieved (needs ground truth)

---
## Example Q&A
**Question:** How does Click handle command groups?

**Answer:**
> Click handles command groups through the `@click.group()` decorator,
> which is used to define a group of commands. A custom `AliasedGroup`
> class in `tests/typing/typing_aliased_group.py:10-31` inherits from
> `click.Group` and overrides `get_command` and `resolve_command` methods
> to provide aliasing functionality.

**Sources:**
- `tests/typing/typing_aliased_group.py:10-31`
- `tests/test_commands.py:520-541`
- `examples/naval/naval.py:6-15`

---

## 🚀 Features

- **AST-aware chunking** — splits code by function/class boundaries, not arbitrary token windows
- **Hybrid search** — combines semantic similarity + BM25 keyword search for better retrieval
- **Conversation memory** — remembers previous questions in the same session
- **Multi-language support** — Python, JavaScript, TypeScript, Java, Go, Rust, C++, and more
- **Zero hallucination policy** — answers grounded only in actual retrieved code
- **RAGAS evaluation** — quantitative metrics to measure RAG quality
- **REST API** — FastAPI backend with Swagger documentation
- **Chat UI** — Streamlit interface with source citations

---

## 🛠️ Tech Stack

| Layer | Tool                                                   |
|-------|--------------------------------------------------------|
| Orchestration | LangChain 0.3                                  |
| LLM | Groq — LLaMA-3.3-70b-versatile                           |
| Embeddings | HuggingFace — BAAI/bge-base-en-v1.5 (free, local) |
| Vector DB | ChromaDB                                           |
| Hybrid Search | EnsembleRetriever (Semantic + BM25)            |
| Evaluation | RAGAS                                             |
| UI | Streamlit                                                 |
| API | FastAPI                                                  |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
