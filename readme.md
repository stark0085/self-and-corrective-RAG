# 🧠 Self-Correcting RAG Agent

> *A research-level Retrieval Augmented Generation system that retrieves knowledge, generates answers, evaluates its own outputs, and iteratively self-corrects — with confidence scoring, hallucination detection, and memory support.*

---

## 🧩 Problem

Standard RAG systems follow a simple pipeline: **retrieve → generate → done.**

That's a serious limitation.

They have **no mechanism to catch**:
- Retrieved documents that are completely off-topic
- Answers that hallucinate facts not present in the source
- Responses that only partially address the question
- Queries that were too vague to retrieve useful context

The result? Users receive confidently wrong answers with no signal that something went wrong.

**This project solves that.** By building a self-correcting feedback loop on top of RAG, the system can detect its own failures and fix them before responding.

---

## 🎯 Approach

Instead of retrieve → generate → done, this system does:

```
Retrieve → Grade Relevance → Generate → Self-Evaluate → Fix if Needed → Score Confidence → Respond
```

If the retrieved documents aren't relevant, it **rewrites the query and retrieves again.**
If the generated answer is poorly grounded or incomplete, it **revises and retries** (up to 2x).
Every final answer comes with a **confidence score** built from three independent graders.

### Core Evaluation Dimensions
| Grader | What it checks |
|--------|---------------|
| **Relevance** | Are the retrieved docs actually useful for this question? |
| **Grounding** | Is the answer supported by the docs, or is it hallucinating? |
| **Completeness** | Does the answer fully address what was asked? |

The three scores combine into a single **weighted confidence percentage** shown to the user.

---

## 🔁 Iterations

### v1 — Basic RAG (Baseline)
Built a simple LangChain pipeline: PDF upload → chunking → ChromaDB storage → retrieval → LLM generation. No feedback, no evaluation. Just retrieve and generate.

**Problem found:** System confidently answered questions using irrelevant chunks. No way to know if the answer was trustworthy.

---

### v2 — Relevance Grading
Added a **relevance grader** that scores retrieved documents before passing them to the LLM. If relevance is below threshold, the query gets rewritten and retrieval is retried.

**Problem found:** Even with relevant docs, the LLM sometimes fabricated details not present in the source.

---

### v3 — Hallucination & Completeness Graders
Introduced two more independent graders post-generation:
- **Grounding grader** — checks if every claim in the answer is supported by retrieved docs
- **Completeness grader** — checks if the answer actually addresses the full question

Each grader runs as a separate LLM call with a focused prompt, scoring 0.0–1.0.

**Problem found:** The three graders were running inside a linear chain — there was no way to loop back and retry on failure.

---

### v4 — LangGraph Cyclic Agent Loop
Replaced the linear LangChain chain with a **LangGraph graph** with conditional branching:
- If relevance fails → rewrite query → retrieve again
- If grounding or completeness fails → revise query → regenerate
- After **max 2 retries** → accept best available answer, flag low confidence

This turned the system from a pipeline into an **agent with a correction loop**.

**Problem found:** ChromaDB would throw stale collection errors if a new PDF was uploaded mid-session. Global object initialization was the culprit.

---

### v5 — Lazy Loading + Confidence Scorer + UI + Memory
- **Lazy loading** for ChromaDB and the retriever — fresh objects on every call, no stale state
- **Weighted confidence scorer** — combines the three grader scores into one user-facing number
- **Streamlit UI** — PDF upload, question input, score visualization, query history display
- **Conversation memory** — tracks prior questions in session for context continuity

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Retrieve Docs  │◄────────────────────────┐
│  (ChromaDB)     │                         │
└────────┬────────┘                         │
         │                                  │ Rewrite Query
         ▼                                  │
┌─────────────────┐                         │
│  Grade Docs     │──── Not Relevant? ──────┘
│  (Relevance)    │
└────────┬────────┘
         │ Relevant
         ▼
┌─────────────────┐
│ Generate Answer │
│  (OpenRouter    │
│     LLM)        │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│      Self-Evaluation         │
│  ┌──────────────────────┐    │
│  │  Relevance Score     │    │
│  │  Grounding Score     │    │
│  │  Completeness Score  │    │
│  └──────────────────────┘    │
└────────┬─────────────────────┘
         │
    ┌────┴────┐
    │         │
  Pass      Fail
    │         │
    │         ▼
    │   ┌──────────────┐
    │   │ Revise Query │──► Retry (max 2x)
    │   └──────────────┘
    │
    ▼
┌─────────────────────┐
│  Confidence Score   │
│  (Weighted Average) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Final Response    │
│  Answer + Scores +  │
│  Query History      │
└─────────────────────┘
```

---

## 🔑 Key Design Choices

### 1. Lazy Loading for ChromaDB
ChromaDB collections throw `NotFoundError` if the retriever is initialized globally and a new PDF is uploaded mid-session (the collection reference goes stale). By instantiating the client and retriever **fresh on every call**, this is completely avoided — no restart needed between uploads.

### 2. Three Separate Graders Instead of One
A single "quality score" prompt would conflate three different failure modes. Splitting into relevance, grounding, and completeness means:
- Each grader has a **tightly focused prompt** → more accurate scoring
- You can **diagnose exactly what went wrong** (bad retrieval vs hallucination vs incomplete answer)
- Weights can be tuned independently per use case

### 3. LangGraph Over Plain LangChain
LangChain is a linear pipeline. LangGraph supports **cycles and conditional edges**, which is exactly what a correction loop needs. The graph structure makes the retry logic explicit, inspectable, and easy to extend.

### 4. OpenRouter for LLM Access
Using OpenRouter as the LLM backend makes the system **model-agnostic** — swapping from GPT-4o-mini to Claude or Mistral is a one-line config change. No vendor lock-in.

### 5. HuggingFace `all-MiniLM-L6-v2` for Embeddings
Local embedding model — no extra API cost, no latency from an external call, and performant enough for document-level semantic search. Keeps the system usable on a free API budget.

### 6. Max 2 Retries with Graceful Degradation
Unlimited retries would create infinite loops on genuinely unanswerable questions. Capping at 2 retries and **always returning a scored answer** (even a low-confidence one) keeps the UX predictable. The confidence score tells the user when to trust the answer and when to verify externally.

---

## 📁 Project Structure

```
self_correcting_rag/
│
├── .env                          # 🔐 API keys (never commit this)
├── .gitignore
├── requirements.txt
├── README.md
│
├── app/
│   ├── config.py                 # LLM + Embeddings setup (OpenRouter + HuggingFace)
│   ├── state.py                  # LangGraph shared state (Pydantic model)
│   │
│   ├── ingestion/
│   │   ├── loader.py             # PDF loader (PyPDF)
│   │   ├── splitter.py           # Text chunker (RecursiveCharacterTextSplitter)
│   │   └── vectorstore.py        # ChromaDB vector store (lazy loading)
│   │
│   ├── retrieval/
│   │   └── retriever.py          # Similarity search retriever (lazy loading)
│   │
│   ├── generation/
│   │   └── generator.py          # LLM answer generation from context
│   │
│   ├── grading/
│   │   ├── relevance.py          # Are retrieved docs relevant?
│   │   ├── grounding.py          # Is answer grounded in docs?
│   │   └── completeness.py       # Does answer fully address the question?
│   │
│   ├── confidence/
│   │   └── confidence_scorer.py  # Weighted confidence score
│   │
│   ├── memory/
│   │   └── memory.py             # Conversation history tracking
│   │
│   ├── graph/
│   │   └── workflow.py           # LangGraph agent loop (CORE)
│   │
│   └── ui/
│       └── streamlit_app.py      # Streamlit frontend
│
└── data/
    ├── chroma_db/                # Persistent vector database (auto-generated)
    └── temp.pdf                  # Temporary uploaded file (auto-generated)
```

---


## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/self-correcting-rag-langgraph
cd self-correcting-rag-langgraph
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_key_here
```
Get your free key at [openrouter.ai](https://openrouter.ai)

### 4. Run
```bash
streamlit run app/ui/streamlit_app.py --server.fileWatcherType none
```
Open `http://localhost:8501`

---

## 📦 Requirements

```
langchain
langgraph
langchain-community
langchain-openai
langchain-text-splitters
chromadb
sentence-transformers
faiss-cpu
pydantic
streamlit
pypdf
numpy
torch
transformers
openai
python-dotenv
```

---

## 📊 Score Interpretation

| Score | 🟢 High (≥0.85) | 🟡 Medium (0.65–0.85) | 🔴 Low (<0.65) |
|-------|-----------------|----------------------|----------------|
| **Relevance** | Retrieved docs are on-topic | Partially relevant | Off-topic retrieval |
| **Grounding** | No hallucination | Some unsupported claims | Hallucinated answer |
| **Completeness** | Fully answered | Partial answer | Incomplete/refused |
| **Confidence** | Trust the answer | Use with caution | Verify externally |

---

## 🧪 Stress Test Cases

| Test | Question | Expected Behaviour |
|------|----------|--------------------|
| ✅ Grounding | *"What was the exact misdiagnosis reduction rate?"* | High grounding, correct number cited |
| 🎯 Hallucination trap | *"What did GPT-5 achieve in medical exams?"* | Low confidence, refusal to fabricate |
| 📉 Completeness | *"How much does AI reduce drug timelines specifically?"* | Low completeness flagged |
| 📊 Relevance | *"What % of US equity trading is algorithmic?"* | Low relevance, query rewritten |
| 🔀 Contradiction | *"What is the consensus AI accuracy for pneumonia?"* | Flags conflicting information |

---

## 🩺 Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: langchain.text_splitter` | `pip install langchain-text-splitters` |
| `chromadb.errors.NotFoundError` | Delete `data/chroma_db/*` and re-upload PDF |
| `RuntimeError: no running event loop` | Add `--server.fileWatcherType none` to run command |
| Duplicate chunks retrieved | Clear `data/chroma_db/` before re-indexing |

---

## 🧩 Standard RAG vs This Project

| Feature | Standard RAG | This Project |
|---------|-------------|--------------|
| Retrieval | ✅ | ✅ |
| Generation | ✅ | ✅ |
| Self-correction loop | ❌ | ✅ |
| Hallucination detection | ❌ | ✅ |
| Multi-criteria grading | ❌ | ✅ |
| Query reformulation | ❌ | ✅ |
| Confidence scoring | ❌ | ✅ |
| Conversation memory | ❌ | ✅ |
| LangGraph agent architecture | ❌ | ✅ |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenRouter (GPT-4o-mini) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Agent Framework | LangGraph |
| RAG Framework | LangChain |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |

---

*Built with LangChain · LangGraph · ChromaDB · HuggingFace · OpenRouter · Streamlit*
