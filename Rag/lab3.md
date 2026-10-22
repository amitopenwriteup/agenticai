# Lab 3: Build a RAG Pipeline with LangChain and ChromaDB

**Section:** 3 | **Time:** about 40 minutes | **Type:** Demonstration + guided coding

## Why RAG for NovaHome?

Customers and marketers now ask **factual questions**: prices, policies, specs, brand rules. A standalone LLM has never seen NovaHome's documents, so it will guess. RAG gives it the right text at question time.

```
Question ─► Embed ─► Search Chroma ─► Top-k chunks ─► Prompt (context + question) ─► LLM ─► Answer
                         ▲
Documents ─► Split ─► Embed ─► Store
```

## Objectives

- Walk through each RAG component: documents, chunking, embeddings, vector store, retrieval, augmentation, generation
- Package the pipeline in a reusable module (`rag_app.py`) so later labs can call it with different settings
- Compare standalone vs RAG answers
- Observe both successful and unsuccessful retrieval

## How this lab works

Each step creates a Python file with a `tee` command, then runs it. Run everything from inside `novahome_eval/` with your virtual environment active, with Ollama running and `data/docs/` created in Lab 0.

> **Running in Jupyter?** Put `%%bash` on the first line of a cell for the `tee` blocks, and use `!python script_name.py` to run a script. The `'EOF'` quotes stop the shell from expanding characters inside the code, and `> /dev/null` hides the echo that `tee` prints. On Windows, use WSL or Git Bash.

| File created | Step | Purpose |
|---|---|---|
| `lab3_01_load_and_chunk.py` | 3.1, 3.2 | Load documents and split them into chunks |
| `lab3_02_embed_store.py` | 3.3 | Embed chunks and store them in ChromaDB |
| `lab3_03_retrieval.py` | 3.4 | Run similarity search and inspect distances |
| `lab3_04_augmented_generation.py` | 3.5 | Paste retrieved context into a prompt and generate |
| `rag_app.py` | 3.6 | Reusable module used by Labs 4 and 5 |
| `lab3_test_rag_app.py` | 3.6 | Quick test of the module |
| `lab3_05_compare.py` | 3.7 | Standalone LLM vs RAG on all questions |
| `lab3_06_failures.py` | 3.8 | Four deliberate failure demonstrations |

---

## 3.1 Load the documents and 3.2 chunk them

```bash
tee lab3_01_load_and_chunk.py > /dev/null << 'EOF'
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3.1 Load the documents
docs = [Document(page_content=p.read_text(), metadata={"source": p.name})
        for p in sorted(Path("data/docs").glob("*.md"))]
print(len(docs), "documents:", [d.metadata["source"] for d in docs])

# 3.2 Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(len(chunks), "chunks")
for c in chunks[:4]:
    print("-" * 60)
    print(c.metadata, "\n", c.page_content)
EOF
```

```bash
python lab3_01_load_and_chunk.py
```

**Explain:** a `Document` is text plus **metadata**. The `source` metadata is what lets us verify later *which file* the answer came from.

| Choice | Effect |
|---|---|
| Smaller chunks | Precise retrieval, but facts may be split across chunks |
| Larger chunks | More context per hit, but more noise and cost |
| Overlap | Protects sentences that straddle a boundary |

> **Keep this in mind for Lab 5:** chunk size is one of the settings we will tune using evaluation results.

## 3.3 Embeddings and ChromaDB

```bash
tee lab3_02_embed_store.py > /dev/null << 'EOF'
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

docs = [Document(page_content=p.read_text(), metadata={"source": p.name})
        for p in sorted(Path("data/docs").glob("*.md"))]
chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50).split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    collection_name="novahome_350_50",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

if vectorstore._collection.count() == 0:       # avoid re-adding on re-runs
    vectorstore.add_documents(chunks)

print("Chunks stored:", vectorstore._collection.count())

# Peek inside Chroma
peek = vectorstore._collection.peek(2)
print(peek["ids"])
print(peek["metadatas"])
print(len(peek["embeddings"][0]), "dimensions per vector")
EOF
```

```bash
python lab3_02_embed_store.py
```

**Explain:** each chunk becomes a vector. Chroma stores vectors plus text plus metadata and persists them in `chroma_db/`.

## 3.4 Retrieval

```bash
tee lab3_03_retrieval.py > /dev/null << 'EOF'
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="novahome_350_50",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="chroma_db",
)

query = "How long does the NovaBrew keep water warm?"

for doc, score in vectorstore.similarity_search_with_score(query, k=3):
    print(f"distance={score:.3f} | {doc.metadata['source']}")
    print(doc.page_content[:200].replace("\n", " "), "\n")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever returned", len(retriever.invoke(query)), "documents")
EOF
```

```bash
python lab3_03_retrieval.py
```

> **Reading the score:** Chroma returns a **distance**, so **lower is closer**. Do not compare distances across different embedding models or collections.

## 3.5 Augmented generation

```bash
tee lab3_04_augmented_generation.py > /dev/null << 'EOF'
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

vectorstore = Chroma(
    collection_name="novahome_350_50",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a NovaHome assistant. Answer ONLY using the context below. "
     "If the context does not contain the answer, reply exactly: "
     "\"I don't know based on the provided documents.\" Be concise."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

llm = ChatOllama(model="llama3.2:3b", temperature=0)
rag_chain = GROUNDED_PROMPT | llm | StrOutputParser()

query = "How long does the NovaBrew keep water warm?"
ctx_docs = retriever.invoke(query)
context = "\n\n".join(d.page_content for d in ctx_docs)

print("CONTEXT SENT TO THE MODEL:\n", context, "\n")
print("ANSWER:", rag_chain.invoke({"context": context, "question": query}))
EOF
```

```bash
python lab3_04_augmented_generation.py
```

**Explain the augmentation step:** the model never "learns" the documents. We simply paste retrieved text into the prompt. The system instruction is what tells the model to stay inside it.

---

## 3.6 Package it: create `rag_app.py`

Evaluation needs the **retrieved contexts** as well as the answer, so our function returns both.

```bash
tee rag_app.py > /dev/null << 'EOF'
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

GEN_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "chroma_db"

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a NovaHome assistant. Answer ONLY using the context below. "
     "If the context does not contain the answer, reply exactly: "
     "\"I don't know based on the provided documents.\" Be concise."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# Deliberately weak prompt used in Labs 4-5 to demonstrate generation failures
LOOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful NovaHome assistant. Answer in one short, confident sentence."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

PLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for NovaHome Appliances. Answer concisely."),
    ("human", "{question}"),
])


def load_docs(path="data/docs"):
    return [Document(page_content=p.read_text(), metadata={"source": p.name})
            for p in sorted(Path(path).glob("*.md"))]


@lru_cache(maxsize=None)
def get_vectorstore(chunk_size=350, chunk_overlap=50):
    name = f"novahome_{chunk_size}_{chunk_overlap}"
    vs = Chroma(collection_name=name,
                embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
                persist_directory=PERSIST_DIR)
    if vs._collection.count() == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vs.add_documents(splitter.split_documents(load_docs()))
    return vs


def rag_answer(question, k=3, prompt=GROUNDED_PROMPT, chunk_size=350, chunk_overlap=50, temperature=0.0):
    """Retrieve k chunks, generate an answer, and return everything needed for evaluation."""
    vs = get_vectorstore(chunk_size, chunk_overlap)
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(question)
    contexts = [d.page_content for d in docs]
    llm = ChatOllama(model=GEN_MODEL, temperature=temperature)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": "\n\n".join(contexts), "question": question}).strip()
    return {"question": question, "answer": answer, "contexts": contexts,
            "sources": [d.metadata["source"] for d in docs]}


def plain_answer(question, temperature=0.0):
    """Standalone LLM with no retrieval (the Lab 1-2 style application)."""
    llm = ChatOllama(model=GEN_MODEL, temperature=temperature)
    return (PLAIN_PROMPT | llm | StrOutputParser()).invoke({"question": question}).strip()
EOF
```

Test the module:

```bash
tee lab3_test_rag_app.py > /dev/null << 'EOF'
from rag_app import rag_answer

r = rag_answer("How long does the NovaBrew keep water warm?")
print(r["answer"])
print(r["sources"])
EOF
```

```bash
python lab3_test_rag_app.py
```

---

## 3.7 Compare: standalone LLM vs RAG

```bash
tee lab3_05_compare.py > /dev/null << 'EOF'
import json
import pandas as pd
from rag_app import plain_answer, rag_answer

rag_eval = json.load(open("data/rag_eval.json"))

rows = []
for item in rag_eval:
    q = item["question"]
    rows.append({"id": item["id"], "question": q,
                 "plain": plain_answer(q),
                 "rag": rag_answer(q)["answer"],
                 "reference": item["reference"]})

cmp = pd.DataFrame(rows)
cmp.to_csv("outputs/lab3_plain_vs_rag.csv", index=False)

for r in rows:
    print(f"[{r['id']}] {r['question']}")
    print("  PLAIN    :", r["plain"])
    print("  RAG      :", r["rag"])
    print("  REFERENCE:", r["reference"], "\n")
EOF
```

```bash
python lab3_05_compare.py
```

**Discussion prompts**

| Question | What to look for |
|---|---|
| Where does the plain model invent facts? | Prices, warranty length, shipping speeds |
| Where does RAG fix this? | Single-fact lookups (q1, q2, q5, q6) |
| Where does RAG still struggle? | q9 (needs two documents plus arithmetic), q7/q8 (out-of-scope and rule-based) |

---

## 3.8 Demonstrate retrieval failures on purpose

One script runs all four failures, so the output can be read side by side.

```bash
tee lab3_06_failures.py > /dev/null << 'EOF'
from rag_app import rag_answer, LOOSE_PROMPT

print("=== Failure A: out-of-scope question (q7) ===")
r = rag_answer("Does NovaHome sell a robot vacuum?")
print(r["answer"])
print(r["sources"], "\n")

print("=== Failure B: the facts live in two places (q9) ===")
q9 = "If I buy a NovaBrew kettle and a NovaAir P2, what is the total price and is standard shipping free?"
for k in (1, 3):
    r = rag_answer(q9, k=k)
    print(f"k={k}", r["sources"])
    print(r["answer"], "\n")

print("=== Failure C: unhelpful chunking (chunk_size=120, overlap=0) ===")
r = rag_answer("How long is the warranty on the NovaAir purifier and does it cover filters?",
               chunk_size=120, chunk_overlap=0)
print(r["sources"])
for c in r["contexts"]:
    print("  CONTEXT:", c.replace("\n", " "))
print(r["answer"], "\n")

print("=== Failure D: a prompt that does not enforce grounding ===")
r = rag_answer("Does NovaHome sell a robot vacuum?", prompt=LOOSE_PROMPT)
print(r["answer"])
EOF
```

```bash
python lab3_06_failures.py
```

**What each failure shows**

| Failure | Lesson |
|---|---|
| A: out-of-scope | A vector search **always returns something**, even when nothing is relevant. The prompt's refusal rule is the only safety net. |
| B: two documents | With `k=1` only one document can be retrieved, so a complete answer is impossible. |
| C: tiny chunks | Tiny chunks can cut a fact away from the heading that gives it meaning. |
| D: loose prompt | Confident one-sentence answers invite confident hallucination. |

> **Bridge to Lab 4:** we can *see* these failures by reading them. Evaluation metrics must make them **measurable**, across hundreds of questions.

---

## 3.9 Connect to the enterprise

This pattern appears in HR policy bots, IT helpdesks, customer-support assistants, sales enablement and compliance Q&A. In every case the stakes are the same: **a fluent wrong answer is worse than "I don't know."**

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Connection refused` or timeout from Ollama | Start Ollama (`ollama serve`) and check `ollama list` |
| `model "nomic-embed-text" not found` | Run `ollama pull nomic-embed-text` |
| Empty or odd results after changing documents | Delete `chroma_db/` and re-run, since collections are only filled when empty |
| `ModuleNotFoundError: rag_app` | Run scripts from `novahome_eval/` and make sure the `rag_app.py` block was run |
| `SyntaxError` right after pasting | The closing `EOF` must be alone on its line with no leading spaces |

## AI copilot prompts

- *"Explain what `RecursiveCharacterTextSplitter` does with `chunk_size` and `chunk_overlap`, using a short example string."*
- *"Chroma returns distances, not similarities. How do I convert a distance to a cosine similarity for my collection settings?"*
- *"Show me how to add metadata filtering to this Chroma retriever so that only `policies.md` is searched."*

## Learner challenge

Use `tee` to add a fifth document, `data/docs/faq_shipping_exceptions.md`, containing a rule that **contradicts** `policies.md` (for example, a different express price). Delete `chroma_db/`, re-run `lab3_05_compare.py`, and describe what the assistant does. Why is this an evaluation problem and not just a retrieval problem?

## Checkpoint

- [ ] `rag_app.py` exists and `rag_answer()` returns `answer`, `contexts` and `sources`
- [ ] `outputs/lab3_plain_vs_rag.csv` exists
- [ ] You observed at least three distinct failure types

**Next:** [Lab 4: Evaluate the RAG application](lab4_rag_evaluation.md)
