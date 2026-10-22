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

---

## 3.1 Load the documents

```python
from pathlib import Path
from langchain_core.documents import Document

docs = [Document(page_content=p.read_text(), metadata={"source": p.name})
        for p in sorted(Path("data/docs").glob("*.md"))]

print(len(docs), "documents:", [d.metadata["source"] for d in docs])
```

**Explain:** a `Document` is text plus **metadata**. The `source` metadata is what lets us verify later *which file* the answer came from.

## 3.2 Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(len(chunks), "chunks")
for c in chunks[:4]:
    print("-" * 60)
    print(c.metadata, "\n", c.page_content)
```

| Choice | Effect |
|---|---|
| Smaller chunks | Precise retrieval, but facts may be split across chunks |
| Larger chunks | More context per hit, but more noise and cost |
| Overlap | Protects sentences that straddle a boundary |

> **Keep this in mind for Lab 5:** chunk size is one of the settings we will tune using evaluation results.

## 3.3 Embeddings and ChromaDB

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    collection_name="novahome_350_50",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

if vectorstore._collection.count() == 0:       # avoid re-adding on re-runs
    vectorstore.add_documents(chunks)

print("Chunks stored:", vectorstore._collection.count())
```

**Explain:** each chunk becomes a vector. Chroma stores vectors plus text plus metadata and persists them in `chroma_db/`.

### Peek inside Chroma

```python
peek = vectorstore._collection.peek(2)
print(peek["ids"])
print(peek["metadatas"])
print(len(peek["embeddings"][0]), "dimensions per vector")
```

## 3.4 Retrieval

```python
query = "How long does the NovaBrew keep water warm?"

for doc, score in vectorstore.similarity_search_with_score(query, k=3):
    print(f"distance={score:.3f} | {doc.metadata['source']}")
    print(doc.page_content[:200].replace("\n", " "), "\n")
```

> **Reading the score:** Chroma returns a **distance**, so **lower is closer**. Do not compare distances across different embedding models or collections.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever.invoke(query)
```

## 3.5 Augmented generation

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a NovaHome assistant. Answer ONLY using the context below. "
     "If the context does not contain the answer, reply exactly: "
     "\"I don't know based on the provided documents.\" Be concise."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

llm = ChatOllama(model="llama3.2:3b", temperature=0)
rag_chain = GROUNDED_PROMPT | llm | StrOutputParser()

ctx_docs = retriever.invoke(query)
context = "\n\n".join(d.page_content for d in ctx_docs)
print(rag_chain.invoke({"context": context, "question": query}))
```

**Explain the augmentation step:** the model never "learns" the documents. We simply paste retrieved text into the prompt. The system instruction is what tells the model to stay inside it.

---

## 3.6 Package it: create `rag_app.py`

Evaluation needs the **retrieved contexts** as well as the answer, so our function returns both. Run this cell to write the file.

```python
%%writefile rag_app.py
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
```

Test the module:

```python
from rag_app import rag_answer, plain_answer

r = rag_answer("How long does the NovaBrew keep water warm?")
print(r["answer"]); print(r["sources"])
```

---

## 3.7 Compare: standalone LLM vs RAG

```python
import json, pandas as pd
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

pd.set_option("display.max_colwidth", 120)
cmp[["id", "plain", "rag", "reference"]]
```

**Discussion prompts**

| Question | What to look for |
|---|---|
| Where does the plain model invent facts? | Prices, warranty length, shipping speeds |
| Where does RAG fix this? | Single-fact lookups (q1, q2, q5, q6) |
| Where does RAG still struggle? | q9 (needs two documents plus arithmetic), q7/q8 (out-of-scope and rule-based) |

---

## 3.8 Demonstrate retrieval failures on purpose

### Failure A: out-of-scope question (q7)

```python
r = rag_answer("Does NovaHome sell a robot vacuum?")
print(r["answer"]); print(r["sources"])
```

Retrieval will still return the *closest* chunks, even though none is relevant. A vector search **always returns something**. The prompt's refusal rule is the only safety net.

### Failure B: the facts live in two places (q9)

```python
for k in (1, 3):
    r = rag_answer("If I buy a NovaBrew kettle and a NovaAir P2, what is the total price and is standard shipping free?", k=k)
    print(f"k={k}", r["sources"]); print(r["answer"], "\n")
```

With `k=1` only one document can be retrieved, so a complete answer is impossible.

### Failure C: unhelpful chunking

```python
r = rag_answer("How long is the warranty on the NovaAir purifier and does it cover filters?", chunk_size=120, chunk_overlap=0)
print(r["sources"]); print(r["contexts"]); print(r["answer"])
```

Tiny chunks can cut a fact away from the heading that gives it meaning.

### Failure D: a prompt that does not enforce grounding

```python
from rag_app import LOOSE_PROMPT
print(rag_answer("Does NovaHome sell a robot vacuum?", prompt=LOOSE_PROMPT)["answer"])
```

Confident one-sentence answers invite confident hallucination.

> **Bridge to Lab 4:** we can *see* these failures by reading them. Evaluation metrics must make them **measurable**, across hundreds of questions.

---

## 3.9 Connect to the enterprise

This pattern appears in HR policy bots, IT helpdesks, customer-support assistants, sales enablement and compliance Q&A. In every case the stakes are the same: **a fluent wrong answer is worse than "I don't know."**

---

## AI copilot prompts

- *"Explain what `RecursiveCharacterTextSplitter` does with `chunk_size` and `chunk_overlap`, using a short example string."*
- *"Chroma returns distances, not similarities. How do I convert a distance to a cosine similarity for my collection settings?"*
- *"Show me how to add metadata filtering to this Chroma retriever so that only `policies.md` is searched."*

## Learner challenge

Add a fifth document, `faq_shipping_exceptions.md`, containing a rule that **contradicts** `policies.md` (for example, a different express price). Ask the same question and describe what the assistant does. Why is this an evaluation problem and not just a retrieval problem?

## Checkpoint

- [ ] `rag_app.py` exists and `rag_answer()` returns `answer`, `contexts` and `sources`
- [ ] `outputs/lab3_plain_vs_rag.csv` exists
- [ ] You observed at least three distinct failure types

**Next:** [Lab 4: Evaluate the RAG application](lab4_rag_evaluation.md)
