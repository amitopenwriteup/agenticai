# Lab 4: Evaluating the RAG Application

**Section:** 4 | **Time:** about 50 minutes | **Type:** Demonstration + guided coding

## Why RAG Needs More Than Answer Quality

A RAG answer can be wrong for several different reasons, and each needs a different fix:

```
Question ─► [Retrieval] ─► Contexts ─► [Generation] ─► Answer
              ▲                           ▲
   Context Precision / Recall /      Faithfulness / Answer Relevance /
   Context Relevance                 Answer Correctness
```

| If this is bad... | ...the problem is probably in |
|---|---|
| Context Recall / Context Relevance | Retrieval (chunking, embeddings, `k`) |
| Context Precision | Ranking and noise in retrieved chunks |
| Faithfulness | Generation ignoring or going beyond the context |
| Answer Relevance | Prompt design or the model not addressing the question |
| Answer Correctness | Any of the above; use the other metrics to locate the cause |

## Objectives

- Run a fast, deterministic retrieval check that needs no judge LLM
- Run RAGAS metrics and generate a per-question report
- Interpret score ranges
- Show retrieval-good/generation-bad and the reverse
- Locate the failing component for each bad answer

---

## 4.1 Generate answers and contexts for all questions

```python
import json, pandas as pd
from rag_app import rag_answer

rag_eval = json.load(open("data/rag_eval.json"))

records = []
for item in rag_eval:
    out = rag_answer(item["question"], k=3)
    records.append({**item, **{"answer": out["answer"], "contexts": out["contexts"], "sources": out["sources"]}})

df = pd.DataFrame(records)
df.to_pickle("outputs/lab4_rag_run.pkl")
df[["id", "question", "answer", "sources"]]
```

## 4.2 Retrieval sanity check (no judge LLM needed)

Because we know which document should contain each answer, we can compute a simple **hit rate** and **reciprocal rank**. These are fast, free and fully deterministic, so run them **before** any LLM-judged metric.

```python
def retrieval_check(row):
    expected = set(row["expected_sources"])
    if not expected:                       # out-of-scope question: nothing should be "found"
        return pd.Series({"hit": None, "all_found": None, "rr": None})
    retrieved = row["sources"]
    hit = any(s in expected for s in retrieved)
    all_found = expected.issubset(set(retrieved))
    rr = next((1 / (i + 1) for i, s in enumerate(retrieved) if s in expected), 0.0)
    return pd.Series({"hit": hit, "all_found": all_found, "rr": rr})

df = pd.concat([df, df.apply(retrieval_check, axis=1)], axis=1)
print(df[["id", "sources", "hit", "all_found", "rr"]])
print("\nHit rate@3:", df["hit"].dropna().astype(float).mean().round(2))
print("All-sources-found rate:", df["all_found"].dropna().astype(float).mean().round(2))
print("MRR:", df["rr"].dropna().mean().round(2))
```

| Metric | Meaning |
|---|---|
| Hit rate@k | At least one expected document was retrieved |
| All-found rate | Every expected document was retrieved (important for multi-document questions such as q9) |
| MRR | How high the first relevant document ranks |

> Note that a document-level hit does not prove the **right chunk** was retrieved. That is what the RAGAS context metrics assess.

---

## 4.3 Set up RAGAS

> **Version note:** RAGAS APIs change between releases. The code below follows the `EvaluationDataset` interface (0.2.x and later). Check `pip show ragas` and the official documentation if an import fails, and ask your AI copilot to adapt the code to your version.

```python
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,                 # "Answer Relevance"
    LLMContextPrecisionWithReference,  # "Context Precision"
    LLMContextRecall,                  # "Context Recall"
    AnswerCorrectness,
)

try:
    from ragas.metrics import ContextRelevance     # "Context Relevance" (no reference needed)
    HAS_CTX_REL = True
except ImportError:
    HAS_CTX_REL = False
    print("ContextRelevance not available in this ragas version; see the fallback in 4.7.")

from langchain_ollama import ChatOllama, OllamaEmbeddings

# The judge should be at least as capable as the generator. A stronger judge gives more reliable scores.
judge_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
```

### The RAGAS data format

```python
def to_ragas(df):
    return EvaluationDataset.from_list([
        {"user_input": r["question"],
         "retrieved_contexts": r["contexts"],
         "response": r["answer"],
         "reference": r["reference"]}
        for _, r in df.iterrows()
    ])

dataset = to_ragas(df)
```

| RAGAS field | Comes from |
|---|---|
| `user_input` | The question |
| `retrieved_contexts` | Chunks returned by Chroma |
| `response` | The generated answer |
| `reference` | Our hand-written ground-truth answer |

## 4.4 Run the evaluation

```python
metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
           LLMContextRecall(), AnswerCorrectness()]
if HAS_CTX_REL:
    metrics.append(ContextRelevance())

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_emb,
    run_config=RunConfig(timeout=300, max_workers=2),   # local models: keep concurrency low
)

report = result.to_pandas()
report.insert(0, "id", df["id"].values)
report.to_csv("outputs/lab4_ragas_report.csv", index=False)      # cache for the live session
report.round(2)
```

> **Instructor tip:** local judge models are slow and sometimes return malformed JSON, which shows up as `NaN` scores or timeouts. Run this once before the session and **load the cached CSV live**:
> `report = pd.read_csv("outputs/lab4_ragas_report.csv")`
> Then re-run only one or two questions live to show the mechanics.

```python
score_cols = [c for c in report.columns if c not in ("id", "user_input", "retrieved_contexts", "response", "reference")]
print(report[score_cols].mean().round(2))
```

---

## 4.5 What each metric measures and how to read it

| Metric | How it is computed (simplified) | High score means | Low score means |
|---|---|---|---|
| **Faithfulness** | Judge splits the answer into claims; checks each against retrieved contexts | Every claim is supported by context | Answer contains claims **not in** the context, a hallucination or outside knowledge |
| **Answer Relevance** | Generates questions from the answer and compares them with the original question | Answer addresses the question | Answer is off-topic, evasive or incomplete |
| **Context Precision** | Are the relevant chunks ranked at the top of the retrieved list? | Useful chunks come first, little noise | Relevant chunks are buried under irrelevant ones |
| **Context Recall** | What share of the **reference** answer's claims can be found in the contexts? | Retrieval fetched what was needed | Key facts were never retrieved |
| **Context Relevance** | Is each retrieved chunk relevant to the question? (no reference needed) | Retrieved text is on-topic | Retrieval brought in noise |
| **Answer Correctness** | Combines factual overlap with the reference and semantic similarity | Answer agrees with ground truth | Answer disagrees or is missing facts |

### Score ranges (heuristics, calibrate on your own data)

| Range | Interpretation | Typical action |
|---|---|---|
| **0.85 to 1.0** | Strong | Monitor |
| **0.60 to 0.85** | Acceptable but inspect failures | Read the lowest rows |
| **0.30 to 0.60** | Weak | Investigate the specific component |
| **Below 0.30** | Failing | Fix before any rollout |

> **Important caveats**
> - Judge-based metrics are themselves LLM outputs. They vary between runs and between judge models.
> - Do not compare scores across different judge models.
> - A refusal such as "I don't know based on the provided documents" on an out-of-scope question can score **low on Answer Relevance** while being the **correct behaviour**. Metrics need business context.

---

## 4.6 Read the report row by row

```python
view = report.merge(df[["id", "question", "answer", "sources"]], on="id", suffixes=("", "_df"))
cols = ["id", "question", "answer"] + score_cols
view[cols].round(2)
```

### Guided diagnosis exercise

For each row with any score below 0.6, fill in:

| id | Lowest metric | Retrieval OK? (hit/all_found) | Likely failing component | Proposed fix |
|----|---------------|-------------------------------|--------------------------|--------------|
| | | | | |

Use this decision flow:

```
Is Context Recall low?  ── yes ─►  Retrieval did not fetch the facts   ─► tune k / chunking / embeddings
          │ no
Is Faithfulness low?    ── yes ─►  Generator added unsupported claims  ─► strengthen grounding prompt, lower temperature
          │ no
Is Answer Relevance low?── yes ─►  Answer misses the question          ─► improve prompt / output format
          │ no
Is Correctness low?     ── yes ─►  Check reference quality or reasoning (e.g., arithmetic in q9)
```

---

## 4.7 Deliberate failure demonstrations

### Case 1: retrieval succeeds, generation fails

Use the loose prompt, which encourages confident short answers and does not enforce grounding. Retrieval is unchanged.

```python
from rag_app import LOOSE_PROMPT

qs = [i for i in rag_eval if i["id"] in ("q4", "q7", "q8")]
rows = []
for item in qs:
    out = rag_answer(item["question"], k=3, prompt=LOOSE_PROMPT)
    rows.append({**item, "answer": out["answer"], "contexts": out["contexts"], "sources": out["sources"]})
df_loose = pd.DataFrame(rows)

res_loose = evaluate(to_ragas(df_loose), metrics=[Faithfulness(), AnswerCorrectness()],
                     llm=judge_llm, embeddings=judge_emb, run_config=RunConfig(timeout=300, max_workers=2))
out_loose = res_loose.to_pandas(); out_loose.insert(0, "id", df_loose["id"].values)
out_loose[["id", "response", "faithfulness", "answer_correctness"]].round(2)
```

**Expected pattern:** the right documents were retrieved (hit = True), but **Faithfulness drops** because the answer adds claims, or the model invents an answer instead of refusing.

### Case 2: retrieval fails, generation is fine

Force poor retrieval (`k=1`, tiny chunks) and keep the strict grounded prompt.

```python
rows = []
for item in rag_eval:
    if item["id"] in ("q4", "q9"):
        out = rag_answer(item["question"], k=1, chunk_size=120, chunk_overlap=0)
        rows.append({**item, "answer": out["answer"], "contexts": out["contexts"], "sources": out["sources"]})
df_bad_ret = pd.DataFrame(rows)

res_bad = evaluate(to_ragas(df_bad_ret),
                   metrics=[LLMContextRecall(), Faithfulness(), AnswerCorrectness()],
                   llm=judge_llm, embeddings=judge_emb, run_config=RunConfig(timeout=300, max_workers=2))
out_bad = res_bad.to_pandas(); out_bad.insert(0, "id", df_bad_ret["id"].values)
out_bad[["id", "response", "context_recall", "faithfulness", "answer_correctness"]].round(2)
```

**Expected pattern:** **Context Recall is low** (facts were never retrieved). Faithfulness may be **high**, because the model honestly said "I don't know" or stuck to what little it had. This is the classic case where an answer is *faithful but wrong*, so you need both context and correctness metrics.

### Summary matrix

| Scenario | Context Recall | Faithfulness | Answer Correctness | Diagnosis |
|---|---|---|---|---|
| Healthy | High | High | High | Good |
| Good retrieval, bad generation | High | **Low** | Low | Fix prompt / model |
| Bad retrieval, honest generation | **Low** | High | Low | Fix retrieval |
| Bad retrieval, hallucinating generation | **Low** | **Low** | Low | Fix both; add refusal rule |

---

## 4.8 Fallback if RAGAS is unavailable or the judge is too slow

A minimal **LLM-as-judge** you can write yourself. It teaches what the metrics actually do.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

judge = ChatOllama(model="llama3.1:8b", temperature=0)

FAITH_PROMPT = ChatPromptTemplate.from_template(
"""You are a strict fact-checker.
Context:
{context}

Answer:
{answer}

List each factual claim in the answer. For each, say SUPPORTED or UNSUPPORTED by the context.
Finish with one line: SCORE = <supported claims>/<total claims>."""
)
faith_chain = FAITH_PROMPT | judge | StrOutputParser()

r = df.iloc[0]
print(faith_chain.invoke({"context": "\n\n".join(r["contexts"]), "answer": r["answer"]}))
```

Ask learners: *what biases might this judge have?* (verbosity, agreeing with fluent text, sensitivity to prompt wording)

---

## AI copilot prompts

- *"My RAGAS evaluate() call returns NaN for some rows with a local Ollama judge. List likely causes (timeouts, JSON parsing, context length) and how to diagnose each."*
- *"Explain the difference between Context Precision and Context Recall using a worked example with 3 retrieved chunks."*
- *"Write a pandas snippet that flags every row where Faithfulness > 0.8 but Answer Correctness < 0.5 and explains what that pattern means."*

## Learner challenge

Find a row where **Answer Relevance is low but the answer is actually right** (hint: look at refusals). Propose how you would change the evaluation so that correct refusals are scored as successes (for example, a separate "should refuse" flag in the dataset).

## Checkpoint

- [ ] Retrieval hit rate and MRR computed
- [ ] `outputs/lab4_ragas_report.csv` saved
- [ ] You can explain faithful-but-wrong vs unfaithful-but-right
- [ ] You completed the diagnosis table for at least three rows

**Next:** [Lab 5: Compare, diagnose and improve](lab5_compare_diagnose_wrapup.md)
