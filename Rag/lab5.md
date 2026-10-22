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

## How this lab works

Each step creates a Python file with a `tee` command, then runs it. Run everything from inside `novahome_eval/` with your virtual environment active and Ollama running. This lab needs `rag_app.py` and `data/rag_eval.json` from Labs 0 and 3, plus the `llama3.1:8b` judge model (see Lab 0).

> **Running in Jupyter?** Put `%%bash` on the first line of a cell for the `tee` blocks, and use `!python script_name.py` to run a script. The `'EOF'` quotes stop the shell from expanding characters inside the code, and `> /dev/null` hides the echo that `tee` prints. On Windows, use WSL or Git Bash.

| File created | Step | Purpose |
|---|---|---|
| `lab4_01_generate_and_retrieval_check.py` | 4.1, 4.2 | Generate answers, then compute hit rate and MRR (no judge LLM) |
| `lab4_common.py` | 4.3 | RAGAS setup shared by later scripts |
| `lab4_02_ragas_eval.py` | 4.4 | Run RAGAS metrics and save a cached report |
| `lab4_03_read_report.py` | 4.6 | Read the report row by row and flag weak scores |
| `lab4_04_failure_cases.py` | 4.7 | Two deliberate failure demonstrations |
| `lab4_05_fallback_judge.py` | 4.8 | A hand-written LLM-as-judge fallback |

---

## 4.1 Generate answers and contexts, and 4.2 retrieval sanity check

Because we know which document should contain each answer, we can compute a simple **hit rate** and **reciprocal rank**. These are fast, free and fully deterministic, so run them **before** any LLM-judged metric.

```bash
tee lab4_01_generate_and_retrieval_check.py > /dev/null << 'EOF'
import json
import pandas as pd
from rag_app import rag_answer

rag_eval = json.load(open("data/rag_eval.json"))

# 4.1 Generate answers and contexts for all questions
records = []
for item in rag_eval:
    out = rag_answer(item["question"], k=3)
    records.append({**item, "answer": out["answer"], "contexts": out["contexts"], "sources": out["sources"]})
df = pd.DataFrame(records)


# 4.2 Retrieval sanity check (no judge LLM needed)
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
df.to_pickle("outputs/lab4_rag_run.pkl")
df[["id", "question", "answer", "sources", "hit", "all_found", "rr"]].to_csv(
    "outputs/lab4_retrieval_check.csv", index=False)

for _, r in df.iterrows():
    print(f"[{r['id']}] {r['question']}")
    print("  ANSWER :", r["answer"])
    print("  SOURCES:", r["sources"], "| hit:", r["hit"], "| all_found:", r["all_found"], "| rr:", r["rr"], "\n")

print("Hit rate@3:", round(df["hit"].dropna().astype(float).mean(), 2))
print("All-sources-found rate:", round(df["all_found"].dropna().astype(float).mean(), 2))
print("MRR:", round(df["rr"].dropna().astype(float).mean(), 2))
EOF
```

```bash
python lab4_01_generate_and_retrieval_check.py
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

```bash
tee lab4_common.py > /dev/null << 'EOF'
from ragas import EvaluationDataset
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
    print("ContextRelevance not available in this ragas version; see the fallback in 4.8.")

from langchain_ollama import ChatOllama, OllamaEmbeddings

# The judge should be at least as capable as the generator. A stronger judge gives more reliable scores.
judge_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

# Local models: keep concurrency low
RUN_CONFIG = RunConfig(timeout=300, max_workers=2)

NON_SCORE_COLS = ("id", "user_input", "retrieved_contexts", "response", "reference")


def to_ragas(df):
    return EvaluationDataset.from_list([
        {"user_input": r["question"],
         "retrieved_contexts": r["contexts"],
         "response": r["answer"],
         "reference": r["reference"]}
        for _, r in df.iterrows()
    ])


def all_metrics():
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
               LLMContextRecall(), AnswerCorrectness()]
    if HAS_CTX_REL:
        metrics.append(ContextRelevance())
    return metrics
EOF
```

### The RAGAS data format

| RAGAS field | Comes from |
|---|---|
| `user_input` | The question |
| `retrieved_contexts` | Chunks returned by Chroma |
| `response` | The generated answer |
| `reference` | Our hand-written ground-truth answer |

## 4.4 Run the evaluation

```bash
tee lab4_02_ragas_eval.py > /dev/null << 'EOF'
import sys
import pandas as pd
from ragas import evaluate
from lab4_common import (to_ragas, all_metrics, judge_llm, judge_emb, RUN_CONFIG, NON_SCORE_COLS)

REPORT = "outputs/lab4_ragas_report.csv"

if "--cached" in sys.argv:
    report = pd.read_csv(REPORT)
    print("Loaded cached report:", REPORT)
else:
    df = pd.read_pickle("outputs/lab4_rag_run.pkl")
    result = evaluate(dataset=to_ragas(df), metrics=all_metrics(),
                      llm=judge_llm, embeddings=judge_emb, run_config=RUN_CONFIG)
    report = result.to_pandas()
    report.insert(0, "id", df["id"].values)
    report.to_csv(REPORT, index=False)          # cache for the live session
    print("Saved", REPORT)

score_cols = [c for c in report.columns if c not in NON_SCORE_COLS]
print("\nPer-question scores:")
print(report[["id"] + score_cols].round(2).to_string(index=False))
print("\nMean scores:")
print(report[score_cols].mean().round(2))
EOF
```

```bash
python lab4_02_ragas_eval.py
```

> **Instructor tip:** local judge models are slow and sometimes return malformed JSON, which shows up as `NaN` scores or timeouts. Run the command above once before the session, then **load the cached CSV live**:
>
> ```bash
> python lab4_02_ragas_eval.py --cached
> ```
>
> Then re-run only one or two questions live to show the mechanics.

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

```bash
tee lab4_03_read_report.py > /dev/null << 'EOF'
import pandas as pd

NON_SCORE_COLS = ("id", "user_input", "retrieved_contexts", "response", "reference")

report = pd.read_csv("outputs/lab4_ragas_report.csv")
df = pd.read_pickle("outputs/lab4_rag_run.pkl")

score_cols = [c for c in report.columns if c not in NON_SCORE_COLS]
view = report.merge(df[["id", "question", "answer", "sources", "hit", "all_found"]], on="id")

for _, r in view.iterrows():
    low = [c for c in score_cols if pd.notna(r[c]) and r[c] < 0.6]
    print(f"[{r['id']}] {r['question']}")
    print("  ANSWER   :", str(r["answer"])[:160])
    print("  RETRIEVAL: hit =", r["hit"], "| all_found =", r["all_found"], "| sources =", r["sources"])
    print("  SCORES   :", {c: round(float(r[c]), 2) for c in score_cols if pd.notna(r[c])})
    print("  NEEDS ATTENTION (<0.6):", low if low else "none", "\n")

view.to_csv("outputs/lab4_report_with_answers.csv", index=False)
EOF
```

```bash
python lab4_03_read_report.py
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

One script runs both cases.

```bash
tee lab4_04_failure_cases.py > /dev/null << 'EOF'
import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerCorrectness, LLMContextRecall
from rag_app import rag_answer, LOOSE_PROMPT
from lab4_common import to_ragas, judge_llm, judge_emb, RUN_CONFIG

rag_eval = json.load(open("data/rag_eval.json"))
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)


def build(ids, **kwargs):
    rows = []
    for item in rag_eval:
        if item["id"] in ids:
            out = rag_answer(item["question"], **kwargs)
            rows.append({**item, "answer": out["answer"], "contexts": out["contexts"], "sources": out["sources"]})
    return pd.DataFrame(rows)


# Case 1: retrieval succeeds, generation fails (loose prompt, same retrieval)
df_loose = build(("q4", "q7", "q8"), k=3, prompt=LOOSE_PROMPT)
res = evaluate(to_ragas(df_loose), metrics=[Faithfulness(), AnswerCorrectness()],
               llm=judge_llm, embeddings=judge_emb, run_config=RUN_CONFIG).to_pandas()
res.insert(0, "id", df_loose["id"].values)
print("=== Case 1: good retrieval, loose prompt ===")
print(res[["id", "response", "faithfulness", "answer_correctness"]].round(2).to_string(index=False))
res.to_csv("outputs/lab4_case1_loose_prompt.csv", index=False)

# Case 2: retrieval fails, generation is fine (k=1, tiny chunks, strict prompt)
df_bad = build(("q4", "q9"), k=1, chunk_size=120, chunk_overlap=0)
res2 = evaluate(to_ragas(df_bad), metrics=[LLMContextRecall(), Faithfulness(), AnswerCorrectness()],
                llm=judge_llm, embeddings=judge_emb, run_config=RUN_CONFIG).to_pandas()
res2.insert(0, "id", df_bad["id"].values)
print("\n=== Case 2: poor retrieval, strict prompt ===")
print(res2[["id", "response", "context_recall", "faithfulness", "answer_correctness"]].round(2).to_string(index=False))
res2.to_csv("outputs/lab4_case2_bad_retrieval.csv", index=False)
EOF
```

```bash
python lab4_04_failure_cases.py
```

**Case 1 expected pattern:** the right documents were retrieved (hit = True), but **Faithfulness drops** because the answer adds claims, or the model invents an answer instead of refusing.

**Case 2 expected pattern:** **Context Recall is low** (facts were never retrieved). Faithfulness may be **high**, because the model honestly said "I don't know" or stuck to what little it had. This is the classic case where an answer is *faithful but wrong*, so you need both context and correctness metrics.

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

```bash
tee lab4_05_fallback_judge.py > /dev/null << 'EOF'
import pandas as pd
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

df = pd.read_pickle("outputs/lab4_rag_run.pkl")
for idx in (0, 2):          # q1 (simple fact) and q3 (policy)
    r = df.iloc[idx]
    print(f"[{r['id']}] {r['question']}")
    print("ANSWER:", r["answer"])
    print(faith_chain.invoke({"context": "\n\n".join(r["contexts"]), "answer": r["answer"]}), "\n")
EOF
```

```bash
python lab4_05_fallback_judge.py
```

Ask learners: *what biases might this judge have?* (verbosity, agreeing with fluent text, sensitivity to prompt wording)

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `FileNotFoundError: outputs/lab4_rag_run.pkl` | Run `lab4_01_generate_and_retrieval_check.py` first |
| `ModuleNotFoundError: ... langchain_community.chat_models.vertexai` on `from ragas import ...` | `langchain-community` is newer than your RAGAS expects. Run `pip install "langchain-community<0.4.2"` (try `<0.4` if it persists), then re-run |
| `ImportError` from `ragas` or `ragas.metrics` | Your RAGAS version differs; check `pip show ragas` and adapt `lab4_common.py` |
| `NaN` scores or timeouts | Local judge is slow or returned malformed JSON; lower `max_workers`, raise `timeout`, use a stronger judge, or load the cached report with `--cached` |
| `KeyError: 'context_recall'` or similar | Column names differ by version; print `res2.columns` and adjust |
| `model "llama3.1:8b" not found` | Run `ollama pull llama3.1:8b` |
| `SyntaxError` right after pasting | The closing `EOF` must be alone on its line with no leading spaces |

## AI copilot prompts

- *"My RAGAS evaluate() call returns NaN for some rows with a local Ollama judge. List likely causes (timeouts, JSON parsing, context length) and how to diagnose each."*
- *"Explain the difference between Context Precision and Context Recall using a worked example with 3 retrieved chunks."*
- *"Write a pandas snippet that flags every row where Faithfulness > 0.8 but Answer Correctness < 0.5 and explains what that pattern means."*

## Learner challenge

Find a row where **Answer Relevance is low but the answer is actually right** (hint: look at refusals in `outputs/lab4_report_with_answers.csv`). Propose how you would change the evaluation so that correct refusals are scored as successes (for example, a separate "should refuse" flag in `data/rag_eval.json`).

## Checkpoint

- [ ] Retrieval hit rate and MRR computed (`outputs/lab4_retrieval_check.csv`)
- [ ] `outputs/lab4_ragas_report.csv` saved
- [ ] You can explain faithful-but-wrong vs unfaithful-but-right
- [ ] You completed the diagnosis table for at least three rows

**Next:** [Lab 5: Compare, diagnose and improve](lab5_compare_diagnose_wrapup.md)
