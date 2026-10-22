# Lab 5: Compare, Diagnose, Improve and Wrap Up

**Section:** 4 (final part) and 5 | **Time:** about 15 minutes of the Section 4 block, then 10 minutes of conclusion and Q&A | **Type:** Demonstration + discussion

## Objectives

- Compare the standalone LLM and the RAG application on the same questions
- Use a diagnosis playbook to trace failures to retrieval, context, prompt or generation
- Run one improvement loop: change a setting, re-evaluate, record the result
- Consolidate how metrics map to business objectives

---

## 5.1 Standalone LLM vs RAG on the same questions

Faithfulness and the context metrics need retrieved contexts, so the standalone model can only be compared on **answer-level** metrics: Answer Correctness (RAGAS) and BERTScore.

```python
import json, pandas as pd
import evaluate
from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AnswerCorrectness
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rag_app import rag_answer, plain_answer

rag_eval = json.load(open("data/rag_eval.json"))
judge_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
bertscore = evaluate.load("bertscore")

def run_system(name, answer_fn):
    rows = []
    for item in rag_eval:
        out = answer_fn(item["question"])
        rows.append({"id": item["id"], "question": item["question"], "reference": item["reference"],
                     "answer": out["answer"], "contexts": out["contexts"]})
    return pd.DataFrame(rows).assign(system=name)

df_plain = run_system("standalone", lambda q: {"answer": plain_answer(q), "contexts": []})
df_rag   = run_system("rag",        lambda q: rag_answer(q, k=3))

def score_answers(df):
    ds = EvaluationDataset.from_list([
        {"user_input": r.question, "retrieved_contexts": r.contexts,
         "response": r.answer, "reference": r.reference} for r in df.itertuples()])
    res = ragas_evaluate(ds, metrics=[AnswerCorrectness()], llm=judge_llm, embeddings=judge_emb,
                         run_config=RunConfig(timeout=300, max_workers=2)).to_pandas()
    bs = bertscore.compute(predictions=df["answer"].tolist(), references=df["reference"].tolist(),
                           lang="en", model_type="distilbert-base-uncased")["f1"]
    return pd.DataFrame({"id": df["id"].values,
                         "answer_correctness": res["answer_correctness"].values,
                         "bertscore_f1": bs})

cmp = score_answers(df_plain).merge(score_answers(df_rag), on="id", suffixes=("_standalone", "_rag"))
cmp.to_csv("outputs/lab5_standalone_vs_rag.csv", index=False)
print(cmp.round(2))
print(cmp.drop(columns="id").mean().round(2))
```

> **Instructor tip:** cache this CSV before the session, as with Lab 4.

### What to expect and discuss

| Observation | Explanation |
|---|---|
| RAG wins clearly on fact lookups (q1, q2, q5, q6) | The standalone model cannot know fictional specs, so it guesses |
| BERTScore gap is much smaller than the Answer Correctness gap | Embeddings reward fluent, same-topic answers even when numbers are wrong |
| q7 and q8 may be close | Refusing a bad request depends on the prompt and rules, not just retrieval |
| q9 may fail in both systems | Multi-document retrieval plus arithmetic is hard for small models |

**Key lesson:** the metric you choose decides whether you notice the improvement. BERTScore alone would have understated the value of RAG.

---

## 5.2 Diagnosis playbook

Use this table whenever an answer is wrong.

| Symptom (scores) | Likely root cause | How to confirm | Typical fix |
|---|---|---|---|
| Context Recall low, Hit rate low | Retrieval missed the document | Print `sources` and `contexts` | Increase `k`, improve chunking, try a better embedding model, add metadata filters |
| Context Recall low, Hit rate high | Right file but wrong chunk | Read the retrieved chunks | Adjust chunk size and overlap, split by section headings |
| Context Precision low | Relevant chunk ranked below noise | Check rank order | Reranking, smaller `k`, hybrid (keyword + vector) search |
| Context Relevance low | Retrieval returns off-topic text | Inspect distances and chunks | Better query rewriting, cleaner documents |
| Faithfulness low, contexts fine | Generator added unsupported claims | Compare answer to context line by line | Strengthen grounding instruction, lower temperature, use a stronger model |
| Faithfulness high, Correctness low | Faithful to **wrong or incomplete** context | Check Context Recall | Fix retrieval, not the prompt |
| Answer Relevance low | Answer drifts from the question or refuses wrongly | Read the answer | Clarify prompt, add examples of desired format |
| Everything high but humans disagree | Reference or metric blind spot | Spot-check by hand | Improve the reference set, add rule checks or human review |
| Scores vary run to run | Judge or generator randomness | Re-run three times | Set temperature 0, use a stronger judge, report mean and spread |

### Practice cases

Classify each case using the table above.

1. Context Recall 0.2, Faithfulness 0.9, Correctness 0.2.
2. Context Recall 0.9, Faithfulness 0.3, Correctness 0.3.
3. Context Precision 0.3, Context Recall 0.9, Faithfulness 0.9, Correctness 0.9.
4. Answer Relevance 0.1 on a question where the model correctly said "I don't know".

<details>
<summary>Answers</summary>

1. Retrieval failure with an honest generator. Fix retrieval.
2. Good retrieval but the generator is unfaithful. Fix the prompt, temperature or model.
3. Answers are fine, but ranking is noisy. Tune reranking or `k` for cost and latency.
4. A metric artefact. Score refusals separately with a "should refuse" flag.

</details>

---

## 5.3 One improvement loop

Change **one thing at a time**, re-measure, and keep a log. Here we test chunk size and `k`.

```python
from ragas.metrics import Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference

def evaluate_config(label, **kwargs):
    df = run_system(label, lambda q: rag_answer(q, **kwargs))
    ds = EvaluationDataset.from_list([
        {"user_input": r.question, "retrieved_contexts": r.contexts,
         "response": r.answer, "reference": r.reference} for r in df.itertuples()])
    res = ragas_evaluate(ds,
        metrics=[Faithfulness(), LLMContextRecall(), LLMContextPrecisionWithReference(), AnswerCorrectness()],
        llm=judge_llm, embeddings=judge_emb,
        run_config=RunConfig(timeout=300, max_workers=2)).to_pandas()
    cols = ["faithfulness", "context_recall", "llm_context_precision_with_reference", "answer_correctness"]
    return res[[c for c in cols if c in res.columns]].mean().rename(label)

experiments = {
    "baseline  k=3 chunk=350": dict(k=3, chunk_size=350, chunk_overlap=50),
    "small chunks k=3 chunk=150": dict(k=3, chunk_size=150, chunk_overlap=20),
    "more context k=5 chunk=350": dict(k=5, chunk_size=350, chunk_overlap=50),
    "tiny k=1 chunk=350": dict(k=1, chunk_size=350, chunk_overlap=50),
}
results = pd.concat([evaluate_config(n, **kw) for n, kw in experiments.items()], axis=1).T
results.to_csv("outputs/lab5_experiments.csv")
results.round(2)
```

> Column names for RAGAS results can differ between versions. If a column is missing, print `res.columns` and adjust.

### Experiment log (fill in as a group)

| Run | Change | Context Recall | Faithfulness | Correctness | Decision |
|---|---|---|---|---|---|
| 1 | Baseline | | | | |
| 2 | Chunk 150 | | | | |
| 3 | k = 5 | | | | |
| 4 | k = 1 | | | | |

**Questions**

1. Did increasing `k` improve recall? Did it hurt precision or faithfulness?
2. What does each additional chunk cost in latency and tokens?
3. With only 9 questions, which differences would you trust? What would you do to trust more?

### Beyond this lab

| Improvement | Metric most likely to move |
|---|---|
| Add a reranker | Context Precision |
| Rewrite queries before retrieval | Context Recall |
| Add citations in the prompt | Faithfulness (easier to audit) |
| Add a "should refuse" test set | Safety and refusal behaviour |
| Use a stronger generator | Faithfulness, Correctness |
| Grow the test set to hundreds of questions | Reliability of every comparison |

---

## 5.4 From metrics to business outcomes

| Business question | Evidence to show |
|---|---|
| Can we let the assistant answer customers unsupervised? | Faithfulness and Correctness on policy and price questions; refusal behaviour on out-of-scope ones |
| Is the marketing copy safe to publish? | Rule-check pass rate plus human review, with ROUGE/BERTScore as regression signals |
| Is RAG worth the extra cost and latency? | Correctness gain vs standalone, against the added tokens and response time |
| What should we fix next? | The lowest-scoring component in the diagnosis playbook |

Metrics are a **monitoring system**, not a one-off grade. Re-run the suite whenever the model, prompt, documents or chunking change.

---

## 5.5 Conclusion (Section 5, 10 minutes)

### Workflow recap

```
Define business goal ─► Build reference + rule checks ─► Choose metrics for the architecture
        ▲                                                          │
        │                                                          ▼
Iterate on prompt / retrieval / data  ◄── Diagnose failing component ◄── Run evaluation + read rows
```

### Standalone LLM vs RAG evaluation

| | Standalone LLM | RAG system |
|---|---|---|
| Main risk | Hallucinated facts, style drift | Retrieval failure, unfaithful generation, noisy context |
| Core metrics | BLEU, ROUGE, METEOR, BERTScore, Accuracy/EM, rule checks | Faithfulness, Answer Relevance, Context Precision/Recall/Relevance, Answer Correctness, plus hit rate and MRR |
| Debug question | "Is the output good?" | "Is the **retrieval** good, the **context** good, or the **generation** good?" |

### Five takeaways

1. No single metric is enough. Combine similarity, correctness, grounding and rule checks.
2. Pick metrics from the **business objective** and the **architecture**, not from habit.
3. Similarity metrics cannot detect wrong facts. Grounding metrics can.
4. Always inspect rows, not just averages.
5. Evaluation is how a prototype becomes a production system: change one thing, re-measure, record.

### Exit ticket (answer individually)

1. Name one situation where BLEU is misleading and what you would use instead.
2. A RAG answer has Faithfulness 0.95 and Correctness 0.2. Where do you look first?
3. Which metric would you add for a legal-compliance assistant, and why?

### Open Q&A prompts

- How large should an evaluation set be before we trust it?
- Can the judge model be the same as the generator?
- How do we evaluate multi-turn conversations?

---

## AI copilot prompts

- *"Given this table of RAGAS results, which component should I fix first and why? `<paste table>`."*
- *"Design a 30-question evaluation set for a returns-policy assistant, including at least five unanswerable and five adversarial questions."*
- *"How can I estimate whether a 0.04 difference in Faithfulness between two configurations is noise?"*

## Final learner deliverable (optional take-home)

Submit a one-page evaluation report for the NovaHome assistant containing:

1. The business goal and the metrics you selected, with reasons
2. A results table for standalone vs RAG
3. Two failures, each traced to a component with evidence
4. One improvement you tested and its measured effect
5. One risk that your metrics do **not** cover

## Checkpoint

- [ ] Standalone vs RAG comparison table produced
- [ ] You can classify the four practice cases
- [ ] You completed at least one experiment and logged the result
- [ ] You can state which metric answers which question

**Back to the** [workshop README](README.md)
