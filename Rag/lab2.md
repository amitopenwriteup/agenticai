# Lab 2: Scoring with Traditional NLP Metrics

**Section:** 2 (second half) | **Time:** about 35 minutes | **Type:** Demonstration + guided coding

## Objectives

- Compute Accuracy, Exact Match, BLEU, ROUGE, METEOR and BERTScore
- Interpret what each score does and does not tell you
- See cases where some metrics look good while others reveal problems
- Use results to improve the prompt and generation settings, then re-measure

---

## 2.1 Load the metrics

```python
import evaluate, nltk
import numpy as np
import pandas as pd

for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

bleu       = evaluate.load("bleu")
rouge      = evaluate.load("rouge")
meteor     = evaluate.load("meteor")
bertscore  = evaluate.load("bertscore")
exact      = evaluate.load("exact_match")
```

> The first load downloads files. This is why we do it before the session.

## 2.2 Closed-answer metrics: Accuracy and Exact Match

```python
df_intent = pd.read_csv("outputs/lab1_intent.csv")

def normalise(s):
    return str(s).strip().lower().strip(".!\"' ")

preds = [p for p in df_intent["prediction"]]
refs  = [l for l in df_intent["label"]]

# Strict Exact Match on the raw output
em_raw = exact.compute(predictions=preds, references=refs)["exact_match"]

# Exact Match after light normalisation (case, punctuation, whitespace)
em_norm = exact.compute(predictions=[normalise(p) for p in preds], references=refs)["exact_match"]

# Accuracy = share of correct labels after normalisation
accuracy = np.mean([normalise(p) == r for p, r in zip(preds, refs)])

print(f"Exact match (raw):        {em_raw:.2f}")
print(f"Exact match (normalised): {em_norm:.2f}")
print(f"Accuracy:                 {accuracy:.2f}")
df_intent.assign(correct=[normalise(p) == r for p, r in zip(preds, refs)])
```

**Interpretation**

| Observation | What it means |
|---|---|
| Raw EM much lower than normalised EM | The model understands the task but fails the **output format**, a prompt or parser problem |
| Normalised EM equals Accuracy here | Both measure exact label match; Accuracy is the usual term for classification |
| Both low | Real classification errors; inspect the confusion (for example warranty vs returns) |

For a fuller picture on classification, add per-class Precision/Recall (for example `sklearn.metrics.classification_report`).

---

## 2.3 Open-ended generation metrics

```python
df = pd.read_csv("outputs/lab1_marketing_v1.csv")
preds = df["prediction"].tolist()
refs  = df["reference"].tolist()

def score_generation(preds, refs):
    """Return a per-sample DataFrame of all generation metrics."""
    rows = []
    # BERTScore in one batch (loading the model per call is slow)
    bs = bertscore.compute(predictions=preds, references=refs,
                           lang="en", model_type="distilbert-base-uncased")
    for i, (p, r) in enumerate(zip(preds, refs)):
        b = bleu.compute(predictions=[p], references=[[r]], smooth=True)["bleu"]
        rg = rouge.compute(predictions=[p], references=[r])
        m = meteor.compute(predictions=[p], references=[r])["meteor"]
        rows.append({
            "bleu": b,
            "rouge1": rg["rouge1"], "rouge2": rg["rouge2"], "rougeL": rg["rougeL"],
            "meteor": m,
            "bertscore_f1": bs["f1"][i],
        })
    return pd.DataFrame(rows)

scores_v1 = score_generation(preds, refs)
report_v1 = pd.concat([df[["id", "prediction"]], scores_v1], axis=1)
report_v1.round(3)
```

```python
print(scores_v1.mean().round(3))
```

### How to read each score

| Metric | What it measures | Rough reading for short marketing copy* |
|---|---|---|
| **BLEU** | Precision of overlapping n-grams (1 to 4) with a brevity penalty | Below 0.2 is common even for good paraphrases |
| **ROUGE-1/2** | Recall of unigrams/bigrams from the reference | ROUGE-1 above about 0.5 means substantial word reuse |
| **ROUGE-L** | Longest common subsequence | Captures word order better than ROUGE-1 |
| **METEOR** | Overlap with stemming, synonyms and word-order penalty | More forgiving than BLEU |
| **BERTScore (F1)** | Cosine similarity of contextual embeddings | Often 0.85 to 0.95 for same-topic text, so look at *differences*, not absolute values |

\*These are heuristics for this tiny dataset, not universal thresholds. Always compare against a baseline and against human judgement.

> **Why BERTScore looks "high" everywhere:** embeddings of any same-domain English text are similar. Use `rescale_with_baseline=True` (with `lang="en"`) to spread scores out, or interpret the score only relative to other systems.

---

## 2.4 When metrics disagree: the demonstration set

Run these hand-written candidates against one reference to see what each metric rewards.

```python
reference = "Meet the NovaBrew Smart Kettle: control it from your phone and keep your tea warm for 2 hours."

candidates = {
  "A. Near copy":
     "Meet the NovaBrew Smart Kettle: control it from your phone and keep your tea warm for 2 hours!",
  "B. Good paraphrase":
     "Run the NovaBrew from your phone and enjoy tea that stays hot for two hours.",
  "C. Fluent but WRONG fact":
     "Meet the NovaBrew Smart Kettle: control it from your phone and keep your tea warm for 8 hours.",
  "D. Off-topic":
     "Discover the NovaChef Air Fryer with eight handy presets for quick dinners.",
}

demo = score_generation(list(candidates.values()), [reference] * len(candidates))
demo.insert(0, "candidate", list(candidates.keys()))
demo.round(3)
```

**What learners should notice**

| Candidate | Expected pattern | Lesson |
|---|---|---|
| A | All metrics very high | Metrics reward copying |
| B | BLEU/ROUGE low, METEOR/BERTScore higher | **Good output, penalised by lexical metrics** |
| C | All metrics still high | **Metrics cannot detect a wrong fact** (2 hours vs 8 hours), which is why faithfulness and correctness metrics exist |
| D | BERTScore still fairly high, others low | Same domain and style fools embedding metrics |

> **Key message:** BLEU/ROUGE/METEOR/BERTScore measure *similarity to a reference*, not *truth*, *brand compliance* or *usefulness*. Use them as regression signals, not as quality guarantees.

### Limitations of traditional metrics for open-ended generation

1. Many valid outputs exist, but a reference covers one phrasing.
2. They cannot verify facts.
3. They ignore style rules, safety and tone.
4. Short texts make n-gram metrics noisy (BLEU can be 0 with no 4-gram match).
5. They need a human-written reference, which is expensive.

---

## 2.5 Use the results: improve the prompt and settings

Hypotheses from our reading of the outputs:

- Temperature 0.7 causes variation, so try 0.2.
- The model ignores the style of our approved copy, so add **few-shot examples**.

```python
from langchain_core.prompts import ChatPromptTemplate
import json

marketing_eval = json.load(open("data/marketing_eval.json"))

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system",
     "You are a marketing copywriter for NovaHome Appliances. "
     "Friendly, confident, simple tone. Never exceed {max_words} words. "
     "Mention the exact product name and one or two concrete specs from the brief. "
     "Do NOT invent specs or make health claims. Reply with the copy only.\n\n"
     "Examples of approved copy:\n"
     "- Dinner made easy: the NovaChef 5L Air Fryer packs 8 one-touch presets and 1700 W of power.\n"
     "- Launch week: save 15% on the NovaChef 5L Air Fryer"),
    ("human", "Write a {format} for this brief:\n{brief}"),
])

def generate(prompt, temperature):
    chain = make_chain(prompt, temperature=temperature)   # defined in Lab 1
    return [chain.invoke({"format": i["format"], "max_words": i["max_words"], "brief": i["brief"]}).strip()
            for i in marketing_eval]

preds_v2 = generate(PROMPT_V2, temperature=0.2)
scores_v2 = score_generation(preds_v2, refs)
```

> The few-shot examples use briefs m3 and m4 from the evaluation set, so scores on those items will be **inflated** (data leakage). Compare on m1, m2 and m5 only, or use separate examples in real projects.

```python
cmp = pd.DataFrame({"v1 (T=0.7, zero-shot)": scores_v1.mean(),
                    "v2 (T=0.2, few-shot)":  scores_v2.mean()}).round(3)
cmp["delta"] = (cmp.iloc[:, 1] - cmp.iloc[:, 0]).round(3)
cmp
```

```python
# Honest comparison: exclude the leaked items (indices 2 and 3 = m3, m4)
keep = [0, 1, 4]
pd.DataFrame({"v1": scores_v1.iloc[keep].mean(), "v2": scores_v2.iloc[keep].mean()}).round(3)
```

Re-run the rule checks from Lab 1 on `preds_v2` (word limit, banned words, health claims). **A prompt change can raise similarity scores while breaking a rule**, so always check both.

---

## 2.6 Discussion

1. Candidate C scored almost as well as A. What would you add to the pipeline to catch it?
2. Which single metric would you put on a dashboard for the marketing team, and why do you still need human review?
3. With only 5 examples, how much should we trust a 0.03 improvement?

<details>
<summary>Suggested answers</summary>

1. A fact-checking step against source data (in Part 2 of the session: faithfulness / answer correctness against retrieved context), or an LLM-as-judge with the brief as evidence.
2. Probably BERTScore or ROUGE-L as a regression signal, plus rule-based compliance checks; human review catches tone and persuasiveness.
3. Very little. Small samples and stochastic generation make small deltas noise. Use more items and repeated runs.

</details>

---

## AI copilot prompts

- *"My BLEU score is 0.0 for a short sentence even though it looks very similar to the reference. Explain why and how smoothing helps."*
- *"Explain the difference between BERTScore precision, recall and F1 with a worked example."*
- *"Write a function that runs each prompt variant 5 times and reports the mean and standard deviation of ROUGE-L."*

## Learner challenge

Write one new candidate that scores **higher on BLEU than candidate B but is worse marketing copy**. What does that tell you about optimising for a metric?

## Checkpoint

- [ ] You have a per-sample score table and a mean-score table for v1
- [ ] You can explain why candidate C is dangerous
- [ ] You compared v1 and v2 and checked rule compliance

**Take the 10-minute break here**, then continue to [Lab 3: Build the RAG pipeline](lab3_rag_pipeline.md)
