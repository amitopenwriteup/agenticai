# Lab 1: Build the Non-RAG LLM Application (Marketing Assistant)

**Section:** 2 (first half) | **Time:** about 20 minutes | **Type:** Demonstration + guided coding

## Business Context

NovaHome's marketing team wants an assistant that drafts **social captions and email subject lines** from short product briefs.

| Business objective | Implication for evaluation |
|---|---|
| Save copywriter time | Outputs must be usable with light editing |
| Stay on-brand | Tone, length and vocabulary rules must be respected |
| Never mislead customers | No invented specs, no health claims |

Write down (as a group) what "good" means **before** choosing metrics. You will revisit this list in Lab 2.

## Objectives

- Build a prompt, model and parser chain with LangChain and a local Ollama model
- Generate outputs for several representative inputs
- Create the output table that later labs will score
- Build a second, closed-answer task (intent classification) for Accuracy and Exact Match

---

## 1.1 The generation chain

```python
import json
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

marketing_eval = json.load(open("data/marketing_eval.json"))

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system",
     "You are a marketing copywriter for NovaHome Appliances. "
     "Write in a friendly, confident tone. "
     "Never exceed {max_words} words. "
     "Do not make medical or health claims. "
     "Reply with the copy only, no quotation marks and no explanations."),
    ("human", "Write a {format} for this brief:\n{brief}"),
])

def make_chain(prompt, temperature=0.7, model="llama3.2:3b"):
    llm = ChatOllama(model=model, temperature=temperature)
    return prompt | llm | StrOutputParser()

chain_v1 = make_chain(PROMPT_V1, temperature=0.7)
```

**Explain while coding:**

| Component | Role |
|---|---|
| `ChatPromptTemplate` | Turns structured inputs into a prompt, so changes to the prompt are versionable |
| `ChatOllama` | Local model; `temperature` controls randomness |
| `StrOutputParser` | Converts the model message into a plain string |
| `|` (pipe) | LangChain Expression Language: chains the components |

## 1.2 Generate outputs

```python
rows = []
for item in marketing_eval:
    out = chain_v1.invoke({
        "format": item["format"], "max_words": item["max_words"], "brief": item["brief"]
    }).strip()
    rows.append({**item, "prediction": out})

df_v1 = pd.DataFrame(rows)
df_v1["pred_words"] = df_v1["prediction"].str.split().str.len()
df_v1.to_csv("outputs/lab1_marketing_v1.csv", index=False)

for r in rows:
    print(f"[{r['id']}] {r['format']}")
    print("  REF :", r["reference"])
    print("  PRED:", r["prediction"], "\n")
```

## 1.3 Read the outputs before computing any metric

Fill in this table by hand. This is **human evaluation**, the ground truth that automated metrics try to approximate.

| id | On-brand? | Within word limit? | Any invented facts? | Would you publish it? |
|----|-----------|-------------------|---------------------|-----------------------|
| m1 | | | | |
| m2 | | | | |
| m3 | | | | |
| m4 | | | | |
| m5 | | | | |

Quick automated check on a rule you *can* test:

```python
df_v1["within_limit"] = df_v1["pred_words"] <= df_v1["max_words"]
df_v1["says_cheap"] = df_v1["prediction"].str.contains("cheap", case=False)
df_v1["health_claim"] = df_v1["prediction"].str.contains(r"\b(cure|treat|prevent|allerg|flu|asthma)", case=False, regex=True)
print(df_v1[["id", "pred_words", "within_limit", "says_cheap", "health_claim"]])
```

> **Teaching point:** some of the most important quality criteria (word limit, banned words, forbidden claims) are **rule-based checks**, not similarity metrics. Real evaluation suites mix both.

---

## 1.4 A closed-answer task: intent classification

Customer messages must be routed to the right team. Here there is exactly one right answer, so **Accuracy** and **Exact Match** are appropriate.

```python
intent_eval = json.load(open("data/intent_eval.json"))
LABELS = ["product_info", "returns", "shipping", "warranty"]

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the customer message into exactly one of: product_info, returns, shipping, warranty. "
     "Reply with the label only, in lowercase, with no punctuation."),
    ("human", "{text}"),
])
intent_chain = make_chain(INTENT_PROMPT, temperature=0.0)

rows = []
for item in intent_eval:
    pred = intent_chain.invoke({"text": item["text"]}).strip()
    rows.append({**item, "prediction": pred})

df_intent = pd.DataFrame(rows)
df_intent.to_csv("outputs/lab1_intent.csv", index=False)
df_intent
```

Look at any mismatches. Is the model wrong, or is the output **format** wrong (for example `"Returns."` instead of `returns`)? Evaluation must separate these two failure types, since they are fixed in different ways.

---

## 1.5 Discussion

1. Which customer messages are genuinely ambiguous? Could a human labeller disagree?
2. Why can we use Exact Match for intents but not for captions?
3. If the model returns `"The label is returns"`, did it fail the task? Which metric would say yes, and which would say no?

---

## AI copilot prompts

- *"Explain this LangChain chain `prompt | llm | StrOutputParser()` line by line and show the equivalent without the pipe operator."*
- *"Suggest 5 additional test cases for a marketing-caption generator that would expose brand-guideline violations."*

## Learner challenge

Add a sixth brief that is deliberately tricky (for example, one that tempts the model to make a health claim). Predict the result **before** you run it.

## Checkpoint

- [ ] `outputs/lab1_marketing_v1.csv` exists
- [ ] `outputs/lab1_intent.csv` exists
- [ ] You wrote your own "what does good look like" list for the business

**Next:** [Lab 2: Score with traditional NLP metrics](lab2_nlp_metrics.md)
