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

## How this lab works

Each step creates a small Python file with a `tee` command, then runs it. Run everything from inside `novahome_eval/` with your virtual environment active, and make sure Lab 0 created `data/marketing_eval.json` and `data/intent_eval.json`.

> **Running in Jupyter?** Put `%%bash` on the first line of a cell for the `tee` blocks, and use `!python script_name.py` to run a script. The `'EOF'` quotes stop the shell from expanding characters inside the code, and `> /dev/null` hides the echo that `tee` prints. On Windows, use WSL or Git Bash.

| File created | Purpose |
|---|---|
| `lab1_common.py` | Shared prompts and the `make_chain()` helper (reused in Lab 2) |
| `lab1_generate_marketing.py` | Generates captions and subject lines |
| `lab1_rule_checks.py` | Rule-based checks (word limit, banned words, claims) |
| `lab1_intent.py` | Intent classification for Accuracy / Exact Match |

---

## 1.1 The generation chain

```bash
tee lab1_common.py > /dev/null << 'EOF'
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system",
     "You are a marketing copywriter for NovaHome Appliances. "
     "Write in a friendly, confident tone. "
     "Never exceed {max_words} words. "
     "Do not make medical or health claims. "
     "Reply with the copy only, no quotation marks and no explanations."),
    ("human", "Write a {format} for this brief:\n{brief}"),
])

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the customer message into exactly one of: product_info, returns, shipping, warranty. "
     "Reply with the label only, in lowercase, with no punctuation."),
    ("human", "{text}"),
])

LABELS = ["product_info", "returns", "shipping", "warranty"]


def make_chain(prompt, temperature=0.7, model="llama3.2:3b"):
    llm = ChatOllama(model=model, temperature=temperature)
    return prompt | llm | StrOutputParser()
EOF
```

**Explain while reading the file:**

| Component | Role |
|---|---|
| `ChatPromptTemplate` | Turns structured inputs into a prompt, so changes to the prompt are versionable |
| `ChatOllama` | Local model; `temperature` controls randomness |
| `StrOutputParser` | Converts the model message into a plain string |
| `\|` (pipe) | LangChain Expression Language: chains the components |

## 1.2 Generate outputs

```bash
tee lab1_generate_marketing.py > /dev/null << 'EOF'
import json
import pandas as pd
from lab1_common import PROMPT_V1, make_chain

marketing_eval = json.load(open("data/marketing_eval.json"))
chain_v1 = make_chain(PROMPT_V1, temperature=0.7)

rows = []
for item in marketing_eval:
    out = chain_v1.invoke({
        "format": item["format"],
        "max_words": item["max_words"],
        "brief": item["brief"],
    }).strip()
    rows.append({**item, "prediction": out})

df_v1 = pd.DataFrame(rows)
df_v1["pred_words"] = df_v1["prediction"].str.split().str.len()
df_v1.to_csv("outputs/lab1_marketing_v1.csv", index=False)

for r in rows:
    print(f"[{r['id']}] {r['format']}")
    print("  REF :", r["reference"])
    print("  PRED:", r["prediction"], "\n")
EOF
```

Run it:

```bash
python lab1_generate_marketing.py
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

Now a quick automated check on rules you *can* test:

```bash
tee lab1_rule_checks.py > /dev/null << 'EOF'
import pandas as pd

df = pd.read_csv("outputs/lab1_marketing_v1.csv")

df["within_limit"] = df["pred_words"] <= df["max_words"]
df["says_cheap"] = df["prediction"].str.contains("cheap", case=False)
df["health_claim"] = df["prediction"].str.contains(
    r"\b(cure|treat|prevent|allerg|flu|asthma)", case=False, regex=True)

checks = df[["id", "pred_words", "within_limit", "says_cheap", "health_claim"]]
print(checks.to_string(index=False))
checks.to_csv("outputs/lab1_marketing_v1_checks.csv", index=False)
EOF
```

```bash
python lab1_rule_checks.py
```

> **Teaching point:** some of the most important quality criteria (word limit, banned words, forbidden claims) are **rule-based checks**, not similarity metrics. Real evaluation suites mix both.

---

## 1.4 A closed-answer task: intent classification

Customer messages must be routed to the right team. Here there is exactly one right answer, so **Accuracy** and **Exact Match** are appropriate.

```bash
tee lab1_intent.py > /dev/null << 'EOF'
import json
import pandas as pd
from lab1_common import INTENT_PROMPT, make_chain

intent_eval = json.load(open("data/intent_eval.json"))
intent_chain = make_chain(INTENT_PROMPT, temperature=0.0)

rows = []
for item in intent_eval:
    pred = intent_chain.invoke({"text": item["text"]}).strip()
    rows.append({**item, "prediction": pred})

df_intent = pd.DataFrame(rows)
df_intent.to_csv("outputs/lab1_intent.csv", index=False)

print(df_intent.to_string(index=False))
print("\nMismatches (raw string comparison):")
print(df_intent[df_intent["prediction"] != df_intent["label"]].to_string(index=False))
EOF
```

```bash
python lab1_intent.py
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

Add a sixth brief to `data/marketing_eval.json` that is deliberately tricky (for example, one that tempts the model to make a health claim). Predict the result **before** you re-run `lab1_generate_marketing.py` and `lab1_rule_checks.py`.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: lab1_common` | You are not in `novahome_eval/`, or the first `tee` block was not run |
| `FileNotFoundError: data/marketing_eval.json` | Re-run the Lab 0 dataset steps |
| `Connection refused` from Ollama | Start Ollama (`ollama serve`) and check `ollama list` |
| `SyntaxError` right after pasting | The closing `EOF` must be alone on its line with no leading spaces |

## Checkpoint

- [ ] `lab1_common.py`, `lab1_generate_marketing.py`, `lab1_rule_checks.py` and `lab1_intent.py` exist
- [ ] `outputs/lab1_marketing_v1.csv` exists
- [ ] `outputs/lab1_intent.csv` exists
- [ ] You wrote your own "what does good look like" list for the business

**Next:** [Lab 2: Score with traditional NLP metrics](lab2_nlp_metrics.md)
