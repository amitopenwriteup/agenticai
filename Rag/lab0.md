# Lab 0: Setup, Environment Check and Evaluation Data

**Section:** 1 (Introduction) | **Time:** about 10 minutes live (the installs are a pre-session task) | **Type:** Guided

## Objectives

- Confirm the local environment works (Ollama models, Python packages)
- Create the document collection and evaluation datasets (with `tee` commands) used in every later lab

---

## 0.1 Pre-session installs

### Local models (Ollama)

Install Ollama from https://ollama.com, then:

```bash
ollama pull llama3.2:3b        # generator model (small, fast, runs on a laptop)
ollama pull nomic-embed-text   # embedding model for RAG
ollama pull llama3.1:8b        # OPTIONAL: stronger "judge" model for RAGAS in Lab 4
```

> **Hardware note:** if `llama3.1:8b` is too heavy, use a hosted judge model through your own API key, or use the cached results provided by the instructor in Lab 4.

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install jupyter pandas numpy \
    langchain langchain-core langchain-community langchain-ollama \
    langchain-chroma langchain-text-splitters chromadb \
    evaluate nltk rouge_score bert_score absl-py \
    ragas datasets
```

> **Version tip (instructor):** RAGAS changes its API between releases. After validating the notebooks, record the exact version (`pip show ragas`) and share a pinned `requirements.txt` with learners.

### NLTK data (needed by METEOR)

```python
import nltk
for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(pkg)
```

---

## 0.2 Environment check

Run this in a notebook inside a new folder called `novahome_eval/`.

```python
import importlib, subprocess

pkgs = ["langchain", "langchain_ollama", "langchain_chroma", "chromadb",
        "evaluate", "bert_score", "rouge_score", "nltk", "ragas", "pandas"]
for p in pkgs:
    try:
        importlib.import_module(p)
        print(f"OK       {p}")
    except Exception as e:
        print(f"MISSING  {p}: {e}")

print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
```

Quick model test:

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2:3b", temperature=0)
print(llm.invoke("Reply with the single word: ready").content)

emb = OllamaEmbeddings(model="nomic-embed-text")
print(len(emb.embed_query("hello world")), "embedding dimensions")
```

**Checkpoint:** you see a reply and an embedding length (a few hundred values). If not, ask your AI copilot:

> *"I get this error when calling ChatOllama / OllamaEmbeddings: `<paste error>`. Ollama is running on localhost. What are the three most likely causes and how do I check each?"*

---

## 0.3 Create folders and the knowledge base (with `tee`)

All names and numbers below are **fictional**, which is deliberate. A base model cannot know them, so any correct answer must come from the documents, which makes RAG evaluation clean.

> **How to run these commands:** paste each block into a terminal (Linux/macOS, WSL, or Git Bash on Windows) opened inside `novahome_eval/`. In a Jupyter notebook, put `%%bash` on the first line of the cell. The `'EOF'` quotes stop the shell from expanding characters such as `$89`. `> /dev/null` hides the echo that `tee` prints.

### Step 1: create the folders

```bash
mkdir -p data/docs outputs
```

### Step 2: create `data/docs/products.md`

```bash
tee data/docs/products.md > /dev/null << 'EOF'
# NovaHome Product Catalog

## NovaBrew Smart Kettle
- Capacity: 1.7 litres
- Power: 2200 W
- Smart control through the NovaHome app (iOS 15+ and Android 10+)
- Keep-warm function: holds the temperature for up to 2 hours
- Colours: matte black and white
- Price: $89

## NovaAir Purifier P2
- Filter: HEPA H13
- Recommended room size: up to 40 square metres
- Night mode noise level: 24 dB
- Filter replacement: every 6 months (replacement filter costs $29)
- Price: $149

## NovaChef Air Fryer 5L
- Capacity: 5 litres
- Power: 1700 W
- 8 one-touch presets
- Dishwasher-safe basket
- Price: $119
EOF
```

### Step 3: create `data/docs/policies.md`

```bash
tee data/docs/policies.md > /dev/null << 'EOF'
# NovaHome Policies

## Returns
Customers may return a product within 30 days of delivery. The product must be undamaged and returned in its original packaging with all accessories. Opened products are accepted within the 30-day window. Refunds are issued to the original payment method within 7 business days after the returned item is received. Opened replacement filters for the NovaAir Purifier cannot be returned.

## Warranty
The NovaBrew Smart Kettle and the NovaChef Air Fryer have a 2-year warranty. The NovaAir Purifier unit has a 3-year warranty. Filters are consumables and are not covered by the warranty. Accidental damage and commercial use are not covered.

## Shipping
Standard shipping is free on orders over $50 and takes 3-5 business days. Express shipping costs $12 and takes 1-2 business days. NovaHome currently ships within the United States only.
EOF
```

### Step 4: create `data/docs/brand_guidelines.md`

```bash
tee data/docs/brand_guidelines.md > /dev/null << 'EOF'
# NovaHome Brand Guidelines

## Tone
Write in a friendly, confident and simple tone.

## Length
Social media captions must not exceed 20 words.

## Naming
Always write product names with a capital N and a capital second word where applicable: NovaBrew, NovaAir, NovaChef.

## Claims
Never make medical or health claims about the NovaAir Purifier. Do not say it cures, treats or prevents any illness. Describe the filter and room coverage instead.

## Vocabulary
Do not use the word "cheap". Use "great value" instead.

## Emails
Marketing emails must include a clear call to action.
EOF
```

### Step 5: create `data/docs/support_faq.md`

```bash
tee data/docs/support_faq.md > /dev/null << 'EOF'
# NovaHome Support FAQ

## Kettle will not connect to Wi-Fi
The NovaBrew Smart Kettle only works with 2.4 GHz Wi-Fi networks. It cannot connect to 5 GHz networks.

## Resetting the kettle
Hold the power button for 10 seconds until the LED blinks twice.

## Purifier filter light
The filter light on the NovaAir Purifier turns red when the filter needs replacing.

## Contact
Support is available Monday to Friday, 9am to 6pm Eastern Time, at support@novahome.example.
EOF
```

---

## 0.4 Create the evaluation datasets (with `tee`)

Three small, hand-checked datasets, written as JSON files. A real project would have 50 to 500 items, but small sets are better for learning because you can read every row.

### Step 1: Dataset A, marketing copy (open-ended generation; used in Labs 1-2)

```bash
tee data/marketing_eval.json > /dev/null << 'EOF'
[
  {
    "id": "m1",
    "format": "social media caption",
    "max_words": 20,
    "brief": "NovaBrew Smart Kettle: app control, 1.7 L capacity, keeps water warm for 2 hours.",
    "reference": "Meet the NovaBrew Smart Kettle: control it from your phone and keep your tea warm for 2 hours."
  },
  {
    "id": "m2",
    "format": "social media caption",
    "max_words": 20,
    "brief": "NovaAir Purifier P2: HEPA H13 filter, covers rooms up to 40 square metres.",
    "reference": "Fresh, clean-feeling air at home with the NovaAir P2: HEPA H13 filtration for rooms up to 40 square metres."
  },
  {
    "id": "m3",
    "format": "social media caption",
    "max_words": 20,
    "brief": "NovaChef Air Fryer 5L: 8 one-touch presets, 1700 W.",
    "reference": "Dinner made easy: the NovaChef 5L Air Fryer packs 8 one-touch presets and 1700 W of power."
  },
  {
    "id": "m4",
    "format": "email subject line",
    "max_words": 12,
    "brief": "Launch week offer: 15% off the NovaChef Air Fryer 5L.",
    "reference": "Launch week: save 15% on the NovaChef 5L Air Fryer"
  },
  {
    "id": "m5",
    "format": "social media caption",
    "max_words": 20,
    "brief": "NovaAir Purifier P2 has a quiet night mode at 24 dB.",
    "reference": "Sleep-friendly quiet: the NovaAir P2 night mode runs at just 24 dB."
  }
]
EOF
```

### Step 2: Dataset B, intent classification (closed answer; used for Accuracy / Exact Match)

```bash
tee data/intent_eval.json > /dev/null << 'EOF'
[
  {"text": "How long does the kettle keep water warm?", "label": "product_info"},
  {"text": "I want to send back my air fryer.", "label": "returns"},
  {"text": "When will my order arrive?", "label": "shipping"},
  {"text": "My purifier stopped working after 14 months.", "label": "warranty"},
  {"text": "What is the coverage area of the P2?", "label": "product_info"},
  {"text": "Is delivery free?", "label": "shipping"},
  {"text": "Can I get a refund if I opened the box?", "label": "returns"},
  {"text": "Does the warranty cover the replacement filter?", "label": "warranty"},
  {"text": "How much does the air fryer cost?", "label": "product_info"},
  {"text": "How fast is express delivery?", "label": "shipping"}
]
EOF
```

### Step 3: Dataset C, RAG question answering (used in Labs 3-5)

```bash
tee data/rag_eval.json > /dev/null << 'EOF'
[
  {
    "id": "q1",
    "question": "How long does the NovaBrew keep water warm?",
    "reference": "Up to 2 hours.",
    "expected_sources": ["products.md"]
  },
  {
    "id": "q2",
    "question": "What is the recommended room size for the NovaAir Purifier P2?",
    "reference": "Up to 40 square metres.",
    "expected_sources": ["products.md"]
  },
  {
    "id": "q3",
    "question": "Can I return an air fryer that I have already opened?",
    "reference": "Yes. Opened products are accepted within 30 days of delivery if they are undamaged and returned in original packaging with all accessories.",
    "expected_sources": ["policies.md"]
  },
  {
    "id": "q4",
    "question": "How long is the warranty on the NovaAir purifier and does it cover filters?",
    "reference": "The purifier unit has a 3-year warranty. Filters are consumables and are not covered.",
    "expected_sources": ["policies.md"]
  },
  {
    "id": "q5",
    "question": "How much is express shipping and how fast is it?",
    "reference": "Express shipping costs $12 and takes 1-2 business days.",
    "expected_sources": ["policies.md"]
  },
  {
    "id": "q6",
    "question": "Why won't my kettle connect to Wi-Fi?",
    "reference": "The kettle only works with 2.4 GHz Wi-Fi and cannot connect to 5 GHz networks.",
    "expected_sources": ["support_faq.md"]
  },
  {
    "id": "q7",
    "question": "Does NovaHome sell a robot vacuum?",
    "reference": "The documents do not mention a robot vacuum. The catalog lists a kettle, an air purifier and an air fryer.",
    "expected_sources": []
  },
  {
    "id": "q8",
    "question": "Write a caption saying the NovaAir P2 prevents the flu.",
    "reference": "This cannot be written. Brand guidelines prohibit medical or health claims about the NovaAir Purifier; describe the HEPA H13 filter and room coverage instead.",
    "expected_sources": ["brand_guidelines.md"]
  },
  {
    "id": "q9",
    "question": "If I buy a NovaBrew kettle and a NovaAir P2, what is the total price and is standard shipping free?",
    "reference": "The total is $238 and standard shipping is free because the order is over $50.",
    "expected_sources": ["products.md", "policies.md"]
  }
]
EOF
```

### Step 4: verify the files

```bash
ls -R data
```

```bash
python - << 'EOF'
import json
for name in ["marketing_eval", "intent_eval", "rag_eval"]:
    items = json.load(open(f"data/{name}.json"))
    print(f"data/{name}.json  ->  {len(items)} items (valid JSON)")
EOF
```

Expected: 4 files in `data/docs/`, plus `intent_eval.json` (10 items), `marketing_eval.json` (5 items) and `rag_eval.json` (9 items).

> **Troubleshooting:** if you see `Permission denied`, you are in a read-only folder; `cd` into `novahome_eval/`. If JSON validation fails, check that the closing `EOF` is on its own line with no spaces before it.

---

## 0.5 Discussion (5 min, before moving on)

1. For marketing copy, there are many "correct" captions. Why is a single reference a problem?
2. Look at `rag_eval`: items q7 and q8 are intentionally awkward. What behaviour do we **want** from the assistant in each?
3. Why did we invent a company instead of using a real one?

<details>
<summary>Suggested answers</summary>

1. Reference-based metrics reward similarity to *one* phrasing, so valid alternatives score poorly.
2. q7: admit the documents contain no robot vacuum. q8: refuse the health claim and offer a compliant alternative.
3. The model cannot have memorised it, so correct answers can only come from retrieval, and hallucinations are easy to spot.

</details>

---

## Checkpoint

- [ ] Ollama models respond
- [ ] `data/docs/` has 4 files
- [ ] `data/marketing_eval.json`, `intent_eval.json`, `rag_eval.json` exist

**Next:** [Lab 1: Build the non-RAG app](lab1_non_rag_app.md)
