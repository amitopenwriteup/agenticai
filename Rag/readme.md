# LLM Evaluation Workshop: From Standalone LLM to RAG

**Format:** Instructor-led demonstration + guided labs | **Duration:** 3 hours (plus a 10-minute break) | **Language/Tools:** Python, LangChain, ChromaDB, Ollama, Hugging Face `evaluate`, RAGAS

---

## What You Will Do

You will evaluate **one business application twice**:

1. **Version 1 (non-RAG):** a marketing-copy assistant for a fictional company, *NovaHome Appliances*, evaluated with classic NLP metrics (BLEU, ROUGE, METEOR, BERTScore, Accuracy, Exact Match).
2. **Version 2 (RAG):** the same assistant extended with a knowledge base (product catalog, policies, brand guidelines, support FAQ) using LangChain + ChromaDB, evaluated with RAG-specific metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall, Context Relevance, Answer Correctness).

The goal is not to memorise framework syntax. It is to learn **which metric answers which question**, how to **read the scores**, and how to **trace a bad answer back to its root cause** (prompt, retrieval, context, or generation).

---

## Learning Outcomes

By the end you will be able to:

- Explain why LLM evaluation differs from traditional ML evaluation
- Run and interpret BLEU, ROUGE, METEOR, BERTScore, Accuracy and Exact Match
- Build a local RAG pipeline with LangChain and ChromaDB
- Run RAGAS-style evaluation and interpret each metric's score range
- Diagnose whether a failure came from retrieval, context quality, prompt design or generation
- Use evaluation results to drive iterative improvements
- Choose metrics based on business objective and system architecture

---

## Prerequisites

- Python and Jupyter / VS Code
- Working knowledge of LLMs, prompting and basic RAG
- Familiarity with embeddings and semantic search
- Basic familiarity with Accuracy, Precision, Recall, BLEU and ROUGE
- LangChain familiarity is helpful but not required

> **Debugging is part of the job.** When something breaks, read the error, then use an LLM assistant as a thinking partner (see the "AI copilot prompts" in each lab).

---

## Session Map

| Section | Lab file | Time | What happens |
|---|---|---|---|
| 1. Introduction to LLM Evaluation | *(slides/discussion)* + [`lab0_setup_and_data.md`](lab0_setup_and_data.md) | 20 min | Why LLM evaluation is hard, metric recap, session roadmap, environment check, data creation |
| 2. Evaluate a non-RAG LLM app | [`lab1_non_rag_app.md`](lab1_non_rag_app.md), [`lab2_nlp_metrics.md`](lab2_nlp_metrics.md) | 55 min | Build the baseline app, create references, score with traditional metrics, improve prompts |
| *Break* | | 10 min | Place it where it fits best (after Lab 2 is natural) |
| 3. Build a RAG pipeline | [`lab3_rag_pipeline.md`](lab3_rag_pipeline.md) | 40 min | Documents, chunking, embeddings, Chroma, retrieval, augmented generation |
| 4. Evaluate the RAG app | [`lab4_rag_evaluation.md`](lab4_rag_evaluation.md) | 50 min | Retrieval checks, RAGAS metrics, report interpretation, failure cases |
| 5. Compare, diagnose, wrap up | [`lab5_compare_diagnose_wrapup.md`](lab5_compare_diagnose_wrapup.md) | 10 min + Q&A | Non-RAG vs RAG comparison, diagnosis playbook, improvement loop |

---

## Project Structure

Everything is created inside one folder, `novahome_eval/`. Run all notebooks from that folder.

```
novahome_eval/
├── data/
│   ├── docs/                    # RAG knowledge base (created in Lab 0)
│   │   ├── products.md
│   │   ├── policies.md
│   │   ├── brand_guidelines.md
│   │   └── support_faq.md
│   ├── marketing_eval.json      # copywriting briefs + reference copy
│   ├── intent_eval.json         # labelled customer messages
│   └── rag_eval.json            # questions + reference answers + expected sources
├── outputs/                     # CSV results from every lab
├── chroma_db/                   # persisted vector store (created in Lab 3)
├── rag_app.py                   # reusable RAG functions (created in Lab 3)
└── lab notebooks (.ipynb)
```

---

## Metric Cheat Sheet (keep this handy)

| Metric | Needs reference? | Needs contexts? | Question it answers | Typical weakness |
|---|---|---|---|---|
| Accuracy / Exact Match | Yes | No | Is the label/short answer exactly right? | Only for closed answers |
| BLEU | Yes | No | How many n-grams overlap with the reference? | Penalises valid paraphrases |
| ROUGE-1/2/L | Yes | No | How much of the reference is covered? | Surface overlap only |
| METEOR | Yes | No | Overlap with stemming and synonyms | Still lexical |
| BERTScore | Yes | No | Is the meaning similar (embeddings)? | Cannot catch factual errors |
| Faithfulness | No | Yes | Is every claim supported by the retrieved context? | Judge-LLM dependent |
| Answer Relevance | No | Optional | Does the answer address the question? | Does not check truth |
| Context Precision | Yes | Yes | Are relevant chunks ranked near the top? | Needs a reference |
| Context Recall | Yes | Yes | Did retrieval fetch everything needed? | Needs a good reference |
| Context Relevance | No | Yes | Is the retrieved context on-topic for the question? | Judge-LLM dependent |
| Answer Correctness | Yes | No | Is the answer factually and semantically right vs. the reference? | Needs a reference |

---

## Business Framing

| Business goal | What to measure | Why |
|---|---|---|
| Brand-safe marketing copy | ROUGE/BERTScore against approved copy; guideline compliance checks | Tone and consistency matter more than exact wording |
| Correct support answers (policies, prices) | Faithfulness, Answer Correctness, Context Recall | A wrong refund policy has legal and trust costs |
| Fast, cheap assistant | Retrieval hit rate, latency, `k` and chunk size trade-offs | More context = more cost and more noise |
| Safe refusals | Out-of-scope question behaviour, Faithfulness | The assistant should say "I don't know" rather than invent |

---

## Facilitation Notes

- **Pre-validate everything.** Pull models, run every notebook end-to-end and cache RAGAS results before the session (each lab shows how).
- **Don't spend time on installs.** Run the Lab 0 environment check as a pre-session task.
- **Show failures on purpose.** Each lab includes a deliberate failure so learners see what bad scores look like.
- **Always tie back to business.** After each metric table ask: *"Would a customer or compliance officer be happy with this?"*
- **Use AI assistants openly.** Each lab includes prompts learners can paste into an LLM assistant for debugging, explaining output and generating variants.
