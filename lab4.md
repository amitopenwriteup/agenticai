# Minimal RAG Pipeline

**Retrieve → Build prompt → Generate** · Linux · Python 3.9+

> Everything is in this one file: setup, data, code and how to run it. Copy the blocks in order.

## How it works

```
question ──► 1. Retrieve ──► 2. Build prompt ──► 3. Generate ──► answer
             (TF-IDF top-k)   (context + question)  (LLM or extractive)
                  │
                  └─ if best score < --min-score ──► "I could not find this in the documents."
```

| Stage | What it does | Where in `rag.py` |
|---|---|---|
| Retrieve | Scores every chunk against the question with TF-IDF and returns the top k | `Retriever.search` |
| Build prompt | Puts the chunks and the question into an "answer only from this context" prompt | `build_prompt` |
| Generate | Sends the prompt to an LLM, or copies the top chunk if no LLM is set | `generate` |
| Abstain | Refuses to answer when nothing relevant was retrieved | `answer` |

---

## Setup (once)

**Step 1: create the environment**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install scikit-learn numpy
# optional, only for --llm anthropic:
# pip install anthropic
```

**Step 2: create the files.** Paste this whole block into your terminal. It creates the folder `rag-pipeline/` with everything needed.

```bash
mkdir -p rag-pipeline
cd rag-pipeline

cat > requirements.txt <<'EOF'
scikit-learn
numpy
# optional, only for --llm anthropic
# anthropic
EOF

cat > corpus.json <<'EOF'
[
{"id":"c01","text":"SSH lets you log in to a remote server securely. Use ssh user@host. Disable password login and use key pairs in /etc/ssh/sshd_config for better security."},
{"id":"c02","text":"chmod changes file permissions. Use chmod 644 file for owner read-write and others read-only. chmod +x makes a script executable."},
{"id":"c03","text":"chown changes the owner and group of a file. Use chown user:group file. Use -R to apply it recursively to a directory."},
{"id":"c04","text":"df -h shows free disk space per filesystem. du -sh folder shows how much space a folder uses."},
{"id":"c05","text":"ps aux lists running processes. top shows live CPU and memory usage. Use kill PID to stop a process; kill -9 PID forces it to stop immediately."},
{"id":"c06","text":"systemctl manages services. systemctl start nginx starts a service, systemctl enable nginx starts it at boot, systemctl status nginx shows its state."},
{"id":"c07","text":"apt install package installs software on Debian and Ubuntu. Run apt update first to refresh the package list. apt remove package uninstalls it."},
{"id":"c08","text":"cron runs scheduled jobs. Edit your schedule with crontab -e. A line like 30 2 * * * /opt/backup.sh runs the script every day at 02:30."},
{"id":"c09","text":"ufw is a simple firewall. ufw allow 22/tcp opens the SSH port. ufw enable turns the firewall on and ufw status lists the rules."},
{"id":"c10","text":"System logs live in /var/log. Use journalctl -u nginx to read service logs and tail -f /var/log/syslog to follow a log file live."},
{"id":"c11","text":"useradd -m alice creates a user with a home directory. passwd alice sets the password. usermod -aG sudo alice gives admin rights."},
{"id":"c12","text":"tar -czf backup.tar.gz folder creates a compressed archive. tar -xzf backup.tar.gz extracts it."},
{"id":"c13","text":"ip addr shows network interfaces and IP addresses. ping host tests whether a machine is reachable. ss -tulpn lists listening ports."},
{"id":"c14","text":"free -h shows used and free memory. If a server runs out of memory the kernel may kill processes. Add swap space to reduce the risk."},
{"id":"c15","text":"find / -name file searches for files by name. grep -r text folder searches inside files for a text string."}
]
EOF

cat > rag.py <<'EOF'
"""Minimal RAG pipeline: retrieve -> build prompt -> generate.

Usage:
  python rag.py "how do I make a script executable"
  python rag.py "stop a frozen process" --k 3 --llm ollama --model llama3.2
  python rag.py "run a script every night" --llm anthropic --model <model-name>
  python rag.py            # interactive mode

LLM backends:
  extractive (default)  no LLM, returns the top chunk (runs with zero setup)
  ollama                local model via http://localhost:11434
  anthropic             needs `pip install anthropic` and ANTHROPIC_API_KEY
"""
import argparse
import json
import os
import urllib.request

from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "I could not find this in the documents."

Context:
{context}

Question: {question}
Answer:"""


# ---------- 1. Retrieve ----------
class Retriever:
    def __init__(self, corpus_path):
        with open(corpus_path) as f:
            self.chunks = json.load(f)
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, question, k=3):
        scores = (self.vec.transform([question]) @ self.matrix.T).toarray()[0]
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


# ---------- 2. Build prompt ----------
def build_prompt(question, hits):
    context = "\n".join(f"[{c['id']}] {c['text']}" for c, _ in hits)
    return PROMPT.format(context=context, question=question)


# ---------- 3. Generate ----------
def generate(prompt, hits, llm, model):
    if llm == "ollama":
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["response"].strip()
    if llm == "anthropic":
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    return hits[0][0]["text"]  # extractive fallback: copy top chunk


# ---------- Pipeline ----------
def answer(retriever, question, k, llm, model, min_score):
    hits = retriever.search(question, k)
    if hits[0][1] < min_score:  # nothing relevant: abstain instead of guessing
        return "I could not find this in the documents.", hits
    return generate(build_prompt(question, hits), hits, llm, model), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--llm", choices=["extractive", "ollama", "anthropic"], default="extractive")
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--min-score", type=float, default=0.05, help="abstain below this top score")
    a = ap.parse_args()

    r = Retriever(a.corpus)

    def run(q):
        ans, hits = answer(r, q, a.k, a.llm, a.model, a.min_score)
        print("\nRetrieved:", ", ".join(f"{c['id']} ({s:.2f})" for c, s in hits))
        print("Answer:", ans, "\n")

    if a.question:
        run(a.question)
    else:
        while True:
            q = input("question (blank to quit)> ").strip()
            if not q:
                break
            run(q)


if __name__ == "__main__":
    main()
EOF
```

**Checkpoint:** run `python rag.py "make a script executable"`. You should see `Retrieved: c02 ...` and the chmod answer.

---

## Run it

**Zero setup (no LLM, returns the top chunk):**

```bash
python rag.py "make a script executable"
python rag.py "stop a process that is frozen"
```

**Expected:**

```
Retrieved: c02 (0.33), c08 (0.13), c15 (0.00)
Answer: chmod changes file permissions. Use chmod 644 file for owner read-write and others read-only. chmod +x makes a script executable.

Retrieved: c05 (0.43), c14 (0.00), c15 (0.00)
Answer: ps aux lists running processes. top shows live CPU and memory usage. Use kill PID to stop a process; kill -9 PID forces it to stop immediately.
```

**Out-of-scope question (abstains):**

```bash
python rag.py "what is the capital of France"
```

```
Retrieved: c15 (0.00), c14 (0.00), c13 (0.00)
Answer: I could not find this in the documents.
```

**Interactive mode:**

```bash
python rag.py
```

**With a real LLM:**

```bash
# Local model with Ollama (ollama must be running, model pulled)
python rag.py "run a script every night" --llm ollama --model llama3.2

# Anthropic API
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python rag.py "run a script every night" --llm anthropic --model <model-name>
```

> With an LLM the answer is written from the retrieved chunks instead of copied. The `extractive` mode is the only one tested here; `ollama` and `anthropic` need your own setup.

---

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--k` | `3` | number of chunks sent to the LLM |
| `--llm` | `extractive` | `extractive`, `ollama` or `anthropic` |
| `--model` | `llama3.2` | model name for the chosen LLM |
| `--min-score` | `0.05` | abstain when the best retrieval score is below this |
| `--corpus` | `corpus.json` | path to your own chunks file |

---

## Use your own documents

Replace `corpus.json` with a list of chunks in the same shape:

```json
[
  {"id": "d01", "text": "First chunk of your document..."},
  {"id": "d02", "text": "Second chunk..."}
]
```

Keep each chunk short (a few sentences) and about one topic. Retrieval quality depends mostly on how the text is split.

---

## Known limits and next steps

- **Keyword matching only.** TF-IDF needs shared words. "run a script every night" finds the cron chunk only because "script" overlaps, and "automate a task at 2am daily" would find nothing. Swap in embeddings (for example `sentence-transformers`) in `Retriever` for meaning-based search.
- **No chunking step.** You supply pre-split chunks.
- **No vector database.** Scores are computed in memory, which is fine for thousands of chunks, not millions.
- **Tune `--min-score`.** A high value refuses answerable questions, a low value lets weak matches through.
