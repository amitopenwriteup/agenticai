# Hands-On Workshop: Evaluating Semantic & Generative Search (RAG)

**Part III of the course · 45 minutes · Runs on any Linux VM (Ubuntu/Debian tested)**

You will build a tiny search system over a Linux-admin knowledge base, then measure it the way the course taught:

| Time (remote timestamp) | Block | What you do |
|---|---|---|
| `T+0:00 – T+0:10` | **1. Data preparation** | Set up the VM, create reference chunks, labelled queries, reference answers and generated answers |
| `T+0:10 – T+0:25` | **2. Retrieval evaluation** | Run retrieval, compute Precision@k, Recall@k, F1@k, MAP, MRR |
| `T+0:25 – T+0:45` | **3. Generation evaluation** | Compute BLEU and ROUGE, compare two "generators", discuss limits |

> **Why a small dataset?** 15 chunks and 12 queries let you check every number by hand. The same code scales to real data: swap the JSON files.

---

## 0. Prerequisites (do this before the session, ~5 min)

- Linux VM, Python 3.9+, ~2 GB RAM, ~1 GB free disk (the optional embedding model is ~90 MB plus libraries).
- Internet access for `pip` (and for the one-time embedding-model download if you choose that path).

```bash
python3 --version
# If python3-venv is missing on Ubuntu/Debian:
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
```

---

## Block 1: Data Preparation (`T+0:00 – T+0:10`)

### 1.1 Create the project and virtual environment

```bash
mkdir -p ~/rag-eval-workshop/{data,src,runs} && cd ~/rag-eval-workshop
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scikit-learn sacrebleu rouge-score
```

Optional (for the semantic/embedding path in Block 2; large install, can run in the background):

```bash
pip install sentence-transformers
```

### 1.2 The four data assets

| File | Role in evaluation |
|---|---|
| `data/corpus.json` | The **chunks** that retrieval searches over (15 chunks) |
| `data/queries.json` | Queries with **relevance judgments**: which chunk IDs are relevant (ground truth for retrieval metrics) |
| `data/references.json` | One **reference answer** per query (ground truth for generation metrics) |
| `data/generated_llm.json` | **Generated texts** from a RAG-style generator (hand-written here so no API key or GPU is needed) |

Create them (copy the whole block into your terminal while inside `~/rag-eval-workshop`):

```bash
cat > data/corpus.json <<'EOF'
[
 {"id":"c01","text":"systemctl manages services on systemd-based Linux distributions. Use 'systemctl status nginx' to check a service, 'systemctl restart nginx' to restart it, and 'systemctl enable nginx' to start it automatically at boot."},
 {"id":"c02","text":"journalctl reads logs collected by systemd-journald. Use 'journalctl -u nginx' to see the logs of one service and 'journalctl -f' to follow new log lines in real time."},
 {"id":"c03","text":"chmod changes file permissions. Permissions are read (4), write (2) and execute (1). For example 'chmod 640 file.txt' gives the owner read and write access, the group read access, and others no access."},
 {"id":"c04","text":"chown changes the owner and group of a file. Run 'sudo chown alice:developers report.txt' to make alice the owner and developers the group."},
 {"id":"c05","text":"df -h shows free and used disk space for each mounted filesystem in human-readable units. du -sh * shows how much space each item in the current directory uses."},
 {"id":"c06","text":"To set up passwordless SSH login, generate a key pair with 'ssh-keygen -t ed25519' and copy the public key to the server using 'ssh-copy-id user@host'."},
 {"id":"c07","text":"On Debian and Ubuntu, apt installs software. Run 'sudo apt update' to refresh package lists and 'sudo apt install htop' to install a package. 'sudo apt upgrade' updates installed packages."},
 {"id":"c08","text":"cron runs scheduled jobs. Edit your schedule with 'crontab -e'. The line '30 2 * * * /opt/backup.sh' runs the script every day at 02:30."},
 {"id":"c09","text":"ps aux lists running processes. To stop a process, find its PID and run 'kill PID'. If it does not exit, 'kill -9 PID' forces it to terminate."},
 {"id":"c10","text":"ufw is a simple firewall front end on Ubuntu. 'sudo ufw allow 22/tcp' opens SSH, 'sudo ufw enable' turns the firewall on, and 'sudo ufw status' shows the active rules."},
 {"id":"c11","text":"grep searches text for a pattern. 'grep -rn \"error\" /var/log' searches recursively and prints line numbers. Add -i for case-insensitive matching."},
 {"id":"c12","text":"tar bundles files into an archive. 'tar -czf backup.tar.gz folder/' creates a compressed archive and 'tar -xzf backup.tar.gz' extracts it."},
 {"id":"c13","text":"'sudo adduser bob' creates a new user with a home directory. 'sudo usermod -aG sudo bob' adds bob to the sudo group so he can run administrative commands."},
 {"id":"c14","text":"free -h shows used and available memory. 'top' or 'htop' shows live CPU and memory use for each process."},
 {"id":"c15","text":"'ip addr' shows network interfaces and IP addresses. 'ss -tulpn' lists listening ports and the processes that own them."}
]
EOF

cat > data/queries.json <<'EOF'
[
 {"qid":"q01","query":"How do I restart a service and make it start on boot?","relevant":["c01"]},
 {"qid":"q02","query":"show nginx logs in real time","relevant":["c02"]},
 {"qid":"q03","query":"give my team read-only access to a file","relevant":["c03","c04"]},
 {"qid":"q04","query":"how much disk space is left","relevant":["c05"]},
 {"qid":"q05","query":"log in over ssh without typing a password","relevant":["c06"]},
 {"qid":"q06","query":"install a program on ubuntu","relevant":["c07"]},
 {"qid":"q07","query":"run a script automatically every night","relevant":["c08"]},
 {"qid":"q08","query":"kill a process that is hanging","relevant":["c09"]},
 {"qid":"q09","query":"open port 22 on the firewall","relevant":["c10","c15"]},
 {"qid":"q10","query":"let a user run admin commands","relevant":["c13"]},
 {"qid":"q11","query":"which process is using a network port","relevant":["c15","c09"]},
 {"qid":"q12","query":"search the logs for errors","relevant":["c11","c02"]}
]
EOF

cat > data/references.json <<'EOF'
{
 "q01":"Run systemctl restart on the service and systemctl enable to start it at boot.",
 "q02":"Use journalctl -u nginx -f to follow the nginx logs in real time.",
 "q03":"Use chmod 640 so the group has read access, and chown to set the group.",
 "q04":"Run df -h to see free and used disk space on each filesystem.",
 "q05":"Generate a key with ssh-keygen and copy it to the server with ssh-copy-id.",
 "q06":"Run sudo apt update and then sudo apt install followed by the package name.",
 "q07":"Add a line to crontab -e, for example 30 2 * * * followed by the script path.",
 "q08":"Find the PID with ps aux and run kill PID, or kill -9 PID if it will not exit.",
 "q09":"Run sudo ufw allow 22/tcp to open SSH and check the rules with sudo ufw status.",
 "q10":"Run sudo usermod -aG sudo username to add the user to the sudo group.",
 "q11":"Run ss -tulpn to list listening ports and the processes that own them.",
 "q12":"Use grep -rn error /var/log to search the logs recursively for errors."
}
EOF

cat > data/generated_llm.json <<'EOF'
{
 "q01":"You can restart it with systemctl restart and enable it with systemctl enable so it starts at boot.",
 "q02":"Run journalctl -u nginx -f to follow the nginx logs live.",
 "q03":"Set the file mode with chmod 640 and change the group using chown.",
 "q04":"Use the df -h command to check free disk space.",
 "q05":"Create a key pair with ssh-keygen, then upload the public key using ssh-copy-id.",
 "q06":"First run sudo apt update, then sudo apt install and the name of the package.",
 "q07":"Use cron. Open crontab -e and add a line such as 0 0 * * * with the path of your script.",
 "q08":"Look up its PID using ps aux and stop it with kill, or kill -9 if needed.",
 "q09":"Use ufw to allow port 22 over tcp, and turn the firewall on with sudo ufw enable.",
 "q10":"Use usermod -aG sudo and the username to give that user admin rights.",
 "q11":"The ss -tulpn command shows which process listens on each port.",
 "q12":"Use grep with the -rn options on the /var/log directory to find errors."
}
EOF
```

### 1.3 Sanity-check the data

```bash
python - <<'EOF'
import json
c = json.load(open("data/corpus.json")); q = json.load(open("data/queries.json"))
ids = {x["id"] for x in c}
print("chunks:", len(c), "| queries:", len(q))
bad = [(x["qid"], r) for x in q for r in x["relevant"] if r not in ids]
print("dangling relevance labels:", bad or "none")
print("avg relevant chunks per query:", sum(len(x["relevant"]) for x in q) / len(q))
EOF
```

Expected: `chunks: 15 | queries: 12`, `dangling relevance labels: none`, average `1.4166...`.

**Discuss (2 min):** Who produced the relevance labels in a real project? How many judges? (Links back to *Manual / Task-Oriented Evaluation*.)

---

## Block 2: Retrieval Evaluation (`T+0:10 – T+0:25`)

### 2.1 The retriever (`src/retrieve.py`)

Ranks **all** chunks for each query and saves the ranking. Two methods: `tfidf` (lexical, no downloads) and `embed` (semantic, uses `all-MiniLM-L6-v2`).

```bash
cat > src/retrieve.py <<'EOF'
"""Build an index over data/corpus.json and rank all chunks for every query."""
import argparse, json
import numpy as np

def load(p):
    with open(p) as f:
        return json.load(f)

def tfidf_scores(docs, queries):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    D = vec.fit_transform(docs)
    Q = vec.transform(queries)
    return (Q @ D.T).toarray()          # cosine (rows are L2-normalised)

def embed_scores(docs, queries):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    D = model.encode(docs, normalize_embeddings=True)
    Q = model.encode(queries, normalize_embeddings=True)
    return Q @ D.T

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tfidf", "embed"], default="tfidf")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    corpus, queries = load("data/corpus.json"), load("data/queries.json")
    ids = [c["id"] for c in corpus]
    docs = [c["text"] for c in corpus]
    qtexts = [q["query"] for q in queries]

    S = tfidf_scores(docs, qtexts) if a.method == "tfidf" else embed_scores(docs, qtexts)

    run = {}
    for q, row in zip(queries, S):
        order = np.argsort(-row)
        run[q["qid"]] = [{"id": ids[i], "score": round(float(row[i]), 4)} for i in order]

    out = a.out or f"runs/{a.method}.json"
    with open(out, "w") as f:
        json.dump(run, f, indent=1)
    print(f"wrote {out}  ({len(run)} queries x {len(ids)} chunks)")
    for q in queries[:3]:
        top = [r["id"] for r in run[q["qid"]][:3]]
        print(f"  {q['qid']} {q['query']!r:55} top3={top}  relevant={q['relevant']}")
EOF
```

```bash
python src/retrieve.py --method tfidf
```

Expected (first lines):

```text
wrote runs/tfidf.json  (12 queries x 15 chunks)
  q01 'How do I restart a service and make it start on boot?'  top3=['c01', 'c04', 'c02']  relevant=['c01']
  ...
```

### 2.2 The metrics (`src/eval_retrieval.py`)

```bash
cat > src/eval_retrieval.py <<'EOF'
"""Precision@k, Recall@k, F1@k, Average Precision, MAP and MRR."""
import argparse, json

def prf(ranked, rel, k):
    hits = sum(1 for d in ranked[:k] if d in rel)
    p = hits / k
    r = hits / len(rel)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f

def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            total += hits / i            # precision at each relevant rank
    return total / len(rel)              # divide by ALL relevant docs

def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1 / i
    return 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/tfidf.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    queries = json.load(open("data/queries.json"))
    run = json.load(open(a.run))
    rows = []
    for q in queries:
        ranked = [r["id"] for r in run[q["qid"]]]
        rel = set(q["relevant"])
        p, r, f = prf(ranked, rel, a.k)
        rows.append((q["qid"], p, r, f, average_precision(ranked, rel), reciprocal_rank(ranked, rel)))

    if a.verbose:
        print(f"{'qid':5} {'P@'+str(a.k):>6} {'R@'+str(a.k):>6} {'F1@'+str(a.k):>6} {'AP':>6} {'RR':>6}")
        for r in rows:
            print(f"{r[0]:5} " + " ".join(f"{x:6.3f}" for x in r[1:]))
    n = len(rows)
    mean = lambda j: sum(r[j] for r in rows) / n
    print(f"\nrun={a.run}  k={a.k}  queries={n}")
    print(f"  Precision@{a.k} = {mean(1):.3f}")
    print(f"  Recall@{a.k}    = {mean(2):.3f}")
    print(f"  F1@{a.k}        = {mean(3):.3f}")
    print(f"  MAP          = {mean(4):.3f}")
    print(f"  MRR          = {mean(5):.3f}")
EOF
```

Run at k = 3 with the per-query table:

```bash
python src/eval_retrieval.py --run runs/tfidf.json --k 3 --verbose
```

Expected output:

```text
qid      P@3    R@3   F1@3     AP     RR
q01    0.333  1.000  0.500  1.000  1.000
q02    0.333  1.000  0.500  1.000  1.000
q03    0.667  1.000  0.800  1.000  1.000
q04    0.333  1.000  0.500  1.000  1.000
q05    0.333  1.000  0.500  1.000  1.000
q06    0.333  1.000  0.500  1.000  1.000
q07    0.333  1.000  0.500  1.000  1.000
q08    0.333  1.000  0.500  1.000  1.000
q09    0.333  0.500  0.400  0.567  1.000
q10    0.333  1.000  0.500  1.000  1.000
q11    0.333  0.500  0.400  0.750  1.000
q12    0.333  0.500  0.400  0.591  1.000

run=runs/tfidf.json  k=3  queries=12
  Precision@3 = 0.361
  Recall@3    = 0.875
  F1@3        = 0.500
  MAP          = 0.909
  MRR          = 1.000
```

### 2.3 Read the results (5 min)

- **Recall@3 is high (0.875) but Precision@3 is low (0.361).** Most queries have only 1 relevant chunk, so with k = 3 the best possible precision is 1/3. *Precision@k is capped by how many relevant items exist.*
- Look at `q09`, `q11`, `q12`: AP drops below 1 because a second relevant chunk is ranked lower or missed. Open `runs/tfidf.json` and find where `c15`, `c09`, `c02` landed.
- **MRR = 1.0** means the first relevant chunk is always at rank 1. MAP is lower because it also rewards finding *all* relevant chunks.

Try different cut-offs and watch the precision/recall trade-off:

```bash
python src/eval_retrieval.py --k 1
python src/eval_retrieval.py --k 5
```

```text
# k=1                          # k=5
Precision@1 = 1.000   Precision@5 = 0.233
Recall@1    = 0.833   Recall@5    = 0.917
```

### 2.4 Compare with semantic retrieval (optional, needs `sentence-transformers`)

```bash
python src/retrieve.py --method embed
python src/eval_retrieval.py --run runs/embed.json --k 3 --verbose
```

Numbers will differ from TF-IDF (they depend on the model). Questions to answer:

1. Which queries improved? (Hint: `q06`, `q07`, `q10` use words like "program", "automatically every night", "admin commands" that do not appear in the chunks.)
2. Did any query get worse? Why might an embedding model rank an unrelated chunk highly?

### 2.5 Exercises (pick one)

- **A. Add a hard query** to `data/queries.json`, e.g. `{"qid":"q13","query":"my server is slow, what is eating RAM","relevant":["c14"]}`, and compare TF-IDF vs embeddings.
- **B. Hand-verify AP** for `q12` on paper: relevant = {c11, c02}. Find their ranks in `runs/tfidf.json`, compute AP, and confirm the script's `0.591`.
- **C. Hybrid:** in `retrieve.py`, add `0.5 * tfidf_scores + 0.5 * embed_scores` (normalise each first) as a third method and evaluate it.

---

## Block 3: Generation Evaluation (`T+0:25 – T+0:45`)

### 3.1 Understand BLEU and ROUGE on a toy example first (5 min)

```bash
cat > src/toy_metrics.py <<'EOF'
"""Hand-checkable examples used in the slides."""
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

ref = "the cat sat on the mat"
hyp = "the cat is on the mat"
print("BLEU-4 (no smoothing):", round(BLEU(smooth_method="none").sentence_score(hyp, [ref]).score / 100, 3))
print("BLEU   (smoothed)    :", round(BLEU(effective_order=True).sentence_score(hyp, [ref]).score / 100, 3))
s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"]).score(ref, hyp)
for k, v in s.items():
    print(f"{k}: P={v.precision:.3f} R={v.recall:.3f} F1={v.fmeasure:.3f}")

ranked = ["d1","d2","d3","d4","d5","d6","d7","d8","d9","d10"]
rel = {"d1","d3","d6","d11"}                      # d11 is never retrieved
hits = [d in rel for d in ranked]
k = 5
p, r = sum(hits[:k]) / k, sum(hits[:k]) / len(rel)
print(f"P@5={p:.2f} R@5={r:.2f} F1@5={2*p*r/(p+r):.3f}")
ap = sum((sum(hits[:i+1]) / (i+1)) for i, h in enumerate(hits) if h) / len(rel)
print(f"AP={ap:.4f}  RR={1/(hits.index(True)+1):.2f}")
EOF
python src/toy_metrics.py 2>/dev/null
```

Expected:

```text
BLEU-4 (no smoothing): 0.0
BLEU   (smoothed)    : 0.38
rouge1: P=0.833 R=0.833 F1=0.833
rouge2: P=0.600 R=0.600 F1=0.600
rougeL: P=0.833 R=0.833 F1=0.833
P@5=0.40 R@5=0.50 F1@5=0.444
AP=0.5417  RR=1.00
```

Check by hand: reference `the cat sat on the mat`, hypothesis `the cat is on the mat`.

- Unigram matches 5/6, bigram 3/5, trigram 1/4, 4-gram 0/3. Unsmoothed BLEU-4 is **0** because one n-gram precision is zero; smoothing fixes this for short sentences.
- ROUGE-1 recall = 5/6, ROUGE-2 recall = 3/5, ROUGE-L uses the longest common subsequence (`the cat on the mat`, length 5).
- The last two lines reproduce the ranking example from the slides: P@5 = 0.40, R@5 = 0.50, AP = 0.5417.

### 3.2 Build a second "generator" for comparison (2 min)

An **extractive baseline** that just returns the top-1 retrieved chunk verbatim:

```bash
cat > src/generate_baseline.py <<'EOF'
"""Extractive baseline 'generator': answer = top-1 retrieved chunk, verbatim."""
import json, sys
run_path = sys.argv[1] if len(sys.argv) > 1 else "runs/tfidf.json"
corpus = {c["id"]: c["text"] for c in json.load(open("data/corpus.json"))}
run = json.load(open(run_path))
out = {qid: corpus[ranked[0]["id"]] for qid, ranked in run.items()}
json.dump(out, open("runs/generated_baseline.json", "w"), indent=1)
print("wrote runs/generated_baseline.json")
EOF
python src/generate_baseline.py runs/tfidf.json
```

### 3.3 Score both systems (`src/eval_generation.py`)

```bash
cat > src/eval_generation.py <<'EOF'
"""Sentence-level BLEU (sacrebleu, smoothed) and ROUGE-1/2/L (rouge-score)."""
import argparse, json
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/generated_llm.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    refs = json.load(open("data/references.json"))
    gens = json.load(open(a.gen))
    bleu = BLEU(effective_order=True)            # smoothing for short sentences
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    tot = {"bleu": 0, "r1": 0, "r2": 0, "rl": 0}
    for qid, ref in refs.items():
        hyp = gens[qid]
        b = bleu.sentence_score(hyp, [ref]).score / 100
        r = sc.score(ref, hyp)                  # (target, prediction)
        row = {"bleu": b, "r1": r["rouge1"].fmeasure,
               "r2": r["rouge2"].fmeasure, "rl": r["rougeL"].fmeasure}
        for k in tot: tot[k] += row[k]
        if a.verbose:
            print(f"{qid}  BLEU={b:.3f}  R1={row['r1']:.3f}  R2={row['r2']:.3f}  RL={row['rl']:.3f}")
    n = len(refs)
    print(f"\ngen={a.gen}  n={n}")
    print(f"  BLEU    = {tot['bleu']/n:.3f}")
    print(f"  ROUGE-1 = {tot['r1']/n:.3f}")
    print(f"  ROUGE-2 = {tot['r2']/n:.3f}")
    print(f"  ROUGE-L = {tot['rl']/n:.3f}")
EOF
```

```bash
python src/eval_generation.py --gen runs/generated_baseline.json 2>/dev/null
python src/eval_generation.py --gen data/generated_llm.json --verbose 2>/dev/null
```

Expected summary lines:

```text

gen=runs/generated_baseline.json  n=12
  BLEU    = 0.104
  ROUGE-1 = 0.488
  ROUGE-2 = 0.293
  ROUGE-L = 0.437

gen=data/generated_llm.json  n=12
  BLEU    = 0.171
  ROUGE-1 = 0.603
  ROUGE-2 = 0.270
  ROUGE-L = 0.500
```

(If you see a sacrebleu warning about `effective_order`, it is harmless; the script already enables it.)

### 3.4 Interpret (8 min): the important lesson

| System | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| Extractive baseline (copy top chunk) | 0.104 | 0.488 | **0.293** | 0.437 |
| LLM-style generator | **0.171** | **0.603** | 0.270 | **0.500** |

Discuss:

1. The copy-paste baseline *beats* the LLM-style answers on ROUGE-2. Is it actually the better answer? Read both for `q07` and `q11`. Overlap rewards copying wording, not answering the question.
2. Find a query where the LLM answer is **correct but paraphrased** (e.g. `q03`, `q11`) and observe a low BLEU. Metrics measure surface overlap, not meaning.
3. Which of the following would these metrics **not** catch? A fluent answer that contradicts the retrieved chunk (hallucination); an answer that is correct but too long; an answer that omits the key step.
4. Which complementary methods would you add? (Manual rubric, semantic similarity such as BERTScore, LLM-as-judge, groundedness/faithfulness checks.)

### 3.5 Exercises (pick one)

- **A.** Edit one answer in `data/generated_llm.json` to be **wrong but lexically similar** (e.g. `kill -9` replaced by `kill -1`) and see how little the scores move.
- **B.** Write a deliberately **good paraphrase** with no shared words and observe the drop.
- **C.** Combine pipelines: generate the baseline from `runs/embed.json` and see how retrieval quality propagates into generation scores.
- **D (stretch).** Replace the hand-written answers with a real LLM (see the optional appendix) and re-score.

---

## Wrap-up (`T+0:45`)

You computed, on one dataset: Precision@k, Recall@k, F1@k, MAP, MRR for retrieval, and BLEU, ROUGE-1/2/L for generation. Key takeaways:

1. Report **retrieval and generation separately**; a bad answer may be a retrieval failure or a generation failure.
2. Precision@k depends on how many relevant items exist; always state k.
3. BLEU/ROUGE are cheap, reproducible and weak: use them as regression signals, not as the final word.

---

## Appendix A: Folder layout when finished

```text
rag-eval-workshop/
├── data/ (corpus.json, queries.json, references.json, generated_llm.json)
├── src/  (retrieve.py, eval_retrieval.py, generate_baseline.py, eval_generation.py, toy_metrics.py)
└── runs/ (tfidf.json, embed.json, generated_baseline.json)
```

## Appendix B: Optional real LLM generator with Ollama (local, no API key)

Needs ~8 GB RAM for a small model. Skip if the VM is small.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
pip install requests
```

```bash
cat > src/generate_ollama.py <<'EOF'
import json, sys, requests
run = json.load(open("runs/tfidf.json"))
corpus = {c["id"]: c["text"] for c in json.load(open("data/corpus.json"))}
queries = {q["qid"]: q["query"] for q in json.load(open("data/queries.json"))}
out = {}
for qid, ranked in run.items():
    ctx = "\n".join(corpus[r["id"]] for r in ranked[:3])
    prompt = (f"Answer in one or two sentences using ONLY the context.\n\n"
              f"Context:\n{ctx}\n\nQuestion: {queries[qid]}\nAnswer:")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "llama3.2:3b", "prompt": prompt, "stream": False})
    out[qid] = r.json()["response"].strip()
json.dump(out, open("runs/generated_ollama.json", "w"), indent=1)
print("wrote runs/generated_ollama.json")
EOF
python src/generate_ollama.py
python src/eval_generation.py --gen runs/generated_ollama.json
```

## Appendix C: Troubleshooting

| Symptom | Fix |
|---|---|
| `python3 -m venv` fails | `sudo apt install -y python3-venv` |
| `ModuleNotFoundError` | Run `source .venv/bin/activate` in the current terminal |
| `pip` timeouts | Retry, or use a mirror; check VM proxy settings |
| Embedding download fails | Needs access to huggingface.co; continue with `--method tfidf` |
| Out of memory with `sentence-transformers` | Skip Block 2.4; the rest does not need it |
| Scores differ slightly from this page | Library versions and (for embeddings) the model differ; TF-IDF numbers should match closely |

## Appendix D: Cleanup

```bash
deactivate
rm -rf ~/rag-eval-workshop
```
