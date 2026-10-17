# Workshop 1 · Search Systems and Evaluation Metrics

**Part A (30 min) + Part B (45 min)** · Linux VM · Python 3.9+

> Everything is in this one file: setup, data, scripts and exercises. Do Part A, then Part B.

## Setup (5 min, do once)

**Step 1: create the environment**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn sacrebleu rouge-score
# optional, heavier: pip install sentence-transformers
```

**Step 2: create the data and scripts.** Paste this whole block into your terminal. It creates the folder `rag-eval-workshop/` with every file the exercises use. Run all later commands from inside that folder.

```bash
mkdir -p rag-eval-workshop/data rag-eval-workshop/src rag-eval-workshop/runs
cd rag-eval-workshop

cat > data/corpus.json <<'EOF'
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

cat > data/queries.json <<'EOF'
[
{"id":"q01","text":"how do I log in to a remote server securely","relevant":["c01"],"intent":"info"},
{"id":"q02","text":"make a script executable","relevant":["c02"],"intent":"task"},
{"id":"q03","text":"change the owner of a folder recursively","relevant":["c03"],"intent":"task"},
{"id":"q04","text":"how much free disk space is left","relevant":["c04"],"intent":"info"},
{"id":"q05","text":"stop a process that is frozen","relevant":["c05"],"intent":"task"},
{"id":"q06","text":"start nginx when the machine boots","relevant":["c06"],"intent":"task"},
{"id":"q07","text":"run a script every night","relevant":["c08"],"intent":"task"},
{"id":"q08","text":"open port 22 on the firewall","relevant":["c09"],"intent":"task"},
{"id":"q09","text":"nginx service logs","relevant":["c10"],"intent":"nav"},
{"id":"q10","text":"give a user admin rights","relevant":["c11"],"intent":"task"},
{"id":"q11","text":"compress a folder into one file","relevant":["c12"],"intent":"task"},
{"id":"q12","text":"server killed my programs because RAM is full","relevant":["c14","c05"],"intent":"info"}
]
EOF

cat > data/references.json <<'EOF'
{
"q01":"Use ssh user@host to log in to a remote server securely, and prefer key pairs over passwords.",
"q02":"Run chmod +x on the script to make it executable.",
"q03":"Use chown -R user:group on the folder to change its owner recursively.",
"q04":"Run df -h to see free disk space per filesystem.",
"q05":"Find its PID with ps aux and run kill -9 PID to force it to stop.",
"q06":"Run systemctl enable nginx so that it starts at boot.",
"q07":"Add a cron job with crontab -e, for example 30 2 * * * /opt/backup.sh to run it every day at 02:30.",
"q08":"Run ufw allow 22/tcp to open port 22 on the firewall.",
"q09":"Use journalctl -u nginx to read the nginx service logs.",
"q10":"Run usermod -aG sudo alice to give the user admin rights.",
"q11":"Use tar -czf backup.tar.gz folder to compress a folder into one archive file.",
"q12":"The kernel kills processes when memory runs out. Check free -h and add swap space to reduce the risk."
}
EOF

cat > data/generated_llm.json <<'EOF'
{
"q01":"You can securely connect to a remote machine with the ssh command, and it is safer to use key pairs than passwords.",
"q02":"Give the file execute permission with chmod +x.",
"q03":"Apply chown with the -R flag and a user:group value to change ownership of the whole directory.",
"q04":"The df -h command reports how much disk space is free on each filesystem.",
"q05":"Look up the process ID using ps aux, then use kill -9 on that ID to terminate it immediately.",
"q06":"Use systemctl enable nginx to make the service launch automatically at boot.",
"q07":"Schedule it with cron: open crontab -e and add a line such as 30 2 * * * /opt/backup.sh, which executes daily at 02:30.",
"q08":"Allow the SSH port through the firewall with ufw allow 22/tcp.",
"q09":"Run journalctl -u nginx to view the logs of the nginx service.",
"q10":"Add the user to the sudo group with usermod -aG sudo alice.",
"q11":"Create a compressed archive with tar -czf backup.tar.gz followed by the folder name.",
"q12":"When RAM runs out, the kernel can kill running programs. Check usage with free -h and configure swap space to lower the risk."
}
EOF

cat > src/common.py <<'EOF'
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(path):
    p = path if os.path.isabs(path) or os.path.exists(path) else os.path.join(ROOT, path)
    with open(p) as f:
        return json.load(f)
def save(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
def corpus():  return load("data/corpus.json")
def queries(): return load("data/queries.json")
EOF

cat > src/retrieve.py <<'EOF'
"""Rank every chunk for every query.  Methods: tfidf | lsa | embed | hybrid"""
import argparse, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from common import corpus, queries, save, ROOT

def tfidf_scores(docs, qs):
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    D = v.fit_transform(docs); Q = v.transform(qs)
    return (Q @ D.T).toarray()

def lsa_scores(docs, qs, dim=8):
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    D = v.fit_transform(docs); Q = v.transform(qs)
    svd = TruncatedSVD(n_components=dim, random_state=0).fit(D)
    d = normalize(svd.transform(D)); q = normalize(svd.transform(Q))
    return q @ d.T

def embed_scores(docs, qs):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Install first:  pip install sentence-transformers")
    m = SentenceTransformer("all-MiniLM-L6-v2")
    d = m.encode(docs, normalize_embeddings=True); q = m.encode(qs, normalize_embeddings=True)
    return q @ d.T

def rrf(score_mats, k=60):
    out = np.zeros_like(score_mats[0])
    for S in score_mats:
        ranks = (-S).argsort(axis=1).argsort(axis=1) + 1
        out += 1.0 / (k + ranks)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tfidf", choices=["tfidf", "lsa", "embed", "hybrid"])
    ap.add_argument("--dense", default="lsa", choices=["lsa", "embed"], help="dense part of hybrid")
    a = ap.parse_args()
    C, Q = corpus(), queries()
    docs = [c["text"] for c in C]; qs = [q["text"] for q in Q]
    if a.method == "tfidf":  S = tfidf_scores(docs, qs)
    elif a.method == "lsa":  S = lsa_scores(docs, qs)
    elif a.method == "embed": S = embed_scores(docs, qs)
    else:
        dense = lsa_scores(docs, qs) if a.dense == "lsa" else embed_scores(docs, qs)
        S = rrf([tfidf_scores(docs, qs), dense])
    run = {}
    for i, q in enumerate(Q):
        order = np.argsort(-S[i], kind="stable")
        run[q["id"]] = [[C[j]["id"], round(float(S[i][j]), 4)] for j in order]
    out = os.path.join(ROOT, "runs", f"{a.method}.json")
    save(run, out); print("saved", out)
EOF

cat > src/eval_retrieval.py <<'EOF'
"""Score a ranking: P@k, R@k, F1@k, AP, RR.   --verbose per query, --by-intent slices."""
import argparse
from collections import defaultdict
from common import load, queries

def metrics(ranking, relevant, k):
    ids = [r[0] for r in ranking]
    top = ids[:k]; hits = sum(1 for d in top if d in relevant)
    p = hits / k; r = hits / len(relevant)
    f1 = 0 if hits == 0 else 2 * p * r / (p + r)
    ap, seen = 0.0, 0
    for i, d in enumerate(ids, 1):
        if d in relevant:
            seen += 1; ap += seen / i
    ap /= len(relevant)
    rr = next((1 / i for i, d in enumerate(ids, 1) if d in relevant), 0.0)
    return dict(P=p, R=r, F1=f1, AP=ap, RR=rr)

def mean(rows): return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--verbose", action="store_true"); ap.add_argument("--by-intent", action="store_true")
    a = ap.parse_args()
    run, Q = load(a.run), queries()
    rows, per = [], {}
    for q in Q:
        per[q["id"]] = metrics(run[q["id"]], set(q["relevant"]), a.k); rows.append(per[q["id"]])
        if a.verbose:
            top = [d for d, _ in run[q["id"]][:a.k]]
            m = per[q["id"]]
            print(f'{q["id"]}  rel={q["relevant"]}  top{a.k}={top}  P={m["P"]:.2f} R={m["R"]:.2f} AP={m["AP"]:.3f} RR={m["RR"]:.2f}')
    m = mean(rows)
    print(f'\nk={a.k}  Precision@k={m["P"]:.3f}  Recall@k={m["R"]:.3f}  F1@k={m["F1"]:.3f}  MAP={m["AP"]:.3f}  MRR={m["RR"]:.3f}')
    if a.by_intent:
        g = defaultdict(list)
        for q in Q: g[q["intent"]].append(per[q["id"]])
        print("\nBy intent:")
        for intent, rs in g.items():
            mm = mean(rs); print(f'  {intent:5s} n={len(rs):2d}  P@{a.k}={mm["P"]:.3f}  R@{a.k}={mm["R"]:.3f}  MAP={mm["AP"]:.3f}  MRR={mm["RR"]:.3f}')
EOF

cat > src/generate_baseline.py <<'EOF'
"""Extractive baseline: the answer is the top retrieved chunk, copied."""
import sys, os
from common import load, corpus, save, ROOT
run = load(sys.argv[1]); text = {c["id"]: c["text"] for c in corpus()}
out = {qid: text[r[0][0]] for qid, r in run.items()}
p = os.path.join(ROOT, "runs", "generated_baseline.json"); save(out, p); print("saved", p)
EOF

cat > src/eval_generation.py <<'EOF'
"""BLEU (sentence-level, smoothed, averaged) and ROUGE-1/2/L F1 against references."""
import argparse
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from common import load

def score_all(gen, ref):
    bleu = BLEU(effective_order=True)
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rows = {}
    for q in ref:
        r = sc.score(ref[q], gen[q])
        rows[q] = dict(BLEU=bleu.sentence_score(gen[q], [ref[q]]).score / 100,
                       R1=r["rouge1"].fmeasure, R2=r["rouge2"].fmeasure, RL=r["rougeL"].fmeasure)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True)
    ap.add_argument("--ref", default="data/references.json"); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rows = score_all(load(a.gen), load(a.ref))
    if a.verbose:
        for q, m in rows.items(): print(q, " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    avg = {k: sum(r[k] for r in rows.values()) / len(rows) for k in ["BLEU", "R1", "R2", "RL"]}
    print(f'BLEU={avg["BLEU"]:.3f}  ROUGE-1={avg["R1"]:.3f}  ROUGE-2={avg["R2"]:.3f}  ROUGE-L={avg["RL"]:.3f}')
EOF

cat > src/toy_metrics.py <<'EOF'
"""The cat-and-mat example from the slides."""
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
ref, gen = "the cat sat on the mat", "the cat is on the mat"
print("reference:", ref); print("generated:", gen)
print("BLEU-4 no smoothing :", round(BLEU(smooth_method="none").sentence_score(gen, [ref]).score / 100, 3))
print("BLEU-4 smoothed     :", round(BLEU().sentence_score(gen, [ref]).score / 100, 3))
sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
for k, v in sc.score(ref, gen).items():
    print(f"{k:7s} P={v.precision:.3f} R={v.recall:.3f} F1={v.fmeasure:.3f}")
EOF

cat > data/key_facts.json <<'EOF'
{"q01":"ssh","q02":"chmod +x","q03":"-R","q04":"df -h","q05":"kill -9","q06":"systemctl enable nginx","q07":"crontab -e","q08":"ufw allow 22/tcp","q09":"journalctl -u nginx","q10":"usermod -aG sudo","q11":"tar -czf","q12":"free -h"}
EOF

cat > data/ratings.json <<'EOF'
{"items":20,
"rater_a":[1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,0,1],
"rater_b":[1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,0,1]}
EOF

cat > src/rag_prompt.py <<'EOF'
"""Build the RAG prompt for one query:  python src/rag_prompt.py runs/tfidf.json q07 --k 3"""
import argparse
from common import load, corpus, queries
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("qid"); ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()
text = {c["id"]: c["text"] for c in corpus()}; q = {x["id"]: x for x in queries()}[a.qid]
ctx = "\n".join(f"[{i}] ({d}) {text[d]}" for i, (d, _) in enumerate(load(a.run)[a.qid][:a.k], 1))
print(f"""You are a Linux assistant. Use ONLY the context below.
Cite sources like [1]. If the context is not enough, say "not found".

Context:
{ctx}

Question: {q["text"]}
Answer:""")
EOF

cat > src/diagnose.py <<'EOF'
"""Was the evidence retrieved?  No -> fix retrieval.  Yes -> fix generation.
An answer is "ok" if it contains the key fact (data/key_facts.json).
python src/diagnose.py --run runs/tfidf.json --gen data/generated_llm.json --k 3"""
import argparse
from common import load, queries
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--gen", required=True)
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()
run, gen, key = load(a.run), load(a.gen), load("data/key_facts.json")
counts = {"OK": 0, "FIX RETRIEVAL": 0, "FIX GENERATION": 0, "LUCKY (answer ok, evidence missed)": 0}
for q in queries():
    got = any(d in q["relevant"] for d, _ in run[q["id"]][:a.k])
    good = key[q["id"]].lower() in gen[q["id"]].lower()
    verdict = "OK" if got and good else "FIX GENERATION" if got else "FIX RETRIEVAL" if not good else "LUCKY (answer ok, evidence missed)"
    counts[verdict] += 1; print(f'{q["id"]}  evidence_in_top{a.k}={str(got):5s}  answer_ok={str(good):5s}  -> {verdict}')
print("\n", counts)
EOF

cat > src/chunk_demo.py <<'EOF'
"""Chunk size and overlap: where does a fact get cut?  python src/chunk_demo.py --size 120 --overlap 0"""
import argparse
DOC = ("To back up a folder, first create an archive with tar -czf backup.tar.gz folder. "
       "Then copy the archive to another machine with scp backup.tar.gz user@host:/backups. "
       "Finally schedule the whole job with cron by running crontab -e and adding a line such as "
       "30 2 * * * /opt/backup.sh so that it runs every night at 02:30.")
ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=120); ap.add_argument("--overlap", type=int, default=0)
a = ap.parse_args(); step = a.size - a.overlap
chunks = [DOC[i:i + a.size] for i in range(0, len(DOC), step)]
for i, c in enumerate(chunks, 1): print(f"chunk {i} ({len(c)} chars): {c!r}")
for fact in ["tar -czf backup.tar.gz", "crontab -e", "30 2 * * * /opt/backup.sh"]:
    print(f"{fact!r:32s} intact in one chunk? {any(fact in c for c in chunks)}")
EOF

cat > src/metrics_playground.py <<'EOF'
"""Edit RANKING / TOTAL_RELEVANT and re-run.  1 = relevant result, 0 = not relevant (rank 1 on the left)."""
import math
RANKING = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]   # relevant at ranks 1, 3, 6
TOTAL_RELEVANT = 4                           # one relevant chunk was never retrieved
GRADES = [3, 0, 2, 0, 0, 1, 0, 0, 0, 0]      # for nDCG (graded relevance 0-3)

print(" k   P@k   R@k   F1@k")
for k in range(1, len(RANKING) + 1):
    h = sum(RANKING[:k]); p, r = h / k, h / TOTAL_RELEVANT
    print(f"{k:2d}  {p:.2f}  {r:.2f}  {0 if h == 0 else 2*p*r/(p+r):.3f}")
ap_, seen = 0.0, 0
for i, x in enumerate(RANKING, 1):
    if x: seen += 1; ap_ += seen / i
print("\nAverage Precision :", round(ap_ / TOTAL_RELEVANT, 3))
print("Reciprocal Rank   :", round(next((1 / i for i, x in enumerate(RANKING, 1) if x), 0), 3))
dcg = lambda g: sum((2**x - 1) / math.log2(i + 1) for i, x in enumerate(g, 1))
print("nDCG@10           :", round(dcg(GRADES) / dcg(sorted(GRADES, reverse=True)), 3))
EOF

cat > src/kappa.py <<'EOF'
"""Cohen's kappa for two raters."""
from common import load
d = load("data/ratings.json"); a, b = d["rater_a"], d["rater_b"]; n = len(a)
po = sum(x == y for x, y in zip(a, b)) / n
pa1, pb1 = sum(a) / n, sum(b) / n
pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
print(f"items={n}  observed agreement={po:.2f}  chance agreement={pe:.2f}  kappa={(po - pe) / (1 - pe):.3f}")
print("disagreements at items:", [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y])
EOF

```

**Checkpoint:** `python src/retrieve.py --method tfidf` prints `saved .../runs/tfidf.json`.

> **Numbers:** expected outputs belong to this dataset (15 chunks, 12 queries). Match the pattern, not necessarily every digit if you change the data.

---

# Part A · Search Systems Refresher

**30 min**

**You will:** see the search pipeline run end to end, compare lexical and semantic retrieval, test how chunking cuts facts, build a RAG prompt, and diagnose whether a wrong answer is a retrieval or a generation problem.

---

## 1. Look at the data (3 min)

Open `data/corpus.json` (15 Linux chunks) and `data/queries.json` (12 queries, each with the relevant chunk ids).

**Task 1.1** For `q07` ("run a script every night") which chunk is relevant? Does the query share any word with it?

**Task 1.2** Which query has **two** relevant chunks, and why?

<details><summary>Answers</summary>
q07 → c08 (cron). It shares no content word with the chunk: "run a script every night" vs "cron runs scheduled jobs". q12 has two relevant chunks (c14 memory, c05 processes), because both explain what happens when RAM is full.
</details>

---

## 2. The pipeline: query → retrieve → rank (5 min)

```bash
python src/retrieve.py --method tfidf
python src/eval_retrieval.py --run runs/tfidf.json --k 3 --verbose
```

Each line shows the query, its relevant chunk, and the top three results.

**Task 2.1** Find the queries where the relevant chunk is **not** at rank 1.

**Expected:** `q07` (relevant chunk at rank 3) and `q11` (not in the top 3 at all).

**Task 2.2** Name the pipeline stage that failed for `q11`: understanding, retrieval, ranking or presentation? (Hint: is the relevant chunk anywhere in the ranking?)

```bash
python3 -c "
import json; r=json.load(open('runs/tfidf.json'))['q11']; print([d for d,_ in r])"
```

<details><summary>Answer</summary>
Chunk c12 is at rank 5, so it was retrieved but ranked too low: a ranking/precision problem, not a recall one. The query says "compress" and the chunk says "compressed"; TF-IDF without stemming treats them as different words.
</details>

---

## 3. Lexical vs semantic (7 min)

`tfidf` counts shared words. `lsa` compresses the same word statistics into 8 dense dimensions, a small stand-in for embeddings that needs no download.

```bash
python src/retrieve.py --method lsa
python src/eval_retrieval.py --run runs/lsa.json --k 3 --verbose | grep -v "AP=1.000"
```

**Task 3.1** Compare `tfidf` and `lsa` for `q07`:

| | rank 1 | rank 2 | rank 3 |
|---|---|---|---|
| tfidf | c07 | c02 | **c08** |
| lsa | c07 | **c08** | c02 |

Dense retrieval moved the cron chunk up one place, but neither method put it first. Why?

<details><summary>Answer</summary>
The query and chunk share almost no words. With only 15 tiny chunks, LSA has little co-occurrence data to learn "script" ≈ "cron". A real embedding model trained on large text does better. That is the case for semantic search.
</details>

**Task 3.2 (optional, needs `pip install sentence-transformers`)**

```bash
python src/retrieve.py --method embed
python src/eval_retrieval.py --run runs/embed.json --k 3 --verbose | grep -v "AP=1.000"
```

Does `q07` improve? Which queries get worse?

**Task 3.3 Cosine similarity by hand.** Two tiny vectors:

```bash
python3 -c "
import numpy as np
a=np.array([1,2,0]); b=np.array([2,1,1])
print(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))"
```

Expected: `0.7302...`. Change `b` to `[2,4,0]`. What happens, and why? (Answer: 1.0. Same direction, different length. Cosine compares direction only.)

**Task 3.4 Hybrid.** Fuse both rankings with reciprocal rank fusion:

```bash
python src/retrieve.py --method hybrid
python src/eval_retrieval.py --run runs/hybrid.json --k 3
```

Expected: `Recall@k=0.917`, `MAP=0.822`.

---

## 4. Chunking: where does a fact get cut? (3 min)

```bash
python src/chunk_demo.py --size 60 --overlap 0
python src/chunk_demo.py --size 60 --overlap 25
```

**Task 4.1** Which fact is split with no overlap? **Expected:** `tar -czf backup.tar.gz` (intact in one chunk? `False`).

**Task 4.2** Does overlap 25 fix it? **Expected:** yes, `True` for all three.

**Task 4.3** Try `--size 300`. What is the downside of one big chunk? (Answer: the chunk covers several topics, so its embedding or word counts are diluted and it matches many queries weakly.)

---

## 5. Build the RAG prompt (3 min)

```bash
python src/rag_prompt.py runs/tfidf.json q07 --k 3
```

You see the three retrieved chunks plus rules: use only the context, cite, say "not found" if unsure.

**Task 5.1** Is the evidence for `q07` inside the prompt? Which citation number? **Expected:** yes, `[3] (c08)`.

**Task 5.2** Re-run with `--k 1`. What does the model see now? (Answer: only the apt chunk. The answer cannot be found, so a good model should say "not found".)

---

## 6. Diagnose: was the evidence retrieved? (4 min)

```bash
python src/generate_baseline.py runs/tfidf.json
python src/diagnose.py --run runs/tfidf.json --gen runs/generated_baseline.json --k 3 | grep -v -e "-> OK"
```

The baseline copies the **top-1** chunk as its answer. An answer counts as correct if it contains the key command (`data/key_facts.json`).

**Expected:**

```
q07  evidence_in_top3=True   answer_ok=False  -> FIX GENERATION
q11  evidence_in_top3=False  answer_ok=False  -> FIX RETRIEVAL
```

**Task 6.1** For `q11` which side do you fix? (Retrieval: the evidence never reached the model.)

**Task 6.2** `q07` says "fix generation", but the evidence was at rank 3 and the baseline only used rank 1. What is the real fix? (Answer: ranking. The baseline throws away rank 2–3. This shows why labels like "retrieval" and "generation" are a first diagnosis, not the final one.)

**Task 6.3** Now diagnose answers that contain a wrong fact:

```bash
python src/diagnose.py --run runs/tfidf.json --gen data/generated_flawed.json | grep -v -e "-> OK"
```

**Expected:** `q05 ... -> FIX GENERATION` (it says `kill -1` instead of `kill -9`) and `q11 ... -> LUCKY`: the answer is right although the evidence was not retrieved. The model answered from memory. Is that safe?

---

## 7. Wrap-up questions (discuss, 2 min)

1. Which of the five pipeline stages has the most failures in this dataset?
2. When would you choose lexical over semantic retrieval?
3. If you could change one of the six RAG knobs (chunking, embedding, k, prompt, generator, guardrails), which would help `q07` most?

**Next:** Part B computes the metrics (P, R, F1, AP, MRR, BLEU, ROUGE) that put numbers on what you just saw.

---

# Part B · Evaluation Metrics by Hand and by Code

**45 min**

**You will:** compute Precision, Recall, F1, AP, MRR and nDCG on the running example, watch how they react to k, measure rater agreement with Cohen's kappa, and compute BLEU and ROUGE on the cat-and-mat pair.

---

## 1. The running example (10 min)

Open `src/metrics_playground.py`. At the top:

```python
RANKING = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]   # relevant at ranks 1, 3, 6
TOTAL_RELEVANT = 4                           # one relevant chunk was never retrieved
```

This is the ranked list from the slides: 1 means relevant.

**Step 1: compute by hand first.** On paper, for k = 5:

| | Your answer |
|---|---|
| Relevant in top 5 | |
| Precision@5 | |
| Recall@5 | |
| F1@5 | |

**Step 2: run it.**

```bash
python src/metrics_playground.py
```

**Expected output:**

```
 k   P@k   R@k   F1@k
 1  1.00  0.25  0.400
 2  0.50  0.25  0.333
 3  0.67  0.50  0.571
 4  0.50  0.50  0.500
 5  0.40  0.50  0.444
 6  0.50  0.75  0.600
 7  0.43  0.75  0.545
 8  0.38  0.75  0.500
 9  0.33  0.75  0.462
10  0.30  0.75  0.429

Average Precision : 0.542
Reciprocal Rank   : 1.0
nDCG@10           : 0.943
```

Did your hand calculation match row k = 5? (P = 2/5 = 0.40, R = 2/4 = 0.50, F1 = 0.444.)

**Task 1.1** Recall never goes down as k grows. Precision does. Find the k where F1 peaks. **Answer:** k = 6 (0.600).

**Task 1.2** AP by hand: precision at each relevant rank is 1/1, 2/3 and 3/6. Add them and divide by **4**. Why 4 and not 3? (Answer: the missing relevant chunk must count against you.)

---

## 2. Break it on purpose (10 min)

Edit `RANKING` and re-run for each case. Predict first, then check.

| # | Change | Predict AP | Predict RR |
|---|---|---|---|
| A | `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`, `TOTAL_RELEVANT = 1` | | |
| B | `[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]`, `TOTAL_RELEVANT = 4` | | |
| C | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`, `TOTAL_RELEVANT = 4` | | |
| D | `[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]`, `TOTAL_RELEVANT = 4` | | |

<details><summary>Answers</summary>

- **A** AP = 0.25 and RR = 0.25. One relevant item at rank 4.
- **B** AP = 1.0, RR = 1.0. A perfect ranking.
- **C** AP = 0, RR = 0. Nothing found. Check that F1 prints `0.000` (no divide-by-zero).
- **D** Same four relevant items as B, but late. AP is 0.365 and RR is 0.20, against 1.0 and 1.0 for B, while Recall@10 is identical (1.0 for both). This is why recall alone is not enough.
</details>

**Task 2.1** Use graded labels for nDCG. Set `GRADES = [0, 0, 2, 0, 0, 3, 0, 0, 0, 0]` (the best chunk is at rank 6) and re-run. Expected nDCG: 0.449, against 0.943 for the original. Why is it so much lower? (Answer: the most relevant item sits low and is discounted.)

---

## 3. Metrics on a real system (10 min)

```bash
python src/retrieve.py --method tfidf
python src/retrieve.py --method lsa
python src/retrieve.py --method hybrid
for k in 1 3 5; do for m in tfidf lsa hybrid; do echo -n "k=$k $m: "; python src/eval_retrieval.py --run runs/$m.json --k $k | tail -1; done; done
```

**Expected:**

| k | Method | P@k | R@k | F1@k | MAP | MRR |
|---|---|---|---|---|---|---|
| 1 | tfidf | 0.833 | 0.792 | 0.806 | 0.850 | 0.878 |
| 1 | lsa | 0.667 | 0.625 | 0.639 | 0.808 | 0.808 |
| 1 | hybrid | 0.750 | 0.708 | 0.722 | 0.822 | 0.836 |
| 3 | tfidf | 0.306 | 0.875 | 0.450 | 0.850 | 0.878 |
| 3 | lsa | 0.333 | 0.917 | 0.483 | 0.808 | 0.808 |
| 3 | hybrid | 0.333 | 0.917 | 0.483 | 0.822 | 0.836 |
| 5 | tfidf | 0.200 | 0.958 | 0.329 | 0.850 | 0.878 |
| 5 | lsa | 0.217 | 1.000 | 0.353 | 0.808 | 0.808 |
| 5 | hybrid | 0.217 | 1.000 | 0.353 | 0.822 | 0.836 |

(MAP and MRR use the whole ranking, so they do not change with k.)

**Task 3.1** At k = 5, `lsa` has perfect recall and `tfidf` does not. At k = 1, `tfidf` clearly wins. Which method would you choose for a RAG system that passes **3 chunks** to the LLM? Why?

**Task 3.2** Why is Precision@3 so low (≈ 0.3) for every method although the systems look good? **Answer:** most queries have exactly **one** relevant chunk, so P@3 can never exceed 1/3. Precision depends on how many relevant items exist.

**Task 3.3** Run per query and find the query where `tfidf` has AP of 0.2:

```bash
python src/eval_retrieval.py --run runs/tfidf.json --k 3 --verbose | grep "AP=0.2"
```

**Expected:** `q11 ... AP=0.200 RR=0.20` (relevant chunk at rank 5).

---

## 4. Human labels: agreement (5 min)

Two raters judged 20 chunks (1 = relevant). The labels are in `data/ratings.json`.

```bash
python src/kappa.py
```

**Expected:**

```
items=20  observed agreement=0.80  chance agreement=0.52  kappa=0.583
disagreements at items: [4, 9, 14, 16]
```

**Task 4.1** 80% agreement sounds high. Why is kappa only 0.58? (Answer: raters agree by chance about 52% of the time. Kappa removes that.)

**Task 4.2** Open `data/ratings.json` and change rater B so that it agrees on items 4, 9, 14 and 16. What is kappa now? (Answer: 1.0.) What would you do in a real project with the four disagreeing items? (Answer: discuss, update the guidelines, add a third rater.)

---

## 5. BLEU and ROUGE on cat and mat (7 min)

```bash
python src/toy_metrics.py 2>/dev/null
```

**Expected:**

```
reference: the cat sat on the mat
generated: the cat is on the mat
BLEU-4 no smoothing : 0.0
BLEU-4 smoothed     : 0.38
rouge1  P=0.833 R=0.833 F1=0.833
rouge2  P=0.600 R=0.600 F1=0.600
rougeL  P=0.833 R=0.833 F1=0.833
```

**Task 5.1** By hand, list the 2-grams of the generated sentence and mark which appear in the reference. You should find 3 of 5, which matches ROUGE-2 precision of 0.600.

**Task 5.2** Edit `src/toy_metrics.py`. Change `gen` and predict the effect before running:

| gen | Prediction | Result |
|---|---|---|
| `"the cat sat on the mat"` | everything 1.0 | BLEU 1.0, all ROUGE 1.0 |
| `"the feline rested on the rug"` | same meaning, few shared words | BLEU 0.18, R-1 F1 0.50, R-2 F1 0.20 (only "the" and "on" match) |
| `"the cat"` | high precision, low recall | BLEU 0.0 (brevity penalty), R-1 P 1.0 but R 0.333 |
| `"the mat sat on the cat"` | same words, wrong meaning | BLEU 0.43, R-1 F1 1.0, R-2 F1 0.80, R-L F1 0.667 |

**Task 5.3** The last row has **perfect ROUGE-1** and a high ROUGE-2, but the wrong meaning. What does this tell you about overlap metrics?

---

## 6. Wrap-up (3 min)

Fill in the table from memory, then check the slides.

| Need | Metric |
|---|---|
| One right answer | |
| Everything relevant | |
| Clean browsable list | |
| Graded relevance | |
| Single tuning number | |

**Answers:** MRR or P@1 · Recall@k · P@k or MAP · nDCG · F1@k or MAP.

**Next:** Part C runs the full retrieval + generation evaluation and compares systems.
