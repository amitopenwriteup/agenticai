# Workshop 2 · Full Evaluation, Challenges and Trust

**Part C (45 min) + Part D (30 min)** · Linux VM · Python 3.9+

> Everything is in this one file: setup, data, scripts and exercises. Do Part C, then Part D.

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

cat > data/generated_flawed.json <<'EOF'
{
 "q01": "You can securely connect to a remote machine with the ssh command, and it is safer to use key pairs than passwords.",
 "q02": "Give the file execute permission with chmod +x.",
 "q03": "Apply chown with the -R flag and a user:group value to change ownership of the whole directory.",
 "q04": "The df -h command reports how much disk space is free on each filesystem.",
 "q05": "Look up the process ID using ps aux, then use kill -1 on that ID to terminate it immediately.",
 "q06": "Use systemctl enable nginx to make the service launch automatically at boot.",
 "q07": "Schedule it with cron: open crontab -e and add a line such as 30 2 * * * /opt/backup.sh, which executes daily at 02:30. You must also run systemctl restart cron before the job works.",
 "q08": "Allow the SSH port through the firewall with ufw allow 22/tcp.",
 "q09": "Run journalctl -u nginx --delete-all to view the logs of the nginx service.",
 "q10": "Add the user to the sudo group with usermod -aG sudo alice.",
 "q11": "Create a compressed archive with tar -czf backup.tar.gz followed by the folder name.",
 "q12": "When RAM runs out, the kernel can kill running programs. Check usage with free -h and configure swap space to lower the risk."
}
EOF

cat > data/ambiguous.json <<'EOF'
[
{"id":"a1","text":"backup","intents":{"schedule a backup":["c08"],"create a backup archive":["c12"]}},
{"id":"a2","text":"memory","intents":{"check free memory":["c14"],"find what uses memory":["c05"]}},
{"id":"a3","text":"port","intents":{"open a firewall port":["c09"],"list listening ports":["c13"]}}
]
EOF

cat > src/perturb.py <<'EOF'
"""Wrong but similar: change one fact, see how little the overlap scores move."""
import copy
from common import load
from eval_generation import score_all
gen, ref = load("data/generated_llm.json"), load("data/references.json")
bad = copy.deepcopy(gen); bad["q05"] = bad["q05"].replace("kill -9", "kill -1")
for name, g in [("original", gen), ("kill -9 -> kill -1 in q05", bad)]:
    r = score_all(g, ref); print(f"{name:28s} q05 R-L={r['q05']['RL']:.3f} BLEU={r['q05']['BLEU']:.3f} | all-queries R-L={sum(x['RL'] for x in r.values())/len(r):.3f}")
print("\nThe answer is now dangerous-wrong (signal 1 = hang up, not force kill) but the score barely moves.")
EOF

cat > src/ambiguity_eval.py <<'EOF'
"""Ambiguous queries: one label set per intent. Does the top-3 cover all intents?"""
import numpy as np
from common import load, corpus
from retrieve import tfidf_scores
C = corpus(); docs = [c["text"] for c in C]; A = load("data/ambiguous.json")
S = tfidf_scores(docs, [a["text"] for a in A])
for i, a in enumerate(A):
    top = [C[j]["id"] for j in np.argsort(-S[i], kind="stable")[:3]]
    print(f'\nquery "{a["text"]}"  top3={top}')
    covered = 0
    for intent, rel in a["intents"].items():
        hit = any(d in top for d in rel); covered += hit
        print(f'   intent "{intent}": relevant={rel}  covered={hit}')
    first = list(a["intents"].values())[0]
    print(f'   single-label recall (only first intent labelled) = {1.0 if any(d in top for d in first) else 0.0}   intent coverage = {covered}/{len(a["intents"])}')
EOF

cat > src/pooling_bias.py <<'EOF'
"""Judge only the old system's top-N (the pool). A different system is then scored on incomplete labels."""
import argparse
from common import load, queries
from eval_retrieval import metrics, mean
ap = argparse.ArgumentParser(); ap.add_argument("--old", default="runs/tfidf.json"); ap.add_argument("--new", default="runs/lsa.json")
ap.add_argument("--depth", type=int, default=2, help="how many results of the old system were judged"); ap.add_argument("--k", type=int, default=3)
a = ap.parse_args(); old, new, Q = load(a.old), load(a.new), queries()
full, pooled = [], []
for q in Q:
    rel = set(q["relevant"]); judged = {d for d, _ in old[q["id"]][:a.depth]}
    pooled_rel = rel & judged                      # unjudged results count as NOT relevant
    full.append(metrics(new[q["id"]], rel, a.k))
    pooled.append(metrics(new[q["id"]], pooled_rel, a.k) if pooled_rel else dict(P=0, R=0, F1=0, AP=0, RR=0))
    if rel != pooled_rel: print(f'{q["id"]}: relevant {sorted(rel)} but only {sorted(pooled_rel)} were ever judged')
f, p = mean(full), mean(pooled)
print(f'\nNew system with FULL labels  : R@{a.k}={f["R"]:.3f} MAP={f["AP"]:.3f} MRR={f["RR"]:.3f}')
print(f'New system with POOLED labels: R@{a.k}={p["R"]:.3f} MAP={p["AP"]:.3f} MRR={p["RR"]:.3f}')
EOF

cat > src/position_bias.py <<'EOF'
"""Simulate clicks: every result is EQUALLY relevant (p=0.5), users look less at lower ranks."""
import random
random.seed(7); N = 20000; K = 5; clicks = [0] * K
for _ in range(N):
    for r in range(K):
        examined = random.random() < 1 / (r + 1)          # examination falls with rank
        if examined and random.random() < 0.5: clicks[r] += 1
print("rank  CTR   (true relevance is identical for all ranks)")
for r, c in enumerate(clicks, 1): print(f"{r:4d}  {c / N:.3f}")
print("\nNaive reading: rank 1 is 'better'.  Truth: position bias.  Fix: randomize order when judging.")
EOF

cat > src/groundedness.py <<'EOF'
"""Crude groundedness check: every command, flag, path or number in the answer must appear in the retrieved top-k chunks.
Also: abstain ("not found") when the best retrieval score is below --abstain."""
import argparse, re
from common import load, corpus, queries
COMMANDS = set("ssh chmod chown df du ps top kill systemctl apt cron crontab ufw journalctl tail useradd passwd usermod tar ip ping ss free find grep nginx sudo".split())
def tokens(s): return [t.rstrip(".,:;") for t in re.findall(r"[a-z0-9\-\./\*:@_+]+", s.lower())]
def technical(t): return t in COMMANDS or bool(re.search(r"[0-9\-/\*:@+]", t)) and len(t) > 1
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/tfidf.json"); ap.add_argument("--gen", default="data/generated_flawed.json")
ap.add_argument("--k", type=int, default=3); ap.add_argument("--abstain", type=float, default=0.0, help="say 'not found' if top score is below this")
a = ap.parse_args(); run, gen = load(a.run), load(a.gen); text = {c["id"]: c["text"] for c in corpus()}
flagged = 0
for q in queries():
    top = run[q["id"]][:a.k]
    if top[0][1] < a.abstain: print(f'{q["id"]}  ABSTAIN  (top score {top[0][1]:.2f} < {a.abstain})'); continue
    ctx = set(tokens(" ".join(text[d] for d, _ in top)))
    bad = [t for t in tokens(gen[q["id"]]) if technical(t) and t not in ctx]
    flagged += bool(bad); print(f'{q["id"]}  {"FLAG" if bad else "ok  "}  unsupported={bad}')
print(f"\n{flagged} answers flagged")
EOF

```

**Checkpoint:** `python src/retrieve.py --method tfidf` prints `saved .../runs/tfidf.json`.

> **Numbers:** expected outputs belong to this dataset (15 chunks, 12 queries). Match the pattern, not necessarily every digit if you change the data.

---

# Part C · Full Evaluation: Retrieval and Generation

**45 min**

**You will:** run the complete evaluation loop on one small RAG system: rank, score retrieval, generate, score generation, then compare systems and stress-test the metrics.

**Time plan:** data check 10 min · retrieval evaluation 15 min · generation evaluation 15 min · exercises 5 min.

> **Note on numbers.** The expected values below belong to this kit's dataset (15 chunks, 12 queries). If you use a different dataset your numbers will differ. What must match is the *pattern*, not every digit.

---

## 1. The four data assets (10 min)

| Asset | File | Answers the question |
|---|---|---|
| Reference chunks | `data/corpus.json` | what can be found |
| Relevance labels | `data/queries.json` | what *should* be found |
| Reference answers | `data/references.json` | what should be said |
| Generated texts | `data/generated_llm.json` + baseline | what the system said |

**Task 1.1** Count how many queries have more than one relevant chunk.

```bash
python3 -c "
import json; q=json.load(open('data/queries.json'))
print([x['id'] for x in q if len(x['relevant'])>1])"
```

**Expected:** `['q12']`.

**Task 1.2** Compare `references.json` with `generated_llm.json` for `q05` and `q07`. Which words differ? Is the meaning the same?

**Task 1.3** Labels are created **before** any system is compared. Why does that matter? (Answer: otherwise the labels would be biased toward whatever the first system returned. Part D shows this effect.)

---

## 2. Retrieval evaluation (15 min)

```bash
python src/retrieve.py --method tfidf
python src/eval_retrieval.py --run runs/tfidf.json --k 3 --verbose
```

**Expected summary line:**

```
k=3  Precision@k=0.306  Recall@k=0.875  F1@k=0.450  MAP=0.850  MRR=0.878
```

**Checkpoint:** paste your Precision@3 in chat.

**Expected per-query lines that are not perfect:**

```
q07  rel=['c08']  top3=['c07', 'c02', 'c08']  P=0.33 R=1.00 AP=0.333 RR=0.33
q11  rel=['c12']  top3=['c15', 'c04', 'c03']  P=0.00 R=0.00 AP=0.200 RR=0.20
q12  rel=['c14', 'c05']  top3=['c14', 'c01', 'c02']  P=0.33 R=0.50 AP=0.667 RR=1.00
```

**Task 2.1** Explain each of the three lines in one sentence.

<details><summary>Answers</summary>

- **q07** found, but at rank 3. No shared words between "run a script every night" and "cron runs scheduled jobs".
- **q11** missed within top 3. "compress" does not match "compressed" without stemming.
- **q12** one of two relevant chunks found. Recall 0.5 and AP 0.667.
</details>

**Task 2.2** Precision@3 is only 0.306. Is this a bad system? (No. Most queries have one relevant chunk, so P@3 is capped at 0.333. Look at Recall@3, MAP and MRR together.)

**Task 2.3** Compare three methods. Fill in the table:

```bash
python src/retrieve.py --method lsa
python src/retrieve.py --method hybrid
for m in tfidf lsa hybrid; do echo -n "$m  "; python src/eval_retrieval.py --run runs/$m.json --k 3 | tail -1; done
```

| Method | P@3 | R@3 | F1@3 | MAP | MRR |
|---|---|---|---|---|---|
| tfidf | 0.306 | 0.875 | 0.450 | 0.850 | 0.878 |
| lsa | 0.333 | 0.917 | 0.483 | 0.808 | 0.808 |
| hybrid | 0.333 | 0.917 | 0.483 | 0.822 | 0.836 |

**Question:** `lsa` has the best recall but the worst MAP and MRR. Which would you ship for a RAG system that sends 3 chunks to the LLM? Which for a search page that shows a single best answer? (Suggested answer: `lsa` or `hybrid` for RAG, because recall at k is what matters when the LLM reads all three. `tfidf` for a one-answer page, because MRR is higher.)

---

## 3. Generation evaluation (15 min)

**Step 1: the toy example** (the same pair as the slides):

```bash
python src/toy_metrics.py 2>/dev/null
```

Expected: BLEU-4 unsmoothed `0.0`, smoothed `0.38`, ROUGE-1 F1 `0.833`, ROUGE-2 F1 `0.600`, ROUGE-L F1 `0.833`.

**Checkpoint:** type "match" in chat if yours are the same.

**Step 2: score two systems**

```bash
python src/generate_baseline.py runs/tfidf.json
python src/eval_generation.py --gen runs/generated_baseline.json 2>/dev/null
python src/eval_generation.py --gen data/generated_llm.json 2>/dev/null
```

**Expected:**

| System | BLEU | R-1 | R-2 | R-L |
|---|---|---|---|---|
| Copy top chunk (baseline) | 0.136 | 0.431 | 0.230 | 0.349 |
| LLM-style | 0.210 | 0.534 | 0.272 | 0.412 |

**Checkpoint:** paste your ROUGE-L for both systems in chat.

**Step 3: look at the worst scores**

```bash
python src/eval_generation.py --gen data/generated_llm.json --verbose 2>/dev/null | sort -t= -k5 | head -4
```

**Expected lowest:** `q02` (R-L 0.222), `q08` (0.333), `q05` (0.343), `q10` (0.348).

**Task 3.1** Open `q02` in both files:

- reference: *Run chmod +x on the script to make it executable.*
- generated: *Give the file execute permission with chmod +x.*

Is the generated answer correct? Why is its score low? (Answer: correct, but paraphrased. "give the file execute permission" shares few words with "make it executable".)

**Task 3.2** The LLM-style system beats the baseline on every metric here. Does that prove it gives better answers? Look at the baseline's answer for `q05` and decide. (Hint: the baseline returns the whole chunk, including extra sentences. ROUGE rewards overlap, so it is a regression signal, not a verdict.)

**Task 3.3** What these metrics miss. Score the flawed answers:

```bash
python src/eval_generation.py --gen data/generated_flawed.json 2>/dev/null
```

**Expected:** `BLEU=0.192  ROUGE-1=0.519  ROUGE-2=0.250  ROUGE-L=0.395`, only a small drop from 0.210 / 0.534 / 0.272 / 0.412.

Open `data/generated_flawed.json`. It contains three harmful errors. Find them: `q05` says `kill -1` instead of `kill -9`, `q07` adds an unsupported `systemctl restart cron` step, and `q09` invents `--delete-all`. The overall scores barely moved. **What does that tell you about using BLEU/ROUGE alone?**

---

## 4. Exercises and stretch goals (5 min, or homework)

### Exercise 1: Semantic vs lexical

```bash
pip install sentence-transformers
python src/retrieve.py --method embed
python src/eval_retrieval.py --run runs/embed.json --k 3 --verbose | grep -v "AP=1.000"
```

Which queries improved compared with `tfidf`? Which got worse?

### Exercise 2: Add a hard query

Append this to the list in `data/queries.json` (keep valid JSON, add a comma after the previous item):

```json
{"id":"q13","text":"automate a task at 2am daily","relevant":["c08"],"intent":"task"}
```

Re-run all three retrieval methods and check `q13`:

```bash
for m in tfidf lsa hybrid; do python src/retrieve.py --method $m >/dev/null; python src/eval_retrieval.py --run runs/$m.json --k 3 --verbose | grep "^q13"; done
```

**Expected for all three:** `top3=['c01', 'c02', 'c03']  P=0.00 R=0.00 AP=0.125`.

The query shares **no** word with the cron chunk, so every score is 0 and the ranking is just corpus order. This is the strongest argument for real embeddings and for an **abstain** rule when the best score is zero. (Part D adds one.)

### Exercise 3: Wrong but similar

```bash
python src/perturb.py 2>/dev/null
```

**Expected:**

```
original                     q05 R-L=0.343 BLEU=0.059 | all-queries R-L=0.412
kill -9 -> kill -1 in q05    q05 R-L=0.286 BLEU=0.048 | all-queries R-L=0.407
```

One changed character, a dangerous answer, and the overall score moves by 0.005.

### Exercise 4: Hybrid and a real LLM

- Blend TF-IDF and embeddings: `python src/retrieve.py --method hybrid --dense embed`.
- Or generate answers with a local model (for example Ollama) from the prompt printed by `python src/rag_prompt.py runs/hybrid.json q07 --k 3`, save the answers as JSON `{"q01": "...", ...}`, and score them with `eval_generation.py --gen your_file.json`.

---

## 5. Lab report (2 min)

Fill in and keep for the discussion:

| Question | Your answer |
|---|---|
| Best retrieval method at k = 3 and why | |
| Query that failed in every method | |
| One thing BLEU/ROUGE rewarded wrongly | |
| One thing BLEU/ROUGE missed | |

**Next:** Part D looks at what can go wrong with the evaluation itself: ambiguity, bias, hallucination and abstaining.

---

# Part D · Challenges, Bias and Trust

**30 min**

**You will:** see ambiguity, intent and data drift break a "good" score; reproduce pooling bias and position bias; catch hallucinated commands with a groundedness check; and test an abstain rule.

**Prerequisite:** the setup block at the top of this file. Create the two rankings this part reads:

```bash
python src/retrieve.py --method tfidf && python src/retrieve.py --method lsa
```

---

## 1. Ambiguity: one query, several right answers (5 min)

Open `data/ambiguous.json`. Each query has **one label set per intent**:

| Query | Intent 1 | Intent 2 |
|---|---|---|
| backup | schedule a backup → c08 | create an archive → c12 |
| memory | check free memory → c14 | find what uses memory → c05 |
| port | open a firewall port → c09 | list listening ports → c13 |

```bash
python src/ambiguity_eval.py
```

**Expected (last lines of each block):**

```
query "backup"  top3=['c12', 'c08', 'c01']  ...  intent coverage = 2/2
query "memory"  top3=['c14', 'c05', 'c01']  ...  intent coverage = 2/2
query "port"    top3=['c09', 'c01', 'c02']  ...  single-label recall = 1.0   intent coverage = 1/2
```

**Task 1.1** For "port", a single-label evaluation (only the first intent labelled) reports recall **1.0**. The system actually covers only **1 of 2** intents. What does this show about single-label test sets? (Answer: a single label hides the missed intent and the score looks perfect.)

**Task 1.2** How could a product handle "port" better? List two options. (Answers: ask a clarifying question; diversify the top results; show both intents.)

---

## 2. Intent: report per slice, not just overall (4 min)

```bash
python src/eval_retrieval.py --run runs/tfidf.json --k 3 --by-intent
```

**Expected:**

```
By intent:
  info  n= 3  P@3=0.333  R@3=0.833  MAP=0.889  MRR=1.000
  task  n= 8  P@3=0.292  R@3=0.875  MAP=0.817  MRR=0.817
  nav   n= 1  P@3=0.333  R@3=1.000  MAP=1.000  MRR=1.000
```

**Task 2.1** Which intent has the lowest MRR? **Task 2.2** `nav` shows perfect scores. Can you trust that? (Answer: no. n = 1. A slice with a tiny sample proves nothing. Report n with every slice.)

**Task 2.3** Run the same command for `runs/lsa.json`. **Expected:** `task` MRR 0.775 and `nav` MRR 0.500 (n = 1), `info` MRR 1.000. The single `nav` query now looks worst, but with n = 1 it is an anecdote, not a finding. `task` is the slice to investigate.

---

## 3. Dynamicity: the test set goes stale (5 min)

A new chunk arrives. Add it to the **end** of `data/corpus.json` (keep valid JSON):

```json
{"id":"c16","text":"To open port 22 on the firewall run ufw allow 22/tcp. This is the recommended way to open the SSH port."}
```

Re-run retrieval and score with the **old** labels:

```bash
python src/retrieve.py --method tfidf
python src/eval_retrieval.py --run runs/tfidf.json --k 3 --verbose | grep -E "^q08|^k="
```

**Expected:**

```
q08  rel=['c09']  top3=['c16', 'c09', 'c01']  P=0.33 R=1.00 AP=0.500 RR=0.50
k=3  Precision@k=0.306  Recall@k=0.875  F1@k=0.450  MAP=0.808  MRR=0.836
```

The new chunk is a perfectly good answer, but it is not in the labels, so the system is **penalised for doing well**: MAP fell from 0.850 to 0.808.

**Task 3.1** Now fix the labels. In `data/queries.json` change q08 to `"relevant":["c09","c16"]` and re-score. **Expected:** `q08 ... AP=1.000`, MAP back to `0.850`, MRR `0.878`.

**Task 3.2** Write down the rule you would adopt. (Answer: version the test set, review labels whenever the corpus changes, and monitor continuously.)

> Optional: undo the corpus change before the next section (remove c16, re-run retrieval), so the numbers below match.

---

## 4. Pooling bias: unjudged does not mean irrelevant (5 min)

Imagine only the **old** system's top 2 results were ever judged. We now evaluate a **new** system (`lsa`) on those partial labels.

```bash
python src/pooling_bias.py --old runs/tfidf.json --new runs/lsa.json --depth 2
```

**Expected:**

```
q07: relevant ['c08'] but only [] were ever judged
q11: relevant ['c12'] but only [] were ever judged
q12: relevant ['c05', 'c14'] but only ['c14'] were ever judged

New system with FULL labels  : R@3=0.917 MAP=0.808 MRR=0.808
New system with POOLED labels: R@3=0.833 MAP=0.708 MRR=0.708
```

**Task 4.1** The new system is unchanged. Why does it look 10 points worse in MAP? (Answer: results it finds that the old system never showed were never judged, so they count as wrong.)

**Task 4.2** Try `--depth 3` and `--depth 5`.

**Expected:** pooled MAP rises from 0.708 (depth 2) to 0.750 (depth 3) and 0.767 (depth 5), against 0.808 with full labels. A deeper pool judges more of the new system's results, but the gap does not fully close, because some relevant chunks are still outside the old system's pool. **Fix in practice:** pool the union of results from several systems before labelling.

---

## 5. Position bias: clicks are not relevance (3 min)

```bash
python src/position_bias.py
```

**Expected:**

```
rank  CTR   (true relevance is identical for all ranks)
   1  0.502
   2  0.249
   3  0.164
   4  0.125
   5  0.094
```

**Task 5.1** In this simulation every result is equally relevant. Why does rank 1 get five times the clicks of rank 5? (Answer: users examine the top result more often.) What would you conclude from raw click data? **Fix:** randomize order for judging, or model the examination probability.

---

## 6. Hallucination: groundedness and abstain (6 min)

The groundedness check requires every **command, flag, path or number** in an answer to appear in the retrieved chunks.

```bash
python src/groundedness.py --gen data/generated_flawed.json
```

**Expected (flagged lines):**

```
q05  FLAG  unsupported=['-1']
q07  FLAG  unsupported=['systemctl']
q09  FLAG  unsupported=['--delete-all']
q11  FLAG  unsupported=['tar', '-czf']
4 answers flagged
```

**Task 6.1** Three flags are real errors (q05, q07, q09). Which one is **correct but ungrounded**? **Answer:** q11. The answer is right, but the retrieved chunks did not contain it, so the model used memory. Should this count as a failure? Why might it still matter? (Answer: the answer cannot be verified from the sources, and next time it may be wrong.)

**Task 6.2** Run the check on the good answers:

```bash
python src/groundedness.py --gen data/generated_llm.json | grep -E "FLAG|flagged"
```

**Expected:** only `q11 FLAG`, `1 answers flagged`. A good checker has few false alarms.

**Task 6.3 Abstain.** Add a rule: say "not found" if the best retrieval score is below a threshold.

```bash
python src/groundedness.py --gen data/generated_llm.json --abstain 0.25 | grep -E "ABSTAIN|flagged"
```

**Expected:**

```
q07  ABSTAIN  (top score 0.17 < 0.25)
q12  ABSTAIN  (top score 0.23 < 0.25)
```

Look at the cost: both abstained queries **did** have the right chunk in the top 3 (q07 at rank 3, q12 at rank 1 for c14). Try thresholds `0.15` and `0.35`. **Expected:** 0 queries abstain at 0.15 and 5 at 0.35. What is the trade-off? (Answer: a high threshold avoids wrong answers but refuses answerable questions.)

**Task 6.4 Your RAG triad.** Match each script to the triad check it approximates:

| Check | Question | Script |
|---|---|---|
| Context relevance | are the chunks right for the query? | |
| Groundedness | is the answer supported by the chunks? | |
| Answer relevance | does the answer address the question? | |

<details><summary>Answers</summary>
Context relevance → `eval_retrieval.py`. Groundedness → `groundedness.py`. Answer relevance → `eval_generation.py` / `diagnose.py` (key fact present). In production you would use an LLM judge for the last one, calibrated on human labels.
</details>

---

## 7. The checklist: score your own project (2 min)

Rate your current search or RAG project 0 (not done), 1 (partly) or 2 (done).

| # | Checklist step | Score |
|---|---|---|
| 1 | Task and users defined | |
| 2 | Labelled golden set exists | |
| 3 | Retrieval and generation scored separately | |
| 4 | Human review on a sample | |
| 5 | Checked bias and sliced by group | |
| 6 | k, dataset and version reported; monitored over time | |

Which step has the lowest score? That is your first action next week.

---

## Summary of what each script shows

| Script | Lesson |
|---|---|
| `ambiguity_eval.py` | single-label sets hide missed intents |
| `eval_retrieval.py --by-intent` | averages hide weak slices; report n |
| corpus change (section 3) | labels go stale; version them |
| `pooling_bias.py` | unjudged results are not irrelevant |
| `position_bias.py` | clicks reflect position, not just relevance |
| `groundedness.py` | catch unsupported commands; abstain costs recall |
