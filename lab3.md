# Simple Lab: Search, RAG and Evaluation Metrics

**60 minutes · Linux VM · Python 3.9+ · no GPU, no API key, no model downloads**

This is a smaller, easier version of Workshop 1. Instead of 12 files there are only **two**: `data.json` (all the data) and `lab.py` (all the commands). Every command below was run, and every "Expected" output was copied from a real run.

| Block | Time | What you do |
|---|---|---|
| **Setup** | 5 min | Create 2 files, run 1 check |
| **Part A: Search and RAG** | 25 min | Search, compare methods, chunking, RAG prompt, diagnose failures |
| **Part B: Metrics** | 30 min | Precision/Recall/F1/AP/MRR/nDCG, kappa, BLEU and ROUGE |

> **How to work:** read the task, **guess the answer first**, then run the command and compare with "Expected". Answers are in the collapsible blocks.

---

## Setup (5 min)

**1. Create the environment**

```bash
python3 --version              # must be 3.9 or newer
# if venv is missing on Ubuntu/Debian:  sudo apt update && sudo apt install -y python3-venv python3-pip
mkdir -p ~/simple-lab && cd ~/simple-lab
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn sacrebleu rouge-score
```

**2. Create the data file** (15 Linux chunks, 12 questions, answers, ratings). Paste the whole block:

```bash
cat > data.json <<'EOF'
{
"corpus": [
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
],
"queries": [
{"id":"q01","text":"how do I log in to a remote server securely","relevant":["c01"],"key":"ssh"},
{"id":"q02","text":"make a script executable","relevant":["c02"],"key":"chmod +x"},
{"id":"q03","text":"change the owner of a folder recursively","relevant":["c03"],"key":"-R"},
{"id":"q04","text":"how much free disk space is left","relevant":["c04"],"key":"df -h"},
{"id":"q05","text":"stop a process that is frozen","relevant":["c05"],"key":"kill -9"},
{"id":"q06","text":"start nginx when the machine boots","relevant":["c06"],"key":"systemctl enable nginx"},
{"id":"q07","text":"run a script every night","relevant":["c08"],"key":"crontab -e"},
{"id":"q08","text":"open port 22 on the firewall","relevant":["c09"],"key":"ufw allow 22/tcp"},
{"id":"q09","text":"nginx service logs","relevant":["c10"],"key":"journalctl -u nginx"},
{"id":"q10","text":"give a user admin rights","relevant":["c11"],"key":"usermod -aG sudo"},
{"id":"q11","text":"compress a folder into one file","relevant":["c12"],"key":"tar -czf"},
{"id":"q12","text":"server killed my programs because RAM is full","relevant":["c14","c05"],"key":"free -h"}
],
"references": {
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
},
"llm": {
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
},
"flawed": {
"q01":"You can securely connect to a remote machine with the ssh command, and it is safer to use key pairs than passwords.",
"q02":"Give the file execute permission with chmod +x.",
"q03":"Apply chown with the -R flag and a user:group value to change ownership of the whole directory.",
"q04":"The df -h command reports how much disk space is free on each filesystem.",
"q05":"Look up the process ID using ps aux, then use kill -1 on that ID to terminate it immediately.",
"q06":"Use systemctl start nginx to launch the service now.",
"q07":"Schedule it with cron: open crontab -e and add a line such as 30 2 * * * /opt/backup.sh, which executes daily at 02:30.",
"q08":"Allow the SSH port through the firewall with ufw allow 22/tcp.",
"q09":"Run journalctl -u nginx to view the logs of the nginx service.",
"q10":"Add the user to the sudo group with usermod -aG sudo alice.",
"q11":"Create a compressed archive with tar -czf backup.tar.gz followed by the folder name.",
"q12":"When RAM runs out, the kernel can kill running programs. Check usage with free -h and configure swap space to lower the risk."
},
"ratings": {
"rater_a":[1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,0,1],
"rater_b":[1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,0,1]
}
}
EOF
```

**3. Create the lab program** (one file, about 150 lines; you do not need to read it):

```bash
cat > lab.py <<'EOF'
"""Simple search + evaluation lab.  Run:  python lab.py <command> ...   (python lab.py -h for the list)"""
import argparse, json, math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

D = json.load(open("data.json"))
CORPUS, QUERIES = D["corpus"], D["queries"]
IDS = [c["id"] for c in CORPUS]
TEXT = {c["id"]: c["text"] for c in CORPUS}
QBY = {q["id"]: q for q in QUERIES}

# ---------- retrieval ----------
def _tfidf():
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    return v, v.fit_transform([c["text"] for c in CORPUS])

def scores(method, qtexts):
    v, X = _tfidf(); Q = v.transform(qtexts)
    lex = (Q @ X.T).toarray()
    if method == "tfidf":
        return lex
    svd = TruncatedSVD(n_components=8, random_state=0).fit(X)
    dense = normalize(svd.transform(Q)) @ normalize(svd.transform(X)).T
    if method == "lsa":
        return dense
    out = np.zeros_like(lex)                       # hybrid = reciprocal rank fusion
    for S in (lex, dense):
        out += 1.0 / (60 + (-S).argsort(axis=1).argsort(axis=1) + 1)
    return out

def rankings(method, qtexts):
    S = scores(method, qtexts)
    return [[IDS[j] for j in np.argsort(-row, kind="stable")] for row in S]

def run(method):
    return dict(zip([q["id"] for q in QUERIES], rankings(method, [q["text"] for q in QUERIES])))

# ---------- metrics ----------
def prf(ranked, rel, k):
    h = sum(d in rel for d in ranked[:k]); p, r = h / k, h / len(rel)
    return p, r, (0 if h == 0 else 2 * p * r / (p + r))

def ap_rr(ranked, rel):
    seen, ap = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel: seen += 1; ap += seen / i
    return ap / len(rel), next((1 / i for i, d in enumerate(ranked, 1) if d in rel), 0.0)

# ---------- commands ----------
def cmd_search(a):
    qs = [a.query]
    print(f"Query: {a.query!r}")
    for m in ("tfidf", "lsa", "hybrid"):
        print(f"  {m:7}", rankings(m, qs)[0][:3])

def cmd_eval(a):
    R = run(a.method); rows = []
    for q in QUERIES:
        rel = set(q["relevant"]); p, r, f = prf(R[q["id"]], rel, a.k); ap, rr = ap_rr(R[q["id"]], rel)
        rows.append((p, r, f, ap, rr))
        if a.verbose:
            print(f'{q["id"]}  relevant={q["relevant"]}  top{a.k}={R[q["id"]][:a.k]}  AP={ap:.3f} RR={rr:.2f}')
    m = np.mean(rows, axis=0)
    print(f"method={a.method} k={a.k}  Precision={m[0]:.3f}  Recall={m[1]:.3f}  F1={m[2]:.3f}  MAP={m[3]:.3f}  MRR={m[4]:.3f}")

def cmd_prompt(a):
    q = QBY[a.qid]; ids = rankings(a.method, [q["text"]])[0][:a.k]
    ctx = "\n".join(f"[{i}] ({d}) {TEXT[d]}" for i, d in enumerate(ids, 1))
    print(f'You are a Linux assistant. Use ONLY the context below.\nCite sources like [1]. If the context is not enough, say "not found".\n\nContext:\n{ctx}\n\nQuestion: {q["text"]}\nAnswer:')

def answers(kind, method):
    if kind == "baseline":
        R = run(method); return {qid: TEXT[r[0]] for qid, r in R.items()}
    return D[kind]

def cmd_diagnose(a):
    R, ans = run(a.method), answers(a.answers, a.method); c = {}
    for q in QUERIES:
        got = any(d in q["relevant"] for d in R[q["id"]][:a.k])
        ok = q["key"].lower() in ans[q["id"]].lower()
        v = "OK" if got and ok else "FIX GENERATION" if got else "FIX RETRIEVAL" if not ok else "LUCKY (right answer, evidence missed)"
        c[v] = c.get(v, 0) + 1
        if v != "OK" or a.all: print(f'{q["id"]}  evidence_in_top{a.k}={got!s:5}  answer_ok={ok!s:5} -> {v}')
    print("Summary:", c)

def text_scores(gen, ref):
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"]); r = sc.score(ref, gen)
    return BLEU(effective_order=True).sentence_score(gen, [ref]).score / 100, r

def cmd_text(a):
    ans, ref, rows = answers(a.answers, a.method), D["references"], []
    for q in ref:
        b, r = text_scores(ans[q], ref[q])
        rows.append((b, r["rouge1"].fmeasure, r["rouge2"].fmeasure, r["rougeL"].fmeasure))
        if a.verbose: print(q, "BLEU=%.3f R1=%.3f R2=%.3f RL=%.3f" % rows[-1])
    m = np.mean(rows, axis=0)
    print(f"answers={a.answers}  BLEU={m[0]:.3f}  ROUGE-1={m[1]:.3f}  ROUGE-2={m[2]:.3f}  ROUGE-L={m[3]:.3f}")

def cmd_toy(a):
    from sacrebleu.metrics import BLEU
    print("reference:", a.ref); print("generated:", a.gen)
    print("BLEU-4 no smoothing : %.3f" % (BLEU(smooth_method="none").sentence_score(a.gen, [a.ref]).score / 100))
    b, r = text_scores(a.gen, a.ref); print("BLEU-4 smoothed     : %.3f" % b)
    for k, v in r.items(): print(f"{k:7} P={v.precision:.3f} R={v.recall:.3f} F1={v.fmeasure:.3f}")

def cmd_metrics(a):
    rk = [int(x) for x in a.ranking.split(",")]; total = a.total
    print(" k   P@k   R@k   F1@k")
    for k in range(1, len(rk) + 1):
        h = sum(rk[:k]); p, r = h / k, h / total
        print(f"{k:2d}  {p:.2f}  {r:.2f}  {0 if h == 0 else 2*p*r/(p+r):.3f}")
    seen, ap = 0, 0.0
    for i, x in enumerate(rk, 1):
        if x: seen += 1; ap += seen / i
    print(f"\nAverage Precision : {ap/total:.3f}\nReciprocal Rank   : {next((1/i for i,x in enumerate(rk,1) if x),0):.3f}")
    if a.grades:
        g = [int(x) for x in a.grades.split(",")]
        dcg = lambda s: sum((2**x - 1) / math.log2(i + 1) for i, x in enumerate(s, 1))
        print(f"nDCG@{len(g)}           : {dcg(g)/dcg(sorted(g, reverse=True)):.3f}")

def cmd_kappa(a):
    x, y = D["ratings"]["rater_a"], D["ratings"]["rater_b"]; n = len(x)
    po = sum(i == j for i, j in zip(x, y)) / n; pa, pb = sum(x) / n, sum(y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    print(f"agreement={po:.2f}  by-chance={pe:.2f}  kappa={(po-pe)/(1-pe):.3f}")
    print("disagree on items:", [i + 1 for i, (u, v) in enumerate(zip(x, y)) if u != v])

DOC = ("To back up a folder, first run tar -czf backup.tar.gz folder. Then copy it with scp backup.tar.gz user@host:/backups. "
       "Finally run crontab -e and add 30 2 * * * /opt/backup.sh so it runs every night.")
def cmd_chunk(a):
    step = a.size - a.overlap; ch = [DOC[i:i + a.size] for i in range(0, len(DOC), step)]
    for i, c in enumerate(ch, 1): print(f"chunk {i}: {c!r}")
    for f in ["tar -czf backup.tar.gz", "crontab -e", "30 2 * * * /opt/backup.sh"]:
        print(f"{f!r:30} intact? {any(f in c for c in ch)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    M = ["tfidf", "lsa", "hybrid"]
    p = sp.add_parser("search"); p.add_argument("query"); p.set_defaults(f=cmd_search)
    p = sp.add_parser("eval"); p.add_argument("--method", choices=M, default="tfidf"); p.add_argument("--k", type=int, default=3); p.add_argument("--verbose", action="store_true"); p.set_defaults(f=cmd_eval)
    p = sp.add_parser("prompt"); p.add_argument("qid"); p.add_argument("--method", choices=M, default="tfidf"); p.add_argument("--k", type=int, default=3); p.set_defaults(f=cmd_prompt)
    p = sp.add_parser("diagnose"); p.add_argument("--answers", choices=["baseline", "llm", "flawed"], default="baseline"); p.add_argument("--method", choices=M, default="tfidf"); p.add_argument("--k", type=int, default=3); p.add_argument("--all", action="store_true"); p.set_defaults(f=cmd_diagnose)
    p = sp.add_parser("text"); p.add_argument("--answers", choices=["baseline", "llm", "flawed"], default="llm"); p.add_argument("--method", choices=M, default="tfidf"); p.add_argument("--verbose", action="store_true"); p.set_defaults(f=cmd_text)
    p = sp.add_parser("toy"); p.add_argument("gen", nargs="?", default="the cat is on the mat"); p.add_argument("--ref", default="the cat sat on the mat"); p.set_defaults(f=cmd_toy)
    p = sp.add_parser("metrics"); p.add_argument("ranking"); p.add_argument("total", type=int); p.add_argument("--grades"); p.set_defaults(f=cmd_metrics)
    p = sp.add_parser("kappa"); p.set_defaults(f=cmd_kappa)
    p = sp.add_parser("chunk"); p.add_argument("--size", type=int, default=45); p.add_argument("--overlap", type=int, default=0); p.set_defaults(f=cmd_chunk)
    a = ap.parse_args(); a.f(a)
EOF
```

**4. Checkpoint.** This must print one line:

```bash
python lab.py eval
```

```text
method=tfidf k=3  Precision=0.306  Recall=0.875  F1=0.450  MAP=0.850  MRR=0.878
```

If you see this line, you are ready. If not, see *Troubleshooting* at the end.

**All commands you will use:**

| Command | What it does |
|---|---|
| `python lab.py search "text"` | Top-3 chunks from 3 retrieval methods |
| `python lab.py eval --method M --k K` | Retrieval scores (P, R, F1, MAP, MRR) |
| `python lab.py prompt q07 --k 3` | Show the prompt a RAG system would send |
| `python lab.py diagnose --answers baseline` | Retrieval problem or generation problem? |
| `python lab.py chunk --size 45 --overlap 0` | See where chunking cuts facts |
| `python lab.py metrics 1,0,1,0,0,1,0,0,0,0 4` | Metrics for any ranked list |
| `python lab.py kappa` | Agreement between two human raters |
| `python lab.py toy "sentence"` | BLEU and ROUGE on a sentence |
| `python lab.py text --answers llm` | BLEU and ROUGE on all 12 answers |

Methods `M`: `tfidf` (lexical, word counts), `lsa` (small "meaning" model), `hybrid` (both merged with RRF).

---

# Part A · Search and RAG (25 min)

## A1. Look at the data (2 min)

Open `data.json`. Each question has a `relevant` chunk (the right answer page) and a `key` (the command a correct answer must contain).

**Task:** For `q07` ("run a script every night"), which chunk is relevant? Does the question share any important word with it?

<details><summary>Answer</summary>
<code>c08</code> (the cron chunk). The question and the chunk share no important word: "run a script every night" vs "cron runs scheduled jobs".
</details>

---

## A2. Search and rank (5 min)

```bash
python lab.py eval --k 3 --verbose
```

(The verbose output lists all 12 questions; look for lines where `AP` is below 1.)

Expected, the three imperfect questions:

```text
q07  relevant=['c08']  top3=['c07', 'c02', 'c08']  AP=0.333 RR=0.33
q11  relevant=['c12']  top3=['c15', 'c04', 'c03']  AP=0.200 RR=0.20
q12  relevant=['c14', 'c05']  top3=['c14', 'c01', 'c02']  AP=0.667 RR=1.00
```

**Task 1:** Which question did not even get its relevant chunk into the top 3?

**Task 2:** Where is the right chunk for `q07`?

<details><summary>Answers</summary>
1) <code>q11</code> ("compress a folder into one file"). Its chunk <code>c12</code> is at rank 5. The question says "compress", the chunk says "compressed", and plain word matching treats them as different words.<br>
2) Rank 3, behind two wrong chunks (apt and chmod), because they contain "script".
</details>

---

## A3. Lexical vs "meaning" search (6 min)

```bash
python lab.py search "run a script every night"
```

```text
Query: 'run a script every night'
  tfidf   ['c07', 'c02', 'c08']
  lsa     ['c07', 'c08', 'c02']
  hybrid  ['c07', 'c02', 'c08']
```

`lsa` moved the cron chunk (`c08`) up one place, but not to the top. With only 15 tiny chunks it has little to learn from; a real embedding model trained on large text does better.

Now score the three methods on all 12 questions:

```bash
for m in tfidf lsa hybrid; do python lab.py eval --method $m --k 3; done
```

```text
method=tfidf k=3  Precision=0.306  Recall=0.875  F1=0.450  MAP=0.850  MRR=0.878
method=lsa k=3  Precision=0.333  Recall=0.917  F1=0.483  MAP=0.808  MRR=0.808
method=hybrid k=3  Precision=0.333  Recall=0.917  F1=0.483  MAP=0.822  MRR=0.836
```

**Task:** Which method finds the right chunk *first* most often (MRR)? Which one has the best recall in the top 3?

<details><summary>Answer</summary>
<code>tfidf</code> has the best MRR (0.878): it often puts the right chunk first. <code>lsa</code> and <code>hybrid</code> have the best Recall@3 (0.917): they catch more right chunks somewhere in the top 3. No method wins everywhere, which is why systems combine them.
</details>

---

## A4. Chunking: where does a fact get cut? (3 min)

```bash
python lab.py chunk --size 45 --overlap 0
```

```text
chunk 1: 'To back up a folder, first run tar -czf backu'
chunk 2: 'p.tar.gz folder. Then copy it with scp backup'
chunk 3: '.tar.gz user@host:/backups. Finally run cront'
chunk 4: 'ab -e and add 30 2 * * * /opt/backup.sh so it'
chunk 5: ' runs every night.'
'tar -czf backup.tar.gz'       intact? False
'crontab -e'                   intact? False
'30 2 * * * /opt/backup.sh'    intact? True
```

Two commands got **cut in half**. Now add overlap:

```bash
python lab.py chunk --size 45 --overlap 25 | tail -3
```

```text
'tar -czf backup.tar.gz'       intact? True
'crontab -e'                   intact? True
'30 2 * * * /opt/backup.sh'    intact? True
```

**Task:** Try `--size 300`. What is the downside of one huge chunk?

<details><summary>Answer</summary>
Everything stays intact, but one chunk now covers several topics (backup, copy, cron), so it matches many questions weakly and the answer is buried in noise.
</details>

---

## A5. The RAG prompt (3 min)

```bash
python lab.py prompt q07 --k 3
```

You see the instructions, the three retrieved chunks numbered `[1] [2] [3]`, and the question. This is exactly what an LLM would receive.

**Task 1:** Is the evidence for `q07` inside the prompt? Which number?
**Task 2:** Run with `--k 1`. What does the model see now?

<details><summary>Answers</summary>
1) Yes: <code>[3] (c08)</code>, the cron chunk.<br>
2) Only <code>[1] (c07)</code>, the apt chunk. The answer is not in the prompt, so a good model should say "not found". A model that answers anyway is guessing.
</details>

---

## A6. Diagnose: was the evidence retrieved? (6 min)

Rule: **evidence not retrieved → fix retrieval. Evidence retrieved but answer wrong → fix generation.**

The `baseline` "generator" simply copies the **top-1** chunk as its answer:

```bash
python lab.py diagnose --answers baseline --k 3
```

```text
q07  evidence_in_top3=True   answer_ok=False -> FIX GENERATION
q11  evidence_in_top3=False  answer_ok=False -> FIX RETRIEVAL
Summary: {'OK': 10, 'FIX GENERATION': 1, 'FIX RETRIEVAL': 1}
```

**Task 1:** `q11` says FIX RETRIEVAL. Why? (Hint: A2.)
**Task 2:** `q07` says FIX GENERATION, but the evidence was at rank 3 and the baseline only uses rank 1. What is the *real* problem?

Now check answers that contain **wrong facts**:

```bash
python lab.py diagnose --answers flawed --k 3
```

```text
q05  evidence_in_top3=True   answer_ok=False -> FIX GENERATION
q06  evidence_in_top3=True   answer_ok=False -> FIX GENERATION
q11  evidence_in_top3=False  answer_ok=True  -> LUCKY (right answer, evidence missed)
Summary: {'OK': 9, 'FIX GENERATION': 2, 'LUCKY (right answer, evidence missed)': 1}
```

**Task 3:** In `flawed`, `q05` says `kill -1` instead of `kill -9`, and `q06` says `systemctl start` instead of `enable`. What does `LUCKY` on `q11` mean, and is it safe?

<details><summary>Answers</summary>
1) The right chunk <code>c12</code> was never in the top 3, so the model never saw it.<br>
2) A ranking problem: the evidence was found but ranked 3rd, and the simple generator ignored it. "Retrieval" and "generation" labels are a first diagnosis, not the whole story.<br>
3) The answer was right but the right chunk was not retrieved, so the model must have answered from its own memory. It is not grounded in your documents, so it is not safe to rely on.
</details>

**Discuss (2 min):** Which would you fix first for `q07`: the prompt, the retriever, or the ranking?

---

# Part B · Metrics by Hand and by Code (30 min)

## B1. The running example (8 min)

Ten results, rank 1 on the left. `1` = relevant, `0` = not. Relevant at ranks 1, 3 and 6, and **4 relevant chunks exist in total** (one was never retrieved).

**Step 1. Calculate by hand first**, for k = 5:

| | Your answer |
|---|---|
| Relevant in top 5 | |
| Precision@5 | |
| Recall@5 | |
| F1@5 | |

**Step 2. Run it:**

```bash
python lab.py metrics 1,0,1,0,0,1,0,0,0,0 4 --grades 3,0,2,0,0,1,0,0,0,0
```

```text
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
Reciprocal Rank   : 1.000
nDCG@10           : 0.943
```

Row k = 5 should match your hand calculation (P = 2/5, R = 2/4, F1 = 0.444).

**Task 1:** At which k is F1 highest? 
**Task 2:** AP is the sum of 1/1, 2/3, 3/6 divided by **4**. Why 4 and not 3?

<details><summary>Answers</summary>
1) k = 6 (0.600). <br>
2) The relevant chunk that was never retrieved must count against you. Dividing by 3 would hide the miss.
</details>

---

## B2. Break it on purpose (6 min)

Predict AP and RR for each list, then run it. The format is `python lab.py metrics <ranking> <total_relevant>`.

| # | Command | Guess AP | Guess RR |
|---|---|---|---|
| A | `python lab.py metrics 0,0,0,1,0,0,0,0,0,0 1` | | |
| B | `python lab.py metrics 1,1,1,1,0,0,0,0,0,0 4` | | |
| C | `python lab.py metrics 0,0,0,0,0,0,0,0,0,0 4` | | |
| D | `python lab.py metrics 0,0,0,0,1,1,1,1,0,0 4` | | |

<details><summary>Answers (look at the last two lines of each output)</summary>

| # | AP | RR | Lesson |
|---|---|---|---|
| A | 0.250 | 0.250 | One relevant item at rank 4 |
| B | 1.000 | 1.000 | Perfect ranking |
| C | 0.000 | 0.000 | Nothing found (and no crash) |
| D | 0.365 | 0.200 | Same four items as B, but late. **Recall@10 is identical (1.0) for B and D**, yet AP is far lower. Recall alone cannot see ranking quality. |
</details>

**Task:** Now use graded labels with the best chunk at rank 6:

```bash
python lab.py metrics 1,0,1,0,0,1,0,0,0,0 4 --grades 0,0,2,0,0,3,0,0,0,0 | tail -1
```

```text
nDCG@10           : 0.449
```

Why is this so much lower than 0.943?

<details><summary>Answer</summary>
The most relevant item (grade 3) sits at rank 6, where its value is discounted. nDCG punishes putting the best result low.
</details>

---

## B3. Metrics on the real system (6 min)

```bash
for k in 1 3 5; do for m in tfidf lsa hybrid; do python lab.py eval --method $m --k $k; done; done
```

```text
method=tfidf k=1  Precision=0.833  Recall=0.792  F1=0.806  MAP=0.850  MRR=0.878
method=lsa k=1  Precision=0.667  Recall=0.625  F1=0.639  MAP=0.808  MRR=0.808
method=hybrid k=1  Precision=0.750  Recall=0.708  F1=0.722  MAP=0.822  MRR=0.836
method=tfidf k=3  Precision=0.306  Recall=0.875  F1=0.450  MAP=0.850  MRR=0.878
method=lsa k=3  Precision=0.333  Recall=0.917  F1=0.483  MAP=0.808  MRR=0.808
method=hybrid k=3  Precision=0.333  Recall=0.917  F1=0.483  MAP=0.822  MRR=0.836
method=tfidf k=5  Precision=0.200  Recall=0.958  F1=0.329  MAP=0.850  MRR=0.878
method=lsa k=5  Precision=0.217  Recall=1.000  F1=0.353  MAP=0.808  MRR=0.808
method=hybrid k=5  Precision=0.217  Recall=1.000  F1=0.353  MAP=0.822  MRR=0.836
```

(MAP and MRR use the whole ranking, so they do not change with k.)

**Task 1:** At k = 1 which method is best? At k = 5 which has perfect recall?
**Task 2:** Why is Precision@3 only about 0.3 for every method, even though the systems look good?
**Task 3:** Your RAG prompt will contain **3 chunks**. Which method would you choose, and why?

<details><summary>Answers</summary>
1) <code>tfidf</code> at k = 1 (P 0.833); <code>lsa</code> and <code>hybrid</code> reach Recall 1.000 at k = 5.<br>
2) Most questions have exactly <b>one</b> relevant chunk, so P@3 can never exceed 1/3. Precision depends on how many relevant items exist.<br>
3) Recall@3 matters most because the LLM only sees 3 chunks: <code>lsa</code> or <code>hybrid</code> (0.917). <code>hybrid</code> is a good default since its ranking quality (MAP 0.822) is also decent.
</details>

---

## B4. Human labels: do raters agree? (4 min)

Two raters judged 20 chunks (1 = relevant).

```bash
python lab.py kappa
```

```text
agreement=0.80  by-chance=0.52  kappa=0.583
disagree on items: [4, 9, 14, 16]
```

**Task:** 80% agreement sounds high. Why is kappa only 0.58?

<details><summary>Answer</summary>
Two raters would agree about 52% of the time just by chance. Kappa removes that luck. 0.58 means "moderate": the guidelines probably need work. In a real project, discuss items 4, 9, 14 and 16, rewrite the guidelines and consider a third rater.
</details>

---

## B5. BLEU and ROUGE on the cat and the mat (6 min)

```bash
python lab.py toy 2>/dev/null
```

```text
reference: the cat sat on the mat
generated: the cat is on the mat
BLEU-4 no smoothing : 0.000
BLEU-4 smoothed     : 0.380
rouge1  P=0.833 R=0.833 F1=0.833
rouge2  P=0.600 R=0.600 F1=0.600
rougeL  P=0.833 R=0.833 F1=0.833
```

(`2>/dev/null` just hides a harmless library warning.)

Now try your own sentences. **Guess first**, then run `python lab.py toy "your sentence" 2>/dev/null | tail -4`:

| Generated sentence | Meaning | Real result |
|---|---|---|
| `the cat sat on the mat` | identical | BLEU 1.000, all ROUGE 1.000 |
| `the feline rested on the rug` | same meaning, different words | BLEU 0.180, ROUGE-1 0.500, ROUGE-2 0.200 |
| `the cat` | far too short | BLEU 0.135, ROUGE-1 precision 1.000 but recall 0.333 |
| `the mat sat on the cat` | **wrong meaning**, same words | BLEU 0.427, ROUGE-1 **1.000**, ROUGE-2 0.800 |

**Task:** The last row has a perfect ROUGE-1 although the meaning is wrong. What does this tell you about overlap metrics?

<details><summary>Answer</summary>
They count matching words, not meaning. A scrambled or false sentence can score high, and a correct paraphrase (row 2) can score low. Use them as quick regression signals, then add human review and groundedness checks.
</details>

---

## B6. Score all 12 generated answers (5 min)

```bash
python lab.py text --answers baseline 2>/dev/null
python lab.py text --answers llm 2>/dev/null
python lab.py text --answers flawed 2>/dev/null
```

```text
answers=baseline  BLEU=0.136  ROUGE-1=0.431  ROUGE-2=0.230  ROUGE-L=0.349
answers=llm  BLEU=0.210  ROUGE-1=0.534  ROUGE-2=0.272  ROUGE-L=0.412
answers=flawed  BLEU=0.197  ROUGE-1=0.509  ROUGE-2=0.242  ROUGE-L=0.387
```

- `baseline`: copies the top retrieved chunk.
- `llm`: well-written paraphrased answers.
- `flawed`: the same as `llm`, but with **two wrong commands** (`kill -1`, `systemctl start`).

**Task 1:** Which system scores best? Is that what you expected?
**Task 2:** `flawed` contains two **wrong, harmful** answers. By how much did its scores drop compared with `llm`?
**Task 3:** What does this say about using BLEU/ROUGE alone?

<details><summary>Answers</summary>
1) <code>llm</code> is best on all four numbers, as expected.<br>
2) Only about 0.01 to 0.03 (for example ROUGE-1 from 0.534 to 0.509), although two answers would break a real system. The metrics barely noticed.<br>
3) They are blind to a single wrong word that changes the meaning. Pair them with the diagnose check from A6 (does the answer contain the key fact?), human review, and groundedness checks.
</details>

---

## Wrap-up (3 min)

Fill in from memory, then check.

| Need | Metric |
|---|---|
| One right answer | |
| Everything relevant | |
| Clean browsable list | |
| Graded relevance | |
| Single tuning number | |

**Answers:** MRR or P@1 · Recall@k · P@k or MAP · nDCG · F1@k or MAP.

**Three things to remember:**
1. Check **retrieval first**, then generation (A6).
2. Always say **which k** (B3). Precision depends on how many relevant items exist.
3. **BLEU/ROUGE miss wrong facts** (B5, B6): never rely on them alone.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3 -m venv` fails | `sudo apt install -y python3-venv` |
| `ModuleNotFoundError` | Run `source .venv/bin/activate` in this terminal, then `pip install numpy scikit-learn sacrebleu rouge-score` |
| `FileNotFoundError: data.json` | Run commands from inside `~/simple-lab` |
| Long warning text from sacrebleu | Harmless. Add `2>/dev/null` at the end of the command |
| `BrokenPipeError` when using `head` | Harmless |
| Numbers differ slightly | Library versions can change the 3rd decimal; patterns should match |

## Cleanup

```bash
deactivate
rm -rf ~/simple-lab
```
