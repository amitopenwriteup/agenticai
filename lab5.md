# Development Essentials: Git for Data Workflows

**Format:** Live session + hands-on lab | **Duration:** ~2 hours | **Level:** Beginner to Intermediate

---

## Workshop Overview

In this workshop you will use Git the way data and ETL teams actually need it: tracking notebooks, handling large data files with **Git LFS**, structuring a data project, reading notebook diffs with **nbdime**, and working in feature branches.

### Learning Outcomes

By the end, you will be able to:

1. Apply Git essentials to data and ETL workflows.
2. Version-control large files using Git LFS.
3. Structure a data-oriented codebase.
4. Diff and merge Jupyter notebooks using nbdime.
5. Apply feature-branch strategies for collaborative development.

### Prerequisites

- Basic command-line familiarity
- Awareness of software development workflows
- Understanding of data project files (notebooks, scripts, configs)

### Setup Check (do this before the session)

```bash
git --version            # 2.28+ recommended
python --version         # 3.9+
pip --version
git lfs version         # installed in Part III if missing
jupyter --version       # or: pip install notebook
```

Configure your identity (once per machine):

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

---

## Agenda

| Part | Topic | Time | Type |
|------|-------|------|------|
| I | Introduction | 5 min | Theory |
| II | Git Essentials for Data Workflows | 20 min | Theory |
| III | Git LFS for Large Data Files | 25 min | Simple hands-on |
| IV | Structuring Data Projects | 25 min | Theory |
| V | Notebook Versioning with nbdime | 25 min | Simple hands-on demo |
| VI | Branching Strategies for Feature Development | 20 min | Theory |
| VII | Doubt Resolution and Q&A | 10 min | Discussion |

> **Note:** Parts II, IV and VI are marked theory in the agenda. This workbook includes short optional exercises for them so you can practise at your own pace after the session.

---

## Part I: Introduction (5 min)

### Why version control for data work?

| Problem | How Git helps |
|---------|---------------|
| "Which version of the notebook produced this result?" | Every change is a commit with author, time and message |
| "I broke the ETL script and can't go back" | Revert or check out any earlier state |
| "Two of us edited the same notebook" | Branches and merges (with the right tooling) |
| "The dataset is 2 GB" | Git LFS stores large files outside normal history |

### What Git is not good at (out of the box)

- **Large binary/data files** bloat the repository, so we use Git LFS.
- **Notebooks (`.ipynb`)** are JSON containing code, outputs, and metadata, which produces noisy diffs, so we use nbdime.
- **Secrets and credentials** should never be committed, so we use `.gitignore` and environment variables.

### Session roadmap

You will build **one repository** across the session: `sales-etl-workshop`. Each part adds to it.

---

## Part II: Git Essentials for Data Workflows (20 min)

### 2.1 Core concepts

- **Working directory:** your files on disk
- **Staging area (index):** what will go into the next commit
- **Repository:** the stored history of commits
- **Remote:** a shared copy (GitHub, GitLab, etc.)

```
working dir  --git add-->  staging  --git commit-->  local repo  --git push-->  remote
```

### 2.2 Everyday commands

| Command | Purpose |
|---------|---------|
| `git init` | Start a new repository |
| `git status` | See what changed |
| `git add <file>` | Stage changes |
| `git commit -m "msg"` | Save a snapshot |
| `git log --oneline --graph` | View history |
| `git diff` / `git diff --staged` | See unstaged / staged changes |
| `git restore <file>` | Discard working-directory changes |
| `git restore --staged <file>` | Unstage |
| `git revert <commit>` | Safely undo a commit by adding a new one |

### 2.3 Hands-on: first repository

```bash
mkdir sales-etl-workshop && cd sales-etl-workshop
git init

cat > etl.py << 'EOF'
import csv

def extract(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def transform(rows):
    return [r for r in rows if r.get("amount")]

if __name__ == "__main__":
    print("ETL placeholder")
EOF

git status
git add etl.py
git commit -m "Add initial ETL script with extract and transform"
git log --oneline
```

### 2.4 `.gitignore` for data projects

Create `.gitignore` **before** adding data or notebooks:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Secrets and local config
.env
*.secret
credentials.json

# Generated outputs
data/processed/
outputs/
logs/

# OS / editor
.DS_Store
.vscode/
.idea/
EOF

git add .gitignore
git commit -m "Add .gitignore for data project"
```

> **Rule of thumb:** commit code, config templates and small reference data. Do not commit secrets, generated artefacts, or large raw data (use LFS or external storage).

### 2.5 Writing good commit messages

- Use the imperative mood: "Add retry logic to loader", not "Added stuff".
- Keep the first line under about 70 characters.
- Commit one logical change at a time.

Example: `Fix null handling in transform step for amount column`

### 2.6 Tracking notebooks: the first look

Notebooks are JSON files that store **source and outputs**. That means:

- Re-running a cell changes the file even if the code did not change.
- Execution counts and embedded images create large, unreadable diffs.

Two common mitigations (covered in Part V):

1. A notebook-aware diff/merge tool (**nbdime**)
2. Optionally stripping outputs before commit (e.g., with `nbstripout`)

### Optional exercise (Part II)

1. Edit `transform()` to also convert `amount` to `float`.
2. Run `git diff` and read the output.
3. Stage and commit with a clear message.
4. Use `git log --oneline` to confirm.
5. Make a deliberate mistake, then use `git restore etl.py` to undo it.

---

## Part III: Git LFS for Large Data Files (25 min)

### 3.1 Why LFS?

Git stores every version of every file in history. A 100 MB CSV changed 10 times can add about 1 GB to the repo. Many hosts also cap file sizes (GitHub rejects files over 100 MB in regular Git).

**Git LFS (Large File Storage)** replaces large files in your repo with small **pointer files**, and stores the real content on an LFS server.

```
Working dir:   data/raw/sales.csv        (real 60 MB file)
Git history:   pointer file (~130 bytes) -> oid sha256:abc123..., size 62914560
LFS storage:   actual file content
```

### 3.2 Install and initialise

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows: included with recent Git for Windows installers
# Or download from https://git-lfs.com

git lfs install        # one-time per machine; sets up the Git hooks
```

### 3.3 Hands-on: track and commit a large file

**Step 1: Create a sample dataset (about 60 MB)**

```bash
mkdir -p data/raw

python - << 'EOF'
import csv, random
random.seed(42)
with open("data/raw/sales.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order_id", "customer_id", "region", "amount", "date"])
    regions = ["north", "south", "east", "west"]
    for i in range(1_500_000):
        w.writerow([i, random.randint(1, 50000), random.choice(regions),
                    round(random.uniform(5, 500), 2), "2025-01-%02d" % random.randint(1, 28)])
EOF

ls -lh data/raw/sales.csv
```

**Step 2: Tell LFS which patterns to manage**

```bash
git lfs track "*.csv"
git lfs track "*.parquet"
git lfs track "data/raw/**"

cat .gitattributes
```

> **Important:** commit `.gitattributes` **before or together with** the data file, so LFS handles it from the start.

**Step 3: Commit**

```bash
git add .gitattributes
git add data/raw/sales.csv
git commit -m "Track raw sales data with Git LFS"
```

**Step 4: Verify it worked**

```bash
git lfs ls-files             # lists files managed by LFS
git lfs status               # shows LFS-related staged/unstaged changes
git show HEAD:data/raw/sales.csv   # prints the small pointer file, not the data
```

You should see something like:

```
version https://git-lfs.github.com/spec/v1
oid sha256:<hash>
size <bytes>
```

### 3.4 Working with LFS day to day

| Task | Command |
|------|---------|
| See tracked patterns | `git lfs track` |
| Stop tracking a pattern | `git lfs untrack "*.csv"` |
| List LFS files | `git lfs ls-files` |
| Download LFS files after clone | `git lfs pull` |
| Clone without downloading LFS content | `GIT_LFS_SKIP_SMUDGE=1 git clone <url>` |
| Convert existing history to LFS | `git lfs migrate import --include="*.csv"` |
| Lock a file (avoid binary conflicts) | `git lfs lock <file>` / `git lfs unlock <file>` |

### 3.5 Common pitfalls

1. **Tracking after committing:** if a big file was already committed normally, `git lfs track` alone will not fix history. Use `git lfs migrate import`, and note that it rewrites history, so coordinate with your team.
2. **Forgetting `.gitattributes`:** it must be committed so teammates get the same behaviour.
3. **Storage and bandwidth quotas:** hosting providers limit LFS usage, so check your plan.
4. **Not everything belongs in LFS:** very large or frequently changing datasets are often better in object storage (S3, GCS, Azure Blob) or a data versioning tool (e.g., DVC), with Git tracking only the references.

### Optional exercise (Part III)

1. Append 1,000 rows to `sales.csv` and commit again.
2. Run `git lfs ls-files` and `git log --stat` and compare what Git records vs. LFS.
3. Add `*.parquet` to LFS and explain why it is a good candidate.

---

## Part IV: Structuring Data Projects (25 min)

### 4.1 Why structure matters

A predictable layout lets teammates find things, lets CI/CD tools run scripts, and separates **exploration** (notebooks) from **production** (tested modules).

### 4.2 Recommended layout

```
sales-etl-workshop/
├── README.md                  # what, why, how to run
├── .gitignore
├── .gitattributes             # LFS patterns
├── requirements.txt           # or pyproject.toml
├── .env.example               # template only; real .env is ignored
│
├── config/
│   ├── base.yaml              # shared defaults
│   ├── dev.yaml
│   └── prod.yaml
│
├── data/
│   ├── raw/                   # immutable source data (LFS)
│   ├── interim/               # intermediate (usually ignored)
│   ├── processed/             # final outputs (usually ignored)
│   └── sample/                # tiny sample committed to Git for tests
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_feature_checks.ipynb
│
├── src/
│   └── sales_etl/
│       ├── __init__.py
│       ├── extract.py
│       ├── transform.py
│       ├── load.py
│       └── pipeline.py
│
├── scripts/
│   └── run_pipeline.py        # thin CLI entry point
│
├── tests/
│   └── test_transform.py
│
└── docs/
```

### 4.3 Design principles

| Principle | Practice |
|-----------|----------|
| **Separate code from data** | Data in `data/`, logic in `src/` |
| **Raw data is immutable** | Never edit files in `data/raw/`; write results elsewhere |
| **Notebooks explore, modules ship** | Move stable logic from notebooks into `src/` |
| **Config out of code** | No hard-coded paths, credentials or environment names |
| **Secrets never in Git** | Use `.env` (ignored) and commit `.env.example` |
| **Reproducibility** | Pin dependencies; document how to run |
| **Number notebooks** | `01_`, `02_` communicates the order |

### 4.4 Configuration example

`config/base.yaml`

```yaml
paths:
  raw: data/raw/sales.csv
  processed: data/processed/sales_clean.parquet
pipeline:
  min_amount: 0
  batch_size: 100000
```

`.env.example`

```
DB_HOST=localhost
DB_USER=changeme
DB_PASSWORD=changeme
```

### 4.5 Hands-on (optional): restructure the repo

```bash
mkdir -p config data/interim data/processed data/sample notebooks \
         src/sales_etl scripts tests docs

git mv etl.py src/sales_etl/pipeline.py
touch src/sales_etl/__init__.py
touch data/interim/.gitkeep data/processed/.gitkeep   # keep empty folders

cat > config/base.yaml << 'EOF'
paths:
  raw: data/raw/sales.csv
  processed: data/processed/sales_clean.parquet
EOF

cat > .env.example << 'EOF'
DB_HOST=localhost
DB_USER=changeme
DB_PASSWORD=changeme
EOF

cat > README.md << 'EOF'
# Sales ETL Workshop

Example data project for practising Git, Git LFS, nbdime and feature branching.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    git lfs pull

## Run
    python scripts/run_pipeline.py
EOF

printf "pandas\npyarrow\npyyaml\nnbdime\njupyter\npytest\n" > requirements.txt

git add .
git commit -m "Restructure project into config, data, notebooks, src and tests"
```

> **Tip:** `.gitignore` excludes `data/processed/`, so `.gitkeep` files in ignored folders need `git add -f` if you want to commit them. Alternatively, ignore the contents instead of the folder (e.g., `data/processed/*` and `!data/processed/.gitkeep`).

### Discussion questions

1. Which of your current projects mix raw data, code and outputs in one folder?
2. What belongs in `data/sample/` and why commit it?
3. When should logic move from a notebook into `src/`?

---

## Part V: Notebook Versioning with nbdime (25 min)

### 5.1 The problem

A notebook is JSON. One small code change plus a re-run can produce a diff like this:

```diff
-  "execution_count": 4,
+  "execution_count": 7,
   "outputs": [ { "data": { "image/png": "iVBORw0KGgoAAAANSUhEUg...
```

Reviewers cannot see what actually changed.

### 5.2 What nbdime does

**nbdime** understands notebook structure and provides:

- `nbdiff` / `nbdiff-web`: content-aware diffs (source, outputs and metadata separately)
- `nbmerge` / `nbmerge-web`: three-way merges that avoid corrupt JSON
- Git integration: so `git diff` and `git merge` use it automatically

### 5.3 Install and enable

```bash
pip install nbdime

# Register nbdime as the diff/merge tool for Git (global)
nbdime config-git --enable --global

# Optional: Jupyter integration (diff view inside Jupyter)
nbdime extensions --enable
```

Check that Git picked it up:

```bash
git config --global --list | grep -i nbdime
```

### 5.4 Hands-on demo

**Step 1: Create a notebook**

```bash
python - << 'EOF'
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Sales Exploration"),
    nbf.v4.new_code_cell("import pandas as pd\ndf = pd.read_csv('../data/raw/sales.csv', nrows=1000)\ndf.head()"),
    nbf.v4.new_code_cell("df['amount'].describe()"),
]
nbf.write(nb, "notebooks/01_exploration.ipynb")
EOF

git add notebooks/01_exploration.ipynb
git commit -m "Add initial exploration notebook"
```

**Step 2: Change the notebook**

Open it in Jupyter (`jupyter notebook`), then:

- Edit the markdown title to "Sales Exploration: Regional View"
- Add a cell: `df.groupby('region')['amount'].mean()`
- Run all cells and save

**Step 3: Compare raw Git vs. nbdime**

```bash
# Plain JSON-level comparison (noisy)
git diff --no-ext-diff notebooks/01_exploration.ipynb

# nbdime-aware (readable)
git diff notebooks/01_exploration.ipynb      # uses nbdime after config-git
nbdiff notebooks/01_exploration.ipynb        # compares to the last commit in a terminal view
nbdiff-web notebooks/01_exploration.ipynb    # opens a rich browser-based diff
```

Observe how nbdime separates **source changes** from **output changes**.

**Step 4: Ignore output noise when you only care about code**

```bash
nbdiff -s notebooks/01_exploration.ipynb      # sources only (-s)
nbdiff-web --ignore-outputs notebooks/01_exploration.ipynb
```

**Step 5: Merge conflict demo**

```bash
git add notebooks/01_exploration.ipynb
git commit -m "Add regional analysis to exploration notebook"

# Branch A: edit the describe cell
git switch -c feature/stats-cell HEAD~1
#   change the cell to: df['amount'].describe(percentiles=[.1,.5,.9])
git commit -am "Add percentiles to describe cell"

# Back on main, edit the same cell differently
git switch main
#   change the cell to: df['amount'].agg(['mean','median'])
git commit -am "Use mean and median aggregation"

# Merge, which conflicts on the same cell
git merge feature/stats-cell
```

Resolve using the notebook-aware tool:

```bash
git mergetool          # launches nbdime's merge tool (nbdime config-git sets this up)
# or
nbmerge-web ...        # if you prefer the web UI directly
```

Choose which side to keep for the conflicting cell, save, then:

```bash
git commit             # completes the merge
```

### 5.5 Team practices for notebooks

| Practice | Why |
|----------|-----|
| Clear outputs before committing, or strip them with `nbstripout` | Smaller, cleaner diffs |
| Keep notebooks small and focused | Fewer merge conflicts |
| Avoid two people editing the same notebook at once | Notebook merges are inherently hard |
| Move reusable logic into `src/` | Plain `.py` files diff and review well |
| Restart kernel and run all before committing | Ensures the notebook is reproducible |

Optional: automatically strip outputs on commit.

```bash
pip install nbstripout
nbstripout --install          # sets up a Git filter in this repo
```

### Optional exercise (Part V)

1. Add a plot cell and re-run. Compare `nbdiff -s` vs. `nbdiff`.
2. Install `nbstripout`, commit again, and see how the diff changes.
3. Discuss: when would you *want* outputs committed (e.g., published reports)?

---

## Part VI: Branching Strategies for Feature-Based Development (20 min)

### 6.1 Why branch?

Branches give you **isolation** so you can experiment, build features, or fix bugs without breaking the stable pipeline, and then **collaborate** through reviews and merges.

### 6.2 Branch basics

```bash
git branch                      # list branches
git switch -c feature/add-region-filter   # create + switch
git switch main                 # switch back
git merge feature/add-region-filter       # merge into the current branch
git branch -d feature/add-region-filter   # delete after merge
```

### 6.3 Common strategies

| Strategy | Idea | Best for |
|----------|------|----------|
| **GitHub Flow** | `main` is always deployable; short-lived feature branches; merge via pull request | Most data teams, simple and fast |
| **Git Flow** | `main`, `develop`, plus `feature/*`, `release/*`, `hotfix/*` | Scheduled releases, multiple versions in production |
| **Trunk-based** | Very short branches (or direct commits) merged to `main` many times a day, usually with feature flags | Mature CI/CD teams |

### 6.4 Naming conventions

```
feature/<short-description>     feature/add-region-filter
fix/<short-description>         fix/null-amount-handling
exp/<short-description>         exp/xgboost-baseline
docs/<short-description>        docs/update-readme
```

### 6.5 Feature-branch workflow

```
main:        A --- B ------------------- F --- G
                    \                   /
feature:             C --- D --- E -----
```

1. Update `main`: `git switch main && git pull`
2. Branch: `git switch -c feature/<name>`
3. Commit small, logical changes
4. Push: `git push -u origin feature/<name>`
5. Open a pull request, get a review, and run tests
6. Merge to `main`, then delete the branch

### 6.6 Merge vs. rebase

| | Merge | Rebase |
|---|-------|--------|
| History | Preserves true branching, adds a merge commit | Linear history |
| Safety | Non-destructive | Rewrites commits |
| Rule | Fine on shared branches | **Only rebase your own, unpushed (or personal) branches** |

```bash
# Bring the latest main into your feature branch
git switch feature/add-region-filter
git merge main            # or: git rebase main (personal branches only)
```

### 6.7 Branching for data work: special considerations

- **Experiments:** use `exp/*` branches and merge only what is validated.
- **Large files:** avoid creating many branches that each change big LFS files, because each version consumes storage.
- **Notebooks:** one person per notebook per branch where possible.
- **Protect `main`:** require pull requests and passing tests.
- **Schema changes:** call them out in the PR description so downstream consumers know.

### 6.8 Hands-on (optional): a full feature cycle

```bash
git switch main
git switch -c feature/add-region-filter

cat >> src/sales_etl/pipeline.py << 'EOF'

def filter_region(rows, region):
    return [r for r in rows if r.get("region") == region]
EOF

git add src/sales_etl/pipeline.py
git commit -m "Add region filter to pipeline"

# Simulate a teammate changing main
git switch main
echo "" >> README.md
echo "Updated by teammate." >> README.md
git commit -am "Update README"

# Integrate and merge
git switch feature/add-region-filter
git merge main
git switch main
git merge feature/add-region-filter
git log --oneline --graph --all
```

### 6.9 Resolving a conflict

When Git reports a conflict, open the file and look for:

```
<<<<<<< HEAD
your side
=======
their side
>>>>>>> feature/add-region-filter
```

Edit to the correct final content, remove the markers, then:

```bash
git add <file>
git commit
```

---

## Part VII: Doubt Resolution and Q&A (10 min)

### Quick self-check

1. What goes into Git history when you commit a file tracked by LFS?
2. Why should `.gitattributes` be committed with the data file (or earlier)?
3. Name two reasons raw notebook diffs are hard to read.
4. Which nbdime command opens a browser-based diff?
5. What is the main difference between GitHub Flow and Git Flow?
6. When is rebasing unsafe?

<details>
<summary>Answers</summary>

1. A small pointer file (version, SHA-256 oid, size). The real content lives in LFS storage.
2. So LFS intercepts the file from its first commit and teammates share the same tracking rules.
3. Outputs/binary blobs (e.g., base64 images) and metadata like execution counts make JSON diffs noisy.
4. `nbdiff-web`.
5. GitHub Flow uses `main` plus short-lived feature branches; Git Flow adds long-lived `develop`, `release` and `hotfix` branches for scheduled releases.
6. On shared or already-pushed branches, because it rewrites history and forces others to reconcile.

</details>

### Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| File shows as a tiny text pointer after clone | Run `git lfs install` then `git lfs pull` |
| Push rejected for file over 100 MB | The file was committed outside LFS. Use `git lfs migrate import --include="<pattern>"` |
| `git lfs: command not found` | Install Git LFS, then `git lfs install` |
| `git diff` on notebook still shows raw JSON | Run `nbdime config-git --enable --global`; or use `nbdiff` directly |
| Notebook merge produced invalid JSON | Abort with `git merge --abort`, then retry using `git mergetool` / `nbmerge-web` |
| `.env` was committed by mistake | Remove with `git rm --cached .env`, add it to `.gitignore`, and **rotate the secrets**, since they remain in history |
| Accidentally committed to `main` | `git branch feature/x` then `git reset --hard origin/main` (only if not yet pushed) |

---

## Command Cheat Sheet

```bash
# Basics
git init | git status | git add . | git commit -m "msg"
git log --oneline --graph --all
git diff | git diff --staged
git restore <file> | git restore --staged <file>

# LFS
git lfs install
git lfs track "*.csv"
git lfs ls-files
git lfs pull
git lfs migrate import --include="*.csv"

# Notebooks
nbdime config-git --enable --global
nbdiff <nb> | nbdiff-web <nb>
nbmerge-web <base> <local> <remote>
git mergetool

# Branching
git switch -c feature/<name>
git merge <branch>
git branch -d <branch>
git push -u origin <branch>
```

---

## Wrap-Up

You have:

- Used Git essentials for data and ETL workflows
- Managed large data files with Git LFS
- Organised a data project into a clean, reproducible structure
- Read and merged notebook changes with nbdime
- Applied feature-branch workflows for collaboration

**What's next:** Upcoming module on **Application Deployment**.

---

## Additional Reading

- Official PySpark documentation
- Tutorials on ETL with PySpark (beginner to advanced)
- Example ETL pipeline using PySpark
- Practical overview and best practices

*(Links are provided in the session agenda PDF.)*

Further references for this workshop:

- Git documentation: https://git-scm.com/doc
- Git LFS: https://git-lfs.com
- nbdime documentation: https://nbdime.readthedocs.io
- GitHub Flow: https://docs.github.com/en/get-started/using-github/github-flow

---

## Instructor Notes (optional)

- **Pacing:** Parts III and V carry the live demos. If time is short, shorten the Part IV discussion and leave Part VI's hands-on as take-home.
- **Pre-stage a GitHub repo** (or local bare repo) to demo `push`, `pull` and LFS pointer behaviour on a clone.
- **Fallback for no network:** use `git init --bare ../remote.git` as a local remote.
- **Large-file demo:** generate the CSV before the session to avoid waiting on learners' machines.
