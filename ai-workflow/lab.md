# Agentic AI Workflow: Live Workshop Guide

**Format:** Teaching demonstration (3 hours), run from a Jupyter notebook
**Stack:** Python, LangGraph, LangChain, Ollama (local lightweight LLM)
**Machine:** Ubuntu Linux, 2 CPU, 8 GB RAM, no GPU
**Running case study:** *ShopEasy Support Desk*, an automated customer-ticket triage and resolution system

---

## 0. How to use this guide

- Create **one notebook**, `agentic_workflow.ipynb`. Every block marked **CELL** is one notebook cell, to be pasted or live-coded in order.
- Blocks marked **SAY** are talking points. Blocks marked **YOUR TURN** are 2-minute learner activities. Blocks marked **PROMPT LAB** are where you demonstrate the LLM as a productivity partner.
- The whole session uses **one business problem** and one growing codebase, so learners see a simple script become a reliable, stateful, human-supervised workflow.

### Timetable (180 min total, break included)

| Clock | Section | Min | Key demo |
|---|---|---|---|
| 0:00 | 1. Overview | 15 | Problem, setup check |
| 0:15 | 2. Workflow Orchestration Foundations | 25 | Pipeline vs agent loop vs hybrid, retry/fallback |
| 0:40 | 3. Architectures & Execution Models | 30 | Pipeline, DAG, state machine, events, HITL |
| 1:10 | **Break** | 5 | Absorbed by trimming Section 4 to ~45 min via pre-built cells |
| 1:15 | 4. Building with LangGraph | 45 | Nodes, edges, branching, parallel, retry, traces, bug lab |
| 2:00 | 5. Stateful Agents & Memory | 35 | Checkpoints, resume, time travel, trimming |
| 2:35 | 6. End-to-End Integration | 15 | Full run with HITL and trace analysis |
| 2:50 | 7. Conclusion | 10 | Recap and Q&A |


---

### 1. System setup (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip curl zstd git

# Local LLM runtime
curl -fsSL https://ollama.com/install.sh | sh

# Small model: ~1 GB on disk and in RAM, fast enough on 2 CPUs
ollama pull qwen2.5:1.5b

# Project
mkdir -p ~/workshop && cd ~/workshop
python3 -m venv .venv
source .venv/bin/activate
pip install -U langgraph langchain-core langchain-ollama \
    langgraph-checkpoint-sqlite jupyterlab ipykernel grandalf
```

> Python 3.10 or newer is required. Ubuntu 22.04 and 24.04 both qualify.

### 2. Tune Ollama for a 2-CPU / 8 GB machine

```bash
sudo systemctl stop ollama     # stop the default service
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m ollama serve &
```

Run that in a second terminal, then start Jupyter in the first:

```bash
source .venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

### 3. Warm up the model

```bash
ollama run qwen2.5:1.5b "Say ready"
```

### 4. Expectations on this hardware

| Item | Expected |
|---|---|
| Model RAM | about 1.5 GB (leaves plenty free) |
| Generation speed | roughly 5 to 12 tokens/sec |
| One classification call | about 1 to 3 s |
| One 2-sentence draft | about 5 to 12 s |
| Full 5-ticket run with the LLM | about 1 to 2 min |

**Safety net:** every LLM call in this workshop goes through one helper with a `USE_LLM` switch. If the machine is slow or Ollama misbehaves, set `USE_LLM = False` and the workflow falls back to rule-based logic. Nothing else breaks, and the fallback itself becomes a teaching moment.

### 5. Other prep

- Run the whole notebook top to bottom once. Save a clean copy as `agentic_workflow_SOLUTION.ipynb`.
- Ask your upGrad Delivery PoC to share the finished notebook and this guide with learners in advance, if you want them to follow along.
- Have a browser tab open on an AI assistant for the 🤖 Prompt Labs.
- Optional: https://mermaid.live for pasting graph diagrams (needs internet).

---

# Section 1: Overview (0:00 to 0:15)

### 1.1 Frame the problem (5 min)

**🗣 SAY:** *"Imagine ShopEasy gets 10,000 support tickets a day. We want AI to handle the routine ones, but a wrong $899 refund is a real business loss. So the question is not 'can an LLM answer tickets?' It is 'how do we make it reliable, controllable, and recoverable?'"*

Draw this on the board:

```
Ticket in -> understand it -> look up facts -> draft reply -> (human approval if risky) -> send
```

Our workflow requirements, which we will meet one by one:

| Requirement | Technique | Section |
|---|---|---|
| Understand ticket type | LLM classification with validation | 2, 4 |
| Do not hang on flaky services | Retry and fallback | 2, 4 |
| Fetch data fast | Parallel paths | 3, 4 |
| Do not auto-refund large amounts | Human-in-the-loop | 3, 5, 6 |
| Survive crashes, resume later | Checkpointing | 5 |
| Keep long chats cheap | Memory pruning | 5 |
| Know what actually happened | Execution traces | 4, 6 |

### 1.2 Environment check

**▶ CELL 1: Imports, LLM helper, data**

```python
import json, time, operator, sqlite3, queue, threading
from collections import defaultdict
from typing import TypedDict, Annotated, Optional

from langchain_ollama import ChatOllama

MODEL = "qwen2.5:1.5b"
USE_LLM = True   # set False to run the whole workshop without an LLM

def call_llm(prompt: str, json_mode: bool = False, temperature: float = 0.0,
             max_tokens: int = 150) -> str:
    """Single choke point for every LLM call (easy to retry, log, or disable)."""
    if not USE_LLM:
        raise RuntimeError("LLM disabled")
    llm = ChatOllama(
        model=MODEL, temperature=temperature, num_ctx=2048,
        num_predict=max_tokens, keep_alive="30m",
        format="json" if json_mode else None,
    )
    return llm.invoke(prompt).content

# ---- ShopEasy data (our "databases") ----
ORDERS = {
    "A100": {"item": "Headphones", "amount": 49.0,  "status": "delivered"},
    "A200": {"item": "Laptop",     "amount": 899.0, "status": "delivered"},
    "A300": {"item": "Phone case", "amount": 15.0,  "status": "in_transit"},
}
ORDER_CACHE = {  # stale copy used as a fallback
    "A300": {"item": "Phone case", "amount": 15.0, "status": "in_transit (cached)"},
}
POLICY = {
    "refund": "Refunds are allowed within 30 days for defective or wrong items. Refunds over $100 need manager approval.",
    "shipping": "Standard delivery takes 5-7 days. If delayed more than 3 days, share the tracking link and offer a 10% voucher.",
    "technical": "Ask for app version and OS. Suggest reinstall and clearing cache. Escalate to Level 2 if it persists.",
}
TICKETS = {
    "T1": {"order_id": "A100", "text": "I received my headphones but the left ear is dead. I want my money back."},
    "T2": {"order_id": "A200", "text": "The laptop I bought is not what I ordered. Please refund me."},
    "T3": {"order_id": "A300", "text": "Where is my phone case? It was supposed to arrive three days ago."},
    "T4": {"order_id": "A100", "text": "Do you sell gift cards, and can I pay in instalments?"},
    "T5": {"order_id": "A100", "text": "The app crashes whenever I try to log in on my Android phone."},
}

def keyword_classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["refund", "money back", "return"]): return "refund"
    if any(k in t for k in ["where is", "delivery", "arrive", "tracking"]): return "shipping"
    if any(k in t for k in ["crash", "error", "log in", "login", "password"]): return "technical"
    return "unknown"

print("Tickets loaded:", list(TICKETS))
```

**▶ CELL 2: Smoke test**

```python
t0 = time.time()
print(call_llm("Reply with exactly: ready"))
print(f"LLM latency: {time.time()-t0:.1f}s")
```

**🗣 SAY:** *"Every LLM call goes through `call_llm`. That one function is where we will later add retries, logging, or switch the model off. Good orchestration starts with a single control point."*

### 1.3 Why orchestration matters (3 min, spoken)

- LLMs are probabilistic. Production systems need **predictable behaviour around** them.
- **Orchestration** decides *what runs, in what order, under what conditions, and what happens when something fails*.
- **State** is the memory of the run. Without explicit state you cannot pause, resume, audit, or debug.

---

# Section 2: Workflow Orchestration Foundations (0:15 to 0:40)

### 2.1 Deterministic workflow (pipeline) (5 min)

**▶ CELL 3**

```python
def pipeline(ticket):
    cat = keyword_classify(ticket["text"])            # step 1
    order = ORDERS[ticket["order_id"]]                # step 2
    policy = POLICY.get(cat, "We will get back to you shortly.")  # step 3
    return f"[{cat}] About your {order['item']}: {policy}"        # step 4

for tid, t in TICKETS.items():
    print(tid, "->", pipeline(t))
```

**🗣 SAY:** *"Same input, same output, every time. Cheap, testable, auditable. But look at T4 (gift cards): it is rigid. It cannot reason about anything it was not coded for."*

### 2.2 Autonomous agent loop (7 min)

The LLM decides what to do next. We cap the steps because loops can run away.

**▶ CELL 4**

```python
def agent_loop(ticket, max_steps=5, temperature=0.9):
    history = []
    for step in range(max_steps):
        prompt = f"""You are a support agent. Ticket: "{ticket['text']}" (order {ticket['order_id']}).
Actions already done: {history}
Choose the next action from: lookup_order, lookup_policy, finish.
Return JSON like {{"action": "lookup_order"}}"""
        raw = call_llm(prompt, json_mode=True, temperature=temperature, max_tokens=30)
        try:
            action = json.loads(raw).get("action", "INVALID")
        except Exception:
            action = "INVALID"
        print(f"  step {step}: {action}")
        if action == "finish":
            break
        history.append(action)
    return history

for run_no in range(3):
    print(f"Run {run_no+1}")
    agent_loop(TICKETS["T1"])
```

**🗣 SAY:** *"Run it three times. Different action sequences, repeated actions, maybe invalid output. Flexible, but we cannot promise what it will do, and we cannot easily add 'never refund over $100 without a human'."*

### 2.3 Hybrid architecture (5 min)

The workflow owns the **skeleton and the rules**. The LLM is used only inside bounded steps (classify, draft), and its output is **validated**.

**▶ CELL 5**

```python
def hybrid(ticket):
    # LLM step, but guarded by validation + deterministic fallback
    try:
        raw = call_llm(
            f'Classify into refund, shipping, technical or unknown. Ticket: "{ticket["text"]}". '
            'Return JSON like {"category": "refund"}', json_mode=True, max_tokens=30)
        cat = json.loads(raw)["category"].strip().lower()
        assert cat in {"refund", "shipping", "technical", "unknown"}, f"bad category {cat}"
        source = "llm"
    except Exception as e:
        cat, source = keyword_classify(ticket["text"]), f"fallback ({type(e).__name__})"
    # Deterministic rule the LLM can never override
    order = ORDERS[ticket["order_id"]]
    needs_human = (cat == "refund" and order["amount"] > 100)
    return cat, source, needs_human

for tid in ["T1", "T2", "T4"]:
    print(tid, hybrid(TICKETS[tid]))
```

| | Pipeline | Agent loop | Hybrid |
|---|---|---|---|
| Predictability | High | Low | High |
| Handles novelty | Poor | Good | Good, inside bounds |
| Testability | Easy | Hard | Moderate |
| Cost | Lowest | Highest | Moderate |
| Best for | Known steps | Open-ended exploration | **Most production systems** |

### 2.4 Retries and fallbacks, in plain Python (5 min)

**▶ CELL 6**

```python
_calls = defaultdict(int)
def reset_services():
    _calls.clear()

def order_api(order_id):
    """Simulated flaky order service."""
    _calls[order_id] += 1
    if order_id == "A200" and _calls[order_id] < 3:
        raise ConnectionError("order service 503")        # transient: worth retrying
    if order_id == "A300":
        raise TimeoutError("order service timeout")       # persistent: use fallback
    return ORDERS[order_id]

def with_retry(fn, attempts=3, base=0.3, fallback=None):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"  attempt {i} failed: {type(e).__name__}: {e}")
            time.sleep(base * 2 ** (i - 1))       # exponential backoff
    print("  all attempts failed -> fallback")
    return fallback() if fallback else None

reset_services()
print("A200 ->", with_retry(lambda: order_api("A200")))
print("A300 ->", with_retry(lambda: order_api("A300"),
                            fallback=lambda: ORDER_CACHE["A300"]))
```

**🗣 SAY:** *"Retry is for problems that go away. Fallback is for problems that do not. Never retry blindly: retrying a bad request just fails three times slower."*

### 🤖 PROMPT LAB 1: LLM as requirements partner

Paste into an AI assistant and show how it speeds up design thinking:

```text
I am building a customer-support ticket workflow with these requirements: classify
tickets, look up order and policy data, draft a reply, and require human approval
for refunds over $100. List the failure modes I should handle (external APIs, LLM
output, approvals, crashes) and suggest, for each, a retry, fallback, or
human-escalation strategy. Keep it as a table.
```

**🗣 SAY:** *"Notice we are not asking it to build everything. We use it to discover edge cases, then we decide."*

### 🧪 YOUR TURN (2 min)

In which of these would you use a pipeline, an agent loop, or a hybrid? (a) nightly invoice export, (b) "research this company and write a brief", (c) refund processing.
*(Expected: pipeline, agent loop with limits, hybrid.)*

---

# Section 3: Workflow Architectures and Execution Models (0:40 to 1:10)

All of Section 3 is plain Python on purpose: learners see the *ideas* before the framework.

### 3.1 Pipeline (3 min)

**▶ CELL 7**

```python
def step_ingest(s):   s["text"] = TICKETS[s["id"]]["text"]; return s
def step_classify(s): s["cat"] = keyword_classify(s["text"]); return s
def step_reply(s):    s["reply"] = f"({s['cat']}) {POLICY.get(s['cat'], 'We will follow up.')}"; return s

PIPELINE = [step_ingest, step_classify, step_reply]
state = {"id": "T5"}
for step in PIPELINE:
    state = step(state)
    print(f"after {step.__name__}: {sorted(state)}")
```

Linear, simple, no branches.

### 3.2 DAG: dependencies and parallelism (8 min)

**▶ CELL 8**

```python
from graphlib import TopologicalSorter
from concurrent.futures import ThreadPoolExecutor

def work(name, secs):
    def _f():
        time.sleep(secs)
    return _f

# node -> set of nodes it depends on
dag = {
    "classify":     {"ingest"},
    "fetch_order":  {"ingest"},      # does NOT need classify, so it can run alongside
    "fetch_policy": {"classify"},
    "draft":        {"fetch_order", "fetch_policy"},
    "send":         {"draft"},
}
tasks = {n: work(n, 1) for n in ["ingest", *dag]}

def run_dag(dag, tasks, parallel=True):
    ts = TopologicalSorter(dag); ts.prepare()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4 if parallel else 1) as ex:
        while ts.is_active():
            ready = list(ts.get_ready())
            print(f"  t={time.time()-t0:3.1f}s ready together: {ready}")
            for f in [ex.submit(tasks[n]) for n in ready]:
                f.result()
            for n in ready:
                ts.done(n)
    print(f"  total {time.time()-t0:.1f}s")

print("Parallel:");   run_dag(dag, tasks, parallel=True)
print("Sequential:"); run_dag(dag, tasks, parallel=False)
```

**🗣 SAY:** *"A DAG is just 'what depends on what'. Everything not connected by a dependency can run in parallel. Fewer waiting steps means lower latency. 'Acyclic' means no loops, which also means it always terminates."*

### 3.3 State machine: controlled transitions (7 min)

**▶ CELL 9**

```python
class IllegalTransition(Exception): pass

TRANSITIONS = {
    ("NEW", "classified"):            "CLASSIFIED",
    ("CLASSIFIED", "auto_ok"):        "RESOLVED",
    ("CLASSIFIED", "needs_approval"): "WAITING_APPROVAL",
    ("CLASSIFIED", "unknown"):        "ESCALATED",
    ("WAITING_APPROVAL", "approved"): "RESOLVED",
    ("WAITING_APPROVAL", "rejected"): "ESCALATED",
}

class TicketFSM:
    def __init__(self): self.state, self.log = "NEW", []
    def fire(self, event):
        nxt = TRANSITIONS.get((self.state, event))
        if nxt is None:
            raise IllegalTransition(f"'{event}' is not allowed in state {self.state}")
        self.log.append((self.state, event, nxt)); self.state = nxt
        return self

fsm = TicketFSM().fire("classified").fire("needs_approval").fire("approved")
print(fsm.log)

try:
    TicketFSM().fire("approved")   # skipping classification and approval!
except IllegalTransition as e:
    print("BLOCKED:", e)
```

**🗣 SAY:** *"The illegal jump from NEW to approved is impossible by construction. A free-form agent loop has no such guarantee. LangGraph is essentially this idea, with typed state and persistence added."*

### 3.4 Event-driven execution (5 min)

Concept: instead of a script *calling* steps, **events trigger handlers**. This decouples producers from consumers, which suits agents that react to the world (a new ticket, an SLA breach, a payment webhook).

**▶ CELL 10**

```python
bus = queue.Queue()

def on_ticket_created(e):  print("   triage ->", keyword_classify(TICKETS[e["ticket_id"]]["text"]))
def on_sla_breached(e):    print("   paging on-call for", e["ticket_id"])
def on_refund_approved(e): print("   issuing refund for", e["ticket_id"])

HANDLERS = {
    "ticket.created":  [on_ticket_created],
    "sla.breached":    [on_sla_breached],
    "refund.approved": [on_refund_approved],
}

def worker():
    while True:
        e = bus.get()
        if e is None:
            bus.task_done(); break
        print("EVENT:", e["type"])
        for h in HANDLERS.get(e["type"], []):
            h(e)
        bus.task_done()

th = threading.Thread(target=worker, daemon=True); th.start()
bus.put({"type": "ticket.created", "ticket_id": "T3"})
bus.put({"type": "sla.breached", "ticket_id": "T3"})
bus.put({"type": "refund.approved", "ticket_id": "T2"})
bus.put(None); bus.join()
```

**🗣 SAY:** *"In §6 the same idea triggers a LangGraph run per `ticket.created` event, which is how real systems are wired to webhooks and message queues."*

### 3.5 Human-in-the-loop checkpoint (4 min)

**▶ CELL 11**

```python
def human_checkpoint(ticket_id, amount):
    ans = input(f"Approve refund of ${amount} for {ticket_id}? [y/N]: ").strip().lower()
    return ans == "y"

if human_checkpoint("T2", ORDERS["A200"]["amount"]):
    print("-> refund issued")
else:
    print("-> escalated to manager")
```

**🗣 SAY:** *"Notice the problem: this blocks the program. A real workflow must pause, save its state, free resources, and resume later, possibly after a restart. That is exactly what checkpointing gives us in Section 5."*

### 3.6 Comparison (3 min)

| Pattern | Flexibility | Control | Reliability | Complexity | Use when |
|---|---|---|---|---|---|
| Pipeline | Low | High | High | Low | Fixed steps |
| DAG | Medium | High | High | Medium | Independent steps, speed matters |
| State machine | Medium | Very high | Very high | Medium | Approvals, lifecycles, compliance |
| Agent loop | Very high | Low | Low | Low to build, high to trust | Open-ended tasks, with guardrails |
| Event-driven | High | Medium | Depends on handlers | High | Reacting to external signals |

**5-minute break at 1:10.**

---

# Section 4: Building Agent Workflows with LangGraph (1:15 to 2:00)

**Concepts (3 min, whiteboard):** `State` (shared typed dict), `Node` (function: state in, partial update out), `Edge` (what runs next), `Conditional edge` (a function picks the next node), `Reducer` (how concurrent updates merge).

### 4.1 State and nodes

**▶ CELL 12**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, interrupt, Command

class TicketState(TypedDict, total=False):
    ticket_id: str
    text: str
    order_id: str
    category: str
    order: Optional[dict]
    order_source: str
    policy: str
    refund_amount: float
    draft: str
    status: str
    approval: dict
    trace: Annotated[list[str], operator.add]   # reducer: updates are appended, not overwritten

CATS = {"refund", "shipping", "technical", "unknown"}

def ingest(state):
    t = TICKETS[state["ticket_id"]]
    return {"text": t["text"], "order_id": t["order_id"], "trace": ["ingest"]}

def classify(state):
    try:
        raw = call_llm(
            f'Classify into refund, shipping, technical or unknown. Ticket: "{state["text"]}". '
            'Return JSON like {"category": "refund"}', json_mode=True, max_tokens=30)
        cat = json.loads(raw)["category"].strip().lower()
        if cat not in CATS:
            raise ValueError(f"bad category {cat!r}")
        src = "llm"
    except Exception as e:
        cat, src = keyword_classify(state["text"]), f"keyword-fallback({type(e).__name__})"
    return {"category": cat, "trace": [f"classify -> {cat} via {src}"]}

def fetch_order(state):
    time.sleep(1)                                   # simulate network latency
    try:
        order, src = order_api(state["order_id"]), "primary"
    except TimeoutError:                            # persistent failure -> fallback to cache
        order, src = ORDER_CACHE.get(state["order_id"]), "stale-cache"
    refund = order["amount"] if (order and state.get("category") == "refund") else 0.0
    return {"order": order, "order_source": src, "refund_amount": refund,
            "trace": [f"fetch_order ({src})"]}
    # NOTE: ConnectionError is deliberately NOT caught here, so LangGraph's RetryPolicy handles it.

def fetch_policy(state):
    time.sleep(1)
    return {"policy": POLICY.get(state["category"], ""), "trace": ["fetch_policy"]}

def draft_reply(state):
    o = state.get("order") or {}
    prompt = f"""You are a polite support agent. Write a reply of at most 2 sentences.
Customer message: {state['text']}
Order: {o}
Policy: {state['policy']}
Do not promise anything the policy does not allow."""
    try:
        draft, src = call_llm(prompt, max_tokens=100), "llm"
    except Exception:
        draft = f"Hello, thanks for contacting us about your {o.get('item', 'order')}. Per our policy: {state['policy']}"
        src = "template-fallback"
    return {"draft": draft, "trace": [f"draft_reply ({src})"]}

def human_approval(state):
    decision = interrupt({                          # pauses the graph here
        "question": "Approve refund?", "ticket": state["ticket_id"],
        "amount": state["refund_amount"], "draft": state["draft"]})
    ok = bool(decision.get("approved"))
    out = {"approval": decision, "trace": [f"human_approval -> {'approved' if ok else 'rejected'}"]}
    if not ok:
        out["status"] = "rejected"
    return out

def send_reply(state):
    return {"status": "resolved", "trace": ["send_reply"]}

def escalate(state):
    return {"status": "escalated_to_human_agent", "trace": ["escalate"]}

# ---- routers (conditional edges) ----
def route_after_classify(state):
    if state["category"] == "unknown":
        return "escalate"
    return ["fetch_order", "fetch_policy"]          # fan out in parallel

def route_after_draft(state):
    return "human_approval" if state.get("refund_amount", 0) > 100 else "send_reply"

def route_after_approval(state):
    return "escalate" if state.get("status") == "rejected" else "send_reply"
```

### 4.2 One builder, progressive features

We build a single function with switches, then turn features on one at a time.

**▶ CELL 13**

```python
def build(*, branch=False, parallel=False, retry=False, hitl=False,
          checkpointer=None, fetch_order_fn=fetch_order, draft_fn=draft_reply):
    g = StateGraph(TicketState)
    g.add_node("ingest", ingest)
    g.add_node("classify", classify)
    g.add_node("fetch_order", fetch_order_fn,
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5,
                                        backoff_factor=2, retry_on=ConnectionError) if retry else None)
    g.add_node("fetch_policy", fetch_policy)
    g.add_node("draft_reply", draft_fn)
    g.add_node("send_reply", send_reply)
    if branch or hitl:
        g.add_node("escalate", escalate)
        g.add_edge("escalate", END)
    if hitl:
        g.add_node("human_approval", human_approval)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")

    # --- after classify ---
    if branch and parallel:
        g.add_conditional_edges("classify", route_after_classify, ["escalate", "fetch_order", "fetch_policy"])
    elif branch:
        g.add_conditional_edges("classify",
            lambda s: "escalate" if s["category"] == "unknown" else "fetch_order",
            ["escalate", "fetch_order"])
    elif parallel:
        g.add_edge("classify", "fetch_order"); g.add_edge("classify", "fetch_policy")
    else:
        g.add_edge("classify", "fetch_order")

    # --- fetch stage ---
    if parallel:
        g.add_edge(["fetch_order", "fetch_policy"], "draft_reply")   # join: wait for BOTH
    else:
        g.add_edge("fetch_order", "fetch_policy"); g.add_edge("fetch_policy", "draft_reply")

    # --- after draft ---
    if hitl:
        g.add_conditional_edges("draft_reply", route_after_draft, ["human_approval", "send_reply"])
        g.add_conditional_edges("human_approval", route_after_approval, ["send_reply", "escalate"])
    else:
        g.add_edge("draft_reply", "send_reply")
    g.add_edge("send_reply", END)
    return g.compile(checkpointer=checkpointer)

def run(app, inp, config=None):
    """Stream node-by-node updates (our live execution trace)."""
    t0, state = time.time(), {}
    for step in app.stream(inp, config, stream_mode="updates"):
        for node, upd in step.items():
            if node == "__interrupt__":
                print(f"  +{time.time()-t0:4.1f}s PAUSED for human: {upd[0].value}")
                continue
            print(f"  +{time.time()-t0:4.1f}s {node:15s} wrote {sorted(upd)}")
            state.update({k: v for k, v in upd.items() if k != "trace"})
            state.setdefault("trace", []).extend(upd.get("trace", []))
    print("  FINAL:", state.get("status"), "|", " > ".join(state.get("trace", [])))
    return state
```

### 4.3 Stage 1: linear graph (5 min)

**▶ CELL 14**

```python
app_v1 = build()
print(app_v1.get_graph().draw_ascii())
reset_services()
run(app_v1, {"ticket_id": "T1"})
```

**🗣 SAY:** *"Nodes, edges, state. That is all of LangGraph's core. Notice the stream shows which node ran and which state keys it wrote."*

### 4.4 Stage 2: conditional branching (5 min)

**▶ CELL 15**

```python
app_v2 = build(branch=True)
print("--- T4 (gift cards, unknown) ---")
run(app_v2, {"ticket_id": "T4"})
print("--- T5 (technical) ---")
run(app_v2, {"ticket_id": "T5"})
```

T4 now skips the lookups and goes straight to `escalate`. Branching depends on **state produced by an earlier node**.

**🧪 YOUR TURN:** ask learners which category the 1.5B model gave T4. Small models can be confidently wrong, and our validation only catches *invalid* labels, not *wrong* ones. That is why high-risk steps need rules or humans on top.

### 4.5 Stage 3: failure, then retry, fallback and parallelism (10 min)

First, the failure. T2's order service returns a transient 503.

**▶ CELL 16: See it fail**

```python
reset_services()
try:
    run(build(branch=True), {"ticket_id": "T2"})        # no retry yet
except Exception as e:
    print("CRASHED:", type(e).__name__, e)
```

Now add retries, then the fallback path (T3's service permanently times out):

**▶ CELL 17: Retry + fallback**

```python
reset_services()
print("--- T2: transient error, retried automatically ---")
app_v3 = build(branch=True, retry=True)
try:
    run(app_v3, {"ticket_id": "T2"})
except Exception as e:
    print("Expected: reached the human-approval step?", type(e).__name__)
```

> `app_v3` has no `hitl`, so T2 simply flows to `send_reply`. We are only proving retry here. The approval gate comes next.

```python
reset_services()
print("--- T3: persistent timeout, stale-cache fallback ---")
run(app_v3, {"ticket_id": "T3"})
```

Look for `fetch_order (stale-cache)` in the trace.

**▶ CELL 18: Parallel fan-out and join**

```python
reset_services()
print("SEQUENTIAL (fetches one after another):")
t0 = time.time(); run(build(branch=True, retry=True), {"ticket_id": "T5"}); seq = time.time() - t0

print("\nPARALLEL (fetch_order and fetch_policy together):")
t0 = time.time(); run(build(branch=True, retry=True, parallel=True), {"ticket_id": "T5"}); par = time.time() - t0
print(f"\nsequential {seq:.1f}s vs parallel {par:.1f}s (saved ~1s of simulated latency)")
```

**🗣 SAY:** *"Two nodes in one step run concurrently. The join edge `['fetch_order','fetch_policy'] -> draft_reply` waits for both. The `trace` reducer is why both nodes can write to it in the same step without conflict."*

### 4.6 Visualisation and execution traces (7 min)

**▶ CELL 19: Final graph, visualised**

```python
app_full = build(branch=True, parallel=True, retry=True, hitl=True)
print(app_full.get_graph().draw_ascii())          # works offline (needs grandalf)
print(app_full.get_graph().draw_mermaid())         # paste into https://mermaid.live
```

Optional, with internet: `from IPython.display import Image; Image(app_full.get_graph().draw_mermaid_png())`

**▶ CELL 20: Low-level debug trace**

```python
reset_services()
for ev in app_full.stream({"ticket_id": "T5"}, stream_mode="debug"):
    print(ev["step"], ev["type"], ev["payload"].get("name", ""))
```

**🗣 SAY:** *"Three trace levels: `updates` (what each node changed), `debug` (every task start and finish), and later `get_state_history` (every saved checkpoint). Optional: LangSmith gives hosted traces by setting `LANGSMITH_TRACING=true` and an API key, but everything we need today is local."*

### 4.7 Bug Lab: a silent failure only the trace reveals (5 min)

**▶ CELL 21: Buggy node**

```python
def fetch_order_buggy(state):
    time.sleep(1)
    order = order_api(state["order_id"])
    refund = order.get("price", 0.0)                # BUG: field is called "amount"
    return {"order": order, "order_source": "primary", "refund_amount": refund,
            "trace": ["fetch_order (primary)"]}

app_buggy = build(branch=True, parallel=True, retry=True, hitl=False, fetch_order_fn=fetch_order_buggy)
```

Build the buggy version **with** the approval gate and a checkpointer (we need one for interrupts):

```python
from langgraph.checkpoint.memory import MemorySaver
app_buggy = build(branch=True, parallel=True, retry=True, hitl=True,
                  checkpointer=MemorySaver(), fetch_order_fn=fetch_order_buggy)
reset_services()
state = run(app_buggy, {"ticket_id": "T2"}, {"configurable": {"thread_id": "buggy-1"}})
```

**What to notice:** the run finishes as `resolved` and **`human_approval` never appears in the trace**. A $899 refund went through unreviewed because `refund_amount` silently became `0.0`. No exception, no error. Only the execution trace shows the gate was skipped.

**▶ CELL 22: Fix and verify**

```python
app_fixed = build(branch=True, parallel=True, retry=True, hitl=True, checkpointer=MemorySaver())
reset_services()
run(app_fixed, {"ticket_id": "T2"}, {"configurable": {"thread_id": "fixed-1"}})
```

Now you should see `PAUSED for human`. The graph stops at `human_approval`.

**🗣 SAY:** *"Lesson: add a trace-based assertion to your tests, such as 'any refund over $100 must contain human_approval in its path'."*

Also show what happens **without** a checkpointer:

```python
try:
    run(build(branch=True, parallel=True, retry=True, hitl=True), {"ticket_id": "T2"})
except Exception as e:
    print("ERROR:", type(e).__name__, "-", str(e)[:200])
```

### 🤖 PROMPT LAB 2: LLM as debugging partner

Copy the error from the cell above, and paste:

```text
I'm using LangGraph. My graph has a node that calls interrupt() for human approval.
When I compile without a checkpointer and run it, I get this error: <paste error>.
Explain the cause, and show the minimal code change to fix it, using an in-memory
checkpointer and a thread_id.
```

Then show a **code-generation** prompt:

```text
Write a LangGraph conditional edge function that routes to "human_approval" when
state["refund_amount"] > 100, otherwise to "send_reply". Use the TypedDict state below: <paste state>.
```

**🗣 SAY:** *"The assistant gets us 80% there quickly. We still verify with the trace. Never trust generated workflow code you have not run."*

---

# Section 5: Stateful Agents and Memory Management (2:00 to 2:35)

### 5.1 Stateless vs stateful (7 min)

**▶ CELL 23**

```python
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage, trim_messages

chat_llm = ChatOllama(model=MODEL, temperature=0.2, num_ctx=2048, num_predict=80, keep_alive="30m")
SYS = SystemMessage("You are a concise ShopEasy support agent. Answer in one or two sentences.")

def chat_node(state: MessagesState):
    return {"messages": [chat_llm.invoke([SYS] + state["messages"])]}

def make_chat(checkpointer=None):
    g = StateGraph(MessagesState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat"); g.add_edge("chat", END)
    return g.compile(checkpointer=checkpointer)

stateless = make_chat()
stateful  = make_chat(MemorySaver())
cfg = {"configurable": {"thread_id": "priya-session-1"}}

q1 = {"messages": [HumanMessage("Hi, I'm Priya and my order number is A300.")]}
q2 = {"messages": [HumanMessage("What is my order number?")]}

print("STATELESS:"); stateless.invoke(q1); print(stateless.invoke(q2)["messages"][-1].content)
print("\nSTATEFUL:");  stateful.invoke(q1, cfg); print(stateful.invoke(q2, cfg)["messages"][-1].content)
```

**🗣 SAY:** *"The only difference is a checkpointer and a `thread_id`. The thread ID **is** the session. Different customer, different thread, isolated memory."*

```python
other = {"configurable": {"thread_id": "rahul-session-1"}}
print(stateful.invoke(q2, other)["messages"][-1].content)   # does not know Priya's order
```

### 5.2 Checkpointing + human-in-the-loop that survives restarts (10 min)

We now persist to **SQLite**, so state outlives the Python process.

**▶ CELL 24: Pause at the approval gate**

```python
import os
from langgraph.checkpoint.sqlite import SqliteSaver

if os.path.exists("workshop.db"): os.remove("workshop.db")      # fresh start
conn = sqlite3.connect("workshop.db", check_same_thread=False)
app = build(branch=True, parallel=True, retry=True, hitl=True, checkpointer=SqliteSaver(conn))

thread = f"T2-{int(time.time())}"
cfg = {"configurable": {"thread_id": thread}}
reset_services()
run(app, {"ticket_id": "T2"}, cfg)

snap = app.get_state(cfg)
print("\nNext node :", snap.next)                       # ('human_approval',)
print("Refund    :", snap.values["refund_amount"])
print("Draft     :", snap.values["draft"])
```

**▶ CELL 25: Simulate a restart, then resume**

```python
conn.close()                                           # pretend the server went down
conn2 = sqlite3.connect("workshop.db", check_same_thread=False)
app2 = build(branch=True, parallel=True, retry=True, hitl=True, checkpointer=SqliteSaver(conn2))

print("State recovered from disk:", app2.get_state(cfg).next)
run(app2, Command(resume={"approved": True, "by": "manager_ravi"}), cfg)
```

**🗣 SAY:** *"The manager could answer in five minutes or five days. The workflow was not holding a thread or a socket. It was just a row in SQLite. Also note: on resume, the interrupted node **re-runs from its start**, so keep side effects (emails, payments) **after** the `interrupt()` call or make them idempotent."*

### 5.3 Crash recovery (5 min)

**▶ CELL 26**

```python
CRASH = {"on": True}
def draft_crashy(state):
    if CRASH["on"]:
        raise RuntimeError("simulated worker crash during draft")
    return draft_reply(state)

app3 = build(branch=True, parallel=True, retry=True, hitl=True,
             checkpointer=SqliteSaver(conn2), draft_fn=draft_crashy)
cfg3 = {"configurable": {"thread_id": f"T1-{int(time.time())}"}}
reset_services()
try:
    run(app3, {"ticket_id": "T1"}, cfg3)
except Exception as e:
    print("CRASH:", e)

print("\nSaved state says next ->", app3.get_state(cfg3).next)   # ('draft_reply',)
CRASH["on"] = False                                               # "deploy the fix"
print("\nResuming (note: fetch nodes do NOT re-run):")
run(app3, None, cfg3)                                             # None = continue from checkpoint
```

**🗣 SAY:** *"We did not re-fetch the order or re-classify. We paid for those once. In a real system, that could be an expensive LLM call or an API with rate limits."*

### 5.4 Checkpoint history and time travel (3 min)

**▶ CELL 27**

```python
history = list(app3.get_state_history(cfg3))
for h in reversed(history):
    print(h.config["configurable"]["checkpoint_id"][:8], "step", h.metadata.get("step"), "next:", h.next)

# Fork: replay from the checkpoint just before draft_reply
before_draft = next(h for h in history if h.next == ("draft_reply",))
print("\nReplaying from earlier checkpoint:")
run(app3, None, before_draft.config)
```

Uses: debugging ("what was the state right before it went wrong?"), what-if experiments, and audit.

### 5.5 Memory management: pruning and context optimisation (10 min)

Long sessions grow the prompt. Prompt size drives **cost, latency, and quality**.

**▶ CELL 28: Build a long conversation and measure three strategies**

```python
history = []
for i in range(1, 13):
    history += [
        HumanMessage(f"Turn {i}: my order A{i}00 has a question about warranty, address and invoice INV-{i}."),
        AIMessage(f"Noted for turn {i}. We will help with order A{i}00 and invoice INV-{i}."),
    ]
final_q = HumanMessage("In five words, what was my most recent order number?")

def measure(label, msgs):
    t0 = time.time()
    r = chat_llm.invoke([SYS] + msgs + [final_q])
    dt = time.time() - t0
    tokens = r.usage_metadata["input_tokens"]
    cost = tokens / 1_000_000 * 3.00        # illustrative price: $3 per 1M input tokens
    print(f"{label:14s} msgs={len(msgs):3d}  input_tokens={tokens:5d}  latency={dt:5.1f}s  "
          f"cost/call=${cost:.6f}  -> {r.content.strip()[:60]}")
    return tokens

# 1) Keep everything
measure("full history", history)

# 2) Trim: keep only the most recent messages (a *view*, original is untouched)
trimmed = trim_messages(history, max_tokens=6, strategy="last", token_counter=len, start_on="human")
measure("trim last 6", trimmed)

# 3) Summarise old turns into one note + keep the last few
old, recent = history[:-4], history[-4:]
summary = call_llm("Summarise these support messages in 2 short sentences, keeping order numbers:\n"
                   + "\n".join(m.content for m in old), max_tokens=80)
measure("summary+recent", [SystemMessage(f"Earlier conversation summary: {summary}")] + recent)
```

**🗣 SAY:** *"Look at input tokens and latency. On our CPU-only box the latency difference is very visible. In production, the same ratio shows up on your invoice. Scale by 10,000 sessions a day and pruning is a budget decision, not an optimisation."*

**▶ CELL 29: Permanent pruning inside the graph (fast, no LLM)**

```python
def echo(state: MessagesState):                 # stand-in for an LLM so this runs instantly
    return {"messages": [AIMessage(f"ack ({len(state['messages'])} msgs in context)")]}

def prune(state: MessagesState):
    old = state["messages"][:-4]                # delete everything except the last 4
    return {"messages": [RemoveMessage(id=m.id) for m in old]}

g = StateGraph(MessagesState)
g.add_node("echo", echo); g.add_node("prune", prune)
g.add_edge(START, "echo"); g.add_edge("echo", "prune"); g.add_edge("prune", END)
pruned_app = g.compile(checkpointer=MemorySaver())

c = {"configurable": {"thread_id": "prune-demo"}}
for i in range(8):
    out = pruned_app.invoke({"messages": [HumanMessage(f"message {i}")]}, c)
    print(f"turn {i}: stored messages = {len(out['messages'])}")
```

Stored messages stay bounded at about 4 instead of growing forever.

### Trade-offs to discuss

| Strategy | Pros | Cons |
|---|---|---|
| Keep everything | Perfect recall | Cost, latency, "lost in the middle", hits context limit |
| Trim (sliding window) | Cheap, simple | Forgets early facts (order number, name) |
| Summarise | Keeps the gist | Extra LLM call, can drop details |
| Extract key facts into state | Precise, cheap | Needs design (what is a "fact"?) |
| Pruning (delete) | Bounded storage | Irreversible; audit loss, so log first |

**Best practice:** store durable facts (customer ID, order ID, decisions) as **explicit state fields**, and treat chat history as disposable.

### 🤖 PROMPT LAB 3: LLM for memory design

```text
My support chatbot's conversations run 50+ turns. Propose a memory strategy that keeps
order IDs and customer decisions, controls token cost, and stays auditable. Compare
sliding window, summarisation, and structured fact extraction in a table.
```

---

# Section 6: End-to-End Workflow Integration (2:35 to 2:50)

Everything together: **event-driven trigger, classify, branch, parallel fetch, retry, fallback, human approval, checkpointing, traces**.

**▶ CELL 30: Full run**

```python
if os.path.exists("final.db"): os.remove("final.db")
conn_f = sqlite3.connect("final.db", check_same_thread=False)
final_app = build(branch=True, parallel=True, retry=True, hitl=True, checkpointer=SqliteSaver(conn_f))

events = queue.Queue()
for tid in TICKETS:
    events.put({"type": "ticket.created", "ticket_id": tid})

results, pending_approvals = {}, []
reset_services()
stamp = int(time.time())
while not events.empty():
    e = events.get()
    tid = e["ticket_id"]
    cfg = {"configurable": {"thread_id": f"{tid}-{stamp}"}}
    print(f"\n=== EVENT ticket.created -> {tid} ===")
    st = run(final_app, {"ticket_id": tid}, cfg)
    snap = final_app.get_state(cfg)
    if snap.next:                                   # paused at a human checkpoint
        pending_approvals.append((tid, cfg))
        print(f"  -> {tid} waiting for human: {snap.next}")
    results[tid] = (st.get("category"), st.get("status"), st.get("trace"))
```

**▶ CELL 31: The manager's inbox, then resume**

```python
print("Pending approvals:", [t for t, _ in pending_approvals])
for tid, cfg in pending_approvals:
    print(f"\n=== Manager approves {tid} ===")
    st = run(final_app, Command(resume={"approved": True, "by": "manager_ravi"}), cfg)
    results[tid] = (results[tid][0], st.get("status"), results[tid][2] + st.get("trace", []))
```

**▶ CELL 32: Outcome table and trace audit**

```python
print(f"{'ticket':6} {'category':10} {'status':26} path")
for tid, (cat, status, trace) in results.items():
    print(f"{tid:6} {str(cat):10} {str(status):26} {' > '.join(trace)}")

# Trace-based policy test (the lesson from the Bug Lab)
for tid, (cat, status, trace) in results.items():
    if cat == "refund" and TICKETS[tid]["order_id"] == "A200":
        assert any("human_approval" in step for step in trace), f"{tid}: large refund skipped approval!"
print("\nPolicy check passed: every refund over $100 passed through human approval.")
```

**▶ CELL 33: Visualise the final system**

```python
print(final_app.get_graph().draw_ascii())
```

### Analysis discussion (use the output on screen)

| Observation in trace | Question | Possible improvement |
|---|---|---|
| T3 used `stale-cache` | Is stale data safe to quote? | Add a "may be outdated" note in the draft; alert on cache use |
| T2 retried twice | Are retries too aggressive? | Add jitter, circuit breaker, metrics |
| T4 escalated | Misclassified or truly unknown? | Review labelled examples, add an eval set |
| Draft came from template fallback | Silent quality drop? | Log fallback rate, alert if above a threshold |
| Approval pending | What if no one answers? | SLA timer, reminder events, auto-escalate |

### Production best practices (reinforce)

1. **Deterministic skeleton, LLM at the edges**, with output validation.
2. **Explicit, typed state**: facts live in state, not in chat history.
3. **Retry only transient errors**, with backoff. Everything else needs a fallback or escalation.
4. **Human gates for irreversible or costly actions**; keep side effects after the gate, or make them idempotent.
5. **Checkpoint everything long-running**; one `thread_id` per session or case.
6. **Cap loops and steps** (`recursion_limit` in the config) and set timeouts.
7. **Test with traces**: assert on paths, not just final answers.
8. **Control context growth**: trim, summarise, or extract facts.
9. **Log fallback and retry rates**; they are your early-warning system.

---

# Section 7: Conclusion (2:50 to 3:00)

### Recap: the journey

```
Plain pipeline -> agent loop -> hybrid -> DAG / state machine -> LangGraph graph
   -> + branching -> + parallel -> + retry/fallback -> + human gate
   -> + checkpoints & sessions -> + memory control -> production-ready, observable workflow
```

| Technique | Problem it solves |
|---|---|
| Conditional branching | Different cases need different paths |
| Parallel execution | Latency |
| Retry | Transient failures |
| Fallback | Persistent failures |
| Human-in-the-loop | Risky or irreversible decisions |
| Checkpointing | Crashes, pauses, resumption, audit |
| Memory management | Cost, latency, context limits |
| Traces | Debugging and verification |

### Structured Q&A (3 to 4 min)

Open with these if the room is quiet:

1. When would you choose a free-form agent over a graph?
2. How would you test a workflow whose nodes call an LLM?
3. What would you log for an audit of refund decisions?
4. How do you stop an approval from waiting forever?

### Common implementation challenges: quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` to Ollama | Server not running | `ollama serve`, then `ollama list` |
| First call takes 20+ s | Cold model load | Warm up; `keep_alive="30m"` |
| Machine freezes or OOM | Model too large | Stay with `qwen2.5:1.5b` or `llama3.2:1b` |
| Interrupt error about checkpointer or thread | No checkpointer, or no `thread_id` in config | Compile with a checkpointer and pass `{"configurable": {"thread_id": ...}}` |
| `InvalidUpdateError` on parallel nodes | Two nodes wrote the same key without a reducer | Use `Annotated[..., reducer]` or write distinct keys |
| Join node never runs | A join edge waits for nodes that did not run | Make all joined nodes run on every path |
| Node ran twice after resume | Interrupted nodes restart from the top | Move side effects after `interrupt()` or make idempotent |
| Weird LLM JSON | Small model drift | `format="json"`, validate, keep a fallback |
| Mermaid PNG fails | No internet | Use `draw_ascii()` or `draw_mermaid()` text |

### Take-home challenges

1. Add a **`billing`** category with its own policy and branch.
2. Add a **timeout node**: if approval is not received in N minutes, auto-escalate.
3. Add a **summary node** that compresses long chats before drafting.
4. Write a **trace-based test suite**: one assertion per business rule.
5. Swap `qwen2.5:1.5b` for `llama3.2:3b` and compare accuracy, latency, and memory.

---

## Appendix A: Instructor troubleshooting while live

- **LLM too slow:** set `USE_LLM = False` and continue. Say it out loud: *"This is exactly the fallback path working."*
- **Learner code diverges:** paste from `agentic_workflow_SOLUTION.ipynb`.
- **Running behind:** skip Cell 27 (time travel), shorten §5.5 to Cell 28 only, and use the pre-built final graph in §6.
- **Running ahead:** do the "approval timeout" take-home challenge live.

## Appendix B: Minimal file layout

```
~/workshop/
├── .venv/
├── agentic_workflow.ipynb            # live-coded notebook
├── agentic_workflow_SOLUTION.ipynb   # pre-run backup
├── workshop.db                       # created in §5
└── final.db                          # created in §6
```
