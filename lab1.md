# Lab: Running, Measuring and Selecting Quantised Models for Agents

*Hands-on workshop for the Model Landscape for Agentic AI session (Sections 3 to 5). Duration: about 90 minutes.*

---

## Table of Contents
1. [Learning goals](#learning-goals)
2. [Prerequisites](#prerequisites)
3. [Part 0: Setup](#part-0-setup-15-min)
4. [Part 1: Predict memory before you measure](#part-1-predict-memory-before-you-measure-10-min)
5. [Part 2: Run the models and measure](#part-2-run-the-models-and-measure-20-min)
6. [Part 3: Test tool calling and structured output](#part-3-test-tool-calling-and-structured-output-20-min)
7. [Part 4: Model-selection workshop](#part-4-model-selection-workshop-20-min)
8. [Deliverable](#deliverable)
9. [Troubleshooting](#troubleshooting-debugging-practice)
10. [Optional extensions](#optional-extensions)
11. [Cleanup](#cleanup)
12. [Instructor notes](#instructor-notes)

---

## Learning goals

By the end of this lab you will be able to:

- Estimate the memory a model needs at different precisions and check the estimate against reality.
- Run a small open-weight model locally in two quantisation levels.
- Measure speed, memory and output quality, and compare the results.
- Test tool-call and structured-output reliability, the capabilities agents depend on.
- Make and justify a model-selection decision for a given agent scenario.

## Prerequisites

- An **Ubuntu 22.04 or 24.04 VM or machine** (the steps below assume Ubuntu), with `sudo` access. On macOS or Windows, install Ollama from ollama.com and adapt the shell commands.
- Python 3.10 or newer and basic command-line skills.
- At least 8 GB of free RAM (16 GB recommended) and 4 or more CPU cores. A GPU is optional.
- About 10 GB of free disk space.
- Familiarity with prompts, tokens and basic tool calling.

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS        | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| CPU       | 4 cores | 8 cores |
| RAM       | 8 GB | 16 GB |
| Disk      | 20 GB | 40 GB |
| GPU       | Not required | NVIDIA (optional, for speed) |

You will use **Ollama** as the local inference runtime and the **ollama** Python package. The model is a small open-weight instruct model, **Llama 3.2 3B**, in two quantised builds. If your machine has 8 GB of RAM or less, use the **1B** variant (tags are given in Part 0). Tag names and sizes can change, so confirm exact tags on the Ollama model library page before you start.

---

## Part 0: Setup (15 min)

> **Tip:** The model downloads are the slowest step. Start Step 0.7 early, and read Part 1 while it runs.

**Step 0.1: Check your machine.**

```bash
nproc          # CPU cores
free -h        # RAM
df -h /        # Disk space
lsb_release -a # OS version
```

**Step 0.2: Update the system and install the tools you need.**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl python3 python3-pip python3-venv htop
python3 --version   # Expected: Python 3.10 or higher
```

**Step 0.3: Install Ollama.**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This script downloads the Ollama binary, installs it to `/usr/local/bin/ollama`, and creates, enables and starts a systemd service called `ollama`.

> **Security tip:** Piping a script from the internet into `sh` runs it with your permissions. That is acceptable on a lab VM. In production, download the script and read it first.

**Step 0.4: Verify the installation and the service.**

```bash
ollama --version
sudo systemctl enable --now ollama
sudo systemctl status ollama     # Expect "active (running)". Press q to exit.
```

If you prefer to run Ollama by hand instead, stop the service first (`sudo systemctl stop ollama`) and then run `ollama serve` in a separate terminal. Do not run both, or the port will already be in use.

> **Leave `OLLAMA_HOST` alone.** By default Ollama listens only on `127.0.0.1:11434`, which is all this lab needs. Ollama has no authentication, so do not bind it to `0.0.0.0` unless you have a firewall in place.

**Step 0.5: Verify the API is running.**

```bash
curl http://localhost:11434/api/tags
```

Expected: a JSON response (an empty model list is fine at this point).

**Step 0.6: Create a Python environment and install the client.**

```bash
python3 -m venv lab-env
source lab-env/bin/activate     # You should now see (lab-env) in your prompt
pip install --upgrade pip
pip install ollama
```

**Step 0.7: Pull two quantised builds of the same model.**

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q8_0
ollama list
```

*Low-memory machines (8 GB or less): use the 1B builds instead, and use these names everywhere below.*

```bash
ollama pull llama3.2:1b-instruct-q4_K_M
ollama pull llama3.2:1b-instruct-q8_0
```

Note the file size shown for each build.

**Step 0.8: Test one model.**

```bash
ollama run llama3.2:3b-instruct-q4_K_M "Say hello in one sentence"
```

Expected: a short greeting. This also loads the model, so later runs start faster.

**Checkpoint:** `ollama list` shows both models, the Q8 file is clearly larger than the Q4 file (roughly 1.5 to 2 times), and the test greeting worked.

---

## Part 1: Predict memory before you measure (10 min)

**Step 1.1:** Use this rule of thumb: weights memory is approximately parameters times bytes per parameter. Llama 3.2 3B has about 3 billion parameters. Use 3 billion for your estimate.

**Step 1.2:** Fill in the table with your predictions for a 3B model (weights only).

| Precision | Bytes per parameter | Predicted weights size |
| --- | --- | --- |
| FP16 | 2 |  |
| INT8 (Q8) | 1 |  |
| INT4 (Q4) | 0.5 |  |

**Step 1.3:** Compare your Q4 and Q8 predictions with the sizes from `ollama list`.

**Reflection:** Why is the real file larger than your estimate? Consider scaling factors, layers kept at higher precision, and metadata. The "Q4_K_M" build is a mixed-precision format that uses a little over 4 bits per weight on average, not exactly 4. What else will use memory at run time besides the weights?

---

## Part 2: Run the models and measure (20 min)

**Step 2.1:** Create a file named `bench.py` with the code below. It runs the same prompts on a given model and records speed from the runtime's own counters.

```python
import sys
import ollama

model = sys.argv[1]
prompts = [
    "Explain what a context window is in two sentences.",
    "Write a Python function that checks whether a string is a palindrome.",
    "A customer says: 'My order arrived damaged and I want my money back.' "
    "List the three steps a support agent should take.",
]

speeds = []
for p in prompts:
    r = ollama.chat(model=model,
                    messages=[{"role": "user", "content": p}],
                    options={"temperature": 0})
    tokens = r["eval_count"]
    secs = r["eval_duration"] / 1e9
    speeds.append(tokens / secs)
    print(f"\n--- {p[:50]}...")
    print(r["message"]["content"])
    print(f"[{tokens} tokens, {tokens / secs:.1f} tokens/sec]")

print(f"\nAverage speed: {sum(speeds) / len(speeds):.1f} tokens/sec")
```

> The speed figure measures generation only, so model loading time does not distort it. The first prompt may still feel slow while the model loads.

**Step 2.2:** Run it for both builds. `tee` lets you read the output on screen and save it:

```bash
python bench.py llama3.2:3b-instruct-q4_K_M | tee q4_results.txt
python bench.py llama3.2:3b-instruct-q8_0 | tee q8_results.txt
```

**Step 2.3:** Check memory use. Right after a run (the model stays loaded for a few minutes), run this in a second terminal:

```bash
ollama ps
free -h
```

`ollama ps` shows the loaded model, its total memory size (which includes working memory such as the KV cache, not just the weights), and how it is split between CPU and GPU. Use `htop` if you want a live view.

**Step 2.4:** Record your results. Score quality from 1 to 5 using the same rubric for both builds: 5 is correct, complete and clear; 3 is mostly correct with gaps; 1 is wrong or unusable. Score without looking at which build produced the answer, if your partner can shuffle the files.

| Measure | Q4 | Q8 |
| --- | --- | --- |
| Memory shown by `ollama ps` |  |  |
| Tokens per second (average) |  |  |
| Quality of answer 1 (1 to 5) |  |  |
| Quality of answer 2 (1 to 5) |  |  |
| Quality of answer 3 (1 to 5) |  |  |

**Checkpoint:** you have two result files and a completed table.

**Reflection:** Is the speed difference what you expected? Is any quality difference visible at temperature 0, or do the outputs look similar? Remember that results depend on your hardware, and that three prompts is a very small sample.

---

## Part 3: Test tool calling and structured output (20 min)

Agents depend on valid tool calls and schema-conforming output. Quantisation can affect these before it affects casual chat, so test them directly.

**Step 3.1:** Create `tools_test.py`:

```python
import sys
import ollama

model = sys.argv[1]

tools = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up the status of an order by its ID",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}]

cases = [
    ("Where is my order A-1042?", "A-1042"),
    ("Can you check on order B-77 for me?", "B-77"),
    ("Status of order ZX-9001 please", "ZX-9001"),
    ("What is the capital of France?", None),   # should NOT call a tool
]

passed = 0
for question, expected in cases:
    r = ollama.chat(model=model,
                    messages=[{"role": "user", "content": question}],
                    tools=tools, options={"temperature": 0})
    calls = r["message"].get("tool_calls") or []
    if expected is None:
        ok = len(calls) == 0
    else:
        ok = (len(calls) == 1
              and calls[0]["function"]["name"] == "get_order_status"
              and calls[0]["function"]["arguments"].get("order_id") == expected)
    passed += ok
    print("PASS" if ok else "FAIL", "|", question, "|", calls)

print(f"{passed}/{len(cases)} passed")
```

**Step 3.2:** Run it for both builds.

```bash
python tools_test.py llama3.2:3b-instruct-q4_K_M
python tools_test.py llama3.2:3b-instruct-q8_0
```

**Step 3.3:** Create `json_test.py`. It tests structured output in two modes, because they measure different things:

- **prompt mode:** you ask for JSON in the prompt, and the model must obey on its own.
- **schema mode:** you pass a JSON schema, and Ollama constrains the output to match it.

Each answer is scored twice: **valid** (parses and matches the schema) and **correct** (the labels are the right ones).

```python
import sys
import json
import ollama

model = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "prompt"   # "prompt" or "schema"

schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "urgent": {"type": "boolean"},
    },
    "required": ["sentiment", "urgent"],
}

# (review text, expected sentiment, expected urgent)
reviews = [
    ("Love it, works perfectly!", "positive", False),
    ("This is the third time it broke. I need a replacement TODAY.", "negative", True),
    ("It arrived on Tuesday.", "neutral", False),
    ("Terrible quality, but no rush on a refund.", "negative", False),
    ("Fantastic service and fast delivery.", "positive", False),
]

instruction = (
    "Classify this customer review. Reply with ONLY a JSON object with keys "
    '"sentiment" (positive, negative or neutral) and "urgent" (true or false).\n\n'
    "Review: "
)

valid = correct = 0
for text, exp_sentiment, exp_urgent in reviews:
    extra = {"format": schema} if mode == "schema" else {}
    r = ollama.chat(model=model, options={"temperature": 0},
                    messages=[{"role": "user", "content": instruction + text}],
                    **extra)
    raw = r["message"]["content"]
    try:
        data = json.loads(raw)
        is_valid = (data.get("sentiment") in schema["properties"]["sentiment"]["enum"]
                    and isinstance(data.get("urgent"), bool))
    except Exception:
        data, is_valid = {}, False
    is_correct = (is_valid
                  and data["sentiment"] == exp_sentiment
                  and data["urgent"] == exp_urgent)
    valid += is_valid
    correct += is_correct
    print("VALID  " if is_valid else "INVALID", "|",
          "CORRECT" if is_correct else "WRONG  ", "|", text, "|", raw)

print(f"{valid}/{len(reviews)} valid, {correct}/{len(reviews)} correct  (mode: {mode})")
```

> The `format` option (structured outputs) needs a reasonably recent Ollama and Python client. If you get an error about `format`, update Ollama by re-running the install script from Step 0.3, and run `pip install --upgrade ollama`.

**Step 3.4:** Run all four combinations.

```bash
python json_test.py llama3.2:3b-instruct-q4_K_M prompt
python json_test.py llama3.2:3b-instruct-q4_K_M schema
python json_test.py llama3.2:3b-instruct-q8_0 prompt
python json_test.py llama3.2:3b-instruct-q8_0 schema
```

**Step 3.5:** Record the scores.

| Test | Q4 | Q8 |
| --- | --- | --- |
| Tool-call test (out of 4) |  |  |
| JSON, prompt mode: valid (out of 5) |  |  |
| JSON, prompt mode: correct (out of 5) |  |  |
| JSON, schema mode: valid (out of 5) |  |  |
| JSON, schema mode: correct (out of 5) |  |  |

**Reflection:**
- Did either build fail a case? Check whether the failure was a wrong tool, a wrong argument, an unneeded tool call, or invalid JSON.
- Compare prompt mode with schema mode. Why is "valid" so much easier to achieve in schema mode? Does schema mode also guarantee "correct"? (Constraining the format forces the structure, not the judgement.)
- Five or ten test cases are a very small sample. What would you do to get trustworthy numbers?

---

## Part 4: Model-selection workshop (20 min)

Work in pairs. Use your measurements and the selection framework from Section 5.

**Scenario:** A company wants an internal agent that answers staff questions about HR policies. It must look up policy documents using a tool, return answers as JSON for a web interface, respond within about 5 seconds, and keep all data on company servers. The only available hardware is one workstation with a consumer GPU.

**Step 4.1:** List the task requirements: complexity, reasoning, tool calling, structured output, context length, modality.

**Step 4.2:** List the system constraints: latency, cost, data governance, hardware and deployment.

**Step 4.3:** Complete the decision matrix. Score each option from 1 (poor) to 5 (strong) for this scenario.

| Criterion | Small model, Q4 | Small model, Q8 | Larger hosted API model |
| --- | --- | --- | --- |
| Meets data-governance requirement |  |  |  |
| Latency |  |  |  |
| Tool-call reliability (use your lab results) |  |  |  |
| Structured-output reliability |  |  |  |
| Cost |  |  |  |
| Fits available hardware |  |  |  |
| Total |  |  |  |

**Step 4.4:** State your decision in two or three sentences. Say what you would test next before committing, such as a larger test set, a mid-sized open-weight model, or a split design where a small model handles routine questions and a larger one handles hard cases.

**Step 4.5:** Present your decision to another pair and compare reasoning.

---

## Deliverable

Submit a short write-up (one page) containing:

- Your completed memory, speed and quality tables from Parts 1 to 3.
- Your completed decision matrix from Part 4.
- A paragraph answering: *When is a quantised small model the right choice for an agent, and when is it not?*

---

## Troubleshooting (debugging practice)

Use these as opportunities to practise diagnosis. Try to identify the cause yourself before reading the hint.

| Symptom | Hint |
| --- | --- |
| `connection refused` from Python, or "Cannot connect to Ollama" | The Ollama service is not running. Check `sudo systemctl status ollama` and start it with `sudo systemctl start ollama` (or run `ollama serve` after stopping the service). |
| Model not found, or an HTTP 404 error | The tag is wrong or not pulled. Compare with `ollama list` and the Ollama library page, then re-run `ollama pull`. |
| First request is very slow or seems to hang | The model may be loading for the first time. Wait 30 to 60 seconds, or pre-warm it with `ollama run <model> "hello"`. |
| Very slow responses or the machine freezes | The model is too large for free memory and is swapping. Use the 1B model, close other applications, and check with `free -h` and `htop`. |
| `pip: command not found` | Run `sudo apt install -y python3-pip`. |
| `(lab-env)` does not appear after activation | Re-create the environment: `python3 -m venv lab-env --clear`, then `source lab-env/bin/activate`. |
| `KeyError: 'eval_count'` | Print the full response object and check the field names for your client version. |
| Tool test returns no `tool_calls` | Confirm the model supports tools, and check that your `tools` list matches the example format. |
| Error about `format` in `json_test.py` | Ollama or the Python client is too old for structured outputs. Update both (see the note under Step 3.3). |
| Results differ between runs | Confirm `temperature` is 0. Some small variation can still occur. |

---

## Optional extensions

- Add a third build, such as a Q2 or Q3 variant, and see where tool-call reliability breaks down.
- Try a larger model at Q4 against a smaller model at Q8, using your own memory budget, and compare results.
- Increase the prompt length with a long pasted document and observe how latency and memory change, relating this to the KV cache.
- Enlarge the tool and JSON test sets to 30 or more cases and report pass rates.

---

## Cleanup

```bash
ollama rm llama3.2:3b-instruct-q4_K_M
ollama rm llama3.2:3b-instruct-q8_0
deactivate                      # leave the Python environment
```

---

## Instructor notes

*Remove this section before sharing the lab with learners.*

**Part 1: expected predictions (3B model, weights only)**

| Precision | Bytes per parameter | Expected prediction |
| --- | --- | --- |
| FP16 | 2 | about 6 GB |
| INT8 (Q8) | 1 | about 3 GB |
| INT4 (Q4) | 0.5 | about 1.5 GB |

Real downloads are typically larger than these (roughly 2 GB for Q4_K_M and 3.4 GB for Q8_0), because the model has slightly more than 3 billion parameters, Q4_K_M averages a little over 4 bits per weight, and some layers stay at higher precision. Check the real sizes on the day, since they can change between releases.

**Part 2 and 3: what to expect (hedged, verify on your hardware)**
- Q4 is usually faster and smaller than Q8, but the size of the speed gain varies a lot with hardware, especially on CPU-only machines.
- Quality differences between Q4 and Q8 at 3B are often small on casual prompts, and may show up first in the tool-call and prompt-mode JSON tests. Do not promise learners a specific outcome. If both builds pass everything, use that to discuss why a larger and harder test set is needed.
- Schema mode should make "valid" near 100% for both builds. The "correct" score is where differences can still appear. This is the key teaching point of Step 3.3.

**Part 4: discussion points**
- The larger hosted API model fails the data-governance requirement outright, which is why constraints are applied before comparing capability.
- Between Q4 and Q8, the trade is memory and speed against possible reliability. Link this to the learners' own measurements.
- A strong answer names what to test next, and may propose routing routine questions to a small model and hard cases to a larger self-hosted model.
