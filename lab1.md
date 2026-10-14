# Lab: Running, Measuring and Selecting Quantised Models for Agents

*Hands-on workshop for the Model Landscape for Agentic AI session (Sections 3 to 5). Duration: about 90 minutes.*

## Learning goals

By the end of this lab you will be able to:

- Estimate the memory a model needs at different precisions and check the estimate against reality.
- Run a small open-weight model locally in two quantisation levels.
- Measure speed, memory and output quality, and compare the results.
- Test tool-call and structured-output reliability, the capabilities agents depend on.
- Make and justify a model-selection decision for a given agent scenario.

## Prerequisites

- Python 3.10 or newer and basic command-line skills.
- A machine with at least 8 GB of free RAM (16 GB recommended). A GPU is optional.
- About 10 GB of free disk space.
- Familiarity with prompts, tokens and basic tool calling.

You will use **Ollama** as the local inference runtime and the **ollama** Python package. The model is a small open-weight instruct model, **Llama 3.2 3B**, in two quantised builds. If your machine is limited, use the 1B variant. Tag names can change, so confirm exact tags on the Ollama model library page before you start.

## Part 0: Setup (10 min)

**Step 0.1: Install Ollama.** Download it from ollama.com and install it. Confirm it works:

```bash
ollama --version
```

**Step 0.2: Create a Python environment and install the client.**

```bash
python -m venv lab-env
source lab-env/bin/activate   # Windows: lab-env\Scripts\activate
pip install ollama psutil
```

**Step 0.3: Pull two quantised builds of the same model.**

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q8_0
ollama list
```

Note the file size shown for each build.

**Checkpoint:** `ollama list` shows both models, and the Q8 file is roughly twice the size of the Q4 file.

## Part 1: Predict memory before you measure (10 min)

**Step 1.1:** Use this rule of thumb: weights memory is approximately parameters times bytes per parameter.

**Step 1.2:** Fill in the table with your predictions for a 3B model (weights only).

| Precision | Bytes per parameter | Predicted weights size |
| --- | --- | --- |
| FP16 | 2 |  |
| INT8 (Q8) | 1 |  |
| INT4 (Q4) | 0.5 |  |

**Step 1.3:** Compare your Q4 and Q8 predictions with the sizes from `ollama list`.

**Reflection:** Why is the real file larger than your estimate? Consider scaling factors, layers kept at higher precision, and metadata. What else will use memory at run time besides the weights?

## Part 2: Run the models and measure (20 min)

**Step 2.1:** Create a file named `bench.py` with the code below. It runs the same prompts on a given model and records speed from the runtime's own counters.

```python
import sys, ollama

model = sys.argv[1]
prompts = [
    "Explain what a context window is in two sentences.",
    "Write a Python function that checks whether a string is a palindrome.",
    "A customer says: 'My order arrived damaged and I want my money back.' "
    "List the three steps a support agent should take.",
]

for p in prompts:
    r = ollama.chat(model=model, messages=[{"role": "user", "content": p}],
                    options={"temperature": 0})
    tokens = r["eval_count"]
    secs = r["eval_duration"] / 1e9
    print(f"\n--- {p[:50]}...")
    print(r["message"]["content"])
    print(f"[{tokens} tokens, {tokens/secs:.1f} tokens/sec]")
```

**Step 2.2:** Run it for both builds, saving the output:

```bash
python bench.py llama3.2:3b-instruct-q4_K_M > q4_results.txt
python bench.py llama3.2:3b-instruct-q8_0 > q8_results.txt
```

**Step 2.3:** While a model is loaded, check its memory use:

```bash
ollama ps
```

**Step 2.4:** Record your results.

| Measure | Q4 | Q8 |
| --- | --- | --- |
| Memory shown by `ollama ps` |  |  |
| Tokens per second (average) |  |  |
| Quality of answer 1 (1 to 5) |  |  |
| Quality of answer 2 (1 to 5) |  |  |
| Quality of answer 3 (1 to 5) |  |  |

**Checkpoint:** you have two result files and a completed table.

**Reflection:** Is the speed difference what you expected? Is any quality difference visible at temperature 0, or do the outputs look similar? Remember that results depend on your hardware.

## Part 3: Test tool calling and structured output (20 min)

Agents depend on valid tool calls and schema-conforming output. Quantisation can affect these before it affects casual chat, so test them directly.

**Step 3.1:** Create `tools_test.py`:

```python
import sys, json, ollama

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
        ok = (len(calls) == 1 and
              calls[0]["function"]["name"] == "get_order_status" and
              calls[0]["function"]["arguments"].get("order_id") == expected)
    passed += ok
    print("PASS" if ok else "FAIL", "|", question, "|", calls)

print(f"{passed}/{len(cases)} passed")
```

**Step 3.2:** Run it for both builds.

```bash
python tools_test.py llama3.2:3b-instruct-q4_K_M
python tools_test.py llama3.2:3b-instruct-q8_0
```

**Step 3.3:** Create `json_test.py` to check structured output. This asks for JSON and validates it:

```python
import sys, json, ollama

model = sys.argv[1]
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "urgent": {"type": "boolean"},
    },
    "required": ["sentiment", "urgent"],
}
reviews = [
    "Love it, works perfectly!",
    "This is the third time it broke. I need a replacement TODAY.",
    "It arrived on Tuesday.",
    "Terrible quality, but no rush on a refund.",
    "Fantastic service and fast delivery.",
]

valid = 0
for text in reviews:
    r = ollama.chat(model=model, format=schema, options={"temperature": 0},
                    messages=[{"role": "user",
                               "content": f"Classify this review as JSON: {text}"}])
    try:
        data = json.loads(r["message"]["content"])
        ok = data.get("sentiment") in schema["properties"]["sentiment"]["enum"] \
             and isinstance(data.get("urgent"), bool)
    except Exception:
        ok = False
    valid += ok
    print("VALID" if ok else "INVALID", "|", text, "|", r["message"]["content"])

print(f"{valid}/{len(reviews)} valid")
```

```bash
python json_test.py llama3.2:3b-instruct-q4_K_M
python json_test.py llama3.2:3b-instruct-q8_0
```

**Step 3.4:** Record the scores.

| Test | Q4 | Q8 |
| --- | --- | --- |
| Tool-call test (out of 4) |  |  |
| JSON schema test (out of 5) |  |  |

**Reflection:** Did either build fail a case? Check whether the failure was a wrong tool, a wrong argument, an unneeded tool call, or invalid JSON. Five or ten test cases are a very small sample, so what would you do to get trustworthy numbers?

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

## Deliverable

Submit a short write-up (one page) containing:

- Your completed memory, speed and quality tables from Parts 1 to 3.
- Your completed decision matrix from Part 4.
- A paragraph answering: *When is a quantised small model the right choice for an agent, and when is it not?*

## Troubleshooting (debugging practice)

Use these as opportunities to practise diagnosis. Try to identify the cause yourself before reading the hint.

| Symptom | Hint |
| --- | --- |
| `connection refused` from Python | The Ollama service is not running. Start it with `ollama serve` or open the Ollama app. |
| Model not found | The tag is wrong or not pulled. Compare with `ollama list` and the Ollama library page. |
| Very slow responses or the machine freezes | The model is too large for free memory and is swapping. Use the 1B model or close other applications. |
| `KeyError: 'eval_count'` | Print the full response object and check the field names for your client version. |
| Tool test returns no `tool_calls` | Confirm the model supports tools, and check that your `tools` list matches the example format. |
| Results differ between runs | Confirm `temperature` is 0. Some small variation can still occur. |

## Optional extensions

- Add a third build, such as a Q2 or Q3 variant, and see where tool-call reliability breaks down.
- Try a larger model at Q4 against a smaller model at Q8, using your own memory budget, and compare results.
- Increase the prompt length with a long pasted document and observe how latency and memory change, relating this to the KV cache.
- Enlarge the tool and JSON test sets to 30 or more cases and report pass rates.

## Cleanup

```bash
ollama rm llama3.2:3b-instruct-q4_K_M
ollama rm llama3.2:3b-instruct-q8_0
```
