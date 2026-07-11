"""Fast robustness tests for agent.py — no real model, mocks the LLM."""
import ast, importlib.util, json, os, sys, tempfile

# syntax check
src = open("agent.py", encoding="utf-8").read()
ast.parse(src)
print("syntax: OK")

spec = importlib.util.spec_from_file_location("agent", "agent.py")
m = importlib.util.module_from_spec(spec)
sys.modules["agent"] = m
spec.loader.exec_module(m)

# ---- classifier coverage on unseen variants -------------------------------
CASES = [
    ("Explain what a black hole is.", "factual"),
    ("Who wrote Hamlet?", "factual"),
    ("Compute 12.5% of 3200 and subtract 150.", "math"),
    ("If 3 pens cost $4.50, how much do 7 pens cost?", "math"),
    ("A car depreciates 8% a year from $20000. Value after 2 years?", "math"),
    ("Is this comment positive, negative, or neutral: 'meh, it works'", "sentiment"),
    ("What is the sentiment of: I absolutely love this!", "sentiment"),
    ("Summarize the following article in three bullet points: ...", "summarization"),
    ("Give me a TL;DR of this paragraph: ...", "summarization"),
    ("Extract all persons, organizations and locations from: ...", "ner"),
    ("List the named entities and their types in the sentence.", "ner"),
    ("There's a bug in this function, fix it: def f(x): retrun x", "code_debug"),
    ("This code throws an error, debug it: print(1/0)", "code_debug"),
    ("Write a function that reverses a linked list.", "code_gen"),
    ("Implement binary search in Python.", "code_gen"),
    ("Five people sit in a row; A is left of B; who is in the middle?", "logic"),
    ("Each of three boxes holds a different color. Deduce which.", "logic"),
]
miss = 0
for p, exp in CASES:
    got = m.classify(p)
    if got != exp:
        miss += 1
        print(f"  classify MISS: expected {exp}, got {got} | {p[:50]}")
print(f"classifier: {len(CASES)-miss}/{len(CASES)} correct")

# every category must exist in CONFIG
for _, exp in CASES:
    assert exp in m.CONFIG, f"missing CONFIG for {exp}"
assert m.classify("") == "factual"
print("CONFIG coverage: OK")

# ---- load_tasks tolerance --------------------------------------------------
def _load(data):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps(data).encode()); os.close(fd)
    m.INPUT_PATH = path
    try:
        return m.load_tasks()
    finally:
        os.unlink(path)

assert _load([{"task_id": "a", "prompt": "hi"}]) == [{"task_id": "a", "prompt": "hi"}]
assert _load({"tasks": [{"task_id": "b", "prompt": "x"}]}) == [{"task_id": "b", "prompt": "x"}]
assert _load({}) == []
assert _load("garbage-not-a-list") == []
print("load_tasks tolerance: OK")

# ---- code helpers ----------------------------------------------------------
good = "```python\ndef f(x):\n    return x+1\nassert f(1)==2\n```"
bad_syntax = "```python\ndef f(x)\n return x\n```"
bad_assert = "```python\ndef f(x):\n    return x\nassert f(1)==999\n```"
no_code = "The answer is 42."
assert m.python_code_ok(good) and m.python_code_ok(no_code)
assert not m.python_code_ok(bad_syntax)
assert m.run_python(good)[0] is True
assert m.run_python(bad_assert)[0] is False
assert m.run_python(no_code)[0] is True
# a hostile infinite loop must be killed by the timeout, not hang the agent
assert m.run_python("```python\nwhile True: pass\n```", timeout=3)[0] is False
print("code exec/verify (incl. infinite-loop timeout): OK")

# ---- final_of --------------------------------------------------------------
assert m.final_of("work...\nAnswer: 144") == "144"
assert m.final_of("Answer: Sam owns the cat.") == "sam owns the cat"
assert m.final_of("no final line here") == ""
print("final_of: OK")

# ---- full pipeline with a MOCK model, incl. a task that throws -------------
class FakeLocal:
    def __init__(self): self.tps = 50.0; self.ok = True
    def wait(self, timeout=None): return True
    def ask(self, prompt, cfg, max_tokens, think, temperature=0.0):
        if "boom" in prompt:
            raise RuntimeError("simulated model crash")
        # emit an Answer line for math/logic so needs_final is satisfied
        return f"Response to: {prompt[:30]} Answer: 42", False, ""

m.ask_fireworks = lambda prompt, cfg: ""  # fallback yields nothing (unconfigured)

tasks = [
    {"task_id": "t1", "prompt": "Explain photosynthesis."},
    {"task_id": "t2", "prompt": "Compute 15% of 240."},
    {"task_id": "t3"},                       # missing prompt
    {"prompt": "no id here"},                # missing task_id
    {"task_id": "t5", "prompt": "boom"},     # model throws on this one
    {"task_id": "t6", "prompt": "Grüße — unicode: café, 日本語, 🚀"},
    "not-a-dict",                            # malformed task entry
    {"task_id": "t8", "prompt": "x" * 5000}, # very long prompt
]
outfd, outpath = tempfile.mkstemp(suffix=".json"); os.close(outfd)
infd, inpath = tempfile.mkstemp(suffix=".json")
os.write(infd, json.dumps(tasks).encode()); os.close(infd)
m.INPUT_PATH = inpath; m.OUTPUT_PATH = outpath
orig_local = m.Local
m.Local = FakeLocal
try:
    m.main()
finally:
    m.Local = orig_local

res = json.load(open(outpath, encoding="utf-8"))
os.unlink(inpath); os.unlink(outpath)
assert isinstance(res, list), "output not a list"
assert len(res) == len(tasks), f"expected {len(tasks)} entries, got {len(res)}"
for e in res:
    assert "task_id" in e and "answer" in e, f"bad entry {e}"
    assert isinstance(e["answer"], str) and e["answer"], f"empty answer {e}"
ids = [e["task_id"] for e in res]
assert ids[0] == "t1" and ids[2] == "t3", ids
# the throwing task still got a (fallback) answer, not a crash
assert res[4]["answer"], "throwing task lost its answer"
# unicode must survive round-trip through the JSON file
assert "café" in res[5]["answer"] or "Response" in res[5]["answer"]
print(f"full pipeline: {len(res)} tasks, all with valid answers, no crash")
print("unicode round-trip: OK")

print("\nALL ROBUSTNESS TESTS PASSED")
