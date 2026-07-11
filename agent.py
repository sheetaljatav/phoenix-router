"""Track 1: Hybrid Token-Efficient Routing Agent.

Strategy: answer every task with a local Qwen3.5-2B model served by llama.cpp
inside the container -> zero Fireworks tokens (best possible token score).
A Fireworks fallback (routed through FIREWORKS_BASE_URL, model taken from
ALLOWED_MODELS) exists purely as a safety net: it fires only if the local
model fails or the 10-minute runtime budget is about to be exceeded.
"""

import json
import os
import re
import sys
import time
import urllib.request

START = time.time()
# hard wall: harness kills us at 10 min (DEADLINE_MIN is a local-test knob)
DEADLINE = START + float(os.environ.get("DEADLINE_MIN", "8.5")) * 60
LLAMA_BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
LLAMA_URL = LLAMA_BASE + "/v1/chat/completions"

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# ---------------------------------------------------------------- categories

CATEGORIES = (
    # (name, compiled regex) ‚Äî first match wins, most specific first
    ("summarization", re.compile(r"\bsummar|\bcondense|\btl;?dr|\babridge|\bshorten\b|\bmain (points?|ideas?)\b|\bkey takeaways?\b", re.I)),
    ("ner", re.compile(r"named entit|entit(y|ies)\b|\b(identify|list|find|extract|label)\b.*\b(people|persons?|organi[sz]ations?|locations?|dates)\b|\b(people|persons?|organi[sz]ations?|locations?|dates)\b.*\b(identify|list|find|extract|label)", re.I | re.S)),
    ("sentiment", re.compile(r"\bsentiment\b|classify.*\b(review|tweet|feedback|comment)|positive.*negative", re.I)),
    ("code_debug", re.compile(r"(bug|fix|debug|error|incorrect|wrong|broken|doesn'?t work|fails?\b).*\b(code|function|def |script|program|implementation)|\b(code|function|def |script|program)\b.*\b(bug|fix|debug|has an? error|incorrect|broken)", re.I | re.S)),
    ("code_gen", re.compile(r"(write|implement|create|build|develop)\b.*\b(function|program|script|class|method|code)", re.I)),
    ("math", re.compile(r"\d.*(how (many|much)|calculate|compute|total|percent|%|remain|left over|cost|price|profit|revenue|average|sum of|difference|product of|per\b|each\b|altogether|in all)|what is \d|\d\s*[-+*/^]\s*\d", re.I | re.S)),
    ("logic", re.compile(r"who (owns|has|is|likes|lives)|each (own|have|like|live)|exactly one|either\b.*\bor\b|neither|logic|puzzle|deduce|seated|sits|taller|older than|younger than|to the (left|right) of", re.I)),
)


def classify(prompt: str) -> str:
    for name, rx in CATEGORIES:
        if rx.search(prompt):
            return name
    return "factual"


# Per-category generation config. Local tokens are free, so budgets are set
# by wall-clock speed and answer quality for the LLM judge, not token cost.
# think=True lets the model reason before answering (math and logic).
CONFIG = {
    "factual":       dict(max_tokens=250, think=False, sys=(
        "You are a precise assistant. Answer the question accurately and "
        "completely in 1-3 sentences. Cover every part of the question.")),
    "math":          dict(max_tokens=800, think=False, sys=(
        "You are a careful mathematician. Show the key calculation steps "
        "briefly in plain text (no LaTeX, no headings), double-check the "
        "arithmetic, and keep the whole reply under 120 words. End with a "
        "final line formatted exactly as: Answer: <result>")),
    "sentiment":     dict(max_tokens=150, think=False, sys=(
        "You are a sentiment analyst. State the sentiment label (Positive, "
        "Negative, Neutral, or Mixed) first, then justify it in one short "
        "sentence citing the relevant wording.")),
    "summarization": dict(max_tokens=250, think=False, sys=(
        "You are an expert summarizer. Follow the requested format and "
        "length constraint EXACTLY (e.g. 'one sentence' means exactly one "
        "sentence). Preserve the key facts; no preamble.")),
    "ner":           dict(max_tokens=250, think=False, sys=(
        "You are a named-entity recognition system. List every entity in "
        "the text with its type, one per line, as: Entity - Type. Use types "
        "Person, Organization, Location, Date, Time, and similar. No extra "
        "commentary.")),
    "code_debug":    dict(max_tokens=900, think=False, sys=(
        "You are an expert developer. Identify the bug in one sentence, "
        "then provide the complete corrected code in a code block.")),
    "logic":         dict(max_tokens=800, think=False, sys=(
        "You are a logical reasoner. Work through the constraints briefly "
        "in plain text, verify the solution satisfies ALL conditions, and "
        "keep the whole reply under 120 words. End with a final line "
        "formatted exactly as: Answer: <result>")),
    "code_gen":      dict(max_tokens=1000, think=False, sys=(
        "You are an expert programmer. Write clean, correct, well-structured "
        "code that fully satisfies the specification, including edge cases. "
        "Return the code in a code block with a one-sentence explanation.")),
}

THINK_RX = re.compile(r"<think>.*?</think>", re.S)
CODE_BLOCK_RX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def python_code_ok(answer: str) -> bool:
    """True if the answer's python code block parses (or there is none)."""
    blocks = CODE_BLOCK_RX.findall(answer)
    if not blocks:
        return True
    import ast
    try:
        for b in blocks:
            ast.parse(b)
        return True
    except SyntaxError:
        return False


def run_python(answer: str, timeout=10):
    """Execute the answer's code blocks (incl. self-asserts). -> (ok, err)"""
    blocks = CODE_BLOCK_RX.findall(answer)
    if not blocks:
        return True, ""
    import subprocess, tempfile
    path = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False) as f:
            f.write("\n\n".join(blocks))
            path = f.name
        r = subprocess.run([sys.executable or "python3", path],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr or "").strip()[-400:]
    except Exception as e:
        return False, str(e)[:200]
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def post_json(url: str, payload: dict, headers: dict, timeout: int):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def clean(text: str) -> str:
    text = THINK_RX.sub("", text)
    # drop an unterminated think block (max_tokens cut mid-thought)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


# ------------------------------------------------------------- local model

class Local:
    def __init__(self):
        self.tps = 10.0  # generation speed estimate, refined per request

    def wait(self, timeout=None):
        # cap the health wait well below the run budget: if the server is
        # not up in 4 min it never will be, and the remaining ~5 min must
        # go to the Fireworks fallback instead of more polling
        end = min(DEADLINE - 60, time.time() + 240) \
            if timeout is None else time.time() + timeout
        up = False
        while time.time() < end:
            try:
                with urllib.request.urlopen(LLAMA_BASE + "/health", timeout=3) as r:
                    if r.status == 200:
                        up = True
                        break
            except Exception:
                time.sleep(1.5)
        if not up:
            # surface the server log in container logs so a dead local
            # model is diagnosable from the outside
            try:
                with open("/tmp/llama.log") as f:
                    tail = "\n".join(f.read().splitlines()[-15:])
                print(f"[llama-server never ready]\n{tail}", file=sys.stderr)
            except Exception:
                print("[llama-server never ready; no log]", file=sys.stderr)
            return False
        # warm-up: pages in cold weights so tps measurement is honest
        try:
            t0 = time.time()
            out = post_json(LLAMA_URL, {
                "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                "max_tokens": 40, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }, {}, timeout=300)
            done = out.get("usage", {}).get("completion_tokens", 0)
            dt = time.time() - t0
            if done >= 20 and dt > 0:
                self.tps = max(1.0, done / dt)
        except Exception as e:
            print(f"[warmup fail] {e}", file=sys.stderr)
        return True

    def ask(self, prompt, cfg, max_tokens, think, temperature=0.0):
        sys_prompt = cfg["sys"]
        # hard wall-clock cap: never let one request outlive the run budget
        cap = max(20, min(600, int(DEADLINE - time.time()) + 30))
        t0 = time.time()
        out = post_json(LLAMA_URL, {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": think},
        }, {}, timeout=cap)
        dt = time.time() - t0
        usage = out.get("usage", {})
        done = usage.get("completion_tokens", 0)
        if done > 20 and dt > 1:
            self.tps = 0.7 * self.tps + 0.3 * (done / dt)
        choice = out["choices"][0]
        text = clean(choice["message"].get("content") or "")
        truncated = choice.get("finish_reason") == "length"
        salvage = ""
        if not text and truncated:
            # cut off mid-reasoning: keep the tail of the thought stream,
            # which often already contains the computed result
            thought = choice["message"].get("reasoning_content") or ""
            if not thought:
                raw = choice["message"].get("content") or ""
                m = re.search(r"<think>(.*)", raw, re.S)
                thought = m.group(1) if m else ""
            if thought.strip():
                salvage = " ".join(thought.strip().splitlines()[-3:]).strip()
        return text, truncated, salvage


def checked_code(local, prompt, cfg, budget):
    """Generate code, test it against separately-generated asserts, retry once."""
    answer, _, _ = local.ask(prompt, cfg, budget, think=False)
    if not answer or DEADLINE - time.time() < 60:
        return answer
    if not python_code_ok(answer):
        retry, _, _ = local.ask(
            prompt + "\n\n(Ensure the code is syntactically valid.)",
            cfg, budget, think=False)
        if retry and python_code_ok(retry):
            answer = retry
    code = "\n\n".join(CODE_BLOCK_RX.findall(answer))
    if not code:
        return answer
    try:
        tests, _, _ = local.ask(
            "Task specification:\n" + prompt +
            "\n\nHere is a candidate solution:\n```python\n" + code +
            "\n```\nWrite exactly 3 assert statements (code only, no prose, "
            "no code fences) that test the function against the "
            "specification. Derive expected values from the specification, "
            "not from the code.",
            cfg, 200, think=False)
    except Exception:
        return answer
    tests = "\n".join(l for l in tests.splitlines()
                      if l.strip().startswith("assert"))
    if not tests:
        return answer
    ok, err = run_python("```python\n" + code + "\n\n" + tests + "\n```")
    if ok or DEADLINE - time.time() < 60:
        return answer
    try:
        retry, _, _ = local.ask(
            prompt + "\n\nA previous solution failed this test:\n" + err +
            "\nWrite a corrected solution.", cfg, budget, think=False)
    except Exception:
        return answer
    rcode = "\n\n".join(CODE_BLOCK_RX.findall(retry or ""))
    if rcode and run_python("```python\n" + rcode + "\n\n" + tests + "\n```")[0]:
        return retry
    return answer


# -------------------------------------------------------- fireworks fallback

def fireworks_models():
    models = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    # prefer the smallest/cheapest model by parameter-count hints in the id
    def size_key(mid):
        hits = re.findall(r"(\d+(?:\.\d+)?)[bB]\b", mid)
        return min([float(h) for h in hits], default=999.0)
    return sorted(models, key=size_key)


# discovered-good endpoint path + consecutive-failure circuit breaker
FW_STATE = {"path": None, "fails": 0}


def ask_fireworks(prompt, cfg):
    base = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
    key = os.environ.get("FIREWORKS_API_KEY", "")
    models = fireworks_models()
    if not base or not models:
        raise RuntimeError("fireworks not configured")
    if FW_STATE["fails"] >= 3:
        raise RuntimeError("fireworks circuit open")
    # the harness may hand us a base with or without /v1 ‚Äî probe both once,
    # then stick with whichever worked
    if FW_STATE["path"]:
        paths = [FW_STATE["path"]]
    elif base.endswith("/v1"):
        paths = ["/chat/completions"]
    else:
        paths = ["/chat/completions", "/v1/chat/completions"]
    last_err = None
    for path in paths:
        for model in models[:3]:
            if DEADLINE - time.time() < 10:
                raise RuntimeError("out of time")
            try:
                out = post_json(base + path, {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": cfg["sys"]},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": min(cfg["max_tokens"], 600),
                    "temperature": 0.2,
                }, {"Authorization": f"Bearer {key}"}, timeout=20)
                text = clean(out["choices"][0]["message"]["content"])
                if text:
                    FW_STATE["path"] = path
                    FW_STATE["fails"] = 0
                    return text
            except Exception as e:
                last_err = e
    FW_STATE["fails"] += 1
    raise last_err or RuntimeError("fireworks failed")


# -------------------------------------------------------------------- main

def solve(task, local, tasks_left):
    prompt = task.get("prompt", "")
    cfg = CONFIG[classify(prompt)]
    remaining = DEADLINE - time.time()
    answer = ""

    if local.ok and remaining > 15:
        # token budget: fair share of remaining wall time at measured speed,
        # clamped so one generation can never outrun the per-request cap
        share = min(max(6.0, remaining / max(tasks_left, 1)), 500.0)
        budget = int(min(cfg["max_tokens"], max(120, share * local.tps)))
        think = cfg["think"] and budget >= 500
        cat = classify(prompt)
        time_rich = remaining / max(tasks_left, 1) > 45
        try:
            # all categories answer at temperature 0: deterministic output,
            # no sampling variance (majority-voting at temp 0.8 was measurably
            # flipping correct logic answers to wrong ones)
            if cat == "code_gen" and time_rich:
                answer = checked_code(local, prompt, cfg, budget)
            else:
                answer, truncated, salvage = local.ask(
                    prompt, cfg, budget, think)
                needs_final = cat in ("math", "logic") \
                    and "answer:" not in answer.lower()
                if truncated and (not answer or needs_final) \
                        and DEADLINE - time.time() > 30:
                    # cut off mid-answer: get a compact conclusion, free
                    short, _, s2 = local.ask(
                        prompt + "\n\nReply in at most 60 words and end "
                        "with a line formatted exactly as: Answer: <result>",
                        cfg, 250, think=False)
                    if short and answer:
                        answer = answer.rstrip() + "\n" + short
                    else:
                        answer = answer or short
                    salvage = salvage or s2
                if not answer:
                    answer = salvage
                if answer and cat in ("code_gen", "code_debug") \
                        and not python_code_ok(answer) \
                        and DEADLINE - time.time() > 45:
                    retry, _, _ = local.ask(
                        prompt + "\n\n(Ensure the code is syntactically "
                        "valid.)", cfg, budget, think)
                    if retry and python_code_ok(retry):
                        answer = retry
        except Exception as e:
            print(f"[local fail] {task.get('task_id')}: {e}", file=sys.stderr)
            if DEADLINE - time.time() > 45:
                try:  # quick direct answer beats no answer, still free
                    answer, _, s2 = local.ask(prompt, cfg, 200, think=False)
                    answer = answer or s2
                except Exception as e2:
                    print(f"[local retry fail] {task.get('task_id')}: {e2}",
                          file=sys.stderr)

    if not answer and DEADLINE - time.time() > 10:
        # safety net ‚Äî the only path that spends Fireworks tokens
        try:
            answer = ask_fireworks(prompt, cfg)
        except Exception as e:
            print(f"[fireworks fail] {task.get('task_id')}: {e}", file=sys.stderr)

    return answer or "Unable to answer within constraints."


def main():
    with open(INPUT_PATH) as f:
        tasks = json.load(f)

    local = Local()
    local.ok = local.wait()
    # startup diagnostics in container logs (no secret values, presence only)
    print(f"[startup] local_ok={local.ok} tasks={len(tasks)} "
          f"fw_base={'set' if os.environ.get('FIREWORKS_BASE_URL') else 'MISSING'} "
          f"fw_key={'set' if os.environ.get('FIREWORKS_API_KEY') else 'MISSING'} "
          f"allowed_models={len(fireworks_models())}", flush=True)

    # easy (no-think) categories first: if time gets tight, only the
    # reasoning-heavy tasks see shrunken budgets
    order = sorted(range(len(tasks)),
                   key=lambda i: CONFIG[classify(tasks[i].get("prompt", ""))]["think"])
    answers = {}
    for n, i in enumerate(order):
        task = tasks[i]
        answer = solve(task, local, len(tasks) - n)
        answers[i] = answer
        print(f"[{n+1}/{len(tasks)}] {task.get('task_id')} "
              f"cat={classify(task.get('prompt', ''))} len={len(answer)} "
              f"tps={local.tps:.1f} t={time.time()-START:.0f}s", flush=True)

    results = [{"task_id": t.get("task_id", str(i)), "answer": answers[i]}
               for i, t in enumerate(tasks)]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"done in {time.time()-START:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # never exit non-zero without writing *something*
        print(f"[fatal] {e}", file=sys.stderr)
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            if not os.path.exists(OUTPUT_PATH):
                with open(OUTPUT_PATH, "w") as f:
                    json.dump([], f)
        except Exception:
            pass
    sys.exit(0)
