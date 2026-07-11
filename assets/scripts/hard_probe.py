import json, time, urllib.request, sys

URL = "http://localhost:18081/v1/chat/completions"

def ask(sys_p, user_p, max_tokens):
    payload = {
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": user_p}],
        "max_tokens": max_tokens, "temperature": 0.2, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3000) as r:
        out = json.loads(r.read())
    ch = out["choices"][0]
    return (round(time.time() - t0), ch.get("finish_reason"),
            out.get("usage", {}).get("completion_tokens"),
            (ch["message"].get("content") or "").strip())

for _ in range(300):
    try:
        urllib.request.urlopen("http://localhost:18081/health", timeout=3); break
    except Exception:
        time.sleep(4)
print("ready", flush=True)

MATH_SYS = ("You are a careful mathematician. Show the key calculation steps "
            "briefly in plain text (no LaTeX, no headings), double-check the "
            "arithmetic, and keep the whole reply under 120 words. End with a "
            "final line formatted exactly as: Answer: <result>")
LOGIC_SYS = ("You are a logical reasoner. Work through the constraints briefly "
             "in plain text, verify the solution satisfies ALL conditions, and "
             "keep the whole reply under 120 words. End with a final line "
             "formatted exactly as: Answer: <result>")
CODE_SYS = ("You are an expert programmer. Write clean, correct, well-structured "
            "code that fully satisfies the specification, including edge cases. "
            "Return the code in a code block with a one-sentence explanation.")

tests = [
    ("math", MATH_SYS, "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", 800),
    ("logic", LOGIC_SYS, "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", 800),
    ("codegen", CODE_SYS, "Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", 1000),
]
for name, s, u, mt in tests:
    try:
        dt, fin, toks, content = ask(s, u, mt)
        print(f"===== {name} dt={dt}s finish={fin} tokens={toks}", flush=True)
        print(content[:1100], flush=True)
    except Exception as e:
        print(f"===== {name} ERROR {e}", flush=True)
