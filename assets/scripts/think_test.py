import json, time, urllib.request, sys

URL = "http://localhost:18080/v1/chat/completions"

def ask(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ch = out["choices"][0]
    msg = ch["message"]
    return {
        "dt": round(dt, 1),
        "finish": ch.get("finish_reason"),
        "completion_tokens": out.get("usage", {}).get("completion_tokens"),
        "content": (msg.get("content") or "")[:300],
        "reasoning": (msg.get("reasoning_content") or "")[:200],
    }

# wait for health
for _ in range(200):
    try:
        urllib.request.urlopen("http://localhost:18080/health", timeout=3)
        break
    except Exception:
        time.sleep(3)
print("server ready")

MATH = "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"

tests = {
 "A_default": {"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}],"max_tokens":200,"temperature":0.2},
 "B_kwarg_nothink": {"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}],"max_tokens":200,"temperature":0.2,"chat_template_kwargs":{"enable_thinking":False}},
 "C_softswitch": {"messages":[{"role":"system","content":"Answer briefly. /no_think"},{"role":"user","content":"What is the capital of France?"}],"max_tokens":200,"temperature":0.2},
 "D_math_think": {"messages":[{"role":"system","content":"Solve step by step. End with 'Answer: <result>'."},{"role":"user","content":MATH}],"max_tokens":1200,"temperature":0.2},
 "E_math_kwarg_nothink": {"messages":[{"role":"system","content":"Solve step by step briefly. End with 'Answer: <result>'."},{"role":"user","content":MATH}],"max_tokens":400,"temperature":0.2,"chat_template_kwargs":{"enable_thinking":False}},
}

for name, p in tests.items():
    try:
        r = ask(p)
    except Exception as e:
        r = {"error": str(e)[:200]}
    print("=====", name)
    for k, v in r.items():
        print(f"  {k}: {v}")
    sys.stdout.flush()
