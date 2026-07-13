"""End-to-end test of agent.py routing against mock llama + Fireworks servers.

Scenarios:
  1. fast-local  : local answers everything, ZERO Fireworks requests
  2. slow-local  : speed probe times out -> everything routes to Fireworks
  3. dead-local  : no llama server at all -> everything routes to Fireworks
  4. all-dead    : no servers -> complete placeholder results, exit 0

Run:  python scripts/mock_e2e.py
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(ROOT, "agent.py")
TASKS = os.path.join(ROOT, "input", "tasks.json")
OUT = os.path.join(ROOT, "scripts", "_mock_results.json")

LLAMA_PORT, FW_PORT = 18080, 18081


class Counter:
    def __init__(self):
        self.chat = 0
        self.lock = threading.Lock()


def make_llama_handler(counter, delay):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        def do_POST(self):
            with counter.lock:
                counter.chat += 1
            body = self.rfile.read(int(self.headers["Content-Length"]))
            req = json.loads(body)
            user = req["messages"][-1]["content"]
            time.sleep(delay)
            if "arithmetic expression" in user:
                content = "240 - 36 - 60"
            elif "assert" in user:
                content = "assert True"
            elif "```" in user or "function" in user.lower():
                content = "```python\ndef f(x):\n    return x\n```"
            else:
                content = "The capital is Canberra, near Lake Burley Griffin."
            resp = {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 24}}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return H


def make_fw_handler(counter):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            with counter.lock:
                counter.chat += 1
            body = self.rfile.read(int(self.headers["Content-Length"]))
            req = json.loads(body)
            assert "Bearer" in self.headers.get("Authorization", ""), "no auth"
            user = req["messages"][-1]["content"]
            if "arithmetic expression" in req["messages"][0].get(
                    "content", "") + user:
                content = "240 - 36 - 60"
            elif "OK" in user:
                content = "OK"
            else:
                content = "```python\ndef f(x):\n    return x\n```" \
                    if "code" in user.lower() else "Concise FW answer."
            resp = {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"total_tokens": 30}}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return H


def start(port, handler):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run_agent(env_extra, timeout=180):
    env = dict(os.environ)
    env.update({
        "INPUT_PATH": TASKS,
        "OUTPUT_PATH": OUT,
        "LLAMA_BASE": f"http://127.0.0.1:{LLAMA_PORT}",
        "FIREWORKS_BASE_URL": f"http://127.0.0.1:{FW_PORT}",
        "FIREWORKS_API_KEY": "test-key",
        "ALLOWED_MODELS": "accounts/fireworks/models/minimax-m3,"
                          "accounts/fireworks/models/kimi-k2p7-code",
        "BEACON_URL": "",
    })
    env.update(env_extra)
    if os.path.exists(OUT):
        os.remove(OUT)
    t0 = time.time()
    r = subprocess.run([sys.executable, AGENT], capture_output=True,
                       text=True, timeout=timeout, env=env)
    return r, time.time() - t0


def check_results(n_tasks=8):
    with open(OUT) as f:
        res = json.load(f)
    assert isinstance(res, list) and len(res) == n_tasks, \
        f"bad results: {len(res)}"
    for r in res:
        assert set(r) == {"task_id", "answer"}, f"bad keys: {r.keys()}"
    return res


def blanks(res):
    return sum("Unable to answer" in r["answer"] for r in res)


def main():
    n_tasks = len(json.load(open(TASKS)))
    fails = []

    # ---- scenario 1: fast local -> all local, zero FW requests
    lc, fc = Counter(), Counter()
    l = start(LLAMA_PORT, make_llama_handler(lc, delay=0.2))
    f = start(FW_PORT, make_fw_handler(fc))
    r, dt = run_agent({"DEADLINE_MIN": "3", "LLAMA_WAIT_S": "20"})
    res = check_results(n_tasks)
    ok = r.returncode == 0 and blanks(res) == 0 and fc.chat == 0
    print(f"S1 fast-local : rc={r.returncode} blanks={blanks(res)} "
          f"fw_reqs={fc.chat} local_reqs={lc.chat} {dt:.0f}s "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(("S1", r.stdout[-2000:], r.stderr[-2000:]))
    l.shutdown(); f.shutdown()

    # ---- scenario 2: glacial local (probe times out) -> all Fireworks
    lc, fc = Counter(), Counter()
    l = start(LLAMA_PORT, make_llama_handler(lc, delay=50))
    f = start(FW_PORT, make_fw_handler(fc))
    r, dt = run_agent({"DEADLINE_MIN": "3", "LLAMA_WAIT_S": "20"})
    res = check_results(n_tasks)
    ok = r.returncode == 0 and blanks(res) == 0 and fc.chat >= n_tasks \
        and lc.chat == 1
    print(f"S2 slow-local : rc={r.returncode} blanks={blanks(res)} "
          f"fw_reqs={fc.chat} local_reqs={lc.chat} {dt:.0f}s "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(("S2", r.stdout[-2000:], r.stderr[-2000:]))
    l.shutdown(); f.shutdown()

    # ---- scenario 3: no local server -> all Fireworks
    fc = Counter()
    f = start(FW_PORT, make_fw_handler(fc))
    r, dt = run_agent({"DEADLINE_MIN": "3", "LLAMA_WAIT_S": "4"})
    res = check_results(n_tasks)
    ok = r.returncode == 0 and blanks(res) == 0 and fc.chat >= n_tasks
    print(f"S3 dead-local : rc={r.returncode} blanks={blanks(res)} "
          f"fw_reqs={fc.chat} {dt:.0f}s {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(("S3", r.stdout[-2000:], r.stderr[-2000:]))
    f.shutdown()

    # ---- scenario 4: everything dead -> full placeholder file, exit 0
    r, dt = run_agent({"DEADLINE_MIN": "1", "LLAMA_WAIT_S": "3"})
    res = check_results(n_tasks)
    ok = r.returncode == 0 and blanks(res) == n_tasks
    print(f"S4 all-dead   : rc={r.returncode} blanks={blanks(res)} "
          f"{dt:.0f}s {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(("S4", r.stdout[-2000:], r.stderr[-2000:]))

    if fails:
        for name, so, se in fails:
            print(f"\n===== {name} stdout =====\n{so}\n===== stderr =====\n{se}")
        sys.exit(1)
    print("all scenarios PASS")


if __name__ == "__main__":
    main()
