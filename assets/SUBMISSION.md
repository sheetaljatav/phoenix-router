# lablab.ai Submission Draft — Track 1

**Project title:** Phoenix Router

**Short description (tagline):**
A hybrid routing agent that answers all 8 task categories with a bundled
local model for ZERO Fireworks tokens, with an adaptive time-budget manager
and a Fireworks safety net it almost never needs.

**Long description:**

Phoenix Router is built around the one number that decides the Track 1
leaderboard: total Fireworks tokens. The rules state local inference inside
the container counts fully toward accuracy but zero toward the token score —
so the optimal router is one that never has to leave the box.

How it works:

1. **Deterministic classification (free).** A regex classifier maps each task
   to one of the 8 capability categories (factual, math, sentiment,
   summarization, NER, code debugging, logic, code generation) in
   microseconds, with zero model calls.
2. **Local-first inference (zero tokens).** A 4-bit quantized Qwen3.5-2B
   (Apache-2.0) served by llama.cpp on CPU answers every task with a
   category-tuned system prompt. Thinking mode is enabled only where it pays
   (math, logic) and disabled where it doesn't, keeping the run fast.
3. **Adaptive time budgeting.** The agent measures real tokens/sec as it runs
   and sizes each task's generation budget so the whole batch always finishes
   inside the 10-minute limit on the 2 vCPU / 4 GB grading VM.
4. **Self-verification (free).** Generated code is syntax-checked with
   Python's AST; failures trigger one free local retry.
5. **Fireworks safety net (the only tokens we can ever spend).** If the local
   path errors or time runs low, the agent calls the smallest model in
   ALLOWED_MODELS through FIREWORKS_BASE_URL — never anything else. In normal
   runs this path is never taken: the submission finishes with the
   ZERO_API_CALLS flag, the best possible token score.

Reliability: the container always writes valid /output/results.json and
exits 0, even on internal errors; it reads FIREWORKS_API_KEY,
FIREWORKS_BASE_URL and ALLOWED_MODELS from the environment at runtime; the
image is linux/amd64 and ~2 GB compressed (well under the 10 GB cap).

**Technology tags:** Fireworks AI, llama.cpp, Qwen3.5, Docker, Python, AMD

**Category tags:** AI Agents, Model Routing, Cost Optimization
