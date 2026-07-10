# ZeroToken Router — Track 1: Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1 submission.**

A general-purpose AI agent that handles all eight capability categories
(factual knowledge, math, sentiment, summarization, NER, code debugging,
logical reasoning, code generation) while spending **zero Fireworks tokens**
in the common case.

## How it works

```
/input/tasks.json
      │
      ▼
regex task classifier  (8 categories, deterministic, instant, free)
      │
      ▼
local Qwen3.5-2B (llama.cpp, CPU, 4-bit)   ←  ZERO Fireworks tokens
  · category-tuned system prompts
  · thinking mode ON for math & logic, OFF for the rest
  · adaptive token budgets from measured tokens/sec so the whole
    batch always finishes inside the 10-minute runtime limit
      │  (only if the local path fails or time runs out)
      ▼
Fireworks fallback via FIREWORKS_BASE_URL
  · model chosen as the smallest entry in ALLOWED_MODELS
      │
      ▼
/output/results.json
```

Per the Track 1 rules, local inference inside the container counts fully
toward accuracy but contributes **zero** to the token score — so the router's
job is to make sure the Fireworks safety net is (almost) never needed.

- **Model:** [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache-2.0),
  Unsloth dynamic Q4_K_XL GGUF (~1.3 GB) — sized for the 4 GB RAM / 2 vCPU
  grading environment.
- **Serving:** `llama.cpp` server (official `ghcr.io/ggml-org/llama.cpp:server`
  image), OpenAI-compatible API on localhost.
- **Agent:** single-file Python 3 (stdlib only, no pip dependencies).

## Reliability guarantees

- Always writes valid `/output/results.json` and exits 0, even on internal
  errors (an imperfect answer beats a `RUNTIME_ERROR`).
- Wall-clock budget manager: measures generation speed and shrinks per-task
  token budgets so the run finishes well before the 10-minute cap.
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from the
  environment at runtime — nothing hardcoded, no `.env` in the image.

## Build

```bash
# linux/amd64 as required by the judging VM; the model weights (~1.3 GB)
# are downloaded from Hugging Face during the build
docker buildx build --platform linux/amd64 -t zerotoken-router:latest .
```

CI: every push to `main` rebuilds and publishes
`ghcr.io/sheetaljatav/zerotoken-router:latest` via GitHub Actions
([.github/workflows/build.yml](.github/workflows/build.yml)).

## Run locally

```bash
docker run --rm --cpus=2 --memory=4g \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  zerotoken-router:latest
cat output/results.json
```

`--cpus=2 --memory=4g` mirrors the grading environment. The repo ships the
official practice tasks in `input/tasks.json`. To exercise the Fireworks
fallback, add `-e FIREWORKS_API_KEY=... -e FIREWORKS_BASE_URL=... -e ALLOWED_MODELS=...`.

## Files

| File | Purpose |
|---|---|
| `agent.py` | classifier, routing, budgeting, I/O |
| `run.sh` | starts llama-server, then the agent |
| `Dockerfile` | llama.cpp server base + model + agent |
| `input/tasks.json` | official practice tasks for local testing |

## License

MIT. Bundled model weights: Qwen3.5-2B, Apache-2.0.
