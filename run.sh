#!/bin/sh
# Start the local llama.cpp server (zero Fireworks tokens), then run the agent.
# Memory-tuned for the 4 GB / 2 vCPU grading VM:
#   --ctx-size 4096  : ample for these tasks; halves the KV-cache footprint
#   (mmap default)   : pages the 1.3 GB model on demand instead of forcing it
#                      fully resident, so we stay well clear of an OOM kill
BIN=/app/llama-server
[ -x "$BIN" ] || BIN="$(command -v llama-server 2>/dev/null || echo /app/llama-server)"

"$BIN" \
  -m /models/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 4096 \
  --parallel 1 \
  --threads "$(nproc)" \
  --jinja \
  --no-webui \
  > /tmp/llama.log 2>&1 &

exec python3 /app/agent.py
