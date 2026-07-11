#!/bin/sh
# start the local llama.cpp server when bundled (fat build); slim builds
# ship neither the binary nor the model, so this block is skipped
BIN=/app/llama-server
[ -x "$BIN" ] || BIN="$(command -v llama-server 2>/dev/null || true)"
if [ -n "$BIN" ] && [ -f /models/model.gguf ]; then
"$BIN" -m /models/model.gguf --host 127.0.0.1 --port 8080 --ctx-size 4096 --parallel 1 --threads "$(nproc)" --jinja --no-webui > /tmp/llama.log 2>&1 &
fi
exec python3 /app/agent.py
