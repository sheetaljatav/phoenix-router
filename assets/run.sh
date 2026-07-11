#!/bin/sh
# start local llama.cpp server (zero Fireworks tokens), then run the agent
/app/llama-server \
  -m /models/model.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 \
  --parallel 1 \
  --no-mmap \
  --threads "$(nproc)" \
  --jinja \
  --no-webui \
  > /tmp/llama.log 2>&1 &

exec python3 /app/agent.py
