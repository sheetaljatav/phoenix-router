#!/bin/sh
# start local llama.cpp server (zero Fireworks tokens), then run the agent
BIN=/app/llama-server
[ -x "$BIN" ] || BIN="$(command -v llama-server)"

# mmap (default) keeps peak RSS low on the 4 GB grading VM - the OS pages
# weights in on demand instead of committing the whole file up front.
# ctx 4096 halves KV-cache memory vs 8192; every task fits comfortably.
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
