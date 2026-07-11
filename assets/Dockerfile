# linux/amd64 — required by the judging VM
FROM ghcr.io/ggml-org/llama.cpp:server

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/*

# model weights: Qwen3.5-2B (Apache-2.0), Unsloth dynamic Q4_K_XL (~1.3 GB)
RUN mkdir -p /models \
 && curl -fL --retry 3 -o /models/model.gguf \
    https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf

COPY agent.py run.sh /app/

ENTRYPOINT ["/bin/sh", "/app/run.sh"]
