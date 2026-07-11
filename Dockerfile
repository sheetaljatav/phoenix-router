# linux/amd64 - slim API-routing build (no bundled model).
# The agent routes tasks through the harness's Fireworks proxy with
# token-frugal prompting; LLAMA_WAIT_S=5 makes the (absent) local-model
# probe a no-op. Rebuild from the llama.cpp base to restore local mode.
FROM python:3.12-slim
ENV LLAMA_WAIT_S=5
COPY agent.py run.sh /app/
ENTRYPOINT ["/bin/sh", "/app/run.sh"]
