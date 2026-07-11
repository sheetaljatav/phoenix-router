#!/bin/sh
# measure llama-server readiness + one test completion inside the container
/app/llama-server -m /models/model.gguf --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --threads 2 --jinja --no-webui > /tmp/l.log 2>&1 &

S=0
while [ "$S" -lt 300 ]; do
  if wget -q -O- http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; then
    echo "READY_AT_${S}s"
    break
  fi
  sleep 5
  S=$((S+5))
done

if [ "$S" -ge 300 ]; then
  echo "NEVER_READY"
  tail -30 /tmp/l.log
  exit 1
fi

cat > /tmp/req.json <<'EOF'
{"messages":[{"role":"system","content":"Answer briefly. /no_think"},{"role":"user","content":"What is the capital of France?"}],"max_tokens":60,"temperature":0.2}
EOF

T0=$(date +%s)
wget -q -O- --header 'Content-Type: application/json' \
  --post-file /tmp/req.json http://127.0.0.1:8080/v1/chat/completions
T1=$(date +%s)
echo ""
echo "COMPLETION_SECONDS=$((T1-T0))"
tail -5 /tmp/l.log
