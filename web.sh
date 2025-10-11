#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/src/LostPetTest"
FRONTEND_DIR="$REPO/frontend"
BACKEND_DIR="$REPO/backend"
DIST_DIR="$FRONTEND_DIR/dist"
HOST=0.0.0.0
PORT=8000
DOMAIN="https://bqylcbfyd6l3yjf.studio.ap-southeast-2.sagemaker.aws"

echo "== 1/4 Build frontend =="
cd "$FRONTEND_DIR"
npm run build

echo "== 2/4 Start backend (FastAPI) =="
# 先杀掉占用端口的旧进程（若有）
if lsof -i :$PORT -t >/dev/null 2>&1; then
  kill -9 $(lsof -i :$PORT -t) || true
fi

cd "$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"

# 后台启动 uvicorn，并把日志写入文件
nohup uvicorn backend.server:app --host "$HOST" --port "$PORT" > "$REPO/uvicorn.log" 2>&1 &

# 等待端口起来
echo "Waiting for uvicorn..."
for i in {1..20}; do
  if curl -s "http://127.0.0.1:${PORT}/docs" >/dev/null; then
    echo "Uvicorn is up."
    break
  fi
  sleep 0.5
done

echo "== 3/4 Probe Studio proxy prefixes =="
# 自动探测可用代理前缀
cands=( "/proxy" "/jupyterlab/proxy" "/lab/proxy" "/jupyter/default/proxy" "/studio/default/proxy" )
FOUND=""
for p in "${cands[@]}"; do
  URL="${DOMAIN}${p}/${PORT}/"
  code=$(curl -sk -o /dev/null -w "%{http_code}" "$URL")
  echo "$code  $URL"
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    FOUND="$URL"
    break
  fi
done

echo "== 4/4 Result =="
if [ -n "$FOUND" ]; then
  echo "✅ Open your web app here:"
  echo "$FOUND"
  echo
  echo "API docs (if needed): ${FOUND}docs"
else
  echo "⚠️ Could not find a working proxy prefix automatically."
  echo "Try these manually in your browser:"
  for p in "${cands[@]}"; do
    echo " - ${DOMAIN}${p}/${PORT}/"
  done
  echo
  echo "Uvicorn logs: $REPO/uvicorn.log"
fi
