#!/bin/bash
# 跑批全出完 -> 重启 server (带 /v1 路由) -> 起壳. 远端机 上跑.
set -e
OURS="${OURS:-$HOME/.ours}"
PY="${OURS_PYTHON:-$HOME/.venvs/h3m3/bin/python}"
SERVER_PORT="${SERVER_PORT:-11250}"
SHELL_PORT="${SHELL_PORT:-18080}"
while pgrep -f "run_full.py" >/dev/null; do
  echo "$(date +%H:%M) 跑批还在出, 等..."; sleep 120
done
echo "$(date +%H:%M) 跑批完成, 重启 server + 起壳"
# 备份旧 server, 用新版
cp "$OURS/mlx_server/server.py" "$OURS/mlx_server/server.py.old"
# 把新版 (带 /v1) 从 .new 位置恢复: 新版已推到 server.py.new, 这里直接拷
[ -f "$OURS/mlx_server/server.py.new" ] && cp "$OURS/mlx_server/server.py.new" "$OURS/mlx_server/server.py"
pkill -f "server.py --port $SERVER_PORT"; sleep 3
cd "$OURS/mlx_server"
nohup "$PY" server.py --port "$SERVER_PORT" >> /tmp/mlx_server.log 2>&1 &
sleep 8
curl -s --max-time 6 "http://127.0.0.1:$SERVER_PORT/v1/models" | head -c 200; echo
echo "=== /v1 路由活了, 壳可用. 本机起壳 ==="
cd "$OURS/mlx_server/shell"
pkill -f "http.server $SHELL_PORT" 2>/dev/null; sleep 1
nohup python3 -m http.server "$SHELL_PORT" > /tmp/shell_serve.log 2>&1 &
HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
echo "壳: http://$HOST_IP:$SHELL_PORT  (server-url 填 http://$HOST_IP:$SERVER_PORT)"
echo "shell: http://127.0.0.1:$SHELL_PORT  (server-url 填 http://$HOST_IP:$SERVER_PORT)"
