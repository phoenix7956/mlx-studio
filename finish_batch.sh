#!/bin/bash
# 跑批全出完 -> 重启 server (带 /v1 路由) -> 起壳. 远端机 上跑.
#
# ⚠️ 2026-11-28 改: 内核 现在 由 launchd 管 (com.ours.mlxstudio, 端口 11251).
#    老 写法 是 `pkill -f "server.py --port $PORT"` + `nohup` 自己 拉 —— 那 样 有 两个 坑:
#      1) 跟 launchd 抢 同一个 端口 → 自己 拉 的 那个 bind 失败, 起不来
#      2) 自己 拉 的 内核 **不 source env.sh** → 拿不到 模型 路径 / venv,
#         出歌 / 出图 引擎 全 用不了 (2026-11-28 在 M5U 上 就 是 这么 半残 的)
#    现在 换成 launchctl kickstart —— 让 launchd 用 正确 的 环境 + 端口 重起.
set -e
OURS="${OURS:-$HOME/.ours}"
PY="${OURS_PYTHON:-$HOME/.venvs/h3m3/bin/python}"
SERVER_PORT="${SERVER_PORT:-11251}"
LABEL="${MLX_LAUNCHD_LABEL:-com.ours.mlxstudio}"
SHELL_PORT="${SHELL_PORT:-18080}"
while pgrep -f "run_full.py" >/dev/null; do
  echo "$(date +%H:%M) 跑批还在出, 等..."; sleep 120
done
echo "$(date +%H:%M) 跑批完成, 重启 server + 起壳"
# 备份旧 server, 用新版
cp "$OURS/mlx_server/server.py" "$OURS/mlx_server/server.py.old"
# 把新版 (带 /v1) 从 .new 位置恢复: 新版已推到 server.py.new, 这里直接拷
[ -f "$OURS/mlx_server/server.py.new" ] && cp "$OURS/mlx_server/server.py.new" "$OURS/mlx_server/server.py"
# 重启内核 —— 走 launchd, 不 自己 pkill / nohup (见 上面 的 说明)
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  echo "$(date +%H:%M) kickstart $LABEL (端口 $SERVER_PORT)"
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
else
  # 没有 launchd 管 的 机器 才 自己 拉 —— 这 回 记得 source env.sh
  echo "$(date +%H:%M) 没有 $LABEL, 自己 拉 (端口 $SERVER_PORT)"
  cd "$OURS/mlx_server"
  ( source "$OURS/mlx_server/env.sh"; exec "$PY" server.py --port "$SERVER_PORT" ) >> /tmp/mlx_server.log 2>&1 &
fi
sleep 8
curl -s --max-time 6 "http://127.0.0.1:$SERVER_PORT/v1/models" | head -c 200; echo
echo "=== /v1 路由活了, 壳可用. 本机起壳 ==="
cd "$OURS/mlx_server/shell"
pkill -f "http.server $SHELL_PORT" 2>/dev/null; sleep 1
nohup python3 -m http.server "$SHELL_PORT" > /tmp/shell_serve.log 2>&1 &
HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
echo "壳: http://$HOST_IP:$SHELL_PORT  (server-url 填 http://$HOST_IP:$SERVER_PORT)"
echo "shell: http://127.0.0.1:$SHELL_PORT  (server-url 填 http://$HOST_IP:$SERVER_PORT)"
