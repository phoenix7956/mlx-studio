#!/bin/bash
# 跑批全出完 -> 重启 server (带 /v1 路由) -> 起壳. 远端机 上跑.
set -e
while pgrep -f "run_full.py" >/dev/null; do
  echo "$(date +%H:%M) 跑批还在出, 等..."; sleep 120
done
echo "$(date +%H:%M) 跑批完成, 重启 server + 起壳"
# 备份旧 server, 用新版
cp /Users/<you>/.ours/mlx_server/server.py /Users/<you>/.ours/mlx_server/server.py.old
# 把新版 (带 /v1) 从 .new 位置恢复: 新版已推到 server.py.new, 这里直接拷
[ -f /Users/<you>/.ours/mlx_server/server.py.new ] && cp /Users/<you>/.ours/mlx_server/server.py.new /Users/<you>/.ours/mlx_server/server.py
pkill -f "server.py --port 11250"; sleep 3
cd /Users/<you>/.ours/mlx_server
nohup /Users/<you>/.venvs/h3m3/bin/python server.py --port 11250 >> /tmp/mlx_server.log 2>&1 &
sleep 8
curl -s --max-time 6 http://127.0.0.1:11250/v1/models | head -c 200; echo
echo "=== /v1 路由活了, 壳可用. 本机起壳 ==="
cd /Users/<you>/.ours/mlx_server/shell
pkill -f "http.server 18080" 2>/dev/null; sleep 1
nohup python3 -m http.server 18080 > /tmp/shell_serve.log 2>&1 &
echo "壳 远端机: http://<node-ip>:18080 (server-url 填 http://<node-ip>:11250)"
echo "shell: http://<node-ip>:18080 (或本机 127.0.0.1:18080) server-url 填 http://<node-ip>:11250"
