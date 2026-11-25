#!/bin/bash
# 壳启动: 本机 serve SPA. 浏览器开 http://127.0.0.1:$PORT
# Connect 时 的 server-url 填 这 台 机器 的 内网 地址 + 内核 端口.
cd "$(dirname "$0")"
PORT="${PORT:-18080}"
SERVER_PORT="${SERVER_PORT:-11251}"
HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
echo "壳在 http://127.0.0.1:$PORT  (浏览器开, Connect 时 server-url 填 http://$HOST_IP:$SERVER_PORT)"
python3 -m http.server $PORT
