#!/bin/bash
# 壳启动: 本机 serve SPA, 浏览器开 http://127.0.0.1:18080, server-url 填 http://<node-ip>:11250
cd "$(dirname "$0")"
PORT=18080
echo "壳在 http://127.0.0.1:$PORT  (浏览器开, Connect 时 server-url 填 http://<node-ip>:11250)"
python3 -m http.server $PORT
