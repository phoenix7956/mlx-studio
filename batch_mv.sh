#!/bin/bash
# 测 试 批: 中 文 前 3 片 + 英 文 前 3 片 (8bit 12-step), 会 话 制 热 出
set -e
SID=$(curl -s -m 10 "http://127.0.0.1:11251/session?kind=video" | python3 -c "import json,sys;print(json.load(sys.stdin)['session'])")
echo "sid=$SID"
ZH=/tmp/mv_zh_shots.txt; EN=/tmp/mv_en_shots.txt
for i in 0 1 2; do
  P=$(sed -n "$((i+1))p" $ZH | sed 's/^\[[0-9]*\] //')
  echo "=== 中 文 片 $i: ${P:0:50}..."
  curl -s -m 1800 "http://127.0.0.1:11251/video?session=$SID&prompt=$(python3 -c "import urllib.parse;print(urllib.parse.quote('$P'))")&sec=5&steps=12&seed=$i"
  echo ""
done
for i in 0 1 2; do
  P=$(sed -n "$((i+1))p" $EN | sed 's/^\[[0-9]*\] //')
  echo "=== 英 文 片 $i: ${P:0:50}..."
  curl -s -m 1800 "http://127.0.0.1:11251/video?session=$SID&prompt=$(python3 -c "import urllib.parse;print(urllib.parse.quote('$P'))")&sec=5&steps=12&seed=$((i+3))"
  echo ""
done
curl -s -m 10 "http://127.0.0.1:11251/session/close?sid=$SID"; echo " 会话 关"
