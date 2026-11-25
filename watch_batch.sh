#!/bin/bash
# 盯 跑 批 结 束 -> 拉 回 合 并 -> 写 完 成 标 记 (纯 重 定 向, 不 用 process substitution, nohup 下 稳)
#
# 【配置】全 走 环境变量, 不 写死 任何 机器 的 路径:
#   MV_BATCH_SSH  = 跑批 那 台 机器 (默认 不 设 = 本 机)
#   MV_BATCH_DIR  = 跑批 产物 目录 (默认 ~/Desktop/rea/soak-results/mlxserve)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH_TARGET="${MV_BATCH_SSH:-}"
OUT_DIR="${MV_BATCH_DIR:-$HOME/Desktop/rea/soak-results/mlxserve}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/auto_finish.log"
if [ -n "$SSH_TARGET" ]; then
  SSH_CMD=(ssh -o ConnectTimeout=10 "$SSH_TARGET")
else
  SSH_CMD=(true)   # 不 设 就 当 本 机 (pgrep 自己)
fi
echo "[$(date +%H:%M)] 开 始 盯 跑 批" >> "$LOG"
while "${SSH_CMD[@]}" pgrep -f run_full.py >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date +%H:%M)] 跑 批 结 束, 等 5min 落 盘" >> "$LOG"
sleep 300
echo "[$(date +%H:%M)] 拉 回 + 合 并 开 始" >> "$LOG"
python3 "$HERE/pull_and_merge.py" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] 完 成 标 记" >> "$LOG"
touch "$OUT_DIR/AUTO_DONE"
