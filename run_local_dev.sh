#!/bin/bash
# 本机开发/测试 mlx_server — 产品全能力(视频/图/歌/LLM), 一台机器一份内核.
# 用户按自己定位用其中 1 / 部分 / 全部功能; 代码不分机器, 只按 env 指权重路径.
# 本机定位 (你的用法): 出歌 / 专辑图 / 主控调度; 要出片也照样能出, 只是约定走 远端机.
# 用法: ./run_local_dev.sh 11251
set -e
cd "$(dirname "$0")"
# 内核代码同一份 (两台机 一致), 只 override 权重/依赖 路径.
# 这 份 脚本 必须 **两 台 机器 都 能 跑** —— 之前 写 死 了 本机 的
# ~/Desktop/rea/ours/_vendor, 拷 到 另一台机 上 就 指 到 不存在 的 地方 (静默 走 空 路径).
# 现在 按 "存在 就 用" 挑: 本机 布局 优先, 远端机 布局 兜底.
pick_dir() {  # pick_dir <候选...> —— 挑 第一个 存在 的 目录; 都 不 存在 就 挑 第一个 (报错 时 能 看 出 想要 什么)
  local c
  for c in "$@"; do [ -d "$c" ] && { echo "$c"; return 0; }; done
  echo "$1"
}
pick_file() {  # pick_file <候选...>
  local c
  for c in "$@"; do [ -x "$c" ] && { echo "$c"; return 0; }; done
  echo "$1"
}
export H3M_PORT_DIR="$(pick_dir \
  "$HOME/Desktop/rea/ours/_vendor/minimax-h3_mlx_src" \
  "$HOME/.ours/vendor/minimax-h3_mlx_src")"
export H3M_MLX_ROOT="$(pick_dir \
  "$HOME/.ours/models/models/MiniMax--MiniMax-H3/snapshots/master/FL2VA" \
  "$HOME/.ours/models/MiniMax/MiniMax-H3/FL2VA")"
export H3M_QUANT_DIR="$(pick_dir \
  "$HOME/.ours/quant/h3-mlx-4bit" \
  "$HOME/.ours/quant/h3-mlx-6bit")"
export H3M_KREA_DIR="$(pick_dir \
  "$HOME/Desktop/rea/ours/_vendor/krea2_engine_src" \
  "$HOME/.ours/vendor/krea2_engine_src")"
# Krea-2-Turbo 权重 (真 目录 —— 老 指向 的 LTX-2.5 空壳 目录 会 让 引擎 报
# "Failed to open file .../turbo.safetensors").
export H3M_KREA_DIR_W="$(pick_dir \
  "$HOME/.ours/models/models/krea--Krea-2-Turbo/snapshots/master" \
  "$HOME/.ours/models/Krea-2-Turbo-MLX-Serve-mixed-4-8")"
export ACE_MUSIC_URL="${ACE_MUSIC_URL:-http://127.0.0.1:7861}"
# 一 首 歌 最长 多少 秒 (引擎 硬 上限) —— 本机 128G, 240 太 短 (词多 的 歌 装 不 下)
export ACE_MAX_DURATION="${ACE_MAX_DURATION:-480}"
# 真 ffmpeg (本机 ~/.local/bin/ffmpeg 是 PyAV shim, 不会编码).
# 没 Homebrew 的 机器 —— 它 的 真 ffmpeg 在 ~/.ours/bin (install 放 的), 一样 加 进 PATH.
export PATH="/opt/homebrew/bin:$HOME/.ours/bin:$HOME/bin-real:$PATH"
# venv 也 按 存在 就 用 挑: 本机 开发 态 用 mtx (全能力), 远端机 用 h3m3.
PY="$(pick_file \
  "$HOME/.venvs/mtx/bin/python" \
  "$HOME/.venvs/h3m3/bin/python" \
  "$HOME/.venvs/mlxstudio/bin/python")"
PORT="${1:-11251}"
export MLX_NODE_ROLE=local
echo "[run_local_dev] :$PORT 全能力起 (视频/图/歌/LLM), 定位由使用者定, 内核同 远端机"
exec $PY server.py --port "$PORT"
