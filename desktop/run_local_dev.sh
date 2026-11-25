#!/bin/bash
# 启动 内核 —— 装 到 别的 mac 上 时 也 用 这个 (自带 在 app 里).
# 用法: ./run_local_dev.sh [端口]
#
# 优先 用 install.sh 装 好 的 运行 目录 (~/.ours/mlx_server),
# 没 装 过 就 用 app 自带 的 kernel (Contents/Resources/kernel).
set -e
cd "$(dirname "$0")"
PORT="${1:-11251}"

RUNTIME="$HOME/.ours/mlx_server"
if [ -f "$RUNTIME/server.py" ]; then
  cd "$RUNTIME"
  [ -f "$RUNTIME/env.sh" ] && . "$RUNTIME/env.sh"
elif [ -f "$HOME/.ours/mlx_server/env.sh" ]; then
  . "$HOME/.ours/mlx_server/env.sh"
fi

# —— 选 一个 带 mlx 的 python ——
PY=""
for c in "$HOME/.venvs/mlxstudio/bin/python" "$HOME/.venvs/mtx/bin/python" \
         "$HOME/.venvs/h3m/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [ -n "$c" ] && [ -x "$c" ] || continue
  if "$c" -c "import mlx.core" >/dev/null 2>&1; then PY="$c"; break; fi
  [ -z "$PY" ] && PY="$c"
done
if [ -z "$PY" ]; then
  echo "[内核] 找 不 到 python —— 先 装 一个 带 mlx 的 环境:"
  echo "        python3 -m venv ~/.venvs/mlxstudio && ~/.venvs/mlxstudio/bin/pip install mlx numpy"
  exit 1
fi

# —— 权重 / 厂商 源码 路径 (有 就 用, 没 就 跳过) ——
[ -d "$HOME/Desktop/rea/ours/_vendor/minimax-h3_mlx_src" ] && \
  export H3M_PORT_DIR="$HOME/Desktop/rea/ours/_vendor/minimax-h3_mlx_src"
[ -d "$HOME/Desktop/rea/ours/_vendor/krea2_engine_src" ] && \
  export H3M_KREA_DIR="$HOME/Desktop/rea/ours/_vendor/krea2_engine_src"
[ -d "$HOME/.ours/models/models/MiniMax--MiniMax-H3/snapshots/master/FL2VA" ] && \
  export H3M_MLX_ROOT="$HOME/.ours/models/models/MiniMax--MiniMax-H3/snapshots/master/FL2VA"
[ -d "$HOME/.ours/quant/h3-mlx-4bit" ] && export H3M_QUANT_DIR="$HOME/.ours/quant/h3-mlx-4bit"
# 出图 (Krea-2-Turbo) 权重 —— 指 真正 有 turbo.safetensors 的 那个 目录
for _kd in "$HOME/.ours/models/Krea-2-Turbo-MLX-Serve-mixed-4-8" \
           "$HOME/.ours/models/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8" \
           "$HOME/.ours/models/models/krea--Krea-2-Turbo/snapshots/master"; do
  if [ -f "$_kd/turbo.safetensors" ] || [ -f "$_kd/transformer_mixed_4_8.safetensors" ]; then
    export H3M_KREA_DIR_W="$_kd"; break
  fi
done
export ACE_MUSIC_URL="${ACE_MUSIC_URL:-http://127.0.0.1:7861}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export MINEPROXY_NODE=local

echo "[内核] :$PORT  (python: $PY · 代码: $(pwd))"
exec "$PY" server.py --port "$PORT"
