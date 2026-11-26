#!/bin/bash
# MLX Studio —— 一键 安装 (别的 mac 上 也 能 装)
#
#   ./install.sh              # 装 全部 核心: 内核代码/出歌+出片引擎/依赖/桌面App/开机自启
#   ./install.sh --no-agent   # 不 装 开机自启 (只 跑 一次)
#   ./install.sh --no-ace     # 不 装 出歌 引擎 的 大 环境 (torch, ~2GB)
#   ./install.sh --app        # 只 装 桌面 App
#   ./install.sh --port 11251
#
# 装 完 之后:
#   · 内核 在 http://127.0.0.1:11251  (开机 自启)
#   · 模型 权重 不 随 安装 包 带 —— 打开 界面「模型」页 点「下载」
#   · 出歌/出片/出图 引擎 代码 已 装 好, 权重 在「模型」页 下
set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"

PORT="${MLX_STUDIO_PORT:-11251}"
AGENT=1
ACE=1
APP=0
APP_INSTALLED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-agent) AGENT=0; shift ;;
    --no-ace)   ACE=0; shift ;;
    --app)      APP=1; shift ;;
    --port=*)   PORT="${1#*=}"; shift ;;
    --port)     PORT="${2:-11251}"; shift 2 ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    *) shift ;;
  esac
done

say()  { printf "\033[36m[装]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[注意]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[失败]\033[0m %s\n" "$*" >&2; exit 1; }

# ================================================================= 0. 检查 系统
[ "$(uname -s)" = "Darwin" ] || die "只 支持 macOS (Apple Silicon)"
ARCH="$(uname -m)"
[ "$ARCH" = "arm64" ] || warn "你 这 是 $ARCH —— MLX 只 在 苹果芯片 上 跑 得 动"
say "macOS $(sw_vers -productVersion) · $ARCH"

# 真实 home —— 有些 环境 (CI/沙箱) 的 $HOME 是 假 的
REAL_HOME="$HOME"
DSCL_HOME="$(dscl . -read "/Users/$(whoami)" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
[ -d "$DSCL_HOME" ] && REAL_HOME="$DSCL_HOME"
OURS="$REAL_HOME/.ours"
RUNTIME="$OURS/mlx_server"
VENV_MAIN="$REAL_HOME/.venvs/mlxstudio"
VENV_ACE="$REAL_HOME/.venvs/ace"
say "HOME = $REAL_HOME"

# ================================================================= 1. 找 内核 代码
CORE=""
for c in "$ROOT/内核代码" "$ROOT/core" "$ROOT" \
         "$ROOT/MLX Serve Studio.app/Contents/Resources/kernel"; do
  if [ -f "$c/server.py" ]; then CORE="$c"; break; fi
done
[ -n "$CORE" ] || die "找 不 到 内核 代码 (server.py) —— 这个 包 不 完整"
say "内核 代码: $CORE"

# ================================================================= 2. 目录
mkdir -p "$OURS/models" "$OURS/ace-songs" "$OURS/music3-songs" "$OURS/vendor" \
         "$OURS/quant" "$RUNTIME" "$REAL_HOME/.venvs" "$REAL_HOME/Library/LaunchAgents" \
         "$REAL_HOME/Documents/MLX Studio 输出"
say "运行 目录: $RUNTIME"
say "模型 目录: $OURS/models   (装 完 后 在 界面 点「下载」)"

sync_dir() {  # sync_dir <源目录> <目标目录>
  local s d
  s="$(cd "$1" 2>/dev/null && pwd)" || return 0
  d="$(cd "$2" 2>/dev/null && pwd || echo "$2")"
  [ "$s" = "$d" ] && return 0          # 同 一个 目录 —— 别 自己 覆盖 自己
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$s/" "$d/"
  else
    mkdir -p "$d"; cp -R "$s/." "$d/"
  fi
}

# —— 2.1 内核 代码 ——
# 手工 维护 清单 容易 漏 —— voice_studio.py (配音工作台) 和 image_engine_mflux.py
# (Z-Image / MOSS 出图) 都 漏 过, 结果 换 台 机器 装 完, 那 两 块 直接 白屏.
# 改 成 拷贝 全部 顶层 模块 (只 排除 探针 / 回归 脚本), 加 什么 都 不会 再 漏.
for f in "$CORE"/*.py; do
  b="$(basename "$f")"
  case "$b" in
    probe_ms*.py|smoke_*.py|test_*.py) continue ;;
  esac
  [ "$f" = "$RUNTIME/$b" ] && continue      # 同 一个 文件 (就地 重装) —— 别 cp, 会 报 identical 而 中断
  cp "$f" "$RUNTIME/" 2>/dev/null || true
done
for f in model_catalog.json; do
  [ -f "$CORE/$f" ] || continue
  [ "$CORE/$f" = "$RUNTIME/$f" ] && continue
  cp "$CORE/$f" "$RUNTIME/" 2>/dev/null || true
done
# 跑批 那 条 路 的 脚本 —— 不装 的 话, 新 机器 上 "批量出片" 无 脚本 可 跑
# (之前 就 漏 在 这: 包 里 有 run_full.py, 但 装 到 第二台机 上 没有).
for f in run_full.py run_batch.py batch_mv.sh finish_batch.sh; do
  [ -f "$CORE/$f" ] || [ -f "$ROOT/$f" ] || continue
  [ -f "$CORE/$f" ] && cp "$CORE/$f" "$RUNTIME/" 2>/dev/null || true
  [ -f "$ROOT/$f" ] && [ ! -f "$CORE/$f" ] && cp "$ROOT/$f" "$RUNTIME/" 2>/dev/null || true
done
# 文档 (标准 工作流) —— 装 完 机器 上 有 一份 能 照 抄 的 SOP
for f in STANDARD_WORKFLOW.md MV_PLAYBOOK.md; do
  [ -f "$CORE/$f" ] || [ -f "$ROOT/$f" ] || continue
  [ -f "$CORE/$f" ] && cp "$CORE/$f" "$RUNTIME/" 2>/dev/null || true
  [ -f "$ROOT/$f" ] && [ ! -f "$CORE/$f" ] && cp "$ROOT/$f" "$RUNTIME/" 2>/dev/null || true
done
for d in ui tools engines; do
  [ -d "$CORE/$d" ] && sync_dir "$CORE/$d" "$RUNTIME/$d"
done
# 厂商 引擎 源码 (出片 MiniMax-H3 / 出图 Krea) —— 视频/图片 引擎 本体
for d in minimax-h3_mlx_src krea2_engine_src port_minimax_h3_mlx; do
  for base in "$CORE" "$CORE/vendor" "$ROOT" "$ROOT/vendor"; do
    [ -d "$base/$d" ] || continue
    sync_dir "$base/$d" "$RUNTIME/$d"
    sync_dir "$base/$d" "$OURS/vendor/$d"
    break
  done
done
# —— 我们 自己 的 补丁层 覆盖 到 厂商 源码 包 上 ——
# port_minimax_h3_mlx = 我们 改过 的 H3 引擎 (参考图 / 没 Homebrew 的 机器 ffmpeg 兜底 …).
# 运行时 import 的 是 厂商 源码 里 的 minimax_h3_mlx 包, **不是** port_minimax_h3_mlx ——
# 不 覆盖 的 话, 补丁 就 是 死代码: 参考图 直接 TypeError, 远端机 上 ffmpeg 找不到.
# 覆盖 必须 放在 上面 那 个 sync 之后 (sync 用 rsync --delete, 会 把 补丁 冲掉).
overlay_h3_patch() {  # overlay_h3_patch <厂商源码目录>
  local src="$1/minimax_h3_mlx" port=""
  [ -d "$src" ] || return 0
  # 补丁层 可能 在 源码目录 **里面**, 也 可能 是 它 的 **同级**
  # (安装 包 的 布局 就是 同级: vendor/minimax-h3_mlx_src + port_minimax_h3_mlx).
  # 之前 只 认 "里面" —— 于 是 overlay 永远 静默 跳过, 补丁 是 死代码
  # (参考图 TypeError / 远端机 上 ffmpeg 找不到), 重装 也 修 不 好.
  local cand
  for cand in "$1/port_minimax_h3_mlx" "$1/../port_minimax_h3_mlx"; do
    [ -d "$cand" ] && { port="$cand"; break; }
  done
  if [ -z "$port" ]; then
    echo "[install] ⚠️  找不到 H3 补丁层 (port_minimax_h3_mlx) —— 参考图 会 不可用" >&2
    return 0
  fi
  for f in "$port"/*.py; do
    [ -f "$f" ] || continue
    cp "$f" "$src/$(basename "$f")" 2>/dev/null || true
  done
  find "$src" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
  # 自检 —— 补丁 漏了 要 看见, 不 要 静默 (静默 的 话 参考图 只 在 用 到 时 才 炸)
  if grep -q "ref_images" "$src/pipeline.py" 2>/dev/null; then
    echo "[install] ✓  H3 补丁 覆盖 上 (参考图 可用)"
  else
    echo "[install] ⚠️  H3 补丁 没 覆盖 上 —— 参考图 会 不可用" >&2
  fi
}
for _t in "$RUNTIME" "$OURS/vendor"; do
  [ -d "$_t/minimax-h3_mlx_src" ] && overlay_h3_patch "$_t/minimax-h3_mlx_src"
done
# —— Wan2.2 引擎 (出片, 开源 Apache-2.0) ——
#   port_wan2_mlx   = 我们 自己 的 包装 层 (DiT 常驻 / 截 帧 / 本地 tokenizer)
#   mlx_video       = 上游 引擎 源码 (Blaizzy/mlx-video, MIT) —— vendor 进来, 原样 没改
for base in "$CORE" "$ROOT"; do
  [ -d "$base/port_wan2_mlx" ] && sync_dir "$base/port_wan2_mlx" "$RUNTIME/port_wan2_mlx" && break
done
for base in "$CORE/vendor_src" "$ROOT/vendor_src" "$CORE/vendor" "$ROOT/vendor"; do
  [ -d "$base/mlx_video" ] || continue
  sync_dir "$base/mlx_video" "$RUNTIME/vendor/mlx_video"
  sync_dir "$base/mlx_video" "$OURS/vendor/mlx_video"
  break
done
if [ "$ROOT/run_local_dev.sh" != "$RUNTIME/run_local_dev.sh" ] && [ -f "$ROOT/run_local_dev.sh" ]; then
  cp "$ROOT/run_local_dev.sh" "$RUNTIME/"
elif [ "$CORE/run_local_dev.sh" != "$RUNTIME/run_local_dev.sh" ] && [ -f "$CORE/run_local_dev.sh" ]; then
  cp "$CORE/run_local_dev.sh" "$RUNTIME/"
fi
chmod +x "$RUNTIME/run_local_dev.sh" 2>/dev/null || true

# —— 2.1.1 ffmpeg (出片 要 用 来 压 mp4) ——
#   没 brew 的 机器 没 ffmpeg → 出片 只 剩 帧 序列, 没有 mp4.
#
#   【开源 注意】ffmpeg 本体 是 LGPL/GPL, **不能 随 仓库 分发** —— 分发 就 会 让
#   整个 项目 被 迫 改成 GPL. 所以 这里 **不 带 二进制**, 改 装 的时候 从
#   imageio-ffmpeg (pip 依赖, 本来 就 要 装) 里 拿 一份, 包 成 一个 转发 壳:
#     ~/.ours/bin/ffmpeg -> python -c "imageio_ffmpeg.get_ffmpeg_exe()"
#   用户 自己 pip/brew 装 的 ffmpeg 不 算 分发, 随便 用.
#   (转发壳 本身 要 等 下面 装 完 imageio-ffmpeg 才 写 —— 见 "ffmpeg 转发壳" 那段.)

# —— 2.2 出歌 引擎 (ACE-Step, MIT) ——
ACE_SRC=""
for c in "$ROOT/出歌引擎/ace-step-engine" "$ROOT/ace-step-engine" \
         "$CORE/ace-step-engine" "$REAL_HOME/.ours/ace-step-engine"; do
  [ -d "$c/acestep_mlx" ] && ACE_SRC="$c" && break
done
if [ -n "$ACE_SRC" ]; then
  if [ "$(cd "$ACE_SRC" && pwd)" = "$(cd "$OURS/ace-step-engine" 2>/dev/null && pwd || echo -)" ]; then
    say "出歌 引擎: $OURS/ace-step-engine (已 在 本机)"
  else
    mkdir -p "$OURS/ace-step-engine"
    sync_dir "$ACE_SRC" "$OURS/ace-step-engine"
    say "出歌 引擎: $OURS/ace-step-engine"
  fi
  # project/checkpoints 是 指向 权重 目录 的 软链 —— 打包机 的 绝对 路径 在 别的 机器 上 是 死链,
  # 这里 按 本机 真实 home 重建 (权重 目录 不 在, 就 指 到 官方 位置, 后面 下载 会 填).
  ACE_CK="$OURS/models/ace-step-mlx"
  mkdir -p "$OURS/ace-step-engine/project"
  rm -f "$OURS/ace-step-engine/project/checkpoints" 2>/dev/null || true
  ln -sfn "$ACE_CK" "$OURS/ace-step-engine/project/checkpoints" 2>/dev/null || true
else
  warn "没 找到 出歌 引擎 (ace-step-engine) —— 出歌 功能 会 缺 引擎"
fi

# —— 2.3 出片/出图 引擎 (厂商 源码) ——
if [ -d "$OURS/vendor/minimax-h3_mlx_src" ]; then
  say "出片 引擎: $OURS/vendor/minimax-h3_mlx_src   (权重 稍后 下载)"
else
  warn "没 找到 出片 引擎 (minimax-h3_mlx_src) —— 视频 功能 不可用"
fi
if [ -d "$OURS/vendor/krea2_engine_src" ]; then
  say "出图 引擎: $OURS/vendor/krea2_engine_src   (权重 稍后 下载)"
else
  warn "没 找到 出图 引擎 (krea2_engine_src) —— 出图 功能 不可用"
fi

# ================================================================= 3. 依赖 (venv)
pick_python() {   # 找 一个 已经 带 mlx 的 环境, 别 白 装
  # ① 最好: 带 mlx **+ mlx_lm** (LLM 对话 要 mlx_lm —— 只 有 mlx 的 环境 装 完 也 对话 不了)
  for c in "$VENV_MAIN/bin/python" "$REAL_HOME/.venvs/mtx/bin/python" \
           "$REAL_HOME/.venvs/h3m3/bin/python" "$REAL_HOME/.venvs/h3m/bin/python"; do
    [ -x "$c" ] || continue
    if "$c" -c "import mlx.core, mlx_lm" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  # ② 退而求其次: 只 要 有 mlx (下面 会 补 装 mlx-lm)
  for c in "$VENV_MAIN/bin/python" "$REAL_HOME/.venvs/mtx/bin/python" \
           "$REAL_HOME/.venvs/h3m3/bin/python" "$REAL_HOME/.venvs/h3m/bin/python"; do
    [ -x "$c" ] || continue
    if "$c" -c "import mlx.core" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

PY="$(pick_python || true)"
if [ -z "$PY" ]; then
  say "装 推理 环境 (mlx) 到 $VENV_MAIN …"
  BASE="$(command -v python3 || true)"
  [ -n "$BASE" ] || die "找 不 到 python3 —— 先 装 Xcode 命令行 工具 (xcode-select --install)"
  "$BASE" -m venv "$VENV_MAIN" 2>/dev/null || die "建 venv 失败"
  "$VENV_MAIN/bin/pip" install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
  "$VENV_MAIN/bin/pip" install -q mlx numpy -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || "$VENV_MAIN/bin/pip" install -q mlx numpy || die "装 mlx 失败"
  "$VENV_MAIN/bin/pip" install -q mlx-lm -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || "$VENV_MAIN/bin/pip" install -q mlx-lm || warn "mlx-lm 装 失败 —— 对话 会 不可用"
  PY="$VENV_MAIN/bin/python"
  say "装 好 了: $PY"
else
  say "复用 已有 环境: $PY"
fi
"$PY" -c "import mlx.core, numpy" >/dev/null 2>&1 || die "这个 环境 装 不好 mlx"

# mlx_lm —— 对话 (LLM) 必需。缺 就 补 装。
if ! "$PY" -c "import mlx_lm" >/dev/null 2>&1; then
  say "补 装 mlx-lm (对话 引擎) 到 $PY …"
  "$PY" -m pip install -q mlx-lm -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || "$PY" -m pip install -q mlx-lm || warn "mlx-lm 装 失败 —— 对话 功能 会 不可用"
fi
"$PY" -c "import mlx_lm" >/dev/null 2>&1 && say "对话 引擎 (mlx_lm) OK" || warn "这个 环境 缺 mlx_lm —— 对话 会 报 no module named mlx_lm"

# 视频/出图 引擎 需要 的 库 (都很 小)
#   huggingface_hub / safetensors = 读 权重
#   imageio / imageio-ffmpeg      = 出片 写 mp4
#   opencv-python-headless (cv2)   = 帧 处理 (不 带 GUI, 服务器 上 能 用)
#   ftfy / einops                  = vendor 引擎 (mlx_video) 的 依赖
#   注意: 别 装 PyAV (av) —— 它 自带 一 份 ffmpeg dylib, 跟 cv2 里 的 那 份 撞,
#         两 个 都 加载 会 报 "Class AVFFrame is implemented in both" 并 可能 崩.
#         vendor 里 其实 用不到 av —— mp4 交给 imageio-ffmpeg 就 行.
for pkg in huggingface_hub safetensors imageio imageio-ffmpeg ftfy einops; do
  "$PY" -c "import $pkg" >/dev/null 2>&1 || \
    "$PY" -m pip install -q "$pkg" -i https://pypi.tuna.tsinghua.edu.cn/simple || true
done

# ffmpeg 转发壳 —— 装 完 imageio-ffmpeg 才 写 (要用 它 的 python).
#   为什么 不 直接 带 一个 ffmpeg 二进制:
#     ffmpeg 是 LGPL/GPL, 随 仓库 分发 = 整个 项目 被迫 改 GPL.
#     imageio-ffmpeg 是 pip 依赖, 每个 用户 自己 装 —— 那 不 算 分发.
mkdir -p "$OURS/bin"
cat > "$OURS/bin/ffmpeg" <<FFW
#!/bin/bash
# MLX Studio 的 ffmpeg 转发壳 —— 优先 用 imageio-ffmpeg 自带 的 那份,
# 找不到 就 用 系统 PATH 里 的 ffmpeg.
EXE="\$("$PY" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)"
if [ -x "\$EXE" ]; then exec "\$EXE" "\$@"; fi
exec ffmpeg "\$@"
FFW
chmod +x "$OURS/bin/ffmpeg" 2>/dev/null || true
say "ffmpeg: ~/.ours/bin/ffmpeg (转发 imageio-ffmpeg / 系统 ffmpeg, 出片 压 mp4 用)"

# ---------------------------------------------------------------- MCP 接入
# agent (Codex / Claude Code / Cursor / Gemini / OpenCode) 直接 驱动 本机 Studio.
# MCP server 是 一 个 **薄壳** —— 活 都 交给 内核 干, 自己 不 加载 模型, 不 占 显存.
#
# **单独 开 一个 venv** (~/.venvs/mlx-mcp) —— 不 装进 Studio 的 venv.
#   原因: mcp 会 带 一 些 自己的 依赖 (pydantic / httpx / starlette …).
#   装进 Studio 的 venv, pip 有 可能 顺手 升 级 那些 包, 把 正在 跑 的 内核 弄 出 问题.
#   独立 venv = 零 污染 —— 装 / 升级 / 卸 MCP 都 不 碰 Studio 一 下.
VENV_MCP="$REAL_HOME/.venvs/mlx-mcp"
if [ ! -x "$VENV_MCP/bin/python" ]; then
  say "建 MCP 专用 venv (~/.venvs/mlx-mcp) …"
  "$BASE" -m venv "$VENV_MCP" 2>/dev/null || warn "建 venv 失败 —— agent 接入 可能 装 不上"
fi
if [ -x "$VENV_MCP/bin/python" ] && ! "$VENV_MCP/bin/python" -c "import mcp" >/dev/null 2>&1; then
  say "装 mcp (agent 接入 用, 独立 venv) …"
  "$VENV_MCP/bin/python" -m pip install -q "mcp>=2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || "$VENV_MCP/bin/python" -m pip install -q "mcp>=2.0" \
    || warn "mcp 装 失败 —— agent 接入 暂不可用, 其它 功能 不受影响"
fi

mkdir -p "$OURS/bin"
cat > "$OURS/bin/mlx-mcp" <<'MCPW'
#!/bin/bash
# MLX Studio —— MCP server 启动器 (给 Codex / Claude Code / Cursor 用).
#
# stdout 必须 干净 (MCP 走 stdio) —— 所以 所有 话 都 写 stderr.
REAL_HOME="$HOME"
if [ ! -d "$REAL_HOME/.ours/mlx_server" ]; then
  DSCL_HOME="$(dscl . -read "/Users/$(whoami)" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
  [ -d "$DSCL_HOME" ] && REAL_HOME="$DSCL_HOME"
fi
SCRIPT="$REAL_HOME/.ours/mlx_server/mcp_server.py"
[ -f "$SCRIPT" ] || { echo "MLX Studio MCP: 找不到 $SCRIPT —— 先 跑 install.sh" >&2; exit 127; }
# 挑 一个 **带 mcp sdk** 的 python —— 优先 MCP 专用 venv (不 污染 Studio 的 依赖).
PY=""
for c in "$REAL_HOME/.venvs/mlx-mcp/bin/python" "$REAL_HOME/.venvs/mlxstudio/bin/python" \
         "$REAL_HOME/.venvs/mtx/bin/python" \
         "$REAL_HOME/.venvs/h3m3/bin/python" "$REAL_HOME/.venvs/h3/bin/python"; do
  [ -x "$c" ] || continue
  "$c" -c "import mcp" >/dev/null 2>&1 || continue
  PY="$c"; break
done
[ -n "$PY" ] || { echo "MLX Studio MCP: 没 装 mcp sdk —— 跑 install.sh" >&2; exit 127; }
exec "$PY" "$SCRIPT" "$@"
MCPW
chmod +x "$OURS/bin/mlx-mcp" 2>/dev/null || true
say "MCP 启动器: ~/.ours/bin/mlx-mcp"

# 「第二台机」(出片 / 出图 节点) 的 连接 配置 ——
#   开源版 不 写死 任何 地址. 装 的时候 带上 REMOTE_SSH_TARGET (或 REMOTE_SSH_HOST)
#   就 写 进 ~/.ours/node.json; 不 带 就 不写 (本机 独立 跑 用 不 到).
_rt="${REMOTE_SSH_TARGET:-}"
_rh="${REMOTE_SSH_HOST:-}"
if [ -n "$_rt" ] || [ -n "$_rh" ]; then
  _t="$_rt"
  if [ -z "$_t" ] && [ -n "$_rh" ]; then
    _t="${REMOTE_SSH_USER:-mlxstudio}@$_rh"
  fi
  printf '{"target": "%s"}\n' "$_t" > "$OURS/node.json" 2>/dev/null || true
  say "第二台机: ~/.ours/node.json -> $_t"
fi

# mflux = 第二 套 出图 内核 (Z-Image-Turbo / MOSS-Image-SFT) ——
#   不 装 的 话, 出图 页 里 那 两个 模型 会 报 "no module named mflux"
if ! "$PY" -c "import mflux" >/dev/null 2>&1; then
  say "装 mflux (出图 引擎: Z-Image-Turbo / MOSS-Image-SFT) …"
  "$PY" -m pip install -q "mflux>=0.23.0" -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || "$PY" -m pip install -q "mflux>=0.23.0" \
    || warn "mflux 装 失败 —— Z-Image / MOSS 出图 暂不可用 (Krea 不受影响)"
fi
"$PY" -c "import mflux" >/dev/null 2>&1 && say "出图 引擎 (mflux) OK" || warn "这个 环境 缺 mflux —— Z-Image / MOSS 出图 会 不可用"
# cv2 = opencv-python-headless 的 import 名 —— 单独 判, 别 装 到 GUI 版 (拖 X11)
if ! "$PY" -c "import cv2" >/dev/null 2>&1; then
  "$PY" -m pip install -q opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple || true
fi

# Wan2.2 出片 权重 (走 魔搭 下载 + 本地 转 MLX) 需要 modelscope + torch
#   modelscope = 魔搭 客户端;  torch = 只 在 转换 那 一刻 用 (读 .pth 权重)
for pkg in modelscope torch; do
  if ! "$PY" -c "import $pkg" >/dev/null 2>&1; then
    say "装 $pkg (Wan2.2 出片 权重 下载/转换 用) …"
    "$PY" -m pip install -q "$pkg" -i https://pypi.tuna.tsinghua.edu.cn/simple \
      || "$PY" -m pip install -q "$pkg" \
      || warn "$pkg 装 失败 —— Wan2.2 权重 装不上 (其它 功能 不受影响)"
  fi
done

# MiniMax Music3 需要 mlx-audio (没 装 也 能 用, 只是 Music3 出歌 不可用)
if ! "$PY" -c "import mlx_audio" >/dev/null 2>&1; then
  say "装 mlx-audio (MiniMax Music3 出歌) …"
  "$PY" -m pip install -q mlx-audio -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || "$PY" -m pip install -q mlx-audio || warn "mlx-audio 装 失败 —— Music3 出歌 暂不可用"
fi

# 出歌 (ACE-Step) 需要 torch + diffusers —— 体积 大, 可以 --no-ace 跳过
ACE_PY=""
if [ "$ACE" = "1" ]; then
  for c in "$VENV_ACE/bin/python" "$REAL_HOME/.venvs/ace/bin/python"; do
    [ -x "$c" ] && "$c" -c "import torch, mlx.core, diffusers" >/dev/null 2>&1 \
      && "$c" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >/dev/null 2>&1 \
      && "$c" -c "import transformers; assert transformers.__version__.startswith('4.')" >/dev/null 2>&1 \
      && "$c" -c "import loguru" >/dev/null 2>&1 \
      && ACE_PY="$c" && break
  done
  if [ -z "$ACE_PY" ]; then
    say "装 出歌 环境 (torch + diffusers, ~2GB) 到 $VENV_ACE …"
    # ACE-Step 源码 用 了 PEP-604 类型 语法 (X | None) —— 必须 Python >= 3.10.
    # macOS 自带 的 python3 是 3.9 —— 会 直接 语法 报错, 这里 挨个 找 一个 新的.
    BASE=""
    for cand in "$REAL_HOME"/.local/share/uv/python/*/bin/python3 \
                /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
                /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
                /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
                "$(command -v python3 || true)"; do
      [ -x "$cand" ] || continue
      "$cand" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >/dev/null 2>&1 || continue
      BASE="$cand"; break
    done
    if [ -z "$BASE" ]; then
      warn "没 找到 Python >= 3.10 —— 出歌 引擎 (PEP-604 语法) 装 不 上; 装 个 python3.11+ 再 重跑"
    else
      say "出歌 环境 用: $BASE ($("$BASE" -V 2>&1))"
      # 旧 的 3.9 环境 先 挪 走 (别 直接 删 —— 留着 回退)
      if [ -x "$VENV_ACE/bin/python" ]; then
        mv "$VENV_ACE" "${VENV_ACE}-py39.bak.$(date +%s)" 2>/dev/null || true
      fi
    if [ -n "$BASE" ]; then
      "$BASE" -m venv "$VENV_ACE" 2>/dev/null || true
      if [ -x "$VENV_ACE/bin/python" ]; then
        "$VENV_ACE/bin/pip" install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
        # —— 版本 是 锁 死 的 ———
        # transformers 5.x 的 from_pretrained 会 把 没 加载 上 的 权重 留 在 meta device,
        # ACE-Step 加载 DiT 时 直接 报 "Tensor.item() cannot be called on meta tensors".
        # mlx-lm 0.29 / diffusers 0.39 是 跟 transformers 4.57 配套 的 那 一 组.
        ACE_PINS="torch torchvision torchaudio mlx==0.32.4 mlx-lm==0.29.1 numpy==2.4.6
                  transformers==4.57.6 tokenizers==0.22.2 huggingface_hub==0.36.2
                  diffusers==0.39.0 accelerate==1.15.0 safetensors==0.8.0
                  einops==0.8.2 scipy==1.17.1 soundfile==0.14.0 numba==0.68.1
                  vector-quantize-pytorch==1.34.0 torchao==0.16.0 loguru==0.8.0
                  matplotlib==3.11.2 pillow==13.0.0 toml==0.10.2 xxhash"
        "$VENV_ACE/bin/pip" install -q $ACE_PINS \
          -i https://pypi.tuna.tsinghua.edu.cn/simple \
          || "$VENV_ACE/bin/pip" install -q $ACE_PINS \
          || warn "出歌 环境 装 失败 —— 可以 之后 补 装"
      fi
    fi
    fi
    # —— 自检: 依赖 齐 不齐 + transformers 版本 对 不对 (5.x 会 让 DiT 加载 失败) ——
    if [ -x "$VENV_ACE/bin/python" ]; then
      "$VENV_ACE/bin/python" -c "import torch, mlx.core, diffusers, loguru" >/dev/null 2>&1 \
        && "$VENV_ACE/bin/python" -c "import transformers; assert transformers.__version__.startswith('4.'), transformers.__version__" >/dev/null 2>&1 \
        && ACE_PY="$VENV_ACE/bin/python" \
        || warn "出歌 环境 版本 不 对 (transformers 要 4.x) —— 重 跑 本 脚本 会 重建 它"
    fi
  else
    say "复用 出歌 环境: $ACE_PY"
  fi
  [ -n "$ACE_PY" ] || warn "出歌 环境 没 装 好 —— ACE-Step 出歌 暂不可用 (之后 可 重跑 本 脚本)"
fi
[ -n "$ACE_PY" ] || ACE_PY="$REAL_HOME/.venvs/ace/bin/python"

# ================================================================= 4. 环境 变量
# —— Krea-2-Turbo 权重 目录 (指 真正 有 turbo.safetensors 的 那个) ——
KREA_W=""
for _kd in "$OURS/models/Krea-2-Turbo-MLX-Serve-mixed-4-8" \
           "$OURS/models/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8" \
           "$OURS/models/models/krea--Krea-2-Turbo/snapshots/master" \
           "$OURS/models/models/krea--Krea-2-Turbo/snapshots/main"; do
  if [ -f "$_kd/turbo.safetensors" ] || [ -f "$_kd/transformer_mixed_4_8.safetensors" ]; then
    KREA_W="$_kd"; break
  fi
done
[ -n "$KREA_W" ] || KREA_W="$OURS/models/models/krea--Krea-2-Turbo/snapshots/master"

# —— MiniMax-H3 权重 目录 (出片; 布局 有 好几 种, 挨个 找) ——
H3_ROOT=""
for _hd in "$OURS/models/models/MiniMax--MiniMax-H3/snapshots/master/FL2VA" \
           "$OURS/models/MiniMax/MiniMax-H3/FL2VA" \
           "$OURS/models/MiniMax-H3/FL2VA" \
           "$OURS/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit"; do
  # ⚠️ 只 看 text_encoder —— DiT(transformer) 已经 量化 成 ~/.ours/quant/h3-mlx-4bit,
  #    FL2VA/transformer 那 份 bf16 (62G) 运行 时 根本 不 读, 别 因为 它 缺 就 走错 目录.
  if [ -d "$_hd/text_encoder" ] || [ -d "$_hd" ]; then H3_ROOT="$_hd"; break; fi
done
[ -n "$H3_ROOT" ] || H3_ROOT="$OURS/models/models/MiniMax--MiniMax-H3/snapshots/master/FL2VA"

ENVF="$RUNTIME/env.sh"
{
  echo "# MLX Studio 运行 环境 —— 由 install.sh 生成 (别 手改)"
  echo "export MLX_STUDIO_HOME=\"$RUNTIME\""
  echo "export ACE_MLX_ENGINE_DIR=\"$OURS/ace-step-engine\""
  echo "export ACE_MLX_PYTHON=\"$ACE_PY\""
  echo "export ACE_MLX_CHECKPOINTS=\"$OURS/models/ace-step-mlx\""
  echo "export ACE_MLX_SONGS=\"$OURS/ace-songs\""
  echo "export MM3_PYTHON=\"$PY\""
  echo "export MM3_MODEL_DIR=\"$OURS/models/MiniMax-Music3-4bit\""
  echo "export MM3_SONGS=\"$OURS/music3-songs\""
  [ -d "$OURS/vendor/minimax-h3_mlx_src" ] && echo "export H3M_PORT_DIR=\"$OURS/vendor/minimax-h3_mlx_src\""
  [ -d "$OURS/vendor/krea2_engine_src" ] && echo "export H3M_KREA_DIR=\"$OURS/vendor/krea2_engine_src\""
  echo "export H3M_KREA_DIR_W=\"$KREA_W\""
  echo "export H3M_MLX_ROOT=\"$H3_ROOT\""
  echo "export H3M_QUANT_DIR=\"$OURS/quant/h3-mlx-4bit\""
  # 真 ffmpeg: 本机 有 Homebrew (/opt/homebrew/bin), 没 brew 的 机器 没有 —— 它 的 在
  # ~/.ours/bin (install 放 的). 两个 都 加进 PATH, 免得 抽帧/预览 静默 失败.
  echo "export PATH=\"/opt/homebrew/bin:$OURS/bin:/usr/local/bin:\$PATH\""
  echo "export MLX_NODE_ROLE=local"
  # 一 首 歌 最长 多少 秒 (引擎 硬 上限) —— 128G 机器 给 到 480
  echo "export ACE_MAX_DURATION=\${ACE_MAX_DURATION:-480}"
} > "$ENVF"
say "环境 文件: $ENVF"

# ================================================================= 5. 桌面 App
if [ "$APP" = "1" ] || [ "$AGENT" = "1" ]; then
  A=""
  for c in "$ROOT/MLX Serve Studio.app" "$ROOT/../desktop/MLX Serve Studio.app"; do
    [ -d "$c" ] && A="$c" && break
  done
  # 已装 的 那份 (/Applications/…) **不 算 候选** —— 拿 它 再 装 回 去 等于 没 升级
  # (老 版本 永远 赢, 界面 停在 老 的). 只 在 实在 没 别 的 时候 才 用 它.
  # 从 desktop/ 源码 现 编 一个 (要 有 swiftc) —— 编 出来 的 一定 比 已装 的 新.
  if [ -z "$A" ] && [ -f "$ROOT/desktop/build_app.sh" ] && command -v swiftc >/dev/null 2>&1; then
    say "编译 桌面 App (从 desktop/ 源码) …"
    ( cd "$ROOT/desktop" && ./build_app.sh ) >/dev/null 2>&1 || true
    B="$REAL_HOME/.ours/build/MLX Serve Studio.app"
    [ -d "$B" ] && A="$B"
  fi
  # 编 不 出来 (没 swiftc) —— 用 上 次 编 的 那份
  if [ -z "$A" ] && [ -d "$REAL_HOME/.ours/build/MLX Serve Studio.app" ]; then
    A="$REAL_HOME/.ours/build/MLX Serve Studio.app"
  fi
  # 没 swiftc / 编 不 出来 —— 退回 已装 的 那份 (至少 保证 能 跑)
  if [ -z "$A" ] && [ -d "/Applications/MLX Serve Studio.app" ]; then
    A="/Applications/MLX Serve Studio.app"
  fi
  if [ -n "$A" ]; then
    D="/Applications/MLX Serve Studio.app"
    if pgrep -f MLXServeStudio >/dev/null 2>&1; then
      osascript -e 'tell application "MLX Serve Studio" to quit' >/dev/null 2>&1 || true
      sleep 2; pkill -f MLXServeStudio >/dev/null 2>&1 || true; sleep 1
    fi
    if [ "$A" != "$D" ]; then
      [ -e "$D" ] && mv "$D" "$REAL_HOME/.Trash/MLX Serve Studio.app.$(date +%s)" 2>/dev/null || true
      cp -R "$A" /Applications/ && xattr -cr "$D" 2>/dev/null || true
    fi
    say "桌面 App: $D"
    APP_INSTALLED=1
  elif [ "$APP" = "1" ]; then
    warn "没 找到 桌面 App, 也 没 装 swiftc —— 跳过 (内核 照 常 能 用)"
  fi
fi

# ================================================================= 6. 开机自启
PLIST="$REAL_HOME/Library/LaunchAgents/com.ours.mlxstudio.plist"
if [ "$AGENT" = "1" ]; then
  cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.ours.mlxstudio</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>source "$ENVF" &amp;&amp; exec "$PY" "$RUNTIME/server.py" --port $PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$RUNTIME</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$RUNTIME/out.log</string>
  <key>StandardErrorPath</key><string>$RUNTIME/err.log</string>
</dict></plist>
PLISTEOF
  launchctl unload "$PLIST" >/dev/null 2>&1 || true
  launchctl load "$PLIST" >/dev/null 2>&1 || true
  say "开机自启 装 好 了 (com.ours.mlxstudio, 端口 $PORT)"
fi

# ================================================================= 7. 自检
# —— 先 清 老 版本 留 的 垃圾 (/tmp 老 产物 · 探针 · 旧 缓存) ——
for _c in "$ROOT/clean.sh" "$CORE/clean.sh"; do
  [ -f "$_c" ] || continue
  say "先 清 一 遍 旧 文件 …"
  bash "$_c" 2>&1 | sed 's/^/   /' || true
  break
done

say "自检 内核 代码 …"
if (cd "$RUNTIME" && "$PY" -c "import ui, server, api_full, music_workbench, music_mlx, outputs; print('   内核 模块 OK')" 2>&1 | tail -1); then
  say "✅ 内核 模块 正常"
else
  warn "内核 模块 import 有 问题 —— 看 $RUNTIME/err.log"
fi

if [ "$AGENT" = "1" ]; then
  say "等 内核 起来 …"
  OK=0
  for i in $(seq 1 45); do
    if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then OK=1; break; fi
    sleep 1
  done
  if [ "$OK" = "1" ]; then
    say "✅ 内核 正常 —— http://127.0.0.1:$PORT"
  else
    warn "内核 没 起来. 手动 跑 这 条 看 报错:"
    warn "   source $ENVF && $PY $RUNTIME/server.py --port $PORT"
  fi
fi

# ---------------------------------------------------------------- agent 接入
# 装 完 直接 把 Studio 挂 到 本机 **已经 装 了 的** agent 上 —— 不用 手抄 配置.
#   Codex        ~/.codex/config.toml
#   Claude Code  ~/.claude.json
#   Cursor       ~/.cursor/mcp.json
#   Gemini CLI   ~/.gemini/settings.json
#   OpenCode     ~/.config/opencode/config.json
if [ -x "$VENV_MCP/bin/python" ] && "$VENV_MCP/bin/python" -c "import mcp" >/dev/null 2>&1 \
   && [ -f "$RUNTIME/tools/register_mcp.py" ]; then
  say "挂 agent (MCP) …"
  "$VENV_MCP/bin/python" "$RUNTIME/tools/register_mcp.py" --command "$OURS/bin/mlx-mcp" 2>&1 | sed 's/^/   /' || true
fi

cat <<EOF

【装 好 了】
  内核     http://127.0.0.1:$PORT
  桌面 App /Applications/MLX Serve Studio.app
  运行 目录 $RUNTIME
  模型 目录 $OURS/models

【下一步 —— 下 模型 (唯一 需要 下 的 东西)】
  打开 上面 的 地址 (或者 桌面 App) -> 左侧「模型」页 -> 点「下载」
  出歌 (ACE-Step / Music3) 权重 走 魔搭, 国内 直连;
  下载 优先 走 魔搭 (国内 最快), hf-mirror 自动 兜底.
  模型页 会 显示 每个 权重 的 许可 —— Apache-2.0 / MIT 随便 用.

【API】
  http://127.0.0.1:$PORT/v1/chat/completions   (OpenAI 兼容)

【agent 接入 (MCP)】
  已 挂 上 的 agent 重开 就 能 看到 "mlx-studio" 工具 (出图/出片/出歌/模型).
  状态: ~/.ours/bin/mlx-mcp --check
  自检: ~/.ours/bin/mlx-mcp --check

【停 内核】
  launchctl unload "$PLIST"
EOF

# 装 完 直接 把 界面 拉 起来 (别 让 人 再 去 点 一次)
if [ "$APP_INSTALLED" = "1" ] && [ -d "/Applications/MLX Serve Studio.app" ]; then
  open "/Applications/MLX Serve Studio.app" 2>/dev/null || true
fi
