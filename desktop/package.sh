#!/bin/bash
# 把 桌面端 Studio 打 成 可以 装 进 mac 的 安装包 (DMG 拖拽版 + ZIP 版).
#
# 安装包里 有:
#   MLX Serve Studio.app   桌面 App (自带 内核)
#   install.sh             一键 安装 (内核代码/引擎/依赖/开机自启)
#   内核代码/              内核 + 出片/出图 引擎 源码
#   出歌引擎/              ACE-Step 出歌 引擎 源码
#   安装到本机.command     双击 就 装
# 模型 权重 不 带 —— 装 完 在「模型」页 点「下载」.
set -e
cd "$(dirname "$0")"
NAME="MLX-Serve-Studio"
VER="1.10"
OUT="$(pwd)/dist"
STAGE="$(mktemp -d)/stage"
ROOT="$(cd .. && pwd)"
[ -f "$ROOT/server.py" ] || ROOT="$ROOT/实现代码"
VENDOR="$ROOT/../_vendor"
[ -d "$VENDOR" ] || VENDOR="$HOME/Desktop/rea/ours/_vendor"
REAL_HOME="$HOME"
DSCL_HOME="$(dscl . -read "/Users/$(whoami)" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
[ -d "$DSCL_HOME" ] && REAL_HOME="$DSCL_HOME"
ACE_ENGINE="$REAL_HOME/.ours/ace-step-engine"
# 构建 产物 在 ~/.ours/build/ (不 在 桌面 上 —— 免得 Launchpad 冒 同名 图标)
OUT_APP="${MLX_APP_OUT:-$REAL_HOME/.ours/build}/MLX Serve Studio.app"

rm -rf "$OUT"; mkdir -p "$OUT"
mkdir -p "$STAGE"

# ---------------------------------------------------------------- 桌面 App
[ -d "$OUT_APP" ] || { echo "❌ 先 跑 build_app.sh" >&2; exit 1; }
cp -R "$OUT_APP" "$STAGE/"

# ---------------------------------------------------------------- 内核 代码
K="$STAGE/内核代码"
mkdir -p "$K"
# 跟 install.sh 一致: 拷贝 全部 顶层 模块, 只 排除 探针 / 回归 脚本.
# 手工 清单 漏 过 msdl.py (魔搭下载) —— 漏 了 装完 点「下载」就 报「缺 msdl.py」.
for f in "$ROOT"/*.py; do
  b="$(basename "$f")"
  case "$b" in
    probe_ms*.py|smoke_*.py|test_*.py) continue ;;
  esac
  cp "$f" "$K/"
done
cp "$ROOT/model_catalog.json" "$K/" 2>/dev/null || true
for d in ui tools engines; do
  [ -d "$ROOT/$d" ] && cp -R "$ROOT/$d" "$K/"
done
# 出片 (MiniMax-H3 MLX) / 出图 (Krea-2-Turbo MLX) 引擎 源码
for d in minimax-h3_mlx_src krea2_engine_src port_minimax_h3_mlx; do
  [ -d "$ROOT/$d" ] && cp -R "$ROOT/$d" "$K/"
  [ -d "$VENDOR/$d" ] && cp -R "$VENDOR/$d" "$K/"
  [ -d "$ROOT/vendor/$d" ] && cp -R "$ROOT/vendor/$d" "$K/"
done
# Wan2.2 出片 引擎 (我们 自研 的 port + 上游 vendor 源码 —— 原样 vendor, 没 改)
[ -d "$ROOT/port_wan2_mlx" ] && cp -R "$ROOT/port_wan2_mlx" "$K/"
if [ -d "$ROOT/vendor_src/mlx_video" ]; then
  mkdir -p "$K/vendor_src"
  cp -R "$ROOT/vendor_src/mlx_video" "$K/vendor_src/mlx_video"
fi
# 别 把 __pycache__ 打进 包 —— 换 python 版本 就 是 一 堆 垃圾
find "$K" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
# ACE-Step 权重 目录 的 软链 —— 指向 打包机 的 绝对路径, 换 台 机器 就 是 死链.
# install.sh 会 按 本机 实际 路径 重 建 一 条, 所以 包 里 不 带.
rm -f "$K/ace-step-engine/project/checkpoints" 2>/dev/null || true
cp "$(dirname "$0")/run_local_dev.sh" "$K/run_local_dev.sh"
chmod +x "$K/run_local_dev.sh"

# ---------------------------------------------------------------- 出歌 引擎 (并 进 内核代码)
ACE_SRC=""
for c in "$ROOT/ace-step-engine" "$ROOT/出歌引擎/ace-step-engine" "$ACE_ENGINE"; do
  [ -d "$c/acestep_mlx" ] && ACE_SRC="$c" && break
done
if [ -n "$ACE_SRC" ]; then
  cp -R "$ACE_SRC" "$K/ace-step-engine"
else
  echo "⚠️  没 找到 $ACE_ENGINE —— 出歌 引擎 不 在 包 里 (装 完 只能 用 MiniMax Music3)"
fi
# ACE-Step 权重 目录 的 软链 —— 指向 打包机 的 绝对路径, 换 台 机器 就 是 死链.
# install.sh 会 按 本机 实际 路径 重 建 一 条, 所以 包 里 不 带.
rm -f "$K/ace-step-engine/project/checkpoints" 2>/dev/null || true

# ---------------------------------------------------------------- 安装脚本
cp "$ROOT/install.sh" "$STAGE/install.sh"
chmod +x "$STAGE/install.sh"
cp "$ROOT/clean.sh" "$STAGE/clean.sh"
chmod +x "$STAGE/clean.sh"
cp "$(dirname "$0")/安装说明.txt" "$STAGE/安装说明.txt"
# 许可 (Apache-2.0 要求: 分发 必须 附带 LICENSE 原文 + NOTICE)
for f in LICENSE NOTICE THIRD_PARTY_NOTICES.md README.md; do
  [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$STAGE/$f"
  [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$K/$f"
done
# 文档 —— 商用合规 / Agent 接入 / 出片规范, 装 完 在 本机 能 查
if [ -d "$ROOT/docs" ]; then
  mkdir -p "$K/docs"
  cp -R "$ROOT/docs/." "$K/docs/"
  rm -rf "$K/docs/screenshots"
fi

# ---------------------------------------------------------------- 清理 (双击)
cat > "$STAGE/清理旧文件.command" <<'CMD'
#!/bin/bash
cd "$(dirname "$0")"
echo "== 清 MLX Studio 的 旧 文件 (/tmp 老 产物 · 探针 · 旧 缓存) =="
bash ./clean.sh
echo
read -r -p "按 回车 关闭 这个 窗口 …" _ || true
osascript -e 'tell application "Terminal" to close (every window whose name contains "清理旧文件")' \
  >/dev/null 2>&1 || osascript -e 'tell application "Terminal" to close front window' \
  >/dev/null 2>&1 || true
CMD
chmod +x "$STAGE/清理旧文件.command"

# ---------------------------------------------------------------- 双击 安装
cat > "$STAGE/安装到本机.command" <<'CMD'
#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "== 装 MLX Serve Studio (内核 + 引擎 + 依赖 + 开机自启) =="
if bash ./install.sh --app; then
  echo
  echo "装 完 了 —— 桌面 App 已 打开; 模型 权重 去 界面「模型」页 点「下载」."
else
  echo
  echo "❌ 装 失败 —— 看 上面的 报错"
  read -r -p "按 回车 关闭 …" _ || true
  exit 1
fi
read -r -p "按 回车 关闭 这个 窗口 …" _ || true
# 关掉 本 窗口 (Terminal 默认 不 会 自动 关)
osascript -e 'tell application "Terminal" to close (every window whose name contains "安装到本机")' \
  >/dev/null 2>&1 || osascript -e 'tell application "Terminal" to close front window' \
  >/dev/null 2>&1 || true
CMD
chmod +x "$STAGE/安装到本机.command"

# ---------------------------------------------------------------- 挂 Agent (MCP)
cat > "$STAGE/注册到 Agent.command" <<'CMD'
#!/bin/bash
cd "$(dirname "$0")"
echo "== 把 MLX Studio 挂 到 本机 已装 的 agent (Codex / Claude Code / Cursor …) =="
REG="内核代码/tools/register_mcp.py"
[ -f "$REG" ] || REG="内核代码/register_mcp.py"
if [ ! -f "$REG" ]; then
  echo "❌ 找不到 注册 脚本 ($REG) —— 先 双击「安装到本机.command」"
  read -r -p "按 回车 关闭 …" _ || true
  exit 1
fi
PY=""
for c in "$HOME/.venvs/mlx-mcp/bin/python" "$HOME/.venvs/mlxstudio/bin/python" "$HOME/.venvs/mtx/bin/python" \
         "$HOME/.venvs/h3m3/bin/python" "$HOME/.venvs/h3/bin/python" "$(command -v python3)"; do
  [ -x "$c" ] && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "❌ 找不到 python3"; read -r -p "按 回车 关闭 …" _ || true; exit 1; }
"$PY" "$REG" --command "$HOME/.ours/bin/mlx-mcp"
echo
echo "重开 agent 就 能 看到 mlx-studio 工具 (出图 / 出片 / 出歌)."
read -r -p "按 回车 关闭 这个 窗口 …" _ || true
osascript -e 'tell application "Terminal" to close (every window whose name contains "注册到 Agent")' \
  >/dev/null 2>&1 || osascript -e 'tell application "Terminal" to close front window' \
  >/dev/null 2>&1 || true
CMD
chmod +x "$STAGE/注册到 Agent.command"

# ---------------------------------------------------------------- 自检 内容
MISS=""
for f in install.sh clean.sh model_catalog.json ui/console.html tools/fetch_h3_mlx.py video_batch.py video_mv.py \
         msdl.py server.py voice_studio.py image_engine_mflux.py music_workbench.py \
         mcp_server.py tools/register_mcp.py; do
  [ -e "$STAGE/内核代码/$f" ] || [ -e "$STAGE/$f" ] || MISS="$MISS $f"
done
[ -d "$STAGE/内核代码/krea2_engine_src" ] || MISS="$MISS krea2_engine_src"
[ -d "$STAGE/内核代码/minimax-h3_mlx_src" ] || MISS="$MISS minimax-h3_mlx_src"
[ -d "$STAGE/内核代码/ace-step-engine" ] || MISS="$MISS ace-step-engine"
[ -d "$STAGE/内核代码/port_wan2_mlx" ] || MISS="$MISS port_wan2_mlx"
[ -d "$STAGE/内核代码/vendor_src/mlx_video" ] || MISS="$MISS vendor_src/mlx_video"
if [ -n "$MISS" ]; then
  echo "❌ 安装包 缺:$MISS" >&2
  exit 1
fi

# ---------------------------------------------------------------- DMG
DMG="$OUT/$NAME-$VER.dmg"
hdiutil detach "/Volumes/$NAME $VER" -force >/dev/null 2>&1 || true
hdiutil create -volname "$NAME $VER" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null

# ---------------------------------------------------------------- ZIP
# 打包 前 最后 清 一 次 __pycache__ (ace-step 引擎 里 也 有)
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true
# ACE-Step 权重 软链 —— 指向 打包机 的 绝对路径, 换 台 机器 就 是 死链
rm -f "$STAGE/内核代码/ace-step-engine/project/checkpoints" 2>/dev/null || true
# zip 要 用 python 的 —— macOS 自带 的 zip 不 打 UTF-8 标志,
# 别的 机器 (尤其 windows) 解压 出来 中文名 全 是 乱码.
ZIP_HELPER="$(cd "$(dirname "$0")" && pwd)/zip_utf8.py"
python3 "$ZIP_HELPER" "$STAGE" "$OUT/$NAME-$VER.zip"

# ---------------------------------------------------------------- 挂载 自检
MP=$(hdiutil attach "$DMG" -nobrowse -readonly | tail -1 | awk -F'\t' '{print $NF}')
echo "挂载: $MP"
ls -1 "$MP"
for f in install.sh "安装到本机.command" "安装说明.txt"; do
  [ -e "$MP/$f" ] || { echo "❌ DMG 里 缺 $f" >&2; exit 1; }
done
[ -e "$MP/清理旧文件.command" ] || { echo "❌ DMG 里 缺 清理旧文件.command" >&2; exit 1; }
[ -e "$MP/注册到 Agent.command" ] || { echo "❌ DMG 里 缺 注册到 Agent.command" >&2; exit 1; }
[ -e "$MP/内核代码/mcp_server.py" ] || { echo "❌ DMG 里 缺 mcp_server.py (agent 接入)" >&2; exit 1; }
[ -e "$MP/内核代码/tools/register_mcp.py" ] || { echo "❌ DMG 里 缺 register_mcp.py" >&2; exit 1; }
[ -d "$MP/内核代码" ] || { echo "❌ DMG 里 缺 内核代码/" >&2; exit 1; }
[ -d "$MP/内核代码/krea2_engine_src" ] || { echo "❌ DMG 里 缺 出图引擎" >&2; exit 1; }
[ -d "$MP/内核代码/minimax-h3_mlx_src" ] || { echo "❌ DMG 里 缺 出片引擎" >&2; exit 1; }
[ -d "$MP/内核代码/ace-step-engine" ] || { echo "❌ DMG 里 缺 出歌引擎" >&2; exit 1; }
head -1 "$MP/内核代码/server.py" >/dev/null || { echo "❌ DMG 里 server.py 读 不 出" >&2; exit 1; }
hdiutil detach "$MP" >/dev/null

echo "built:"
ls -lh "$OUT"
