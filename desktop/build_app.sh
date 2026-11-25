#!/bin/bash
# 构建 MLX Serve Studio.app —— 自带 图标 + 内核 代码, 可以 直接 拷 到 别的 mac 上 装.
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"                       # 内核 代码 目录 (mlx_server/)
# 归档 副本 放 在 别的 目录 时, 内核 代码 其实 在 "实现代码/" 里
[ -f "$ROOT/server.py" ] || ROOT="$ROOT/实现代码"
VENDOR="$ROOT/../_vendor"                   # 厂商 源码 (出图 / 出片)
[ -d "$VENDOR" ] || VENDOR="$HOME/Desktop/rea/ours/_vendor"
APP="MLX Serve Studio.app"
ICON="$(pwd)/AppIcon.icns"                  # 图标

# 构建 产物 **不 放 在 桌面** —— 桌面 上 每 多 一个 .app, Launchpad 就 多 一 个 同名 图标.
# 统一 放 到 ~/.ours/build/ ; 要 装 的 话 从 那个 路径 拷 (或 直接 用 DMG).
REAL_HOME="$HOME"
DSCL_HOME="$(dscl . -read "/Users/$(whoami)" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
[ -d "$DSCL_HOME" ] && REAL_HOME="$DSCL_HOME"
OUT_DIR="${MLX_APP_OUT:-$REAL_HOME/.ours/build}"
OUT_APP="$OUT_DIR/MLX Serve Studio.app"
mkdir -p "$OUT_DIR"
rm -rf "$OUT_APP"
mkdir -p "$OUT_APP/Contents/MacOS" "$OUT_APP/Contents/Resources"
APP="$OUT_APP"

# 1) 编译 壳
swiftc -O -o "$APP/Contents/MacOS/MLXServeStudio" Sources/main.swift \
  -framework SwiftUI -framework WebKit -framework AppKit

# 2) Info.plist
cp Info.plist "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

# 3) 图标
if [ -f "$ICON" ]; then
  mkdir -p "$APP/Contents/Resources"
  cp "$ICON" "$APP/Contents/Resources/AppIcon.icns"
fi

# 4) 内核 代码 —— 装 到 别的 mac 上 时, app 自己 就 带 着 内核
K="$APP/Contents/Resources/kernel"
mkdir -p "$K"
for f in server.py api_full.py cap_engines.py engine.py image_engine.py image_engine_mflux.py image_advanced.py \
         llm_engine.py music_engine.py music_mlx.py music_proxy.py music_workbench.py music_duration.py qwen3_tts_engine.py voice_studio.py \
         voice_convert.py ui.py outputs.py model_catalog.json video_batch.py video_mv.py msdl.py; do
  [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$K/"
done
# 魔搭 拉 权重 的 脚本 (模型 不 在 huggingface)
[ -d "$ROOT/tools" ] && cp -R "$ROOT/tools" "$K/tools"
# 我们 自研 的 出歌 引擎 (MiniMax Music3 的 服务 端)
[ -d "$ROOT/engines" ] && cp -R "$ROOT/engines" "$K/engines"
cp -R "$ROOT/ui" "$K/ui"
cp -R "$ROOT/port_minimax_h3_mlx" "$K/port_minimax_h3_mlx" 2>/dev/null || true
# Wan2.2 出片 引擎 (我们 自研 的 port + 上游 vendor 源码)
cp -R "$ROOT/port_wan2_mlx" "$K/port_wan2_mlx" 2>/dev/null || true
mkdir -p "$K/vendor_src"
cp -R "$ROOT/vendor_src/mlx_video" "$K/vendor_src/mlx_video" 2>/dev/null || true
# 厂商 源码 (出图 / 出片 用; 不在 就 自动 跳过)
for v in minimax-h3_mlx_src krea2_engine_src; do
  [ -d "$VENDOR/$v" ] && cp -R "$VENDOR/$v" "$K/$v"
  [ -d "$ROOT/vendor/$v" ] && cp -R "$ROOT/vendor/$v" "$K/$v"
done
cp "$(dirname "$0")/run_local_dev.sh" "$K/run_local_dev.sh"
chmod +x "$K/run_local_dev.sh"

# 6) 自检 —— 内核 缺 任何 一个 模块, 界面 就 白屏 (根路由 抛 异常).
#    这 类 问题 在 别的 mac 上 才 暴露, 所以 打包 时 就 验 一次.
MISS=""
for f in server.py api_full.py cap_engines.py engine.py image_engine.py image_engine_mflux.py image_advanced.py \
         llm_engine.py music_engine.py music_mlx.py music_proxy.py music_workbench.py music_duration.py \
         qwen3_tts_engine.py voice_studio.py voice_convert.py ui.py outputs.py model_catalog.json video_batch.py video_mv.py msdl.py; do
  [ -f "$K/$f" ] || MISS="$MISS $f"
done
[ -f "$K/ui/console.html" ] || MISS="$MISS ui/console.html"
if [ -n "$MISS" ]; then
  echo "❌ 内核 文件 缺:$MISS —— 界面 会 白屏, 别 装" >&2
  rm -rf "$OUT_APP" 2>/dev/null || true      # 别 把 半个 App 留 在 那儿 (装 上去 就 白屏)
  exit 1
fi
PY=""
for c in "$HOME/.venvs/mtx/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [ -x "$c" ] && PY="$c" && break
done
if [ -n "$PY" ]; then
  ( cd "$K" && "$PY" -c "import ui,server,music_workbench,music_mlx,outputs,video_batch,video_mv,voice_studio,msdl,image_engine_mflux" >/dev/null 2>&1 ) || {
    echo "❌ 内核 模块 import 失败 —— 装 上去 也 是 白屏" >&2
    rm -rf "$OUT_APP" 2>/dev/null || true
    exit 1
  }
fi

# 别 把 __pycache__ 打进 app —— 自检 import 会 生成 一 堆, 换 python 版本 就 是 垃圾
find "$K" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# 5) 签名 (ad-hoc) —— 没 开发者 证书 时 也能 在 自己 机器 上 跑
codesign --force --deep -s - "$APP" 2>/dev/null || true
codesign --verify --deep --verbose=1 "$APP" 2>&1 | tail -1
echo "built $APP"
