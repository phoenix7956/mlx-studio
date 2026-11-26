#!/bin/bash
# MLX Studio —— 清 旧 文件 (老 版本 / 闭源 残留 留下 的 垃圾).
#
#   ./clean.sh            # 只 清 安全 的 垃圾 (推荐)
#   ./clean.sh --deep     # 连 闭源 MLX-Serve 残留 / 开发 用 的 内核 服务 一起 清 (会 问 一 遍)
#   ./clean.sh --slim     # 连 冗余 的 模型 副本 一起 清 (省 100G+ 硬盘; 见 第 6 步)
#
# 清 什么:
#   · /tmp 里 的 老 产物 (mx_*) 与 探针 (mlx_probe_*) —— 资料库 会 把 前缀 撞 上 的 当成 产物
#   · 输出 目录 里 的 测试 文件 (mx_tts_probe_* / mx_tts_ref_* / mx_tts_blend_* / mx_smoke_*)
#   · /Applications 里 的 旧 版 App 备份 (MLX Serve Studio.app.old.*)
#   · 旧 的 内核 缓存 (__pycache__ / .pyc)
#   · 旧 的 用户 缓存 (~/Library/Caches/com.ours.*)
#   · --deep: 闭源 MLX-Serve 的 残留 (~/.mlx-serve) + 开发 用 的 内核 launchd
#
# **不 动** 模型 权重 (~/.ours/models) 与 产物 (~/Documents/MLX Studio 输出) 里 你 的 东西.
set -e

DEEP=0
SLIM=0
[ "${1:-}" = "--deep" ] && DEEP=1
[ "${1:-}" = "--slim" ] && SLIM=1

say()  { printf "\033[36m[清]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[注意]\033[0m %s\n" "$*"; }

REAL_HOME="$HOME"
DSCL_HOME="$(dscl . -read "/Users/$(whoami)" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
[ -d "$DSCL_HOME" ] && REAL_HOME="$DSCL_HOME"
OURS="$REAL_HOME/.ours"
OUTDIR="$REAL_HOME/Documents/MLX Studio 输出"

N=0
purge() {  # purge <说明> <文件...>
  local what="$1"; shift
  local k=0 f
  for f in "$@"; do
    [ -e "$f" ] || continue
    rm -rf "$f" 2>/dev/null && k=$((k+1)) || true
  done
  if [ "$k" -gt 0 ]; then say "$what: $k 个"; N=$((N+k)); fi
}

say "HOME = $REAL_HOME"

# —— 1. /tmp 里 的 老 产物 / 探针 ——
TMP_OLD=()
for f in /tmp/mx_* /tmp/mlx_probe_* /tmp/mx_smoke_*; do
  [ -e "$f" ] && TMP_OLD+=("$f")
done
purge "/tmp 老 产物/探针" "${TMP_OLD[@]}"

# —— 2. 输出 目录 里 的 测试 文件 ——
TESTF=()
for f in "$OUTDIR"/*/mx_tts_probe_* "$OUTDIR"/*/mx_tts_ref_* "$OUTDIR"/*/mx_tts_blend_* \
         "$OUTDIR"/*/mx_smoke_* "$OUTDIR"/*/*probe*; do
  [ -e "$f" ] && TESTF+=("$f")
done
purge "输出 目录 里 的 测试 文件" "${TESTF[@]}"

# —— 3. 旧 版 App 备份 ——
purge "旧 版 App 备份" "/Applications/MLX Serve Studio.app.old."* "/Applications/MLX-Serve.app.old."* 2>/dev/null || true

# —— 4. 内核 缓存 ——
find "$OURS/mlx_server" -name "__pycache__" -type d -maxdepth 3 2>/dev/null | while read -r d; do rm -rf "$d" 2>/dev/null; done
find "$OURS/mlx_server" -name "*.pyc" -type f -delete 2>/dev/null || true
say "内核 缓存 (__pycache__ / .pyc) 已 清"

# —— 5. 用户 缓存 ——
purge "旧 的 用户 缓存" "$REAL_HOME/Library/Caches/com.ours.mlxstudio" \
     "$REAL_HOME/Library/Caches/com.ours.mlxsserve"* 2>/dev/null || true

# —— 6. 冗余 的 模型 副本 (默认 只 报告; --slim 才 真 删) ——
#    这 几 份 是 明确 冗余 的, 删 了 不 影响 出片:
#      · FL2VA/transformer  的 bf16 (62G) —— DiT 已经 量化 成 ~/.ours/quant/h3-mlx-4bit,
#        出片 时 根本 不 读 那 份 bf16 (只有 重新 量化 时 才 需要, 可 从 HF 重 下).
#      · models/mt/          的 魔搭 下载 缓存 (15G) —— 顶层 已有 正本.
#      · mlm/Qwen3-4B-8bit   的 重复 档 —— catalog 用 的是 mlm/Qwen3-4B-MLX.
SLIM_PATHS=(
  "$OURS/models/models/MiniMax--MiniMax-H3/snapshots/master/FL2VA/transformer"
  "$OURS/models/MiniMax/MiniMax-H3/FL2VA/transformer"
  "$OURS/models/MiniMax-H3/FL2VA/transformer"
  "$OURS/models/mt"
  "$OURS/models/mlm/Qwen3-4B-8bit"
  "$OURS/models/local"
)
SLIM_HITS=()
for f in "${SLIM_PATHS[@]}"; do
  [ -e "$f" ] && SLIM_HITS+=("$f")
done
# ddalcu/Krea-... 是 软链 —— 魔搭 缓存 删 了 就 断 了, 先 改指 顶层 正本
_LINK="$OURS/models/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8"
_REAL="$OURS/models/Krea-2-Turbo-MLX-Serve-mixed-4-8"
if [ -L "$_LINK" ] && [ -d "$_REAL" ]; then
  rm -f "$_LINK" && ln -s "$_REAL" "$_LINK"
  say "ddalcu/Krea 软链 已 改指 顶层 正本"
fi
if [ "${#SLIM_HITS[@]}" -gt 0 ]; then
  echo
  say "冗余 的 模型 副本 (删 了 不 影响 出片):"
  for f in "${SLIM_HITS[@]}"; do
    echo "    $(du -sh "$f" 2>/dev/null | cut -f1)  $f"
  done
  if [ "$SLIM" = "1" ]; then
    purge "冗余 模型 副本" "${SLIM_HITS[@]}"
  else
    warn "想 一起 清 就 跑: ./clean.sh --slim"
  fi
fi

# —— 7. 报告 (不 自动 删) ——
echo
if [ -d "$REAL_HOME/.mlx-serve" ]; then
  warn "还 留 着 闭源 MLX-Serve 的 残留: $REAL_HOME/.mlx-serve  ($(du -sh "$REAL_HOME/.mlx-serve" 2>/dev/null | cut -f1))"
  warn "   里面 可能 有 你 之前 下 的 模型 —— 要 彻底 清 就 跑: ./clean.sh --deep"
fi
for p in "$REAL_HOME/Library/LaunchAgents/com.codex.mlxserver-dev.plist" \
         "$REAL_HOME/Library/LaunchAgents/com.ours.mlxsserve-ref.plist" \
         "$REAL_HOME/Library/LaunchAgents/com.ours.mlxsserve-ref2.plist"; do
  [ -e "$p" ] && warn "还 留 着 旧 的 内核 服务: $p  (卸载: launchctl unload \"$p\")"
done
[ -d "$OURS/models" ] && say "模型 权重 保留 在: $OURS/models  ($(du -sh "$OURS/models" 2>/dev/null | cut -f1))"

# —— 8. --deep: 闭源 残留 + 开发 服务 ——
if [ "$DEEP" = "1" ]; then
  echo
  printf "要 连 闭源 残留 (~/.mlx-serve) 和 开发 用 的 内核 服务 一起 清 掉? [y/N] "
  read -r ans || ans=n
  if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
    purge "闭源 MLX-Serve 残留" "$REAL_HOME/.mlx-serve"
    for p in "$REAL_HOME/Library/LaunchAgents/com.codex.mlxserver-dev.plist" \
             "$REAL_HOME/Library/LaunchAgents/com.ours.mlxsserve-ref.plist" \
             "$REAL_HOME/Library/LaunchAgents/com.ours.mlxsserve-ref2.plist"; do
      [ -e "$p" ] || continue
      launchctl unload "$p" >/dev/null 2>&1 || true
      purge "旧 的 内核 服务" "$p"
    done
  fi
fi

echo
say "清 完 了 (共 $N 项)"
