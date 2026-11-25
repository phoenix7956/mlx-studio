#!/bin/bash
# 构建 MLX Serve Studio.app (SwiftUI + WKWebView, 无签名, 本机 自跑)
set -e
cd "$(dirname "$0")"
APP="MLX Serve Studio.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -o "$APP/Contents/MacOS/MLXServeStudio" Sources/main.swift \
  -framework SwiftUI -framework WebKit -framework AppKit
cp Info.plist "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --deep -s - "$APP" 2>/dev/null || true
echo "built $APP"
