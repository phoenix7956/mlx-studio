#!/usr/bin/env python3
"""扫描 仓库 里 的 私人 痕迹 —— 不 点名 任何 人, 只 按 规则 认.

要 挡 的:
  1. 写死 的 某台 机器 的 家目录  (/Users/<真名字>/…)
  2. 内网 IP  (192.168.* / 10.* / 172.16-31.*)
  3. 疑似 API key (mlx_ / ghp_ / github_pat_ / AKIA …)

放过 的:
  - /Users/$(whoami)/  —— 脚本 里 取 当前 用户
  - /Users/<名字>/    —— 文档 里 的 占位
  - ~ 开 头 的 路径   (~/.ours/… 是 本项目 自己的 运行 布局, 不是 私人 信息)
  - vendor/ vendor_src/ ace-step-engine/  —— 第三方 源码, 原样 保留

用法: python3 tools/check_private_paths.py [要 扫 的 目录, 默认 当前 仓库]
退出码 0 = 通过, 1 = 有 问题.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SKIP_DIRS = {".git", "vendor", "vendor_src", "ace-step-engine", "ref_kernels",
             "__pycache__", "node_modules", "dist", "build", ".venv", "venv"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".icns", ".mp4", ".mov",
            ".mp3", ".wav", ".zip", ".dmg", ".safetensors", ".bin", ".pyc"}

# 家目录: /Users/<名字>/  —— 名字 里 不能 带 变量/占位
RE_HOME = re.compile(r"/Users/([A-Za-z0-9._-]+)/")
# 内网 IP
RE_LAN = re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"
                    r"|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
                    r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b")
# 疑似 key
RE_KEY = re.compile(r"\b(?:mlx_[A-Za-z0-9]{24,}|ghp_[A-Za-z0-9]{20,}"
                    r"|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b")
# 私钥 文件 头
RE_PEM = re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY")

# 这些 名字 算 占位/变量, 不 算 私人
OK_HOME_NAMES = {"$(whoami)", "<you>", "<名字>", "<name>", "<user>", "$USER",
                  "{user}", "username", "you"}

# 文档 里 用 的 通用 例子 地址 (RFC 风格 的 示例, 不是 任何 真实 主机)
ALLOWED_LITERALS = ("192.168.1.5",)


def scan_text(path, text):
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in RE_HOME.finditer(line):
            name = m.group(1)
            if name in OK_HOME_NAMES or name.startswith(("<", "$", "{")):
                continue
            hits.append((i, "家目录 %s" % m.group(0), line.strip()[:110]))
        for m in RE_LAN.finditer(line):
            if m.group(0) in ALLOWED_LITERALS:
                continue
            hits.append((i, "内网 IP %s" % m.group(0), line.strip()[:110]))
        for m in RE_KEY.finditer(line):
            hits.append((i, "疑似 key", line.strip()[:110]))
        if RE_PEM.search(line):
            hits.append((i, "私钥", line.strip()[:110]))
    return hits


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SKIP_EXT:
                continue
            yield os.path.join(dirpath, fn)


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    bad = 0
    for path in walk(root):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        rel = os.path.relpath(path, root)
        # 这 个 脚本 自己 会 命中 自己 的 正则 —— 跳过
        if rel == os.path.basename(__file__):
            continue
        hits = scan_text(rel, text)
        for line, kind, ctx in hits:
            print("❌ %s:%d  %s —— %s" % (rel, line, kind, ctx))
            bad += 1
    if bad:
        print("\n%d 处 私人 痕迹 —— 别 push, 先 清掉" % bad)
        return 1
    print("✅ 无 私人 路径 / 内网 地址 / 疑似 key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
