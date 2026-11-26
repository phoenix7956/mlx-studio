#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 MLX Studio 注册 成 agent 的 MCP server.

装完 MLX Studio, 这 个 脚本 会 往 本机 **已经 装 了 的** agent 配置 里
加 一 条 mlx-studio —— 不用 用户 手抄 配置:

    Codex        ~/.codex/config.toml          [mcp_servers.mlx-studio]
    Claude Code  ~/.claude.json                 mcpServers["mlx-studio"]
    Cursor       ~/.cursor/mcp.json             mcpServers["mlx-studio"]
    Gemini CLI   ~/.gemini/settings.json        mcpServers["mlx-studio"]
    OpenCode     ~/.config/opencode/config.json mcp["mlx-studio"]

用法:
    python tools/register_mcp.py             # 注册 (幂等, 重复 跑 不会 乱)
    python tools/register_mcp.py --check     # 只 看 状态, 不 改
    python tools/register_mcp.py --remove    # 摘掉
    python tools/register_mcp.py --print     # 打 出手抄 版 配置

只 改 **已经 存在** 的 agent 配置 —— 没 装 的 agent 不 乱 创建 目录.
每次 改 都 先 备份 (.bak-<时间戳>).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

SERVER_NAME = "mlx-studio"
DEFAULT_CMD = os.path.expanduser("~/.ours/bin/mlx-mcp")


def _real_home() -> str:
    h = os.environ.get("HOME") or ""
    if h and h.startswith("/Users/"):
        return h
    try:
        import subprocess
        out = subprocess.run(["dscl", ".", "-read", "/Users/%s" % os.environ.get("USER", ""),
                              "NFSHomeDirectory"], capture_output=True, text=True, timeout=10).stdout
        parts = out.strip().split()
        if parts:
            return parts[-1]
    except Exception:
        pass
    return h or os.path.expanduser("~")


HOME = _real_home()


def _path(*p) -> str:
    return os.path.join(HOME, *p)


def _backup(path: str) -> None:
    try:
        shutil.copy2(path, path + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
    except Exception:
        pass


# ---------------------------------------------------------------- 配置 形态


def _codex_toml_set(text: str, body: str) -> str:
    """往 TOML 里 写 / 换 一 个 [mcp_servers.<name>] 段."""
    lines = text.splitlines()
    header = "[mcp_servers.%s]" % SERVER_NAME
    start = None
    end = len(lines)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s in (header, '[mcp_servers."%s"]' % SERVER_NAME, "[mcp_servers.'%s']" % SERVER_NAME):
            start = i
            break
    if start is not None:
        for j in range(start + 1, len(lines)):
            if lines[j].lstrip().startswith("["):
                end = j
                break
        return "\n".join(lines[:start] + body.splitlines() + lines[end:]).rstrip("\n") + "\n"
    out = text.rstrip("\n")
    if out:
        out += "\n\n"
    return out + body.rstrip("\n") + "\n"


def _codex_toml_remove(text: str) -> str:
    lines = text.splitlines()
    out = []
    skipping = False
    for ln in lines:
        s = ln.strip()
        if s in ("[mcp_servers.%s]" % SERVER_NAME,
                 '[mcp_servers."%s"]' % SERVER_NAME,
                 "[mcp_servers.'%s']" % SERVER_NAME):
            skipping = True
            continue
        if skipping:
            if s.startswith("["):
                skipping = False
            else:
                continue
        out.append(ln)
    return "\n".join(out).rstrip("\n") + "\n"


def _codex_toml_has(text: str) -> bool:
    return any(ln.strip() in ("[mcp_servers.%s]" % SERVER_NAME,
                              '[mcp_servers."%s"]' % SERVER_NAME,
                              "[mcp_servers.'%s']" % SERVER_NAME)
               for ln in text.splitlines())


def _codex_entry(cmd: str) -> str:
    return ("[mcp_servers.%s]\n"
            'command = "%s"\n'
            "args = []\n"
            "startup_timeout_sec = 30\n"
            "enabled = true\n" % (SERVER_NAME, cmd))


def _json_entry(cmd: str) -> dict:
    return {"command": cmd, "args": [], "env": {}}


AGENTS = [
    ("Codex", _path(".codex", "config.toml"), "toml"),
    ("Claude Code", _path(".claude.json"), "json-root"),
    ("Cursor", _path(".cursor", "mcp.json"), "json-root"),
    ("Gemini CLI", _path(".gemini", "settings.json"), "json-root"),
    ("OpenCode", _path(".config", "opencode", "config.json"), "opencode"),
]


def _json_load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _json_dump(path: str, d: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def apply_one(kind: str, path: str, cmd: str, remove: bool):
    """返回 (状态, 详情). 状态: ok / skip / missing / err"""
    if not os.path.exists(path):
        return "missing", "没 装 (没 找到 %s)" % path
    try:
        if kind == "toml":
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if remove:
                if not _codex_toml_has(text):
                    return "skip", "本来 就 没 注册"
                _backup(path)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(_codex_toml_remove(text))
                return "ok", "已 摘掉"
            _backup(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_codex_toml_set(text, _codex_entry(cmd)))
            return "ok", "已 注册"

        d = _json_load(path)
        if kind == "json-root":
            servers = d.setdefault("mcpServers", {})
            if remove:
                if SERVER_NAME not in servers:
                    return "skip", "本来 就 没 注册"
                servers.pop(SERVER_NAME)
            else:
                servers[SERVER_NAME] = _json_entry(cmd)
        elif kind == "opencode":
            servers = d.setdefault("mcp", {})
            if remove:
                servers.pop(SERVER_NAME, None)
            else:
                servers[SERVER_NAME] = {"type": "local", "command": [cmd], "enabled": True}
        else:
            return "err", "未知 类型 %s" % kind
        _backup(path)
        _json_dump(path, d)
        return "ok", "已 摘掉" if remove else "已 注册"
    except Exception as e:
        return "err", str(e)


def status_of(kind: str, path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        if kind == "toml":
            with open(path, encoding="utf-8") as f:
                return _codex_toml_has(f.read())
        d = _json_load(path)
        if kind == "json-root":
            return SERVER_NAME in (d.get("mcpServers") or {})
        if kind == "opencode":
            return SERVER_NAME in (d.get("mcp") or {})
    except Exception:
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--print", dest="print_only", action="store_true")
    ap.add_argument("--command", default=DEFAULT_CMD, help="MCP 启动 命令")
    args = ap.parse_args()
    cmd = args.command

    if args.print_only:
        print("# Codex —— ~/.codex/config.toml")
        print(_codex_entry(cmd))
        print("# Claude Code / Cursor / Gemini —— ~/.claude.json 等 的 mcpServers")
        print(json.dumps({"mcpServers": {SERVER_NAME: _json_entry(cmd)}},
                         ensure_ascii=False, indent=2))
        return 0

    if args.check:
        print("MLX Studio —— MCP 注册 状态")
        print("  命令: %s%s" % (cmd, "" if os.path.exists(cmd) else "   ⚠️ 启动器 不存在"))
        for name, path, kind in AGENTS:
            if not os.path.exists(path):
                print("  —  %-12s 没 装" % name)
            else:
                print("  %s %-12s %s" % ("✅" if status_of(kind, path) else "⚠️ ", name, path))
        return 0

    print("MLX Studio —— 注册 MCP")
    if not os.path.exists(cmd):
        print("  ⚠️ 启动器 不存在: %s —— 先 跑 install.sh" % cmd)
    n = 0
    for name, path, kind in AGENTS:
        st, msg = apply_one(kind, path, cmd, args.remove)
        if st == "ok":
            n += 1
            print("  ✅ %-12s %s" % (name, msg))
        elif st == "skip":
            print("  —  %-12s %s" % (name, msg))
        elif st == "missing":
            print("  —  %-12s %s" % (name, msg))
        else:
            print("  ❌ %-12s %s" % (name, msg))
    print("  → %s 个 agent 挂 上 了" % n)
    if n:
        print("  重开 agent 生效 (Codex / Claude Code 退 出 再 进).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
