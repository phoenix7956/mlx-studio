"""remote_node.py —— 「第二台机」(出片 / 出图 节点) 的 连接 配置.

开源版 不 写死 任何 机器 的 地址. 连接 信息 依次 从 三 处 读:

  1. 环境变量 ``REMOTE_SSH_TARGET`` —— 完整 的 "user@host"
  2. 环境变量 ``REMOTE_SSH_HOST`` (+ 可选 ``REMOTE_SSH_USER``)
  3. 本机 配置文件 ``~/.ours/node.json`` —— ``{"target": "user@host"}``
     (这个 文件 在 本机, **不进 仓库**)

读 不 到 就 返回 "" —— 本机 独立 跑 时 用 不 到 第二台机.
"""

import json
import os

DEFAULT_USER = "<you>"
CFG_PATH = os.path.expanduser("~/.ours/node.json")


def _from_cfg():
    try:
        if not os.path.isfile(CFG_PATH):
            return ""
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = str(d.get("target") or d.get("host") or "").strip()
        if t and "@" not in t:
            u = str(d.get("user") or DEFAULT_USER).strip() or DEFAULT_USER
            t = f"{u}@{t}"
        return t
    except Exception:
        return ""


def target():
    """返回 "user@host"; 没有 配置 就 返回 ""."""
    t = (os.environ.get("REMOTE_SSH_TARGET") or "").strip()
    if t:
        return t if "@" in t else f"{DEFAULT_USER}@{t}"
    host = (os.environ.get("REMOTE_SSH_HOST") or "").strip()
    if host:
        user = (os.environ.get("REMOTE_SSH_USER") or DEFAULT_USER).strip() or DEFAULT_USER
        return f"{user}@{host}"
    return _from_cfg()


def host():
    """只 返回 host (没有 "@user" 前缀)."""
    t = target()
    return t.split("@", 1)[1] if "@" in t else t


def user():
    """只 返回 user."""
    t = target()
    return t.split("@", 1)[0] if "@" in t else ""


def scp(src, dst, timeout=300):
    """把 节点 上 的 文件 拉 回 本机. 没有 配置 就 返回 None."""
    t = target()
    if not t or not src:
        return None
    import subprocess as sp
    try:
        return sp.run(["scp", "-o", "ConnectTimeout=10", f"{t}:{src}", dst],
                      capture_output=True, timeout=timeout)
    except Exception:
        return None
