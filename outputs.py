#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 输出 目录 —— 视频 / 图片 / 音频 / 语音 / 3D 都 落 这 里.

之前 所有 产物 都 丢 在 /tmp (mx_xxx_*), macOS 定期 清 /tmp, 资料库 就 指向
不 存在 的 文件 了. 现在 统一 落 到:

    ~/Documents/MLX Studio 输出/
        音频/    图片/    视频/    语音/    3D/    其他/

文件名 还是 唯一 的 (mx_<类型>_<时间戳>_<随机>.<后缀>), 资料库 靠 这个 认.

想 换 位置: 设 环境变量 MLX_OUT_DIR, 或者 改 这个 文件 顶部 的 ROOT.
"""
import os
import time
import uuid


def _home():
    """真实 用户 的 home —— 不 信 $HOME (有些 运行环境 HOME 是 临时 的)."""
    try:
        import pwd
        h = pwd.getpwuid(os.getuid()).pw_dir
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return os.path.expanduser("~")


HOME = _home()
ROOT = os.environ.get("MLX_OUT_DIR") or os.path.join(HOME, "Documents", "MLX Studio 输出")

KIND_DIR = {
    "audio": "音频",
    "image": "图片",
    "video": "视频",
    "tts":   "语音",
    "3d":    "3D",
    "other": "其他",
}

# 类型 -> 前缀 (跟 老 的 /tmp 命名 保持 一致, 资料库 的 前缀 白名单 不用 改)
KIND_PREFIX = {
    "audio": "mx_audio",
    "image": "mx_img",
    "video": "mx_video",
    "tts":   "mx_tts",
    "3d":    "mx_3d",
    "other": "mx_out",
}


def dir_for(kind):
    """某 一 类 产物 的 目录 (自动 建)."""
    d = os.path.join(ROOT, KIND_DIR.get(kind, "其他"))
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def path_for(kind, prefix=None, ext="", sub=None):
    """给 一个 唯一 的 输出 路径 (目录 会 自动 建)."""
    d = dir_for(kind)
    if sub:
        d = os.path.join(d, sub)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    name = "%s_%d%s" % (prefix or KIND_PREFIX.get(kind, "mx_out"),
                        int(time.time()), uuid.uuid4().hex[:8])
    if ext:
        name += "." + str(ext).lstrip(".")
    return os.path.join(d, name)


def base_for(kind, prefix=None, ext=""):
    """跟 path_for 一样, 只是 不 带 后缀 —— 给 那种 后面 自己 拼 .mp4/.glb 的 地方."""
    return path_for(kind, prefix, "")


def rel(path):
    """输出 文件 -> /out/ 下 的 相对 路径 (子目录 + 文件名)."""
    if not path:
        return ""
    try:
        r = os.path.relpath(os.path.abspath(str(path).split("file://")[-1]), ROOT)
        if not r.startswith(".."):
            return r.replace(os.sep, "/")
    except Exception:
        pass
    return os.path.basename(str(path))


def abs_from_rel(rel_path):
    """相对 路径 -> 绝对 路径 —— 只 允许 落在 ROOT 里面 (防 目录 穿越)."""
    if not rel_path:
        return None
    root = os.path.abspath(ROOT)
    p = os.path.abspath(os.path.join(root, str(rel_path).lstrip("/")))
    if p != root and not p.startswith(root + os.sep):
        return None
    return p


def ensure_root():
    try:
        os.makedirs(ROOT, exist_ok=True)
    except Exception:
        pass
    return ROOT


def describe():
    """给 /v1/settings 或者 界面 用 的 描述."""
    return {"root": ROOT, "kinds": KIND_DIR,
            "exists": os.path.isdir(ROOT)}
