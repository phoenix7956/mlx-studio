#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 闭源 MLX-Serve 留下 的 模型 目录 并 进 我们 自己 的 ~/.ours/models.

闭源 mlx-serve 删了 之后, ~/.mlx-serve/models 里 还 留着 它 下载 的 权重.
那些 权重 现在 照样 在 用 (我们 的 内核 认 这个 目录), 但 放 两 个 地方 太 乱 ——
所以 整个 搬 进 ~/.ours/models, 保持 原来 的 相对 结构:

    ~/.mlx-serve/models/mt/<org>/<repo>        -> ~/.ours/models/mt/<org>/<repo>
    ~/.mlx-serve/models/<org>/<repo>            -> ~/.ours/models/<org>/<repo>
    ~/.mlx-serve/models/README.md               -> ~/.ours/models/README.md

搬 完 会 把 指向 老 绝对路径 的 软链 重 指 到 新 位置.
同 名字 的 东西 两边 都 有 时: 保留 我们 这边 的, 跳过 闭源 那 份.
"""
import os
import shutil
import sys


def home():
    try:
        import pwd
        h = pwd.getpwuid(os.getuid()).pw_dir
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return os.path.expanduser("~")


H = home()
SRC = os.path.join(H, ".mlx-serve", "models")
DST = os.path.join(H, ".ours", "models")
DRY = "--dry" in sys.argv


def merge_dir(s_src, s_dst, report):
    """把 s_src 里 的 每 一 项 搬 进 s_dst (同 结构)."""
    os.makedirs(s_dst, exist_ok=True)
    for name in sorted(os.listdir(s_src)):
        src = os.path.join(s_src, name)
        dst = os.path.join(s_dst, name)
        if os.path.islink(src):
            # 软链: 目标 在 老 目录 里 的 -> 重 指 到 新 目录
            tgt = os.readlink(src)
            if tgt.startswith(SRC):
                new_tgt = tgt.replace(SRC, DST, 1)
                if not os.path.exists(dst):
                    os.symlink(new_tgt, dst)
                    report.append("软链 %s/%s -> %s" % (os.path.basename(s_dst), name,
                                                       os.path.relpath(new_tgt, DST)))
                else:
                    report.append("跳过(已存在) %s/%s" % (os.path.basename(s_dst), name))
                continue
            if not os.path.exists(dst):
                os.symlink(tgt, dst)
                report.append("软链 %s/%s -> %s" % (os.path.basename(s_dst), name, tgt))
            continue
        if os.path.isdir(src):
            if os.path.exists(dst):
                # 两边 同名 目录 -> 往 里 递归 合并
                merge_dir(src, dst, report)
                continue
            if DRY:
                report.append("[dry] 搬 目录 %s" % os.path.relpath(src, SRC))
            else:
                shutil.move(src, dst)
                report.append("搬 目录 %s" % os.path.relpath(src, SRC))
        else:
            if os.path.exists(dst):
                report.append("跳过(已存在) %s" % os.path.relpath(src, SRC))
                continue
            if DRY:
                report.append("[dry] 搬 文件 %s" % os.path.relpath(src, SRC))
            else:
                shutil.move(src, dst)
                report.append("搬 文件 %s" % os.path.relpath(src, SRC))


def fix_links(root, report):
    """搬 完 之后 把 断掉 的 软链 修 好 (目标 指向 老 绝对路径 / 已 不存在 的 目录)."""
    for dirpath, dirs, files in os.walk(root):
        for name in dirs + files:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                continue
            tgt = os.readlink(p)
            if os.path.exists(p):
                continue
            new = None
            if tgt.startswith(SRC):
                new = tgt.replace(SRC, DST, 1)
            elif tgt.startswith(os.path.join(H, ".lmstudio", "models")):
                # LM Studio 那 份 已经 不在了 -> 指 回 我们 自己 的 目录
                new = os.path.join(DST, os.path.relpath(tgt,
                           os.path.join(H, ".lmstudio", "models")))
            if new and os.path.exists(new):
                os.remove(p)
                os.symlink(new, p)
                report.append("修 软链 %s -> %s"
                              % (os.path.relpath(p, DST), os.path.relpath(new, DST)))
            elif new:
                report.append("软链 还 断 着 (目标 没有): %s" % os.path.relpath(p, DST))
            else:
                report.append("软链 还 断 着: %s -> %s"
                              % (os.path.relpath(p, DST), tgt))


def main():
    if not os.path.isdir(SRC):
        print("没有 %s —— 不用 合" % SRC)
        return 0
    print("源: %s" % SRC)
    print("到: %s" % DST)
    report = []
    merge_dir(SRC, DST, report)
    if not DRY:
        fix_links(DST, report)
    for line in report:
        print("  " + line)
    left = []
    for root, dirs, files in os.walk(SRC):
        for n in dirs + files:
            left.append(os.path.join(root, n))
    print("\n剩 在 老 目录 里 的: %d 项" % len(left))
    for p in left[:20]:
        print("  " + os.path.relpath(p, SRC))
    if DRY:
        print("\n(dry-run —— 没 真 动)")
    else:
        # 老 目录 空 了 就 收 掉 (只 收 models, 别 的 留给 clean.sh --deep)
        try:
            os.rmdir(SRC)
            print("\n✓ 老 的 models 目录 已 收 掉")
        except OSError as e:
            print("\n(老 目录 还有 东西, 留着: %s)" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
