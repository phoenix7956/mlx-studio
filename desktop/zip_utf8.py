#!/usr/bin/env python3
"""打包 zip —— **中文 文件名 必须是 UTF-8** (不然 别的 机器 解压 全 是 乱码).

macOS 自带 的 `zip` 会 把 中文名 写成 "UTF-8 字节 + Unicode extra field",
一部分 解压 工具 (win 自带 的、python 的 zipfile) 不 认 那个 extra field,
于是 中文名 变 成 乱码 (内核代码 → σåàµá╕Σ╗úτáü).
python 的 zipfile 默认 会 给 非 ASCII 名 打上 UTF-8 标志 (bit 11), 谁 都 认.

用法: python3 zip_utf8.py <要 打包 的 目录> <输出 的 zip>
"""
import os
import sys
import zipfile


def main():
    if len(sys.argv) != 3:
        print("用法: python3 zip_utf8.py <目录> <输出 zip>")
        return 2
    src = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    if not os.path.isdir(src):
        print("目录 不 存在: %s" % src)
        return 1
    if os.path.exists(out):
        os.remove(out)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(src):
            dirs.sort()
            rel = os.path.relpath(root, src)
            base = "" if rel == "." else rel + "/"
            # 空 目录 也 要 保留 (解出来 才 和 原目录 一 样)
            if base and not dirs and not files:
                z.writestr(zipfile.ZipInfo(base), b"")
            for f in sorted(files):
                p = os.path.join(root, f)
                # 符号链接 存成 链接 本身, 不 展开 (不然 死链 会 变 成 一 大坨 文件)
                if os.path.islink(p):
                    zi = zipfile.ZipInfo(base + f)
                    zi.create_system = 3
                    zi.external_attr = (0o120777 << 16)
                    z.writestr(zi, os.readlink(p).encode("utf-8"))
                    n += 1
                    continue
                z.write(p, base + f)
                n += 1
    print("打包 %d 个 文件 -> %s (%.1f MB)"
          % (n, out, os.path.getsize(out) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
