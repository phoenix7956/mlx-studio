#!/usr/bin/env python3
"""校验 model_catalog.json —— 模型 页 的 数据 源.

每 条 至少 要 有: id / name / cap / license_spdx / license_free / local.
cap 只能 是 已知 的 那 几种 (不然 前端 会 渲染 不出来).
local 路径 不 允许 写死 某台 机器 (不许 带 /Users/<名字>).

用法: python3 tools/check_catalog.py
退出码 0 = 通过, 1 = 有 问题.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAT = os.path.join(ROOT, "model_catalog.json")

KNOWN_CAPS = {
    "image", "video", "music", "audio", "llm", "tts", "stt",
    "voice", "3d", "sound", "embeddings", "decisions", "gen",
}
REQUIRED = ("id", "name", "cap", "license_spdx", "license_free")

# 文档 里 允许 的 通用 例子
ALLOWED_HOSTS = ("127.0.0.1", "0.0.0.0", "192.168.1.5", "mac.local")


def main():
    if not os.path.exists(CAT):
        print("找不到 model_catalog.json")
        return 1
    try:
        data = json.load(open(CAT, encoding="utf-8"))
    except Exception as e:
        print("model_catalog.json 解析 失败: %s" % e)
        return 1

    items = data.get("catalog") or []
    if not items:
        print("catalog 是 空 的")
        return 1

    errs = []
    seen = set()
    for i, m in enumerate(items):
        tag = m.get("id") or ("#%d" % i)
        for k in REQUIRED:
            if k not in m:
                errs.append("[%s] 缺 字段 %s" % (tag, k))
        # 内置 引擎 (程序化 / 退 系统 自带) 没有 权重 —— 不 需要 local
        if not m.get("builtin") and "local" not in m:
            errs.append("[%s] 缺 字段 local" % tag)
        mid = m.get("id")
        if mid in seen:
            errs.append("[%s] id 重复" % tag)
        seen.add(mid)
        cap = m.get("cap")
        if cap and cap not in KNOWN_CAPS:
            errs.append("[%s] cap=%r 不在 已知 列表 里" % (tag, cap))
        lf = m.get("license_free")
        if not isinstance(lf, bool):
            errs.append("[%s] license_free 必须是 true/false" % tag)
        local = str(m.get("local") or "")
        if local and not local.startswith("~"):
            errs.append("[%s] local 必须 用 ~ 开头 (别 写死 绝对路径): %s" % (tag, local))
        if re.search(r"/Users/(?!<)", local):
            errs.append("[%s] local 里 出现 了 真实 用户名: %s" % (tag, local))
        # URL 里 的 地址 —— 只 允许 文档 里 明确 写过 的 通用 例子
        for field in ("ms", "pull", "license_url"):
            v = str(m.get(field) or "")
            for host in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", v):
                if host not in ALLOWED_HOSTS:
                    errs.append("[%s] %s 里 出现 了 内网/私人 地址: %s" % (tag, field, host))

    if errs:
        print("model_catalog.json 有 %d 个 问题:" % len(errs))
        for e in errs:
            print("   " + e)
        return 1

    caps = {}
    for m in items:
        caps[m.get("cap")] = caps.get(m.get("cap"), 0) + 1
    dist = ", ".join("%sx%s" % (k, v) for k, v in sorted(caps.items(), key=lambda x: -x[1]))
    print("model_catalog.json OK —— %d 条, 能力 分布: %s" % (len(items), dist))
    return 0


if __name__ == "__main__":
    sys.exit(main())

