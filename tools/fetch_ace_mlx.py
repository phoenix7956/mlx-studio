#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 魔搭 拉 ACE-Step 1.5 (MLX 自研引擎 用 的 权重).

不 依赖 huggingface —— 走 modelscope (魔搭) 网关.
拉 两 个 源:
  ACE-Step/Ace-Step1.5            -> acestep-v15-turbo/ + vae/ + Qwen3-Embedding-0.6B/
  ACE-Step/acestep-5Hz-lm-0.6B     -> 5Hz 语言模型 (编曲 planner)
"""
import os, sys, time

DEST = os.environ.get("ACE_MLX_DIR", os.path.expanduser("~/.ours/models/ace-step-mlx"))
os.makedirs(DEST, exist_ok=True)

from modelscope import snapshot_download

JOBS = [
    ("ACE-Step/Ace-Step1.5", ["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*"], ""),
    ("ACE-Step/acestep-5Hz-lm-0.6B", None, "acestep-5Hz-lm-0.6B"),
]

def main():
    for repo, pats, subdir in JOBS:
        tgt = os.path.join(DEST, subdir) if subdir else DEST
        os.makedirs(tgt, exist_ok=True)
        print(f"[拉取] {repo} -> {tgt}", flush=True)
        t0 = time.time()
        for attempt in range(1, 6):
            try:
                snapshot_download(repo, allow_file_pattern=pats, local_dir=tgt)
                print(f"[完成] {repo}  {time.time()-t0:.0f}s", flush=True)
                break
            except Exception as e:
                print(f"[重试 {attempt}] {repo}: {e}", flush=True)
                time.sleep(10)
        else:
            print(f"[失败] {repo}", flush=True); sys.exit(1)
    # 记 标记 (acestep-mlx 的 models.py 用 .acestep-mlx/*.ok 判断)
    mk = os.path.join(DEST, ".acestep-mlx"); os.makedirs(mk, exist_ok=True)
    import json
    for name, rev in (("main", "19671f406d603126926c1b7e2adc169acbcade22"),
                      ("acestep-5Hz-lm-0.6B", "148d8ea0225bdab342ee1ae3a354275ccd60ca80")):
        with open(os.path.join(mk, f"{name}-{rev}.ok"), "w") as f:
            json.dump({"repo": name, "revision": rev, "via": "modelscope"}, f)
    print("[全部完成] ", DEST, flush=True)

if __name__ == "__main__":
    main()
