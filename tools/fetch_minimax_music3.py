#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 魔搭 拉 MiniMax Music3 的 MLX 权重 (mlx-audio 格式).

mlx-audio 0.6.4 已经 内置 minimax_music3 实现 —— 这里 只 负责 把 权重 落到 本地.
"""
import os, sys, time
from modelscope import snapshot_download

REPO = os.environ.get("MM3_REPO", "mlx-community/MiniMax-Music3-4bit")
DEST = os.environ.get("MM3_DIR", os.path.expanduser("~/.ours/models/MiniMax-Music3-4bit"))

def main():
    os.makedirs(DEST, exist_ok=True)
    for attempt in range(1, 6):
        try:
            print(f"[拉取] {REPO} -> {DEST}", flush=True)
            t0 = time.time()
            snapshot_download(REPO, local_dir=DEST)
            print(f"[完成] {time.time()-t0:.0f}s", flush=True)
            return 0
        except Exception as e:
            print(f"[重试 {attempt}] {e}", flush=True); time.sleep(10)
    print("[失败]", flush=True); return 1

if __name__ == "__main__":
    sys.exit(main())
