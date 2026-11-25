#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉 MiniMax-H3 权重 (出片/视频 用), 并 量化 出 4bit DiT.

两步:
  ① 从 hf-mirror 拉 "MiniMax/MiniMax-H3" (revision=master) 到
     ~/.ours/models/models/MiniMax--MiniMax-H3/snapshots/master/   (HF 缓存 布局)
  ② 跑 我们 的 量化 脚本 (厂商 源码 minimax-h3_mlx_src/scripts/build_quant.py)
     把 bf16 DiT 量化 成 4bit -> ~/.ours/quant/h3-mlx-4bit

用法:
    python tools/fetch_h3_mlx.py                # 下 + 量化 4bit
    python tools/fetch_h3_mlx.py --no-quant     # 只 下 权重
    python tools/fetch_h3_mlx.py --bits 4 6     # 量化 多 个 位宽
"""
import argparse
import glob
import os
import subprocess
import sys
import time

REPO = os.environ.get("H3_REPO", "MiniMax/MiniMax-H3")
REV = os.environ.get("H3_REV", "master")
MODELS = os.path.expanduser(os.environ.get("H3_MODELS", "~/.ours/models/models"))
QUANT_OUT = os.path.expanduser(os.environ.get("H3_QUANT_OUT", "~/.ours/quant/h3-mlx-4bit"))


def home():
    try:
        import pwd
        h = pwd.getpwuid(os.getuid()).pw_dir
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return os.path.expanduser("~")


def find_vendor():
    """厂商 源码 (含 scripts/build_quant.py) 在 哪."""
    cands = [
        os.environ.get("H3M_PORT_DIR", ""),
        os.path.join(home(), ".ours/vendor/minimax-h3_mlx_src"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minimax-h3_mlx_src"),
        os.path.join(home(), "Desktop/rea/ours/_vendor/minimax-h3_mlx_src"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "scripts", "build_quant.py")):
            return c
    return None


def snapshot_dir():
    """HF 缓存 布局: <cache>/models/<org>--<name>/snapshots/<rev>"""
    name = REPO.replace("/", "--")
    return os.path.join(MODELS, name, "snapshots", REV)


def pick_python():
    """要 一个 装 了 huggingface_hub 的 python —— 没有 就 换 一个 重来."""
    if os.environ.get("_H3_REEXEC"):
        return None
    for c in [sys.executable] + sorted(glob.glob(os.path.join(home(), ".venvs", "*", "bin", "python"))):
        try:
            r = subprocess.run([c, "-c", "import modelscope"], capture_output=True, timeout=60)
            if r.returncode == 0:
                return c
        except Exception:
            pass
        try:
            r = subprocess.run([c, "-c", "import huggingface_hub"], capture_output=True, timeout=60)
            if r.returncode == 0:
                return c
        except Exception:
            pass
    return None


def download():
    py = pick_hf_python()
    if not py:
        print("[失败] 没 找到 带 huggingface_hub 的 python —— "
              "先 装: pip install huggingface_hub", file=sys.stderr)
        return False
    if os.path.abspath(py) != os.path.abspath(sys.executable):
        env = dict(os.environ)
        env["_H3_REEXEC"] = "1"
        print(f"[换 解释器] {py}", flush=True)
        return subprocess.call([py, os.path.abspath(__file__)] + sys.argv[1:], env=env) == 0

    # (1) 魔搭 —— 国内 最快
    try:
        from modelscope import snapshot_download as ms_download
    except Exception:
        ms_download = None
    if ms_download is not None:
        tgt = snapshot_dir()
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        print(f"[拉取] 魔搭 {REPO} @{REV}  -> {tgt}", flush=True)
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                ms_download(REPO, revision=REV, local_dir=tgt,
                            allow_file_pattern=["FL2VA/*"])
                print(f"[完成] {time.time() - t0:.0f}s", flush=True)
                return True
            except Exception as e:
                print(f"[重试 {attempt}] {e}", flush=True)
                time.sleep(10)

    from huggingface_hub import snapshot_download

    ep = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[拉取] {REPO} @{REV}  (镜像 {ep})", flush=True)
    t0 = time.time()
    for attempt in range(1, 6):
        try:
            snapshot_download(
                REPO,
                revision=REV,
                cache_dir=MODELS,
                endpoint=ep,
                max_workers=4,
            )
            print(f"[完成] {time.time() - t0:.0f}s", flush=True)
            return True
        except Exception as e:
            print(f"[重试 {attempt}] {e}", flush=True)
            time.sleep(10)
    print("[失败] 拉 不 到", REPO, file=sys.stderr)
    return False


def quantize(bits):
    vendor = find_vendor()
    if not vendor:
        print("[失败] 没 找到 厂商 源码 (minimax-h3_mlx_src) —— 不 能 量化",
              file=sys.stderr)
        return False
    ckpt = os.path.join(snapshot_dir(), "FL2VA")
    if not os.path.isdir(os.path.join(ckpt, "transformer")):
        print(f"[失败] 权重 不 全 —— 缺 {ckpt}/transformer (先 跑 下载 那 一步)",
              file=sys.stderr)
        return False
    cmd = [sys.executable, os.path.join(vendor, "scripts", "build_quant.py"),
           "--checkpoint", ckpt, "--out", QUANT_OUT, "--bits"] + [str(b) for b in bits]
    print("[量化] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4])
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    if not a.no_download:
        if not download():
            return 1
    if not a.no_quant:
        if not quantize(a.bits):
            return 1
    print("[全部完成]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
