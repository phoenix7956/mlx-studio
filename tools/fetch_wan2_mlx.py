#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 魔搭 拉 Wan2.2 TI2V-5B, 本地 转成 MLX 权重 (q8) —— 给 我们 的 出片 引擎 用.

为什么 自己 转, 不 直接 下 现成 的 MLX 权重:
  现成 的 MLX 权重 只在 HF 镜像 有 (Anes1032 / SceneWorks / MLXCreator 那 几 个).
  我们 定 的 规矩 是 权重 统一 走 魔搭 —— 所以 拉 官方 原始 权重, 用 上游
  mlx-video 的 convert 脚本 在 本地 转, 顺带 拷 tokenizer.

拉 的 源:
  Wan-AI/Wan2.2-TI2V-5B-BF16   (Apache-2.0, 原始 Wan 格式 —— convert 脚本 认 这个)
      diffusion_pytorch_model-*.safetensors  ~10G   (DiT bf16)
      models_t5_umt5-xxl-enc-bf16.pth          11.4G (UMT5-XXL 文本编码器)
      Wan2.2_VAE.pth                            2.8G  (z_dim=48 的 VAE, 带 编码器)
      google/umt5-xxl/                         tokenizer (本地 装 不 要 联网)

转 出 的 MLX 权重 (~19.6G):
      model.safetensors      ~5.4G  (q8 DiT)
      t5_encoder.safetensors ~11.4G (UMT5-XXL)
      vae.safetensors        ~2.8G  (float32)
      config.json + tokenizer.json
"""
import os, sys, shutil, time, subprocess, argparse

ROOT = os.path.expanduser("~/.ours/models")
SRC_DIR = os.path.join(ROOT, "wan2-ti2v-5b-src")
OUT_DIR = os.environ.get("WAN2_MLX_DIR", os.path.join(ROOT, "wan2-ti2v-5b"))

REPO = "Wan-AI/Wan2.2-TI2V-5B-BF16"
NEED = [
    "diffusion_pytorch_model*.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "config.json",
    "google/umt5-xxl/*",
]
TOK_FILES = ["tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "spiece.model"]


def say(*a):
    print("[wan2]", *a, flush=True)


def pull():
    from modelscope import snapshot_download
    os.makedirs(SRC_DIR, exist_ok=True)
    say("拉取", REPO, "->", SRC_DIR)
    t0 = time.time()
    for attempt in range(1, 6):
        try:
            snapshot_download(REPO, allow_file_pattern=NEED, local_dir=SRC_DIR)
            say("完成 %.0fs" % (time.time() - t0))
            return
        except Exception as e:
            say("重试 %d: %s" % (attempt, e))
            time.sleep(10)
    raise SystemExit("拉取失败: " + REPO)


def convert(bits=8, group_size=64):
    """用 上游 mlx-video 的 convert 脚本 转 MLX 权重."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    vendor = os.path.join(root, "vendor_src")
    if not os.path.isdir(os.path.join(vendor, "mlx_video")):
        raise SystemExit("找不到 vendor 引擎: %s" % vendor)
    out = os.path.join(ROOT, "wan2-ti2v-5b-bf16")
    say("转换 -> MLX (bf16)", out)
    cmd = [sys.executable, "-m", "mlx_video.models.wan_2.convert",
           "--checkpoint-dir", SRC_DIR, "--output-dir", out,
           "--dtype", "bfloat16", "--model-version", "2.2"]
    env = dict(os.environ)
    env["PYTHONPATH"] = vendor + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)
    # 量化 DiT -> q8 (省 内存 + 快)
    q = os.path.join(ROOT, "wan2-ti2v-5b-bf16-q%d" % bits)
    say("量化 DiT -> q%d" % bits, q)
    cmd = [sys.executable, "-m", "mlx_video.models.wan_2.convert",
           "--checkpoint-dir", out, "--output-dir", OUT_DIR,
           "--quantize-only", "--bits", str(bits), "--group-size", str(group_size)]
    subprocess.run(cmd, check=True, env=env)
    shutil.rmtree(out, ignore_errors=True)
    # tokenizer 拷 过去 —— 我们 的 port 会 从 权重 目录 读 它 (不联网)
    src_tok = os.path.join(SRC_DIR, "google", "umt5-xxl")
    for f in TOK_FILES:
        s = os.path.join(src_tok, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT_DIR, f))
    say("tokenizer 已就位:", OUT_DIR)


def verify():
    need = ["model.safetensors", "t5_encoder.safetensors",
            "vae.safetensors", "config.json", "tokenizer.json"]
    missing = [f for f in need if not os.path.isfile(os.path.join(OUT_DIR, f))]
    if missing:
        raise SystemExit("权重 不 全, 缺: %s" % missing)
    tot = 0
    for f in os.listdir(OUT_DIR):
        p = os.path.join(OUT_DIR, f)
        if os.path.isfile(p):
            tot += os.path.getsize(p)
    say("权重 完整, 共 %.1f GB -> %s" % (tot / 1e9, OUT_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128])
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--clean-src", action="store_true",
                    help="转 完 删 掉 原始 权重 (省 24G 盘)")
    a = ap.parse_args()
    if not a.skip_pull:
        pull()
    if not a.skip_convert:
        convert(a.bits, a.group_size)
    verify()
    if a.clean_src:
        shutil.rmtree(SRC_DIR, ignore_errors=True)
        say("原始 权重 已 清理:", SRC_DIR)


if __name__ == "__main__":
    main()
