#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉 **FastWan2.2-TI2V-5B** (DMD 蒸馏版) —— 我们 的 快 出片 引擎 用.

为什么 是 它, 不 是 Wan2.2 原版:
   原版 (40 步) 我们 实测 720P/121帧 = **33 分钟** (去噪 30 分 + VAE 4.5 分) ——
   比 H3 还 慢! 因为 mlx-video 用 的 是 全注意力 (seq 27,280), 而 我们 的 H3 port
   有 自研 稀疏注意力 Metal kernel.
   FastWan 是 同 一个 DiT 的 DMD 蒸馏版: **3 步 + 不用 CFG** -> 实测 **3.4 分钟**,
   快 10 倍, 画质 依然 很 能打 (我们 实跑 过: 雨夜霓虹 街景 又 稳 又 电影感).

拉 的 源:
   hf-mirror  lBroth/FastWan2.2-TI2V-5B-MLX   (Apache-2.0, 已经 是 MLX 格式)
       model.safetensors      10.0G  (DMD 蒸馏 DiT, bf16)
       t5_encoder.safetensors 11.4G  (UMT5-XXL)
       vae.safetensors        2.8G   (Wan2.2 z_dim=48 VAE)
   魔搭      Wan-AI/Wan2.2-TI2V-5B-BF16/google/umt5-xxl   (tokenizer, 17M)

为什么 走 hf-mirror 而 不 走 魔搭:
   魔搭 上 的 FastWan 是 **diffusers 格式** (键名 跟 mlx-video 认 的 差 一 大截),
   转换 得 自己 写 键名 映射 —— 错 一个 键 就 静默 出 垃圾 画面, 风险 高.
   lBroth 这份 已经 是 现成 的 MLX 格式, DiT/T5/VAE 三 件套 同源 同版本,
   不用 装 torch, 不用 转. (我们 的 H3 底模 也是 走 hf-mirror 的.)
"""
import os, sys, json, shutil, time, argparse, subprocess

ROOT = os.path.expanduser("~/.ours/models")
OUT_DIR = os.environ.get("WAN2_FAST_DIR", os.path.join(ROOT, "wan2-fastwan-q8"))
MLX_REPO = "lBroth/FastWan2.2-TI2V-5B-MLX"
MS_REPO = "Wan-AI/Wan2.2-TI2V-5B-BF16"
TOK_FILES = ["tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "spiece.model"]


def say(*a):
    print("[wan2fast]", *a, flush=True)


def pull_mlx():
    """拉 MLX 权重 (hf-mirror)."""
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in ("model.safetensors", "t5_encoder.safetensors",
              "vae.safetensors", "config.json"):
        dst = os.path.join(OUT_DIR, f)
        if os.path.isfile(dst):
            say("已有", f, "-- 跳过")
            continue
        say("拉取", f, "...")
        t0 = time.time()
        for attempt in range(1, 6):
            try:
                p = hf_hub_download(MLX_REPO, f, local_dir=OUT_DIR)
                gb = os.path.getsize(p) / 1e9
                say("  完成 %.0fs (%.1f GB)" % (time.time() - t0, gb))
                break
            except Exception as e:
                say("  重试 %d: %s" % (attempt, e))
                time.sleep(10)
        else:
            raise SystemExit("拉取失败: " + f)


def pull_tokenizer():
    """tokenizer 走 魔搭 (小 文件, 顺带 让 离线 安装 也能 用)."""
    try:
        from modelscope import snapshot_download
        tmp = os.path.join(ROOT, ".wan2_tok")
        snapshot_download(MS_REPO, allow_file_pattern=["google/umt5-xxl/*"],
                          local_dir=tmp)
        src = os.path.join(tmp, "google", "umt5-xxl")
        for f in TOK_FILES:
            s = os.path.join(src, f)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(OUT_DIR, f))
        shutil.rmtree(tmp, ignore_errors=True)
        say("tokenizer 到位 (魔搭)")
    except Exception as e:
        say("魔搭 tokenizer 拉取 失败 (%s) -- 试 hf-mirror" % e)
        from huggingface_hub import hf_hub_download
        for f in TOK_FILES:
            try:
                hf_hub_download(MLX_REPO, f, local_dir=OUT_DIR)
                say("  +", f)
            except Exception:
                pass


def quantize(bits=8, group_size=64):
    """DiT 量化 成 q8 —— 内存/带宽 都 小 一半."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    vendor = os.path.join(root, "vendor_src")
    if not os.path.isdir(os.path.join(vendor, "mlx_video")):
        raise SystemExit("找不到 vendor 引擎: %s" % vendor)
    cfg_path = os.path.join(OUT_DIR, "config.json")
    orig_cfg = json.load(open(cfg_path))
    bf16_dir = os.path.join(ROOT, "wan2-fastwan-bf16")
    q_dir = OUT_DIR + ".q%d" % bits
    env = dict(os.environ)
    env["PYTHONPATH"] = vendor + os.pathsep + env.get("PYTHONPATH", "")
    if not os.path.isdir(bf16_dir):
        os.makedirs(bf16_dir, exist_ok=True)
        for f in ("model.safetensors", "config.json"):
            shutil.copy2(os.path.join(OUT_DIR, f), os.path.join(bf16_dir, f))
    say("量化 DiT -> q%d (group %d)" % (bits, group_size))
    cmd = [sys.executable, "-m", "mlx_video.models.wan_2.convert",
           "--checkpoint-dir", bf16_dir, "--output-dir", q_dir,
           "--quantize-only", "--bits", str(bits), "--group-size", str(group_size)]
    subprocess.run(cmd, check=True, env=env)
    # convert 会 重写 config.json -- 把 fastwan_dmd 配方 补 回去 (DMD 采样器 靠 它)
    q_cfg_path = os.path.join(q_dir, "config.json")
    q_cfg = json.load(open(q_cfg_path))
    for k in ("fastwan_dmd",):
        if k in orig_cfg:
            q_cfg[k] = orig_cfg[k]
    json.dump(q_cfg, open(q_cfg_path, "w"), indent=2)
    for f in ("t5_encoder.safetensors", "vae.safetensors", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        s = os.path.join(OUT_DIR, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(q_dir, f))
    shutil.rmtree(bf16_dir, ignore_errors=True)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.rename(q_dir, OUT_DIR)
    say("量化 完成 ->", OUT_DIR)


def verify():
    need = ["model.safetensors", "t5_encoder.safetensors",
            "vae.safetensors", "config.json", "tokenizer.json",
            "tokenizer_config.json"]
    missing = [f for f in need if not os.path.isfile(os.path.join(OUT_DIR, f))]
    if missing:
        raise SystemExit("权重 不 全, 缺: %s" % missing)
    cfg = json.load(open(os.path.join(OUT_DIR, "config.json")))
    if "fastwan_dmd" not in cfg:
        raise SystemExit("config.json 里 没有 fastwan_dmd -- DMD 采样器 会 不 生效")
    tot = 0
    for f in os.listdir(OUT_DIR):
        p = os.path.join(OUT_DIR, f)
        if os.path.isfile(p):
            tot += os.path.getsize(p)
    say("权重 完整, 共 %.1f GB -> %s" % (tot / 1e9, OUT_DIR))
    say("DMD 配方:", cfg["fastwan_dmd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128])
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-quant", action="store_true")
    a = ap.parse_args()
    if not a.skip_pull:
        pull_mlx()
        pull_tokenizer()
    if not a.skip_quant:
        quantize(a.bits, a.group_size)
    verify()


if __name__ == "__main__":
    main()
