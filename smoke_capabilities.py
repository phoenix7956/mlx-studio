#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真能力 冒烟 —— 验 第七轮 新 做 出来 的 能力 (不是 形状, 是 效果).

用法: python3 smoke_capabilities.py [base_url]
  1. 图像 edit = **真 SDEdit** (VAE 编码 -> 加噪 -> 重采样): 结构 保留
  2. 图像 guidance_scale = **真 CFG**: 1.0 与 8.0 出 明显 不同 的 图
  3. 图像 transparent = **真 RGBA**: 有 alpha 通道, 且 背景 半透明
  4. 音色 目录 = **真 清单**: GET /v1/audio/voices 有 14 精选 + 系统声音
  5. 不同 音色 id = **不同 声音** (基频/波形 都 不同)
  6. ref_audio = **真 音色迁移**: 基频 对齐 参考, 共振峰 落到 参考 位置
  7. 视频: LTX 专属 字段 / turbo (缺 LoRA) = 明确 400 指名
"""
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
import wave

BASE = "http://127.0.0.1:11251"
argv = sys.argv[1:]
for a in argv:
    if not a.startswith("--"):
        BASE = a

PASS, FAIL = [], []


def call(method, path, body=None, raw=False):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            b = r.read()
            if raw:
                return r.status, b
            try:
                return r.status, json.loads(b)
            except Exception:
                return r.status, b.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        b = e.read()
        if raw:
            return e.code, b
        try:
            return e.code, json.loads(b)
        except Exception:
            return e.code, b.decode("utf-8", "replace")
    except Exception as e:
        return 0, str(e)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           "" if cond else "  <- " + str(detail)[:220]))


def png_bytes(b64):
    return base64.b64decode(b64)


def pil_img(b64):
    from PIL import Image
    return Image.open(io.BytesIO(png_bytes(b64)))


def wav_of(js):
    return base64.b64decode(js["data"][0]["b64_json"])


def f0_of(wav):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import voice_convert as vc
    with wave.open(io.BytesIO(wav), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return vc._med_f0(a, sr), a, sr


print("== 1. 图像: 真 SDEdit (结构 保留) ==")
st, js = call("POST", "/v1/images/generations",
              {"model": "krea-turbo", "prompt": "a red apple on a wooden table",
               "size": "512x512", "seed": 7})
src_b64 = (js.get("data") or [{}])[0].get("b64_json") if st == 200 else None
check("出图 (基准图)", bool(src_b64), "%s %s" % (st, str(js)[:120]))
if src_b64:
    import numpy as np
    src = pil_img(src_b64).convert("RGB")
    a = np.asarray(src, dtype=np.float32)
    st2, js2 = call("POST", "/v1/images/generations",
                    {"model": "krea-turbo", "prompt": "make it warmer and more vivid",
                     "size": "512x512", "seed": 7, "mode": "edit",
                     "image": png_bytes(src_b64).hex()})
    ed_b64 = (js2.get("data") or [{}])[0].get("b64_json") if st2 == 200 else None
    ok = False
    if ed_b64:
        b = np.asarray(pil_img(ed_b64).convert("RGB"), dtype=np.float32)
        diff = float(np.mean(np.abs(a - b)))
        lo_a = a[::8, ::8].mean(axis=2)
        lo_b = b[::8, ::8].mean(axis=2)
        lo_diff = float(np.mean(np.abs(lo_a - lo_b)))
        ok = 0 < diff < 120 and lo_diff < diff
        print("      全图差 %.1f / 低频差 %.1f" % (diff, lo_diff))
    check("mode=edit 是 真 SDEdit (低频 结构 保留)", ok, "%s %s" % (st2, str(js2)[:150]))

print("== 2. 图像: 真 CFG ==")
st3, js3 = call("POST", "/v1/images/generations",
                 {"model": "krea-turbo", "prompt": "a red apple on a wooden table",
                  "size": "512x512", "seed": 11, "guidance_scale": 1.0})
st4, js4 = call("POST", "/v1/images/generations",
                 {"model": "krea-turbo", "prompt": "a red apple on a wooden table",
                  "size": "512x512", "seed": 11, "guidance_scale": 8.0})
b3 = (js3.get("data") or [{}])[0].get("b64_json") if st3 == 200 else None
b4 = (js4.get("data") or [{}])[0].get("b64_json") if st4 == 200 else None
ok = False
if b3 and b4:
    import numpy as np
    x = np.asarray(pil_img(b3).convert("RGB"), dtype=np.float32)
    y = np.asarray(pil_img(b4).convert("RGB"), dtype=np.float32)
    d = float(np.mean(np.abs(x - y)))
    ok = d > 6.0
    print("      CFG 1.0 vs 8.0 平均差 %.1f" % d)
check("guidance_scale 真 进 采样 (1.0 vs 8.0 出 不同 图)", ok,
      "%s %s / %s %s" % (st3, b3 is None, st4, b4 is None))

print("== 3. 图像: 真 transparent (RGBA) ==")
st5, js5 = call("POST", "/v1/images/generations",
                 {"model": "krea-turbo", "prompt": "a single red apple, plain white background",
                  "size": "512x512", "seed": 3, "transparent": True})
b5 = (js5.get("data") or [{}])[0].get("b64_json") if st5 == 200 else None
ok = False
if b5:
    im = pil_img(b5)
    if im.mode == "RGBA":
        import numpy as np
        al = np.asarray(im)[:, :, 3]
        ok = al.min() < 250 and (al < 250).mean() > 0.001
        print("      模式 RGBA, alpha min %d max %d, 半透明 占比 %.1f%%"
              % (al.min(), al.max(), 100.0 * (al < 250).mean()))
    else:
        print("      模式 =", im.mode)
check("transparent 出 的 是 真 RGBA 且 有 alpha", ok, "%s %s" % (st5, str(js5)[:120]))

print("== 4. 音色 目录 (真 清单) ==")
st6, js6 = call("GET", "/v1/audio/voices")
vs = (js6 or {}).get("voices") or []
sysv = (js6 or {}).get("system") or []
ids = [v["id"] for v in vs]
check("GET /v1/audio/voices 有 精选 音色 清单", st6 == 200 and len(vs) >= 14, str(st6))
check("音色 id 覆盖 中/英 与 男女", len(ids) >= 14 and
      any(i.startswith("zf_") for i in ids) and any(i.startswith("am_") for i in ids),
      str(ids[:20]))
check("系统声音 清单 也 给 (macOS say)", len(sysv) >= 50, str(len(sysv)))

print("== 5. 不同 音色 id = 不同 声音 ==")
en_txt = "Hello there, this is a voice test."
st7, js7 = call("POST", "/v1/audio/speech", {"input": en_txt, "voice": "af_heart"})
st8, js8 = call("POST", "/v1/audio/speech", {"input": en_txt, "voice": "am_michael"})
w7 = wav_of(js7) if st7 == 200 else b""
w8 = wav_of(js8) if st8 == 200 else b""
f7 = f0_of(w7)[0] if w7 else 0.0
f8 = f0_of(w8)[0] if w8 else 0.0
check("af_heart / am_michael 出 不同 波形", bool(w7) and bool(w8) and w7 != w8)
check("不同 音色 的 基频 明显 不同", f7 > 0 and f8 > 0 and abs(f7 - f8) > 20,
      "af=%.1f am=%.1f" % (f7, f8))
st7b, js7b = call("POST", "/v1/audio/speech", {"input": "你好，这是音色测试。", "voice": "af_heart"})
w7b = wav_of(js7b) if st7b == 200 else b""
check("中文 文本 不会 静默 (英文 音色 自动 换 中文 声音)", len(w7b) > 20000,
      "bytes=%d" % len(w7b))

print("== 6. ref_audio = 真 音色迁移 ==")
txt = "Hello there, this is a voice conversion test."
st9, js9 = call("POST", "/v1/audio/speech", {"input": txt, "voice": "af_heart"})
st10, js10 = call("POST", "/v1/audio/speech", {"input": txt, "voice": "am_michael"})
ref_b64 = base64.b64encode(wav_of(js10)).decode() if st10 == 200 else ""
st11, js11 = call("POST", "/v1/audio/speech", {"input": txt, "voice": "af_heart",
                                               "ref_audio": ref_b64})
ok = False
if st11 == 200:
    f_ref = f0_of(wav_of(js10))[0]
    f_out = f0_of(wav_of(js11))[0]
    print("      参考基频 %.1f -> 输出基频 %.1f" % (f_ref, f_out))
    ok = f_ref > 0 and f_out > 0 and abs(f_out - f_ref) / f_ref < 0.15
check("ref_audio 后 基频 跟 参考 对齐 (±15%)", ok, "%s %s" % (st11, str(js11)[:120]))

print("== 7. 视频: 不支持 的 参数 明确 400 ==")
# turbo 那 条: 以前 是 "缺 LoRA -> 400 指名". 现在 turbo_lora.safetensors **已经 装 上 了**
# (Minimax-H3-Turbo-Lora, Apache-2.0), 所以 前提 变 了 —— 没 LoRA 才 应该 400.
# 有 LoRA 的 话, 应该 放 行 (真 去 出片), 不 该 再 报 "缺 LoRA".
_turbo_dir = os.path.expanduser("~/.ours/quant/h3-mlx-4bit")
_turbo_present = os.path.exists(os.path.join(_turbo_dir, "turbo_lora.safetensors"))
_bad_fields = [("cfg_scale", 7.0), ("stg_scale", 0.5), ("pipeline", "two_stage")]
if not _turbo_present:
    _bad_fields.append(("turbo", True))
for field, val in _bad_fields:
    stx, jsx = call("POST", "/v1/video/generations", {"prompt": "x", field: val})
    param = jsx.get("error", {}).get("param") if isinstance(jsx, dict) else None
    ok = stx == 400 and param == field
    check("video %s -> 400 且 指名" % field, ok, "%s %s" % (stx, str(jsx)[:140]))
if _turbo_present:
    print("      (turbo LoRA 已 装 —— 跳过 '缺 LoRA -> 400' 那 条, 前提 已 变)")

# 图像 引擎 和 视频 引擎 **互斥**: 视频 模型 常驻 时 装 图像 模型 会 被 拒
# ("视频 模型 还 在 用 ... 占着内存"). 下面 几 节 全 是 图像/音频, 先 把 视频 卸 了.
print("      卸 视频 模型 (图像/音频 引擎 和 它 互斥) ...")
for _m in ("h3-4bit", "h3-6bit", "h3-8bit", "wan2-fast"):
    call("POST", "/v1/unload-model", {"model": _m})

print("== 8. 参考图 真 进 潜变量 (cond_weights / cond_gain 真 生效) ==")


def solid_b64(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gen_img(body):
    st, js = call("POST", "/v1/images/generations", body)
    b = (js.get("data") or [{}])[0].get("b64_json") if st == 200 else None
    return st, b


def mean_diff(a_b64, b_b64):
    import numpy as np
    a = np.asarray(pil_img(a_b64).convert("RGB"), dtype=np.float32)
    b = np.asarray(pil_img(b_b64).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return 999.0
    return float(np.mean(np.abs(a - b)))


REF_RED = solid_b64((220, 30, 30))
REF_BLUE = solid_b64((30, 30, 220))
BASE_IMG = {"model": "krea-turbo", "prompt": "a photo of a plain studio wall",
            "size": "256x256", "steps": 3, "seed": 99}

# (a) 不同 参考图 -> 输出 必须 不同 (bug 时 是 完全 相同)
_, b_red = gen_img(dict(BASE_IMG, ref_images=[REF_RED]))
_, b_blue = gen_img(dict(BASE_IMG, ref_images=[REF_BLUE]))
d_ref = mean_diff(b_red, b_blue) if (b_red and b_blue) else 0.0
print("      红参考 vs 蓝参考 平均差 %.1f" % d_ref)
check("ref_images 真 进 潜变量 (不同 参考图 出 不同 图)",
      bool(b_red) and bool(b_blue) and d_ref > 5.0, "diff=%.2f" % d_ref)

# (b) cond_weights 改 相对 权重 -> 输出 必须 不同
_, b_w1 = gen_img(dict(BASE_IMG, ref_images=[REF_RED, REF_BLUE],
                       cond_weights=[1, 0]))
_, b_w2 = gen_img(dict(BASE_IMG, ref_images=[REF_RED, REF_BLUE],
                       cond_weights=[0, 1]))
d_w = mean_diff(b_w1, b_w2) if (b_w1 and b_w2) else 0.0
print("      权重 [1,0] vs [0,1] 平均差 %.1f" % d_w)
check("cond_weights 真 生效 (相对 权重 改 变 结果)",
      bool(b_w1) and bool(b_w2) and d_w > 5.0, "diff=%.2f" % d_w)

# (c) cond_gain 改 参考 特征 强度 -> 输出 必须 不同
_, b_g1 = gen_img(dict(BASE_IMG, ref_images=[REF_RED], cond_gain=0.2))
_, b_g2 = gen_img(dict(BASE_IMG, ref_images=[REF_RED], cond_gain=2.0))
d_g = mean_diff(b_g1, b_g2) if (b_g1 and b_g2) else 0.0
print("      gain 0.2 vs 2.0 平均差 %.1f" % d_g)
check("cond_gain 真 生效 (参考 特征 强度 改 变 结果)",
      bool(b_g1) and bool(b_g2) and d_g > 2.0, "diff=%.2f" % d_g)

print("== 9. 变体 mode=variation (Reinhard Lab 色彩迁移) ==")
stv, jsv = call("POST", "/v1/images/generations",
                 {"model": "krea-turbo", "prompt": "a photo of a plain studio wall",
                  "size": "256x256", "seed": 21, "mode": "variation",
                  "strength": 0.6, "image": REF_RED})
b_var = (jsv.get("data") or [{}])[0].get("b64_json") if stv == 200 else None
applied = jsv.get("applied") or {} if isinstance(jsv, dict) else {}
check("mode=variation 出 图 (不 500)", bool(b_var),
      "%s %s" % (stv, str(jsv)[:180]))
check("mode=variation 真 生效 (applied.mode=variation)",
      applied.get("mode") == "variation", str(applied)[:180])

print()
print("== 结果: %d 通过 / %d 失败 ==" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败 项:")
    for f in FAIL:
        print("  -", f)
sys.exit(1 if FAIL else 0)
