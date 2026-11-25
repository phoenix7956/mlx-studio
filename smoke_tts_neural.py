#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第九轮 冒烟 —— 神经 TTS (Qwen3-TTS) + 闭源 契约 对拍 修复.

用法: python3 smoke_tts_neural.py [base_url]

验 的 是 真 效果, 不 是 形状:
  1. GET /v1/audio/voices 报 出 神经 TTS 段 (Qwen3-TTS Base, 可 克隆)
  2. POST /v1/audio/speech (model=qwen3-tts-base) -> 真 神经 合成 (24kHz)
  3. 声纹克隆: 给 ref_audio -> 基频 明显 比 "不克隆" 更 贴近 参考
  4. 闭源 契约 (对拍 得到 的):
       未知 端点 -> 404 {"type":"not_found"}
       /v1/responses/compact 没 model -> 422 "model is required"
       /v1/responses/compact 没 input -> 400 "'input' is a required field"
       /api/show 没 model -> 400 {"error":"model is required"}
       /api/show 未知 model -> 404 {"error":"model not found"}
       /props 有 batching 键
       非法 voice 不 报错 (回落 默认)
       不带 model 的 chat 用 默认 模型
"""
import base64
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import wave

BASE = "http://127.0.0.1:11251"
for a in sys.argv[1:]:
    if not a.startswith("--"):
        BASE = a

PASS, FAIL = [], []


def call(method, path, body=None, raw=False, timeout=900):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
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


def wav_of(js):
    return base64.b64decode(js["data"][0]["b64_json"])


def f0_of(wav):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import voice_convert as vc
    with wave.open(io.BytesIO(wav), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return vc._med_f0(a, sr), a, sr


print("== 1. 音色 清单: 报 出 神经 TTS 段 ==")
st, js = call("GET", "/v1/audio/voices")
neu = (js or {}).get("neural") or {}
check("GET /v1/audio/voices 有 neural 段", st == 200 and bool(neu),
      "%s %s" % (st, str(js)[:150]))
check("neural.backend = qwen3-tts", neu.get("backend") == "qwen3-tts", str(neu)[:150])
check("neural.cloning = true", neu.get("cloning") is True, str(neu)[:150])
check("neural.speakers = [] (Base 模型 确实 没 具名 音色)",
      neu.get("speakers") == [], str(neu.get("speakers"))[:120])

print("== 2. 神经 TTS: 真 合成 ==")
txt = "今天天气不错，我们去公园走走吧。"
body1 = {"model": "qwen3-tts-base", "input": txt, "lang": "zh"}
body1["temperature"] = 0        # 贪心 解码 -> 输出 可 复现 (对比 才 有 意义)
st1, js1 = call("POST", "/v1/audio/speech", body1)
ok = st1 == 200 and (js1 or {}).get("engine") == "qwen3-tts"
check("POST /v1/audio/speech (model=qwen3-tts-base) -> 200 + engine=qwen3-tts",
      ok, "%s %s" % (st1, str(js1)[:180]))
w_plain = b""
if st1 == 200:
    w_plain = wav_of(js1)
    with wave.open(io.BytesIO(w_plain), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
    check("神经 输出 是 24kHz 单声道 WAV (%d 样本)" % n,
          sr == 24000 and ch == 1 and n > 24000, "sr=%s ch=%s n=%s" % (sr, ch, n))
    check("神经 输出 有 真实 波形 (不是 静默)", len(set(w_plain[44:])) > 20,
          "bytes=%d" % len(w_plain))
check("神经 响应 回报 cloned=false", st1 == 200 and (js1 or {}).get("cloned") is False,
      str(js1)[:150])

print("== 3. 声纹克隆: ref_audio -> 基频 贴近 参考 ==")
ref = "/tmp/mx_smoke_ref.wav"
ref_line = "This is a reference voice for testing the clone feature."
# 用 男声 (Daniel ~120Hz): 和 模型 默认 说话人 (女声, ~240Hz) 基频 差 得 开,
# 克隆 有 没有 真的 学 到 参考 的 音色, 一 看 就知道.
say_args = ["say", "-o", ref, "--data-format=LEI16@22050", "-v", "Daniel", ref_line]
made = True
try:
    subprocess.run(say_args, check=True, capture_output=True, timeout=60)
except Exception as e:
    made = False
    print("      macOS say 失败: %s" % e)
if made:
    with open(ref, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode()
    body2 = {"model": "qwen3-tts-base", "input": txt, "lang": "zh"}
    body2["ref_audio"] = ref_b64
    body2["temperature"] = 0
    st2, js2 = call("POST", "/v1/audio/speech", body2)
    check("带 ref_audio -> 200", st2 == 200, "%s %s" % (st2, str(js2)[:180]))
    check("响应 回报 cloned=true", st2 == 200 and (js2 or {}).get("cloned") is True,
          str(js2)[:180])
    if st2 == 200 and w_plain:
        w_clone = wav_of(js2)
        with open(ref, "rb") as f:
            ref_raw = f.read()
        f_ref = f0_of(ref_raw)[0]
        f_cl = f0_of(w_clone)[0]
        f_pl = f0_of(w_plain)[0]
        d_cl = abs(f_cl - f_ref) / f_ref if f_ref else 9.9
        d_pl = abs(f_pl - f_ref) / f_ref if f_ref else 9.9
        print("      参考 F0 %.1f | 克隆 F0 %.1f (差 %.0f%%) | 不克隆 F0 %.1f (差 %.0f%%)"
              % (f_ref, f_cl, d_cl * 100, f_pl, d_pl * 100))
        check("克隆 后 基频 比 不克隆 更 贴近 参考", d_cl < d_pl,
              "clone %.0f%% vs plain %.0f%%" % (d_cl * 100, d_pl * 100))
        check("克隆 波形 与 不克隆 波形 不同", w_clone != w_plain,
              "%d vs %d bytes" % (len(w_clone), len(w_plain)))

print("== 4. 闭源 契约 (对拍 修复 项) ==")
st, js = call("POST", "/v1/nope", {})
e = (js or {}).get("error") or {}
check("未知 端点 -> 404 + type=not_found + 'Unknown endpoint'",
      st == 404 and e.get("type") == "not_found" and e.get("message") == "Unknown endpoint",
      "%s %s" % (st, str(js)[:150]))
st, js = call("GET", "/v1/nope")
e = (js or {}).get("error") or {}
check("GET 未知 端点 -> 404 + type=not_found",
      st == 404 and e.get("type") == "not_found", "%s %s" % (st, str(js)[:150]))
st, js = call("POST", "/v1/responses/compact", {})
e = (js or {}).get("error") or {}
check("compact 没 model -> 422 'model is required' (type=invalid_request_error)",
      st == 422 and e.get("message") == "model is required"
      and e.get("type") == "invalid_request_error", "%s %s" % (st, str(js)[:150]))
st, js = call("POST", "/v1/responses/compact", {"model": "qwen3-4b"})
e = (js or {}).get("error") or {}
check("compact 没 input -> 400 'input is a required field'",
      st == 400 and e.get("message") == "'input' is a required field",
      "%s %s" % (st, str(js)[:150]))
st, js = call("POST", "/v1/responses/compact", {"model": "qwen3-4b", "input": "hello world"})
out0 = ((js or {}).get("output") or [{}])[0]
ok = (st == 200 and (js or {}).get("object") == "response.compaction"
      and out0.get("type") == "compaction" and out0.get("encrypted_content"))
check("compact 成功 -> response.compaction + encrypted_content", ok,
      "%s %s" % (st, str(js)[:180]))
st, js = call("POST", "/api/show", {})
check("/api/show 没 model -> 400 model is required",
      st == 400 and (js or {}).get("error") == "model is required",
      "%s %s" % (st, str(js)[:150]))
st, js = call("POST", "/api/show", {"model": "nope-xyz"})
check("/api/show 未知 model -> 404 model not found",
      st == 404 and (js or {}).get("error") == "model not found",
      "%s %s" % (st, str(js)[:150]))
st, js = call("GET", "/props")
bat = (js or {}).get("batching")
check("/props 有 batching 键", st == 200 and isinstance(bat, dict)
      and "max_group" in bat, "%s %s" % (st, str(js)[:150]))
body3 = {"model": "ours/tts", "input": "hello", "voice": "zzz_no_such_voice"}
st, js = call("POST", "/v1/audio/speech", body3)
check("非法 voice 不 报错 (回落 默认, 200)",
      st == 200 and (js or {}).get("data"), "%s %s" % (st, str(js)[:150]))
body4 = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
st, js = call("POST", "/v1/chat/completions", body4)
check("chat 不带 model -> 200 (用 默认 模型)",
      st == 200 and (js or {}).get("model"), "%s %s" % (st, str(js)[:150]))

print("")
print("== 结果: %d 通过 / %d 失败 ==" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败 项:")
    for f in FAIL:
        print("  - " + f)
sys.exit(1 if FAIL else 0)

