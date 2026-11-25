"""smoke_contract.py — 闭源 契约 参数 的 回归 测试.

对照 闭源 Studio 内嵌 的 API Reference (从 bundle 里 抽 出 的 权威 文本):
  /v1/chat/completions   temperature / top_p / top_k / tool_choice / stream_options
  /v1/images/generations mode=edit|variation, image, ref_images[], strength,
                         guidance_scale, negative_prompt, cond_gain, cond_weights
  /v1/images/edits       multipart/form-data (image[] 重复) + maskless 约束
  /v1/audio/speech       voice 混合, speed, ref_audio, stream
  /v1/audio/music-*      instrumental, vocal_language, bpm, seed, steps, 时长 边界
  /v1/audio/sound-*      steps, seed, 时长 上限
  /v1/video/generations  width/height/num_frames/preview*/LTX+H3 参数
  /v1/library/export     归档 导出 / 导入
"""
import base64
import json
import sys
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:11251"
for a in sys.argv[1:]:
    if not a.startswith("--"):
        BASE = a
PASS, FAIL = [], []


def call(method, path, body=None, raw=False, data=None, ctype=None):
    url = BASE + path
    d = None
    headers = {}
    if body is not None:
        d = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if data is not None:
        d = data
        if ctype:
            headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=d, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = r.read()
            st = r.status
    except urllib.error.HTTPError as e:
        payload = e.read()
        st = e.code
    except Exception as e:
        return 0, str(e)
    if raw:
        return st, payload
    try:
        return st, json.loads(payload.decode("utf-8", "replace"))
    except Exception:
        return st, payload.decode("utf-8", "replace")


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    tail = ""
    if not ok and detail:
        tail = "  -> " + str(detail)[:220]
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, tail))


def err400(name, path, body, param=None):
    st, js = call("POST", path, body)
    ok = st == 400 and isinstance(js, dict)
    if ok:
        ok = js.get("error", {}).get("type") == "invalid_request"
    if ok and param:
        ok = js["error"].get("param") == param
    check(name, ok, "%s %s" % (st, str(js)[:200]))


PNG_1x1 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
           "AAAABJRU5ErkJggg==")


print("== 1. chat 采样 参数 ==")
st, js = call("POST", "/v1/chat/completions",
              {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}],
               "temperature": 9})
check("POST /v1/chat/completions temperature 越界 -> 400", st == 400, st)
st, js = call("POST", "/v1/chat/completions",
              {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}],
               "top_p": 1.7})
check("POST /v1/chat/completions top_p 越界 -> 400", st == 400, st)
st, js = call("POST", "/v1/chat/completions",
              {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}],
               "tool_choice": "nope"})
check("POST /v1/chat/completions tool_choice 非法 -> 400", st == 400, st)
st, js = call("POST", "/v1/chat/completions",
              {"model": "qwen3-4b", "messages": [{"role": "user", "content": "说一句话"}],
               "temperature": 0.2, "top_p": 0.8, "top_k": 40, "max_tokens": 32})
check("POST /v1/chat/completions 带 采样 参数 -> 200", st == 200, "%s %s" % (st, str(js)[:160]))

print("== 2. images/generations 契约 参数 ==")
err400("images/generations size 太小 -> 400", "/v1/images/generations",
       {"prompt": "x", "size": "64x64"}, "size")
err400("images/generations steps 越界 -> 400", "/v1/images/generations",
       {"prompt": "x", "steps": 99}, "steps")
err400("images/generations seed 负数 -> 400", "/v1/images/generations",
       {"prompt": "x", "seed": -1}, "seed")
err400("images/generations mode 无图 -> 400", "/v1/images/generations",
       {"prompt": "x", "mode": "edit"}, "mode")
err400("images/generations 参考图 超 4 张 -> 400", "/v1/images/generations",
       {"prompt": "x", "image": PNG_1x1, "ref_images": [PNG_1x1] * 4}, "ref_images")
err400("images/generations cond_weights 非数字 -> 400", "/v1/images/generations",
       {"prompt": "x", "cond_weights": ["a"]}, "cond_weights")
st, js = call("POST", "/v1/images/generations",
              {"prompt": "x", "mode": "variation", "image": PNG_1x1, "strength": 0.5,
               "guidance_scale": 7, "negative_prompt": "blurry", "cond_gain": 1.5})
check("images/generations 带 全套 参数 不 报 参数 错",
      st != 400 or "must be" not in str(js), "%s %s" % (st, str(js)[:160]))

print("== 3. images/edits (multipart + maskless) ==")
err400("images/edits 带 mask -> 400", "/v1/images/edits",
       {"prompt": "x", "image": PNG_1x1, "mask": "m"}, "mask")
err400("images/edits n>1 -> 400", "/v1/images/edits",
       {"prompt": "x", "image": PNG_1x1, "n": 2}, "n")
err400("images/edits response_format=url -> 400", "/v1/images/edits",
       {"prompt": "x", "image": PNG_1x1, "response_format": "url"}, "response_format")
err400("images/edits output_format!=png -> 400", "/v1/images/edits",
       {"prompt": "x", "image": PNG_1x1, "output_format": "jpeg"}, "output_format")
err400("images/edits stream=true -> 400", "/v1/images/edits",
       {"prompt": "x", "image": PNG_1x1, "stream": True}, "stream")


def multipart(fields, files):
    b = "----" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(("--%s\r\n" % b).encode())
        parts.append(('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode())
        parts.append(str(v).encode() + b"\r\n")
    for k, fn, ct, data in files:
        parts.append(("--%s\r\n" % b).encode())
        parts.append(('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                      % (k, fn)).encode())
        parts.append(("Content-Type: %s\r\n\r\n" % ct).encode())
        parts.append(data + b"\r\n")
    body = b"".join(parts) + ("--%s--\r\n" % b).encode()
    return body, "multipart/form-data; boundary=%s" % b


body, ct = multipart({"model": "krea-turbo", "prompt": "warmer and more vivid"},
                      [("image[]", "reference-0.png", "image/png", base64.b64decode(PNG_1x1))])
st, js = call("POST", "/v1/images/edits", data=body, ctype=ct)
check("images/edits multipart image[] -> 200 + b64",
      st == 200 and js.get("data") and js["data"][0].get("b64_json"),
      "%s %s" % (st, str(js)[:160]))
body, ct = multipart({"prompt": "x"},
                      [("image[]", "a.png", "image/png", base64.b64decode(PNG_1x1)),
                       ("image[]", "b.png", "image/png", base64.b64decode(PNG_1x1))])
st, js = call("POST", "/v1/images/edits", data=body, ctype=ct)
check("images/edits multipart 多张 image[] -> references=2",
      st == 200 and js.get("references") == 2, "%s %s" % (st, str(js)[:160]))

print("== 4. audio/speech ==")
err400("audio/speech speed 越界 -> 400", "/v1/audio/speech",
       {"input": "hi", "speed": 9}, "speed")
st, js = call("POST", "/v1/audio/speech",
              {"model": "ours/tts", "input": "混合 音色 测试", "voice": "af_heart,af_bella",
               "speed": 1.0})
check("audio/speech 音色 混合 -> 200 + WAV",
      st == 200 and js.get("data") and js["data"][0].get("b64_json"),
      "%s %s" % (st, str(js)[:160]))
check("audio/speech 回 报 voice 列表",
      st == 200 and isinstance(js.get("voice"), list) and len(js["voice"]) == 2,
      "%s %s" % (st, str(js)[:160]))

print("== 5. audio/music-generations ==")
err400("music 时长 < 10 -> 400", "/v1/audio/music-generations",
       {"prompt": "x", "duration_seconds": 5}, "duration_seconds")
err400("music 时长 > 600 -> 400", "/v1/audio/music-generations",
       {"prompt": "x", "duration_seconds": 900}, "duration_seconds")
err400("music bpm 越界 -> 400", "/v1/audio/music-generations",
       {"prompt": "x", "bpm": 400}, "bpm")
st, js = call("POST", "/v1/audio/music-generations",
              {"prompt": "lofi", "duration_seconds": 10, "instrumental": True,
               "bpm": 90, "vocal_language": "en", "seed": 42, "steps": 8})
check("music 带 全套 参数 -> 200 (不 报 参数 错)",
      st != 400 or "must be" not in str(js), "%s %s" % (st, str(js)[:160]))

print("== 6. audio/sound-generations ==")
err400("sound 时长 > 120 -> 400", "/v1/audio/sound-generations",
       {"prompt": "x", "duration_seconds": 500}, "duration_seconds")
err400("sound steps 越界 -> 400", "/v1/audio/sound-generations",
       {"prompt": "x", "steps": 99}, "steps")
st, js = call("POST", "/v1/audio/sound-generations",
              {"prompt": "rain", "duration_seconds": 2, "steps": 8, "seed": 7})
check("sound 带 steps/seed -> 200 + WAV",
      st == 200 and js.get("data") and js["data"][0].get("b64_json"),
      "%s %s" % (st, str(js)[:160]))

print("== 7. video/generations ==")
err400("video num_frames 越界 -> 400", "/v1/video/generations",
       {"prompt": "x", "num_frames": 2}, "num_frames")
err400("video width 越界 -> 400", "/v1/video/generations",
       {"prompt": "x", "width": 64}, "width")
err400("video 参考图 超 9 张 -> 400", "/v1/video/generations",
       {"prompt": "x", "ref_images": [PNG_1x1] * 10}, "ref_images")
err400("video pipeline 非法 -> 400", "/v1/video/generations",
       {"prompt": "x", "pipeline": "nope"}, "pipeline")
err400("video chain_prompts 超 chain_windows -> 400", "/v1/video/generations",
       {"prompt": "x", "chain_windows": 2, "chain_prompts": ["a", "b", "c"]},
       "chain_prompts")
err400("video cfg_scale 越界 -> 400", "/v1/video/generations",
       {"prompt": "x", "cfg_scale": 99}, "cfg_scale")

print("== 8. 资料库 归档 ==")
st, js = call("GET", "/v1/library")
check("GET /v1/library 有 items", st == 200 and isinstance(js.get("items"), list), st)
st, js = call("GET", "/v1/library/export")
check("GET /v1/library/export 有 items + version",
      st == 200 and js.get("version") == 1 and isinstance(js.get("items"), list),
      "%s %s" % (st, str(js)[:120]))
st, js = call("POST", "/v1/library/import",
              {"items": [{"type": "image", "model": "test", "prompt": "归档 测试",
                          "b64": PNG_1x1}]})
check("POST /v1/library/import 导入 1 项", st == 200 and js.get("imported") == 1,
      "%s %s" % (st, str(js)[:160]))
st, js = call("POST", "/v1/library/import", {"items": []})
check("POST /v1/library/import 空 items -> 400", st == 400, st)

print()
print("== 结果: %d 通过 / %d 失败 ==" % (len(PASS), len(FAIL)))
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
sys.exit(1 if FAIL else 0)
