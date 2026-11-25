#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭源 mlx-serve 功能面 冒烟 —— 打 我们 自研 内核, 逐 条 验 契约 + 真 跑.

用法: python3 smoke_full.py [base_url] [--live]
  默认: 契约 + 轻量 能力 (embeddings/sound/3d/tts, 不 载 大模型)
  --live: 再 载 LLM 验 对话 / 决策 / 流式
"""
import base64
import json
import os
import sys
import re
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:11259"
LIVE = False
argv = sys.argv[1:]
for a in argv:
    if a == "--live":
        LIVE = True
    elif not a.startswith("--"):
        BASE = a

PASS, FAIL = [], []


def call(method, path, body=None, raw=False, data=None, ctype=None):
    url = BASE + path
    hdr = {}
    if data is None:
        hdr["Content-Type"] = "application/json"
        if body is not None:
            data = json.dumps(body).encode()
    else:
        hdr["Content-Type"] = ctype or "application/octet-stream"
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
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
    print(("  [PASS] " if cond else "  [FAIL] ") + name +
          (("  -> " + str(detail)[:200]) if not cond else ""))


def err_contract(name, st, js, want_code=None, want_msg=None):
    ok = isinstance(js, dict) and "error" in js and isinstance(js["error"], dict) \
        and "message" in js["error"]
    if want_code is not None:
        ok = ok and st == want_code
    if want_msg is not None:
        ok = ok and (want_msg in js["error"]["message"] if ok else False)
    check(name, ok, "%s %s" % (st, str(js)[:140]))



def ws_responses_check(base, model, timeout=120):
    """WS /v1/responses: 握手 + 发 response.create + 收 事件 名 集合."""
    import socket, struct
    host, port = base.replace("http://", "").split(":")
    port = int(port)
    key = base64.b64encode(os.urandom(16)).decode()
    try:
        sk = socket.create_connection((host, port), timeout=timeout)
        sk.sendall(("GET /v1/responses HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
                    "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n" % (host, port, key)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = sk.recv(4096)
            if not d:
                return False, set()
            buf += d
        head, rest = buf.split(b"\r\n\r\n", 1)
        if b"101" not in head.split(b"\r\n")[0]:
            return False, set()
        msg = json.dumps({"type": "response.create",
                          "response": {"model": model, "input": "说: 你好",
                                       "max_output_tokens": 32,
                                       "reasoning_effort": "none"}}).encode()
        mask = os.urandom(4)
        n = len(msg)
        hdr = bytes([0x81, 0x80 | n]) if n < 126 else bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
        sk.sendall(hdr + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(msg)))
        evs = set()
        text = ""
        while True:
            while len(rest) < 2:
                d = sk.recv(65536)
                if not d:
                    sk.close()
                    return ("response.completed" in evs), evs
                rest += d
            b0, b1 = rest[0], rest[1]
            op = b0 & 0x0F
            ln = b1 & 0x7F
            off = 2
            if ln == 126:
                while len(rest) < 4:
                    rest += sk.recv(65536)
                ln = struct.unpack(">H", rest[2:4])[0]
                off = 4
            elif ln == 127:
                while len(rest) < 10:
                    rest += sk.recv(65536)
                ln = struct.unpack(">Q", rest[2:10])[0]
                off = 10
            while len(rest) < off + ln:
                d = sk.recv(65536)
                if not d:
                    break
                rest += d
            payload = rest[off:off + ln]
            rest = rest[off + ln:]
            if op == 8:
                break
            if op != 1:
                continue
            try:
                ev = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            t = ev.get("type")
            if t:
                evs.add(t)
            if t == "response.output_text.delta":
                text += ev.get("delta", "")
            if t == "response.completed":
                break
        sk.close()
        return ("response.completed" in evs and bool(text.strip())), evs
    except Exception:
        return False, set()


print("== 1. 运维 / 元 ==")
st, js = call("GET", "/health")
check("GET /health 200 + status ok", st == 200 and js.get("status") == "ok", js)
st, js = call("GET", "/v1/models")
_row = (js.get("data") or [{}])[0] if isinstance(js, dict) else {}
check("GET /v1/models 行 有 闭源 字段 (alias/engine/architecture/error)",
      all(k in _row for k in ("alias", "engine", "architecture", "error",
                               "bytes_resident", "bytes_on_disk")),
      str(_row)[:200])
check("GET /v1/models 行 的 meta 有 vocab_size/num_layers",
      all(k in (_row.get("meta") or {}) for k in ("vocab_size", "num_layers",
                                                  "context_length", "quantization")),
      str(_row.get("meta"))[:200])
rows = (js.get("data") or []) if isinstance(js, dict) else []
chat_model = next((r["id"] for r in rows if "chat" in (r.get("capabilities") or [])), "")
ok = st == 200 and rows and all(
    set(["id", "object", "capabilities", "state", "loaded", "meta"]) <= set(r) for r in rows)
check("GET /v1/models 原版 形状 (id/object/capabilities/state/loaded/meta)", ok, str(js)[:200])
caps = sorted({c for r in rows for c in (r.get("capabilities") or [])})
check("capabilities 词表 含 chat/image/video/music/embeddings/audio/sound/3d/decisions",
      {"chat", "image", "video", "music", "embeddings", "audio", "sound", "3d",
       "decisions"} <= set(caps), caps)
states = {r.get("state") for r in rows}
check("state 只 有 loaded/ready/absent (壳 只 认 这 三个)",
      states <= {"loaded", "ready", "absent"}, states)
st, js = call("GET", "/props")
ok = st == 200 and isinstance(js, dict) and "settings" in js and "memory" in js
check("GET /props 原版 形状 (settings/memory/default_generation_settings)",
      ok and "default_generation_settings" in js, str(js)[:160])
st, body = call("GET", "/metrics", raw=True)
body = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
check("GET /metrics Prometheus 指标名 (requests_running/generation_tokens_total)",
      st == 200 and "requests_running" in body and "generation_tokens_total" in body, body[:160])
st, js = call("GET", "/metrics.json")
check("GET /metrics.json 有 gen_count/rss_gib/free_gib/engines",
      st == 200 and all(k in js for k in ["gen_count", "rss_gib", "free_gib", "engines"]), str(js)[:160])
st, js = call("GET", "/v1/providers")
check("GET /v1/providers 有 providers 数组", st == 200 and isinstance(js.get("providers"), list), js)
st, js = call("GET", "/status")
check("GET /status 200 (壳 用)", st == 200 and isinstance(js, dict), str(js)[:120])

print("== 2. Ollama 兼容 ==")
st, js = call("GET", "/api/tags")
check("GET /api/tags 有 models 数组", st == 200 and isinstance(js.get("models"), list), str(js)[:160])
st, js = call("GET", "/api/version")
check("GET /api/version 有 version", st == 200 and "version" in js, js)
st, js = call("GET", "/api/ps")
check("GET /api/ps 有 models 数组", st == 200 and isinstance(js.get("models"), list), js)
st, js = call("POST", "/api/show", {"model": chat_model})
check("POST /api/show 出 model_info (闭源 契约)",
      st == 200 and isinstance(js.get("model_info"), dict)
      and "context_length" in (js.get("model_info") or {}), "%s %s" % (st, str(js)[:160]))
st, js = call("POST", "/api/show", {"model": "nope-xyz"})
# 闭源 契约: /api/show 未知 model -> 404 {"error":"model not found"} (裸 字符串,
# 不 是 标准 error 对象) —— 对拍 得到 的, 照 抄.
check("POST /api/show 未知 model -> 404 {\"error\":\"model not found\"}",
      st == 404 and js.get("error") == "model not found", "%s %s" % (st, str(js)[:140]))
st, js = call("POST", "/api/show", {})
check("POST /api/show 没 model -> 400 {\"error\":\"model is required\"}",
      st == 400 and js.get("error") == "model is required", "%s %s" % (st, str(js)[:140]))
st, body = call("POST", "/api/pull", {"model": chat_model}, raw=True)
pulltxt = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
check("POST /api/pull 出 success 状态",
      st == 200 and "success" in pulltxt, pulltxt[-160:])
st, js = call("POST", "/api/push", {"model": chat_model})
err_contract("POST /api/push -> 501 unsupported ollama endpoint", st, js, 501, None)
st, js = call("POST", "/api/delete", {"model": chat_model})
err_contract("POST /api/delete -> 501 unsupported ollama endpoint", st, js, 501, None)

print("== 3. 错误 契约 (400 点名字段) ==")
st, js = call("POST", "/v1/chat/completions", {})
err_contract("POST /v1/chat/completions 缺 messages -> 400", st, js, 400, "messages")
st, js = call("POST", "/v1/chat/completions",
              {"model": "nope-xyz", "messages": [{"role": "user", "content": "hi"}]})
err_contract("未知 model -> 404 且 点名 model", st, js, 404, "nope-xyz")
st, js = call("POST", "/v1/completions", {})
err_contract("POST /v1/completions 缺 prompt -> 400", st, js, 400, "prompt")
st, js = call("POST", "/v1/embeddings", {})
err_contract("POST /v1/embeddings 缺 input -> 400", st, js, 400, "input")
st, js = call("POST", "/v1/audio/music-generations", {})
err_contract("POST /v1/audio/music-generations 缺 prompt -> 400", st, js, 400, "prompt")
st, js = call("POST", "/v1/audio/speech", {})
err_contract("POST /v1/audio/speech 缺 input -> 400", st, js, 400, "input")
st, js = call("POST", "/v1/audio/sound-generations", {})
err_contract("POST /v1/audio/sound-generations 缺 prompt -> 400", st, js, 400, "prompt")
st, js = call("POST", "/v1/video/generations", {})
err_contract("POST /v1/video/generations 缺 prompt -> 400", st, js, 400, "prompt")
st, js = call("POST", "/v1/images/generations", {})
err_contract("POST /v1/images/generations 缺 prompt -> 400", st, js, 400, "prompt")
st, js = call("POST", "/v1/images/edits", {"prompt": "x"})
err_contract("POST /v1/images/edits 缺 image -> 400", st, js, 400, "image")
st, js = call("POST", "/v1/decisions", {"model": chat_model, "questions": {"q1": "?"}})
err_contract("POST /v1/decisions 能力 不 匹配 -> 400 点名 decisions", st, js, 400, "decisions")
st, js = call("POST", "/v1/messages", {"model": "nope-xyz",
                                       "messages": [{"role": "user", "content": "hi"}]})
err_contract("POST /v1/messages 未知 model -> 404", st, js, 404, "nope-xyz")
st, js = call("POST", "/v1/responses", {})
err_contract("POST /v1/responses 缺 input -> 400", st, js, 400, "input")
st, js = call("GET", "/v1/responses/resp_doesnotexist")
err_contract("GET /v1/responses/{id} 不存在 -> 404", st, js, 404)

print("== 4. 能力 真 跑 (轻量, 不 载 大模型) ==")
emb_model = next((r["id"] for r in rows if "embeddings" in (r.get("capabilities") or [])), "")
st, js = call("POST", "/v1/embeddings", {"model": emb_model, "input": ["一只猫", "a cat"]})
vecs = (js.get("data") or []) if isinstance(js, dict) else []
ok = st == 200 and len(vecs) == 2 and isinstance(vecs[0].get("embedding"), list) \
    and len(vecs[0]["embedding"]) > 100
check("POST /v1/embeddings 出 真 向量 (2x%d 维)" % (len(vecs[0].get("embedding", [])) if vecs else 0),
      ok, "%s %s" % (st, str(js)[:160]))
if ok:
    a, b = vecs[0]["embedding"], vecs[1]["embedding"]
    dot = sum(x * y for x, y in zip(a, b))
    check("向量 相似度 合理 (中英 同义 > 0.5)", dot > 0.5, "dot=%.3f" % dot)

st, js = call("POST", "/v1/audio/sound-generations",
              {"model": "ours/sound", "prompt": "rain on a window", "seconds": 2})
ok = st == 200 and isinstance(js.get("data"), list) and js["data"] and \
    len(js["data"][0].get("b64_json", "")) > 1000
check("POST /v1/audio/sound-generations 出 WAV (base64)", ok, "%s %s" % (st, str(js)[:160]))

st, js = call("POST", "/v1/3d/generations", {"model": "ours/3d", "prompt": "a torus ring"})
ok = st == 200 and js.get("url", "").endswith(".glb") and \
    (js.get("mesh") or {}).get("triangles", 0) > 0
check("POST /v1/3d/generations 出 GLB (含 三角形 数)", ok, "%s %s" % (st, str(js)[:160]))

st, js = call("POST", "/v1/audio/speech",
              {"model": "ours/tts", "input": "你好, 这是语音合成测试。", "response_format": "wav"},
              raw=True)
ok = st == 200 and isinstance(js, bytes) and len(js) > 1000 and js[:4] == b"RIFF"
check("POST /v1/audio/speech 出 裸 WAV (RIFF 头)", ok, "%s %s" % (st, str(js)[:80]))

st, js = call("POST", "/v1/images/edits",
              {"model": "krea-turbo", "prompt": "make it warmer and more vivid",
               "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="})
ok = st == 200 and js.get("data") and js["data"][0].get("b64_json") and js.get("operations")
check("POST /v1/images/edits 出 编辑 后 的 图", ok, "%s %s" % (st, str(js)[:160]))

st, js = call("POST", "/api/embed", {"model": emb_model, "input": "一只猫"})
ok = st == 200 and isinstance(js.get("embeddings"), list) and js["embeddings"]
check("POST /api/embed (Ollama 形状) 出 向量", ok, "%s %s" % (st, str(js)[:160]))

if LIVE:
    print("== 5. 实跑 (载 大模型, 慢) ==")
    llm = next((r for r in rows if "chat" in (r.get("capabilities") or [])), None)
    if llm:
        st, js = call("POST", "/v1/chat/completions",
                      {"model": llm["id"], "messages": [{"role": "user", "content": "用一句话说: 你好"}],
                       "max_tokens": 40})
        txt = ""
        try:
            txt = js["choices"][0]["message"]["content"]
        except Exception:
            pass
        check("POST /v1/chat/completions 出 文本", st == 200 and bool(txt), "%s %s" % (st, str(js)[:160]))
        st, body = call("POST", "/v1/chat/completions",
                        {"model": llm["id"], "messages": [{"role": "user", "content": "数到三"}],
                         "max_tokens": 20, "stream": True}, raw=True)
        body = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
        check("POST /v1/chat/completions stream=True -> SSE chunk + [DONE]",
              st == 200 and "chat.completion.chunk" in body and "[DONE]" in body, body[:200])
        # 回归: 思考 模型 的 流式 必须 真的 吐 正文 (delta.content),
        # 光 有 reasoning_content 不 算 数 (那 是 "模型 不 回话" 的 老 毛病)
        try:
            has_c = '"content"' in body
            has_r = '"reasoning_content"' in body
        except Exception:
            has_c = has_r = False
        check("POST /v1/chat/completions stream=True 吐 正文 delta.content",
              st == 200 and has_c, body[-300:])
        st, body2 = call("POST", "/v1/chat/completions",
                         {"model": llm["id"], "messages": [{"role": "user", "content": "说: 你好"}],
                          "max_tokens": 64, "enable_thinking": False, "stream": True}, raw=True)
        body2 = body2.decode("utf-8", "replace") if isinstance(body2, bytes) else body2
        check("POST /v1/chat/completions enable_thinking=false 也 吐 正文",
              st == 200 and '"content"' in body2, body2[-300:])
        # 回归: 空 回答 是 概率性 的 (随机 采样 首 token 撞 EOS) ——
        # 连 跑 5 次, 每次 都 必须 有 正文 (防 "模型 不 回话" 复发)
        empty = 0
        for _i in range(5):
            st3, b3 = call("POST", "/v1/chat/completions",
                           {"model": llm["id"], "messages": [{"role": "user", "content": "说: 你好"}],
                            "max_tokens": 64, "enable_thinking": False}, raw=True)
            b3 = b3.decode("utf-8", "replace") if isinstance(b3, bytes) else str(b3)
            try:
                c = (json.loads(b3)["choices"][0]["message"].get("content") or "").strip()
            except Exception:
                c = ""
            if not c:
                empty += 1
        check("POST /v1/chat/completions 连 跑 5 次 都 有 正文 (无 空 回答)",
              empty == 0, "空 %d/5" % empty)
        st, js = call("POST", "/v1/messages",
                      {"model": llm["id"], "max_tokens": 40,
                       "messages": [{"role": "user", "content": "说: 你好"}]})
        check("POST /v1/messages (Anthropic 形状) 有 content[0].text",
              st == 200 and isinstance(js.get("content"), list) and js["content"][0].get("type") == "text",
              "%s %s" % (st, str(js)[:160]))
        st, js = call("POST", "/v1/tokenize", {"text": "hello"})
        check("POST /v1/tokenize 出 token 列表", st == 200 and isinstance(js.get("tokens"), list), str(js)[:120])
        st, js = call("POST", "/v1/responses", {"model": llm["id"], "input": "说: 你好"})
        rid = js.get("id") if isinstance(js, dict) else None
        check("POST /v1/responses 出 output[0].content[0].text",
              st == 200 and isinstance(js.get("output"), list), "%s %s" % (st, str(js)[:160]))
        if rid:
            st2, js2 = call("GET", "/v1/responses/" + rid)
            check("GET /v1/responses/{id} 取 回 存 的", st2 == 200 and js2.get("id") == rid, str(js2)[:120])
            st3, js3 = call("DELETE", "/v1/responses/" + rid)
            check("DELETE /v1/responses/{id} 删 掉", st3 == 200 and js3.get("deleted") is True, str(js3)[:120])
        st, body = call("POST", "/api/chat",
                        {"model": llm["id"], "messages": [{"role": "user", "content": "你好"}],
                         "stream": False}, raw=True)
        body = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
        check("POST /api/chat (Ollama 形状) 出 message.content",
              st == 200 and '"message"' in body and '"content"' in body, body[:160])
        # 回归: Responses 流式 必须 发 闭源 那 套 事件 名
        st, body = call("POST", "/v1/responses",
                        {"model": llm["id"], "input": "说: 你好", "max_output_tokens": 32,
                         "stream": True}, raw=True)
        body = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
        evs = set(re.findall(r"^event: (\S+)", body, re.M))
        need_ev = {"response.created", "response.in_progress", "response.output_item.added",
                   "response.content_part.added", "response.output_text.delta",
                   "response.output_text.done", "response.content_part.done",
                   "response.output_item.done", "response.completed"}
        check("POST /v1/responses stream=True 发 完整 事件 序列",
              st == 200 and need_ev <= evs, "缺: %s" % sorted(need_ev - evs))
        # 回归: WS /v1/responses (闭源 有 这 个 端点): 握手 + 事件
        ws_ok, ws_ev = ws_responses_check(BASE, llm["id"])
        check("WS /v1/responses 握手 + 事件 序列",
              ws_ok and "response.completed" in ws_ev, str(sorted(ws_ev))[:200])
        st, js = call("POST", "/v1/decisions",
                      {"model": "ours/decisions",
                       "questions": {"q1": "该不该 现在 就 上线 这个 功能?", "q2": "风险 是 什么?"},
                       "state": "功能 已 实现, 未 上线"})
        dec = (js.get("decisions") or {}) if isinstance(js, dict) else {}
        ok = st == 200 and set(dec.keys()) == {"q1", "q2"} and \
            all(d.get("answer") and "(未 回答)" not in d.get("answer", "") and
                0 <= d.get("confidence", -1) <= 1 for d in dec.values())
        check("POST /v1/decisions 出 逐 题 答案 + 置信度", ok, "%s %s" % (st, str(js)[:200]))

print()
print("== 结果: %d 通过 / %d 失败 ==" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败 项:")
    for f in FAIL:
        print("  -", f)
sys.exit(1 if FAIL else 0)
