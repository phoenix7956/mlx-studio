#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""music_workbench.py — 出歌工作台 的 后端 (并 进 我们 自研 内核).

原来 这个 能力 住在 独立的 桌面端 (MC 出歌工作台, tkinter). 现在 整 个 并 进
MLX Serve Studio, 服务端 提供 三 组 端点, 前端 一 个 「音乐」 页 全 接管:

  本地 出歌   POST /v1/music/generate   —— 走 MLX (ACE-Step Turbo), 秒 级 出 歌
  广场          GET  /v1/music/square     —— 我们 自己 模型 出 的 歌
  作品库      GET  /v1/music/works      —— 本地 成品
  批量 出歌   POST /v1/music/batch      —— 清单 批量, 后台 跑

出歌 引擎 有 两 级:
  1) MLX  (首选) —— 闭源 mlx-serve 的 acestep 架构 (Zig/MLX 原生 重实现),
     30 秒 歌 ≈ 12 秒, 60 秒 带 词 ≈ 4 秒. 快 60 倍.
  2) ACE-Step :7861 (兜底) —— 本机 PyTorch CPU 那 套, 慢 (14 分钟/首),
     但 支持 5Hz LM 的 CoT 编曲. MLX 挂 了 才 走 它.

所有 端点 都 免 登录 —— 全部 走 本机 引擎, 不 连 任何 云端.
"""
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import outputs
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))

# 我们 自研 的 MLX 出歌 引擎 (ACE-Step 1.5 · 纯 MLX) —— 主 路.
# 代码/权重 都在 本机 (~/.ours/ace-step-engine, ~/.ours/models/ace-step-mlx), 不 依赖 闭源.
try:
    import music_mlx as _OURS
except Exception:                                    # 内核 还没 装 这 个 模块 时 不 炸
    _OURS = None

# 闭源 mlx-serve (逆向 出来 的 那 个) —— 只 作为 兼容 兜底 (用户 已经 把它 删 了)
MLX_SERVE = os.environ.get("MLX_SERVE_URL", "http://127.0.0.1:11296")
MLX_MUSIC_MODEL = os.environ.get(
    "MLX_MUSIC_MODEL", "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit")

# 本地 成品 扫描 目录 —— 与 原来 桌面端 的 LOCAL_MUSIC_DIRS 一致
LOCAL_MUSIC_DIRS = [
    ("产出", os.path.join(os.path.dirname(HERE), "产出")),
    ("Studio 输出", os.path.join(outputs.ROOT, "音频")),
    ("每日创作歌曲", os.path.expanduser("~/Desktop/每日创作歌曲")),
    ("ComfyUI 输出", os.path.expanduser("~/Desktop/workspace/ComfyUI_Output")),
    ("ACE-Step 缓存", os.path.expanduser(
        "~/Applications/ace-step/repo-v15/.cache/acestep/tmp/api_audio")),
    ("ACE-Step 缓存2", os.path.expanduser("~/.cache/acestep/tmp/api_audio")),
]
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")

# —— 本地 广场 —— 我们 自己 的 模型 出 的 歌 ——
# 参照 「模型」页 的 做法: 每 首 都 标 上 是 哪个 模型 出 的, 能 试听,
# 能 拿 提示词 直接 再 跑 一 首. 引擎 原始 目录 里 的 成品 也 一并 收进来.
LOCAL_SONG_DIRS = [
    ("ours/ace-step-v15-turbo-mlx", "ACE-Step 1.5",
     os.path.join(outputs.HOME, ".ours/ace-songs")),
    ("ours/minimax-music3-4bit", "MiniMax Music3",
     os.path.join(outputs.HOME, ".ours/music3-songs")),
]

MODEL_LABELS = {
    "ours/ace-step-v15-turbo-mlx": "ACE-Step 1.5",
    "ours/minimax-music3-4bit": "MiniMax Music3",
}

_batch_jobs = {}
_batch_lock = threading.Lock()


def _log(*a):
    print("[music_workbench]", *a, file=sys.stderr, flush=True)


def _http_json(url, data=None, method=None, timeout=60, headers=None):
    """小 HTTP 帮手 —— 返回 (status, obj). obj 是 dict 或 原始 文本."""
    hdrs = {"User-Agent": "MLXMusicWorkbench/1.0"}
    if headers:
        hdrs.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs,
                                 method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            st = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        st = e.code
    except Exception as e:
        raise RuntimeError("连不上 %s：%s" % (url, e))
    try:
        return st, json.loads(raw)
    except ValueError:
        return st, raw


def _http_bin(url, data=None, timeout=900, headers=None):
    """下载 二进制 (WAV). 返回 (status, bytes)."""
    hdrs = {"User-Agent": "MLXMusicWorkbench/1.0"}
    if headers:
        hdrs.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise RuntimeError("连不上 %s：%s" % (url, e))


def _wav_to_mp3(path):
    """WAV -> MP3. 有 ffmpeg 就 转, 没 就算了 (WAV 也 能 播)."""
    try:
        import subprocess
        out = os.path.splitext(path)[0] + ".mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-codec:a", "libmp3lame", "-q:a", "2", out],
                       timeout=180, check=True)
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            os.remove(path)
            return out
    except Exception as e:
        _log("wav->mp3 失败 (留 wav):", e)
    return path


def _mlx_model():
    """在 MLX 内核 的 模型 表 里 找 出歌 模型 (返回 dict 或 None)."""
    try:
        st, d = _http_json(MLX_SERVE + "/v1/models", timeout=6)
        if st != 200:
            return None
        items = d.get("data") if isinstance(d, dict) else d
        for m in items or []:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id") or "")
            caps = m.get("capabilities") or []
            if MLX_MUSIC_MODEL.split("/")[-1].lower() in mid.lower() or \
                    ("music" in caps and "acestep" in mid.lower()):
                return m
    except Exception:
        return None
    return None


def mlx_ready():
    """出歌 引擎 在 不 在 —— 我们 自研 的 MLX 引擎 (主) , 闭源 那个 (兜底)."""
    if _OURS is not None:
        try:
            if _OURS.available():
                return True
        except Exception:
            pass
    return _mlx_model() is not None


def mlx_loaded():
    """出歌 权重 现在 有没有 挂 在 内存 里 (卸载 后 = False)."""
    if _OURS is not None:
        try:
            if _OURS.available():
                return bool(_OURS.running())
        except Exception:
            pass
    m = _mlx_model()
    if not m:
        return False
    if "loaded" in m:
        return bool(m.get("loaded"))
    return str(m.get("state") or "").lower() in ("loaded", "ready")


def stop_our_mlx():
    """停 我们 自研 的 MLX 出歌 引擎 (闲置 自动 卸载 / 手动 卸载 都 用 它)."""
    if _OURS is None:
        return False
    try:
        if _OURS.available():
            _OURS.stop()
            return True
    except Exception:
        pass
    return False


def our_mlx_status():
    """我们 自研 引擎 的 详细 状态 (给 /v1/music/health)."""
    if _OURS is None:
        return {"available": False, "running": False, "engine": "none"}
    try:
        return _OURS.status()
    except Exception as e:
        return {"available": False, "running": False, "error": str(e)}


def _stage_audio(path):
    """把 成品 复制 到 统一 输出 目录 (~/Documents/MLX Studio 输出/音频/).

    我们 自研 的 引擎 把 歌 存在 ~/.ours/ace-songs/... (带 song.json 元数据),
    那个 目录 不 在 /out/ 的 搜索 范围 里 —— 直接 给 basename 会 404,
    界面 里 就 播 不 出 声音. 这里 复制 一 份 到 输出 目录 (命名 跟 别的 产物 一致),
    顺便 把 song.json 也 带 过去 (改 同名 .json), 资料库 跨 重启 还能 扫 到.
    """
    if not path or not os.path.isfile(path):
        return path
    try:
        ext = os.path.splitext(path)[1] or ".mp3"
        dst = outputs.path_for("audio", "mx_audio", ext)
        shutil.copyfile(path, dst)
        meta = os.path.join(os.path.dirname(path), "song.json")
        if os.path.isfile(meta):
            try:
                shutil.copyfile(meta, os.path.splitext(dst)[0] + ".json")
            except Exception:
                pass
        return dst
    except Exception:
        return path

def _dur_reason(info):
    """时长 依据 -> 一句 人话 (界面 显示 「自动 为什么 是 这个 时长」)"""
    if not info:
        return ""
    try:
        import music_duration as MD
        return MD.explain(info)
    except Exception:
        return info.get("capped") or ""


def generate(payload, out_dir=None, timeout=1200):
    """出一首歌. 返回 dict:
       {"ok":bool, "path":..., "sec":..., "engine":"mlx"|"ace",
        "title":..., "duration":..., "error":...}
    """
    prompt = (payload.get("prompt") or payload.get("caption") or "").strip()
    lyrics = (payload.get("lyrics") or "").strip()
    instrumental = bool(payload.get("instrumental"))
    # —— 时长: 给 数字 (手动), 或者 "auto" (按 歌词 结构 自动 算) ——
    raw_dur = payload.get("duration_seconds")
    if raw_dur is None:
        raw_dur = payload.get("audio_duration")
    auto = bool(payload.get("duration_auto")) or (
        isinstance(raw_dur, str) and raw_dur.strip().lower() in ("auto", "自动"))
    max_dur = 480.0
    try:
        if _OURS is not None:
            max_dur = float(_OURS.max_duration())
    except Exception:
        pass
    dur_info = None
    if auto:
        import music_duration as MD
        dur, dur_info = MD.estimate(lyrics, instrumental=instrumental, max_s=max_dur)
    else:
        try:
            dur = float(raw_dur)
        except (TypeError, ValueError):
            dur = 60.0
        dur = max(10.0, min(max_dur, dur))
        if dur_info is None and float(raw_dur or 0) > max_dur:
            dur_info = {"capped": "超过 引擎 上限 %d 秒, 已 按 上限" % int(max_dur)}
    seed = payload.get("seed")
    title = (payload.get("title") or "").strip() or "未命名"

    if not prompt and not lyrics:
        return {"ok": False, "error": "提示词 和 歌词 都 是 空 的, 出不了歌"}

    t0 = time.time()
    body = {"model": MLX_MUSIC_MODEL, "prompt": prompt, "duration_seconds": dur}
    if instrumental:
        body["lyrics"] = ""
    elif lyrics:
        body["lyrics"] = lyrics
    for k in ("vocal_language", "bpm", "keyscale", "timesignature"):
        v = payload.get(k)
        if v not in (None, ""):
            body[k] = v
    if seed is not None:
        try:
            body["seed"] = int(seed)
        except (TypeError, ValueError):
            pass

    # ---- 0) 我们 自研 的 MLX 引擎 (ACE-Step 1.5 · 纯 MLX) —— 主 路 ----
    if _OURS is not None:
        try:
            if _OURS.available():
                # UI 下拉 选 的 出歌 模型 —— 只 认 我们 自己 的 ("ours/...")
                want = (payload.get("model") or "").strip()
                ours_model = want if want.startswith("ours/") else None
                path, msg = _OURS.generate(
                    prompt, lyrics=lyrics, duration=dur, instrumental=instrumental,
                    bpm=payload.get("bpm"), keyscale=payload.get("keyscale"),
                    language=payload.get("vocal_language"), model_id=ours_model,
                    seed=seed,
                    timeout=timeout)
                if path:
                    path = _stage_audio(path)
                    return {"ok": True, "path": path, "sec": round(time.time() - t0, 1),
                            "engine": "mlx-ours", "title": title, "duration": dur,
                            "model": ours_model or _OURS.MODEL_ID, "note": msg,
                            "duration_auto": bool(auto), "duration_reason": _dur_reason(dur_info)}
                _log("我们 的 MLX 引擎 出不了 (降级):", msg)
        except Exception as e:
            _log("我们 的 MLX 引擎 异常 (降级):", e)

    # ---- 1) 闭源 MLX-Serve (如果 还 在) ----
    if mlx_ready():
        try:
            st, raw = _http_bin(MLX_SERVE + "/v1/audio/music-generations",
                                 data=body, timeout=timeout)
            if st == 200 and raw[:4] == b"RIFF":
                out_dir = out_dir or outputs.dir_for("audio")
                os.makedirs(out_dir, exist_ok=True)
                p = os.path.join(out_dir, "mx_music_%d.wav" % int(time.time()))
                with open(p, "wb") as f:
                    f.write(raw)
                p = _wav_to_mp3(p)
                return {"ok": True, "path": p, "sec": round(time.time() - t0, 1),
                        "engine": "mlx", "title": title, "duration": dur,
                        "model": MLX_MUSIC_MODEL,
                        "duration_auto": bool(auto), "duration_reason": _dur_reason(dur_info)}
            _log("MLX 出歌 返回 异常:", st, raw[:120])
        except Exception as e:
            _log("MLX 出歌 异常 (降级 ACE):", e)

    # ---- 2) ACE-Step :7861 (慢, 但 有 CoT 编曲) ----
    ace = os.environ.get("ACE_MUSIC_URL", "http://127.0.0.1:7861")
    try:
        ab = {"prompt": prompt, "lyrics": "" if instrumental else lyrics,
              "thinking": True, "task_type": "text2music",
              "vocal_language": payload.get("vocal_language") or "en",
              "inference_steps": int(payload.get("steps") or 8),
              "guidance_scale": float(payload.get("cfg") or 7.0),
              "use_random_seed": seed is None,
              "seed": int(seed or -1), "audio_format": "mp3",
              "audio_duration": dur, "use_cot_caption": True,
              "use_cot_language": True, "lm_backend": "pt"}
        st, d = _http_json(ace + "/release_task", data=ab, timeout=120)
        if st >= 300 or not isinstance(d, dict) or not d.get("task_id"):
            raise RuntimeError("ACE 建任务 失败：%s" % str(d)[:200])
        tid = d["task_id"]
        deadline = t0 + timeout
        while time.time() < deadline:
            time.sleep(3)
            st, qr = _http_json(ace + "/query_result",
                                data={"task_id_list": [tid]}, timeout=60)
            if not isinstance(qr, list) or not qr:
                continue
            item = qr[0]
            code = int(item.get("status") or 0)
            if code == 1:
                raw = item.get("result")
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except ValueError:
                        raw = []
                fp = ""
                for r in (raw or []):
                    if isinstance(r, dict) and r.get("file"):
                        fp = r["file"]
                        break
                if not fp:
                    raise RuntimeError("ACE 完成 但 没 音频")
                if fp.startswith("/v1/audio"):
                    au = urllib.request.urlopen(ace + fp, timeout=120).read()
                elif fp.startswith("http"):
                    au = urllib.request.urlopen(fp, timeout=120).read()
                else:
                    q = urllib.parse.quote(fp, safe="")
                    au = urllib.request.urlopen(ace + "/v1/audio?path=" + q,
                                               timeout=120).read()
                out_dir = out_dir or outputs.dir_for("audio")
                os.makedirs(out_dir, exist_ok=True)
                p = os.path.join(out_dir, "mx_music_%d.mp3" % int(time.time()))
                with open(p, "wb") as f:
                    f.write(au)
                return {"ok": True, "path": p, "sec": round(time.time() - t0, 1),
                        "engine": "ace", "title": title, "duration": dur,
                        "duration_auto": bool(auto), "duration_reason": _dur_reason(dur_info)}
            if code == 2:
                raise RuntimeError("ACE 失 败: %s" % str(item.get("result"))[:200])
        raise RuntimeError("ACE 超 时")
    except Exception as e:
        return {"ok": False, "error": str(e), "sec": round(time.time() - t0, 1)}


def model_label(model_id):
    """模型 id -> 好 看 的 名字 (参照 模型 页 的 口径)."""
    if not model_id:
        return "未知模型"
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    for mid, lbl, _d in LOCAL_SONG_DIRS:
        if model_id == mid:
            return lbl
    return model_id


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _song_model(meta, fallback_id=""):
    """song.json -> 模型 id (和 模型 页 认 的 id 对齐)."""
    e = str(meta.get("engine") or "").lower()
    d = str(meta.get("dit_model") or "").lower()
    if "minimax" in e or "music3" in e or "music3" in d:
        return "ours/minimax-music3-4bit"
    if "acestep" in e or "acestep" in d or meta.get("lm_model"):
        return "ours/ace-step-v15-turbo-mlx"
    return fallback_id


def audio_url(path):
    """本地 成品 的 播放 地址 —— 输出 目录 里 的 走 /out/, 别 的 走 受控 的 /v1/music/audio."""
    try:
        ap = os.path.abspath(str(path))
        if ap.startswith(os.path.abspath(outputs.ROOT) + os.sep):
            return "/out/" + outputs.rel(ap)
    except Exception:
        pass
    return "/v1/music/audio?path=" + urllib.parse.quote(str(path))


def audio_path(p):
    """受控 的 本地 音频 读取 —— 只 允许 落在 允许 的 目录 里 (防 目录 穿越)."""
    if not p:
        return None
    try:
        ap = os.path.abspath(str(p))
    except Exception:
        return None
    roots = [os.path.abspath(outputs.ROOT)]
    for _mid, _lbl, d in LOCAL_SONG_DIRS:
        roots.append(os.path.abspath(d))
    for _lbl, d in LOCAL_MUSIC_DIRS:
        roots.append(os.path.abspath(d))
    for r in roots:
        if ap == r or ap.startswith(r + os.sep):
            return ap if os.path.isfile(ap) else None
    return None


def local_square(limit=200):
    """本地 广场 —— 我们 的 模型 出 的 歌 (标 模型 / 可 试听 / 可 复刻)."""
    rows = []
    seen = set()

    def add(path, meta, model_id, src):
        try:
            st = os.stat(path)
        except OSError:
            return
        # 同 一 首 歌 在 「输出目录」 和 「引擎 原始目录」 各 有 一 份 ——
        # 用 song.json 里 的 audio 名 (复制 时 保留) 做 键, 只 留 一 条.
        key = str(meta.get("audio") or os.path.basename(path))
        if key in seen:
            return
        seen.add(key)
        prompt = (meta.get("prompt") or meta.get("caption") or "").strip()
        try:
            dur = float(meta.get("duration") or meta.get("audio_duration") or 0)
        except (TypeError, ValueError):
            dur = 0.0
        title = (meta.get("title") or "").strip() or (prompt[:18] or "未命名")
        rows.append({
            "title": title, "tags": prompt[:160], "prompt": prompt,
            "duration": round(dur, 1), "model": model_label(model_id),
            "model_id": model_id, "src": src, "url": audio_url(path),
            "path": path, "size": st.st_size, "mtime": st.st_mtime,
            "ts": time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime)),
        })

    # 1) 统一 输出 目录 —— Studio 里 出 的 都 在 这
    adir = outputs.dir_for("audio")
    if os.path.isdir(adir):
        for fn in sorted(os.listdir(adir)):
            if not fn.lower().endswith(AUDIO_EXT):
                continue
            fp = os.path.join(adir, fn)
            meta = _read_json(os.path.splitext(fp)[0] + ".json")
            add(fp, meta, _song_model(meta), "输出目录")

    # 2) 引擎 原始 目录 —— CLI / 批次 出 的, 没 进 输出 目录 的
    for mid, lbl, d in LOCAL_SONG_DIRS:
        if not os.path.isdir(d):
            continue
        for sub in os.listdir(d):
            sp = os.path.join(d, sub)
            if not os.path.isdir(sp):
                continue
            meta = _read_json(os.path.join(sp, "song.json"))
            af = (meta.get("audio") or "").strip()
            cand = os.path.join(sp, af) if af else ""
            if not (cand and os.path.isfile(cand)):
                cands = [f for f in os.listdir(sp) if f.lower().endswith(AUDIO_EXT)]
                cand = os.path.join(sp, cands[0]) if cands else ""
            if not cand or not os.path.isfile(cand):
                continue
            add(cand, meta, _song_model(meta, mid), "引擎 产出")

    rows.sort(key=lambda r: r["mtime"], reverse=True)
    total = len(rows)
    rows = rows[:max(1, int(limit or 200))]
    return {"ok": True, "items": rows, "count": len(rows), "total": total,
            "source": "local"}


def square(code="trending", page=1, page_size=50, source="local"):
    """广场 —— 我们 自己 模型 出 的 歌 (本地, 不联网)."""
    return local_square(limit=max(50, int(page_size or 50) * 4))


def works(token=None, page=1, page_size=100):
    """作品库 —— 本地 成品 (我们 自己 的 引擎 出 的)."""
    rows = []
    for label, d in LOCAL_MUSIC_DIRS:
        if not os.path.isdir(d):
            continue
        for dirpath, _dirs, files in os.walk(d):
            for fn in files:
                if not fn.lower().endswith(AUDIO_EXT):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    stt = os.stat(p)
                except OSError:
                    continue
                rows.append({"name": fn, "path": p, "src": label,
                             "size": stt.st_size,
                             "mtime": stt.st_mtime,
                             "ts": time.strftime("%m-%d %H:%M",
                                                time.localtime(stt.st_mtime))})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    local = rows[:500]
    return {"ok": True, "local": local, "online": [],
            "local_count": len(local), "online_count": 0}


def batch(items, out_dir=None, concurrency=2, model=None, backend="mlx"):
    """批量 出歌 —— 后台 跑, 返回 job_id. 进度 用 batch_status 查."""
    if isinstance(items, dict):
        items = items.get("items") or []
    items = [i for i in (items or []) if isinstance(i, dict)]
    if not items:
        return {"ok": False, "error": "清单 是 空 的"}
    out_dir = out_dir or os.path.join(
        outputs.ROOT, "音频",
        "批次%s" % time.strftime("%m%d%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    job_id = "batch_%d" % int(time.time() * 1000)
    with _batch_lock:
        _batch_jobs[job_id] = {"id": job_id, "status": "running", "done": 0,
                               "total": len(items), "out_dir": out_dir,
                               "items": [], "log": [], "started": time.time()}

    def _one(it, idx):
        title = (it.get("title") or "").strip() or "未命名"
        desc = (it.get("description") or "").strip()
        lyr = (it.get("lyrics") or "").strip()
        tags = it.get("tags") or ""
        if isinstance(tags, list):
            tags = ", ".join(tags)
        if it.get("styles"):
            tags = (tags + ", " + str(it["styles"])).strip(", ")
        prompt = desc or tags or title
        if tags and desc:
            prompt = desc + ", " + tags
        payload = {"prompt": prompt, "lyrics": lyr,
                   "instrumental": bool(it.get("instrumental")),
                   "duration_seconds": it.get("duration") or 120,
                   "title": title}
        if it.get("duration_auto"):
            payload["duration_seconds"] = "auto"
        if it.get("seed") is not None:
            payload["seed"] = it["seed"]
        sub = os.path.join(out_dir, "%03d-%s" % (idx, title))
        os.makedirs(sub, exist_ok=True)
        r = generate(payload, out_dir=sub)
        return {"title": title, "ok": r.get("ok"), "path": r.get("path"),
                "engine": r.get("engine"), "sec": r.get("sec"),
                "duration": r.get("duration"), "duration_reason": r.get("duration_reason"),
                "error": r.get("error")}

    def _run():
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(4, int(concurrency or 1)))) as ex:
            futs = [(i, ex.submit(_one, it, i + 1)) for i, it in enumerate(items)]
            for _i, f in futs:
                try:
                    res = f.result()
                except Exception as e:
                    res = {"ok": False, "error": str(e)}
                with _batch_lock:
                    j = _batch_jobs.get(job_id)
                    if j:
                        j["items"].append(res)
                        j["done"] = len(j["items"])
                        j["log"].append("%s %s" % ("✅" if res.get("ok") else "❌",
                                                   res.get("title")))
                        if not res.get("ok"):
                            j["log"].append("   %s" % res.get("error"))
        with _batch_lock:
            j = _batch_jobs.get(job_id)
            if j:
                j["status"] = "succeeded" if j["done"] == j["total"] else "failed"
                j["ended"] = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id, "total": len(items), "out_dir": out_dir}


def batch_status(job_id):
    with _batch_lock:
        j = _batch_jobs.get(job_id)
        if not j:
            return {"ok": False, "error": "没 这个 批次"}
        return {"ok": True, **j}


def unload_music():
    """卸载 出歌 模型 —— 放 掉 显存."""
    out = []
    if _OURS is not None:
        try:
            if _OURS.available():
                _OURS.stop()
                out.append("我们 的 MLX 引擎: ✅ 已 卸载 (进程 停了, 内存 放 掉)")
        except Exception as e:
            out.append("我们 的 MLX 引擎: ❌ %s" % e)
    try:
        st, _d = _http_json(MLX_SERVE + "/v1/unload-model", timeout=30,
                             data={"model": MLX_MUSIC_MODEL})
        out.append("闭源 MLX-Serve: %s" % ("✅ 已 卸载" if st == 200 else "❌ %s" % st))
    except Exception as e:
        if "refused" in str(e) or "连不上" in str(e):
            out.append("闭源 MLX-Serve: 没 装 (正常)")
        else:
            out.append("闭源 MLX-Serve: ❌ %s" % e)
    try:
        ace = os.environ.get("ACE_MUSIC_URL", "http://127.0.0.1:7861")
        _http_json(ace + "/health", timeout=5)
        out.append("ACE-Step: 服务 还 在 (它 常驻 进程, 卸载 = 停 服务)")
    except Exception:
        out.append("ACE-Step: 没 起 来")
    return "\n".join(out)


def load_music(model_id=None):
    """加载 出歌 模型 —— 把 权重 拉 进 内存 (免 得 第一次 出歌 等 半天)."""
    out = []
    if _OURS is not None:
        try:
            if _OURS.available():
                mid = (model_id or "").strip()
                ok = _OURS.start(mid if mid.startswith("ours/") else None)
                out.append("我们 的 MLX 引擎: %s" % ("✅ 已 加载" if ok else "❌ 起 不 来"))
            else:
                out.append("我们 的 MLX 引擎: ❌ 组件 不 全 (权重/代码/解释器)")
        except Exception as e:
            out.append("我们 的 MLX 引擎: ❌ %s" % e)
    try:
        st, _d = _http_json(MLX_SERVE + "/v1/load-model", timeout=900,
                            data={"model": MLX_MUSIC_MODEL, "capability": "music"})
        out.append("闭源 MLX-Serve: %s" % ("✅ 已 加载" if st == 200 else "❌ %s" % st))
    except Exception as e:
        # 闭源 那 个 App 用户 已经 删 了 —— 连 不 上 是 正常 的, 别 报 错 吓人
        if "refused" in str(e) or "连不上" in str(e):
            out.append("闭源 MLX-Serve: 没 装 (正常 —— 我们 用 自己 的 引擎)")
        else:
            out.append("闭源 MLX-Serve: ❌ %s" % e)
    try:
        ace = os.environ.get("ACE_MUSIC_URL", "http://127.0.0.1:7861")
        _http_json(ace + "/health", timeout=8)
        out.append("ACE-Step: ✅ 服务 在 跑")
    except Exception:
        out.append("ACE-Step: 没 起 来 (MLX 够 用)")
    return "\n".join(out)


def health():
    ours = our_mlx_status()
    return {"ok": True, "mlx": mlx_ready(), "loaded": mlx_loaded(),
            "mlx_url": MLX_SERVE,
            "model": (ours.get("model") if ours.get("available") else MLX_MUSIC_MODEL),
            "ours": ours,
            "model_name": (ours.get("name") if ours.get("available") else MLX_MUSIC_MODEL),
            "suno": False}


def reveal(path):
    """在 访达 中 定位 这个 文件."""
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": "文件 不 在 了"}
    try:
        import subprocess
        subprocess.run(["/usr/bin/open", "-R", path], timeout=20, check=True)
        return {"ok": True, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}
