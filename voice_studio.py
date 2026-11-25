#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配音工作台 —— 一键克隆声音 + 批量自动配音 (内核).

参照 VoiceStudio (github.com/debpalash/VoiceStudio, AGPL-3.0) 的 **交互结构**:
    音色库 (克隆一个 声纹 -> 存 成 音色)  +  批量配音 (一 表 台词 -> 逐条 出 音轨)
但是:
    · **不 抄 它 的 代码** (AGPL-3.0 有 传染性, 我们 自己 的 项目 要 开源);
    · 引擎 全 是 **本机 MLX** (Apple Silicon / Metal), 不 依赖 CUDA;
    · 一切 落 在 本机, 不 出网, 不 要 账号.

引擎 走 mlx-audio (pip install mlx-audio): 一个 统一 的 generate 接口,
各家 开源 TTS (Qwen3-TTS / OmniVoice / VoxCPM2 / IndexTTS / Chatterbox /
CosyVoice) 都 能 用 —— 我们 只 按 各自 的 generate 签名 塞 参数.
"""
import glob
import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid


def _home():
    """真实 用户 的 home —— 不 信 $HOME (有些 运行环境 HOME 是 临时 的)."""
    try:
        import pwd
        h = pwd.getpwuid(os.getuid()).pw_dir
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return os.path.expanduser("~")


HOME = _home()
ROOT = os.path.join(HOME, ".ours", "voices")
REFS = os.path.join(ROOT, "refs")
STORE = os.path.join(ROOT, "profiles.json")


# ============================================================ 引擎 清单
# cloning   : 能不能 拿 参考 音频 克隆 声纹
# ref_text  : 克隆 时 是否 强制 要 参考 音频 的 逐字 文本 (没 给 就 本机 ASR 转)
# dirs      : 权重 可能 在 的 位置 (挨个 找)
# pull/local: 给 「模型」页 下载 用 (hf-mirror)
ENGINES = [
    {
        "id": "qwen3-tts-base",
        "name": "Qwen3-TTS 0.6B Base",
        "note": "声纹克隆 · 跨语种 · 最稳 (已经 在 本机)",
        "cloning": True, "ref_text": True,
        "size": "1.2G",
        "pull": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "local": "~/.ours/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "dirs": ["~/.ours/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
                 "~/.ours/models/Qwen3-TTS-12Hz-0.6B-Base-8bit"],
        "langs": ["zh", "en", "ja", "ko", "yue", "auto"],
    },
    {
        "id": "omnivoice",
        "name": "OmniVoice",
        "note": "VoiceStudio 同款 引擎 · 克隆 + 声音 设计",
        "cloning": True, "ref_text": False,
        "size": "~2.3G",
        "pull": "mlx-community/OmniVoice-8bit",
        "local": "~/.ours/models/omnivoice-8bit",
        "dirs": ["~/.ours/models/omnivoice-8bit",
                 "~/.ours/models/OmniVoice-8bit"],
        "langs": ["zh", "en", "auto"],
    },
    {
        "id": "voxcpm2",
        "name": "VoxCPM2",
        "note": "零样本 克隆 · 中文 稳 · 快",
        "cloning": True, "ref_text": False,
        "size": "~2G",
        "pull": "mlx-community/VoxCPM2-8bit",
        "local": "~/.ours/models/voxcpm2-8bit",
        "dirs": ["~/.ours/models/voxcpm2-8bit",
                 "~/.ours/models/VoxCPM2-8bit"],
        "langs": ["zh", "en", "auto"],
    },
    {
        "id": "indextts2",
        "name": "IndexTTS 2",
        "note": "情绪 语气 控制 最强 · 克隆",
        "cloning": True, "ref_text": False,
        "size": "~6G",
        "pull": "mlx-community/IndexTTS-2-MLX",
        "local": "~/.ours/models/indextts2-mlx",
        "dirs": ["~/.ours/models/indextts2-mlx",
                 "~/.ours/models/IndexTTS-2-MLX"],
        "langs": ["zh", "en", "auto"],
    },
    {
        "id": "chatterbox",
        "name": "Chatterbox",
        "note": "轻 (0.5B) · 克隆 · 出片 快",
        "cloning": True, "ref_text": False,
        "size": "~1.7G",
        "pull": "mlx-community/Chatterbox-TTS-8bit",
        "local": "~/.ours/models/chatterbox-tts-8bit",
        "dirs": ["~/.ours/models/chatterbox-tts-8bit",
                 "~/.ours/models/Chatterbox-TTS-8bit"],
        "langs": ["en", "auto"],
    },
    {
        "id": "cosyvoice2",
        "name": "CosyVoice 2 (0.5B)",
        "note": "零样本 克隆 · 指令 跟随 好",
        "cloning": True, "ref_text": False,
        "size": "~1.5G",
        "pull": "mlx-community/CosyVoice2-0.5B-8bit",
        "local": "~/.ours/models/cosyvoice2-0.5b-8bit",
        "dirs": ["~/.ours/models/cosyvoice2-0.5b-8bit",
                 "~/.ours/models/CosyVoice2-0.5B-8bit"],
        "langs": ["zh", "en", "auto"],
    },
]

ENGINE_BY_ID = {e["id"]: e for e in ENGINES}


# 常见 语言 -> 各家 的 lang_code (qwen3 用 "Chinese"/"English", 其它 用 "zh"/"en")
LANG_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese",
    "ko": "Korean", "yue": "Cantonese", "ru": "Russian",
    "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "auto": "auto",
}


def _expand(p):
    return os.path.expanduser(p) if isinstance(p, str) and p.startswith("~") else p


def _lang_code(lang, engine_id=""):
    lang = (lang or "zh").strip().lower()
    if lang in ("auto", "多语种", "自动"):
        return "auto"
    if engine_id in ("qwen3-tts-base", "qwen3-tts-custom"):
        return LANG_NAMES.get(lang, lang)
    return lang


# ============================================================ 引擎 工具
def find_model_dir(engine_id):
    """找 引擎 权重 目录 —— 挨个 试, 再 glob 兜底."""
    e = ENGINE_BY_ID.get(engine_id)
    if not e:
        return ""
    for d in e.get("dirs", []):
        p = _expand(d)
        if p and os.path.isfile(os.path.join(p, "config.json")):
            return p
    for d in e.get("dirs", []):
        base = os.path.basename(_expand(d) or "")
        if not base:
            continue
        pat = os.path.join(HOME, ".ours", "models", "**", base)
        for hit in sorted(glob.glob(pat, recursive=True)):
            if os.path.isfile(os.path.join(hit, "config.json")):
                return hit
    return ""


def engine_present(engine_id):
    return bool(find_model_dir(engine_id))


def engine_list():
    """给 界面 / API 用 的 引擎 清单."""
    out = []
    for e in ENGINES:
        d = find_model_dir(e["id"])
        row = dict(e)
        row["present"] = bool(d)
        row["model_dir"] = d
        row.pop("dirs", None)
        out.append(row)
    return out


def _accepts(sig, key):
    if key in sig:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in sig.values())


def _gen_kwargs(model, text, ref_audio=None, ref_text=None, lang=None,
                speed=1.0, instruct=None, duration_s=None, extra=None):
    """按 各家 的 generate 签名 拼 参数 —— 只 塞 它 认识 的."""
    try:
        sig = inspect.signature(model.generate).parameters
    except Exception:
        sig = {}
    kw = {"text": text}

    def put(k, v):
        if v is None:
            return
        if _accepts(sig, k):
            kw[k] = v

    put("ref_audio", ref_audio)
    put("ref_text", ref_text)
    put("speed", speed)
    put("instruct", instruct)
    put("duration_s", duration_s)
    put("lang_code", lang)
    put("language", lang)
    put("prompt_text", ref_text)
    for k, v in (extra or {}).items():
        put(k, v)
    return kw


def _collect_audio(results):
    """model.generate 返回 的 是 生成器 (可能 多 段) —— 收成 一 条 numpy 波形."""
    import numpy as np
    if results is None:
        raise RuntimeError("模型 没 返回 音频")
    if not hasattr(results, "__iter__") or isinstance(results, (bytes, bytearray)):
        results = [results]
    chunks = []
    sr = 24000
    for r in results:
        a = getattr(r, "audio", r)
        if a is None:
            continue
        s = getattr(r, "sample_rate", None)
        if s:
            sr = int(s)
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1) if arr.shape[-1] <= 8 else arr.mean(axis=0)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        raise RuntimeError("模型 没 出 音频")
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    audio = np.nan_to_num(audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    return audio, sr


def write_wav(path, audio, sr):
    import numpy as np
    import wave
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((a * 32767.0).astype("<i2").tobytes())
    return path


def read_wav(path):
    import numpy as np
    import wave
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


# ============================================================ 主 类
class VoiceStudio:
    """音色库 + 批量配音 —— 一个 进程 内 只 挂 一个 TTS 模型 (省 内存)."""

    def __init__(self):
        self.lock = threading.RLock()
        self._model = None
        self._model_id = ""
        self._model_dir = ""
        self._profiles = []
        self._asr = None
        os.makedirs(REFS, exist_ok=True)
        self._load_store()

    # ------------------------------------------------------------ 音色库 存储
    def _load_store(self):
        try:
            with open(STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._profiles = data if isinstance(data, list) else []
        except Exception:
            self._profiles = []

    def _save_store(self):
        try:
            os.makedirs(ROOT, exist_ok=True)
            tmp = STORE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._profiles, f, ensure_ascii=False, indent=1)
            os.replace(tmp, STORE)
        except Exception as e:
            print("[voice] 音色库 保存 失败: %s" % e, file=sys.stderr, flush=True)

    def profiles(self):
        with self.lock:
            return [dict(p) for p in self._profiles]

    def profile(self, pid):
        with self.lock:
            for p in self._profiles:
                if p.get("id") == pid:
                    return dict(p)
        return None

    def delete_profile(self, pid):
        with self.lock:
            keep = []
            gone = None
            for p in self._profiles:
                if p.get("id") == pid:
                    gone = p
                else:
                    keep.append(p)
            if gone is None:
                return False
            self._profiles = keep
            self._save_store()
            ref = gone.get("ref_audio") or ""
            if ref and os.path.isfile(ref):
                try:
                    os.remove(ref)
                except Exception:
                    pass
            return True

    # ------------------------------------------------------------ 引擎 加载 / 卸载
    def state(self):
        with self.lock:
            return {
                "loaded": self._model is not None,
                "engine": self._model_id,
                "model_dir": self._model_dir,
                "profiles": len(self._profiles),
                "root": ROOT,
            }

    def unload(self):
        """置闲 30min 后 由 内核 调 —— 真 释放 权重."""
        with self.lock:
            self._model = None
            self._model_id = ""
            self._model_dir = ""
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        print("[voice] 配音 引擎 已 卸载", file=sys.stderr, flush=True)

    def _load_engine(self, engine_id):
        with self.lock:
            if self._model is not None and self._model_id == engine_id:
                return self._model
            if self._model is not None:
                # 换 引擎 先 卸 旧 的 —— 两个 TTS 权重 同时 挂 着 白 占 内存
                self.unload()
            d = find_model_dir(engine_id)
            if not d:
                raise FileNotFoundError(
                    "%s 权重 不 在本机 —— 先 去「模型」页 下载" % (engine_id,))
            from mlx_audio.tts.utils import load_model
            t0 = time.time()
            self._model = load_model(d)
            self._model_id = engine_id
            self._model_dir = d
            print("[voice] 引擎 %s 加载 完成 %.1fs: %s"
                  % (engine_id, time.time() - t0, d), file=sys.stderr, flush=True)
            return self._model

    # ------------------------------------------------------------ ASR (给 参考 音频 自动 转写)
    def _transcribe(self, path, language=None):
        try:
            if self._asr is None:
                import cap_engines as C
                self._asr = C.ASREngine()
            r = self._asr.transcribe(path, language=language) or {}
            return (r.get("text") or "").strip()
        except Exception as e:
            print("[voice] 参考 音频 转写 失败: %s" % e, file=sys.stderr, flush=True)
            return ""

    # ------------------------------------------------------------ 一键 克隆
    def clone(self, name, ref_audio, engine_id="qwen3-tts-base", ref_text="",
              language="zh", kind="clone", instruct="", sample_text="",
              make_sample=True):
        """参考 音频 -> 一个 新 音色. 没 给 ref_text 就 本机 ASR 自动 转写.

        返回 profile dict. make_sample=True 时 顺便 出 一 句 试听样音.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("音色 要 有 名字")
        e = ENGINE_BY_ID.get(engine_id)
        if not e:
            raise ValueError("没有 这个 配音 引擎: %s" % engine_id)
        if not os.path.isfile(ref_audio):
            raise FileNotFoundError("参考 音频 不 在: %s" % ref_audio)
        import outputs as O
        os.makedirs(REFS, exist_ok=True)
        # 参考音频 / 试听样音 都 落 在 统一 输出 目录 —— 这样 /out/ 能 直接 播,
        # 资料库 也 看得见 (不然 路径 在 ~/.ours/voices, 前端 播 不了).
        vdir = os.path.join(O.dir_for("tts"), "音色库")
        os.makedirs(vdir, exist_ok=True)
        pid = "vp_" + uuid.uuid4().hex[:10]
        dst = os.path.join(vdir, "%s.wav" % pid)
        shutil.copyfile(ref_audio, dst)
        # 参考 音频 归一化 成 单声道 24k wav —— 各家 引擎 都 吃 这个 格式
        try:
            self._normalize_ref(dst)
        except Exception as e:
            print("[voice] 参考 音频 归一化 失败 (照 原样 用): %s" % e,
                  file=sys.stderr, flush=True)
        rt = (ref_text or "").strip()
        if not rt and e.get("ref_text"):
            rt = self._transcribe(dst, language=None if language in ("auto", "zh") else language)
        prof = {
            "id": pid,
            "name": name,
            "engine": engine_id,
            "kind": kind,
            "language": language or "zh",
            "ref_audio": dst,
            "ref_text": rt,
            "instruct": instruct or "",
            "created_at": int(time.time()),
            "sample": "",
        }
        if make_sample:
            try:
                prof["sample"] = self.synthesize(
                    prof, sample_text or "你好，这是我的声音克隆测试。")
            except Exception as e:
                print("[voice] 试听样音 生成 失败 (音色 仍 会 保存): %s" % e,
                      file=sys.stderr, flush=True)
        with self.lock:
            self._profiles.insert(0, prof)
            self._save_store()
        return dict(prof)

    def _normalize_ref(self, path):
        """参考 音频 -> 单声道 24k wav (各引擎 都 兼容)."""
        import numpy as np
        try:
            import soundfile as sf
            a, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            a, sr = read_wav(path)
        a = np.asarray(a, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 24000 and a.size:
            # 简单 重采样 (线性) —— 参考 音频 短, 精度 够用
            n = int(round(len(a) * 24000.0 / sr))
            if n > 8:
                x_old = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
                a = np.interp(x_new, x_old, a).astype(np.float32)
            sr = 24000
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        if peak > 0.99:
            a = a * (0.99 / peak)
        write_wav(path, a, 24000)

    # ------------------------------------------------------------ 合成
    def synthesize(self, profile, text, out_path=None, speed=1.0, lang=None,
                   engine_id=None):
        """按 音色 profile 合成 一 条 语音, 返回 wav 路径."""
        if isinstance(profile, str):
            profile = self.profile(profile)
        if not profile:
            raise ValueError("没 这个 音色")
        text = (text or "").strip()
        if not text:
            raise ValueError("台词 是 空的")
        eid = engine_id or profile.get("engine") or "qwen3-tts-base"
        e = ENGINE_BY_ID.get(eid)
        if not e:
            raise ValueError("没有 这个 配音 引擎: %s" % eid)
        model = self._load_engine(eid)
        lang = lang or profile.get("language") or "zh"
        lc = _lang_code(lang, eid)
        ref_audio = profile.get("ref_audio") or ""
        ref_text = (profile.get("ref_text") or "").strip() or None
        if ref_audio and e.get("ref_text") and not ref_text:
            ref_text = self._transcribe(ref_audio) or None
        kw = _gen_kwargs(
            model, text,
            ref_audio=ref_audio if os.path.isfile(ref_audio) else None,
            ref_text=ref_text,
            lang=lc,
            speed=float(speed or 1.0),
            instruct=(profile.get("instruct") or "").strip() or None,
        )
        t0 = time.time()
        results = model.generate(**kw)
        audio, sr = _collect_audio(results)
        sp = float(speed or 1.0)
        if abs(sp - 1.0) > 0.01 and len(audio) > 2048:
            try:
                from voice_convert import wsola_time_stretch
                audio = wsola_time_stretch(audio, sr, 1.0 / max(sp, 0.25))
            except Exception as ex:
                print("[voice] 变速 失败: %s" % ex, file=sys.stderr, flush=True)
        if not out_path:
            import outputs as O
            out_path = O.path_for("tts", "mx_tts_clone", "wav")
        write_wav(out_path, audio, sr)
        print("[voice] %s %.1fs 生成 %s (%.2fs 音频)"
              % (eid, time.time() - t0, os.path.basename(out_path), len(audio) / float(sr)),
              file=sys.stderr, flush=True)
        return out_path

    # ------------------------------------------------------------ 批量 配音
    def batch(self, lines, profile_id=None, engine_id=None, speed=1.0,
              lang=None, fmt="wav", join=True, video=None, keep_original=False,
              out_dir=None, name="批量配音", progress=None):
        """一 表 台词 -> 逐条 出 音轨 (+ 拼成 一 条 + 可选 贴 回 视频)."""
        import outputs as O
        lines = [l for l in (lines or []) if (l.get("text") or "").strip()]
        if not lines:
            raise ValueError("没有 台词")
        base = out_dir or O.dir_for("tts")
        job_dir = os.path.join(base, "%s_%s" % (name, time.strftime("%H%M%S")))
        os.makedirs(job_dir, exist_ok=True)
        items = []
        all_audio = []
        all_sr = 24000
        for i, ln in enumerate(lines):
            txt = (ln.get("text") or "").strip()
            pid = ln.get("profile_id") or profile_id
            prof = self.profile(pid) if pid else None
            if prof is None and profile_id:
                prof = self.profile(profile_id)
            if prof is None:
                items.append({"index": i, "text": txt, "path": "", "url": "",
                              "seconds": 0, "error": "没 指定 音色"})
                continue
            eid = ln.get("engine") or engine_id or prof.get("engine")
            sp = ln.get("speed") or speed
            lg = ln.get("lang") or lang
            p = os.path.join(job_dir, "%03d.wav" % (i + 1))
            try:
                self.synthesize(dict(prof, engine=eid), txt, out_path=p,
                                speed=sp, lang=lg)
                a, sr = read_wav(p)
                all_audio.append(a)
                all_sr = sr
                items.append({"index": i, "text": txt, "path": p,
                              "url": O.rel(p), "seconds": round(len(a) / float(sr), 2),
                              "error": ""})
            except Exception as e:
                items.append({"index": i, "text": txt, "path": "", "url": "",
                              "seconds": 0, "error": str(e)})
            if progress:
                try:
                    progress(i + 1, len(lines))
                except Exception:
                    pass
        out = {"dir": job_dir, "dir_url": O.rel(job_dir), "items": items,
               "count": len(items), "engine": engine_id or ""}
        if join and all_audio:
            import numpy as np
            gap = np.zeros(int(all_sr * 0.18), dtype=np.float32)
            merged = []
            for a in all_audio:
                merged.append(a)
                merged.append(gap)
            jp = os.path.join(job_dir, "合集.wav")
            write_wav(jp, np.concatenate(merged), all_sr)
            out["joined"] = jp
            out["joined_url"] = O.rel(jp)
        else:
            out["joined"] = ""
            out["joined_url"] = ""
        out["mp3"] = ""
        out["mp3_url"] = ""
        if fmt in ("mp3", "both") and shutil.which("ffmpeg"):
            src = out.get("joined")
            if not src:
                src = next((it["path"] for it in items if it.get("path")), "")
            if src:
                try:
                    mp = os.path.join(job_dir, "合集.mp3")
                    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
                    cmd += ["-b:a", "192k", mp]
                    subprocess.run(cmd, check=True, timeout=600)
                    out["mp3"] = mp
                    out["mp3_url"] = O.rel(mp)
                except Exception as e:
                    print("[voice] mp3 转码 失败: %s" % e, file=sys.stderr, flush=True)
        out["video_out"] = ""
        out["video_out_url"] = ""
        if video and os.path.isfile(video) and out.get("joined"):
            try:
                out["video_out"] = self._mux_video(video, out["joined"], keep_original)
                out["video_out_url"] = O.rel(out["video_out"])
            except Exception as e:
                print("[voice] 视频 合成 失败: %s" % e, file=sys.stderr, flush=True)
                out["video_error"] = str(e)
        return out

    def _mux_video(self, video, audio, keep_original):
        """新 音轨 贴 回 视频 (ffmpeg). keep_original=True -> 原声 音量 压 到 15%."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("没 ffmpeg")
        import outputs as O
        outp = O.path_for("video", "mx_video", "mp4")
        args = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio]
        if keep_original:
            args += ["-filter_complex",
                     "[0:a]volume=0.15[a0];[a0][1:a]amix=inputs=2:duration=first[a]",
                     "-map", "0:v", "-map", "[a]"]
        else:
            args += ["-map", "0:v", "-map", "1:a"]
        args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", outp]
        subprocess.run(args, check=True, timeout=1800)
        return outp


_STUDIO = None
_STUDIO_LOCK = threading.Lock()


def get_studio():
    """进程 内 单例."""
    global _STUDIO
    with _STUDIO_LOCK:
        if _STUDIO is None:
            _STUDIO = VoiceStudio()
        return _STUDIO


