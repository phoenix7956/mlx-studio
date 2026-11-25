#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-TTS 神经 TTS 引擎 —— 自研内核 的 神经 语音 / 声纹克隆 能力.

为什么 要 这个 引擎:
    闭源 mlx-serve 的 /v1/audio/speech 背后 是 Qwen3-TTS (0.6B Base, 8bit),
    能 做 两 件 我们 原来 做 不 到 的 事:
      1. 真 神经 TTS —— 波形 是 模型 合成 的, 不 是 macOS `say` 拼 出来 的;
      2. 声纹克隆 —— ref_audio (+ref_text) 直接 复刻 说话人 的 音色.
    原来 我们 只 有 Kokoro/macOS-say, 音色 面板 是 "假 的" (所有 id 落 到 同一 个 声音).

模型 (本机):
    ~/.ours/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
    —— mlx_audio 自带 的 qwen3_tts 实现, 官方 README 明确 写
       "Voice cloning is only supported on the Base models".

坑 (踩 过 的):
    - 参数名 是 `lang_code="Chinese"`, 不是 `language=` —— 传错 克隆 音色 对不上;
    - Base 模型 没有 具名 说话人 (supported_speakers 为空), 传 voice 会 直接 抛
      ValueError —— 所以 引擎 忽略 voice (闭源 也 忽略);
    - 生成 慢, 所以 懒加载 + 缓存 必须 做.
"""
import base64
import glob
import os
import re
import sys
import time

SR = 24000                     # Qwen3-TTS 12Hz codec 输出 24kHz

# 语言 名 -> 模型 认识 的 lang_code
LANG_MAP = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "zh-tw": "Chinese",
    "cmn": "Chinese", "yue": "Chinese",
    "en": "English", "en-us": "English", "en-gb": "English",
    "ja": "Japanese", "jp": "Japanese",
    "ko": "Korean", "kr": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "ms": "Malay",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch", "uk": "Ukrainian",
}

# 候选 权重 目录 (第一个 存在 的 赢)
def _home():
    """真实 home —— 有些运行环境的 $HOME 是假的 (沙箱 / 后台服务)."""
    try:
        import pwd
        h = pwd.getpwuid(os.getuid()).pw_dir
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return os.path.expanduser("~")


_H = _home()
MODEL_DIRS = [
    os.path.join(_H, ".ours/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"),
    os.path.join(_H, ".ours/models/local/Qwen3-TTS-12Hz-0.6B-Base-8bit"),
    os.path.join(_H, ".ours/models/models/Qwen3-TTS-12Hz-0.6B-Base-8bit"),
    os.path.join(_H, ".ours/models/mlm/Qwen3-TTS-12Hz-0.6B-Base-8bit"),
    os.path.join(_H, ".ours/models/mlm/Qwen3-TTS-12Hz-0.6B-Base-bf16"),
]

# 参考 音频 的 文件 头 -> 扩展名 (上传 的 可能 是 mp3/m4a/ogg, 不 一定 是 wav)
_MAGIC = [
    (b"RIFF", ".wav"),
    (b"OggS", ".ogg"),
    (b"fLaC", ".flac"),
    (b"FORM", ".aiff"),
    (b"\x1a\x45\xdf\xa3", ".webm"),
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
    (b"\xff\xf2", ".mp3"),
]


def sniff_ext(data):
    """给 参考 音频 猜 一个 扩展名 (让 解码器 认 得 格式)."""
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".m4a"
    if len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    return ".wav"


def b64_to_file(b64, prefix="mx_ref"):
    """base64 -> 临时 文件 路径 (猜 扩展名). 返回 路径."""
    if not b64:
        return ""
    if isinstance(b64, str):
        m = re.match(r"^data:[^;]+;base64,(.*)$", b64, re.S)
        if m:
            b64 = m.group(1)
        raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
    else:
        raw = bytes(b64)
    path = "/tmp/%s_%d%s" % (prefix, int(time.time() * 1000), sniff_ext(raw))
    with open(path, "wb") as f:
        f.write(raw)
    return path


def lang_code(lang):
    """'zh' -> 'Chinese'; 认 不 出 的 -> 'auto' (模型 自己 判断)."""
    if not lang:
        return "auto"
    s = str(lang).strip().lower().replace("_", "-")
    if s in LANG_MAP:
        return LANG_MAP[s]
    return LANG_MAP.get(s.split("-")[0], "auto")


class Qwen3TTSEngine:
    """神经 TTS (Qwen3-TTS Base) —— 合成 + 声纹克隆."""

    model_id = "qwen3-tts-base"
    backend = "qwen3-tts"

    def __init__(self):
        self.model = None
        self.model_dir = ""
        self.last_voice_used = ""
        self.last_substituted = ""
        self.last_cloned = False
        self.last_ref_samples = 0
        self.last_lang_code = ""
        self.last_seconds = 0.0
        self._asr = None

    # ------------------------------------------------------------ 权重 定位
    @staticmethod
    def find_model_dir():
        """找 权重 目录: 先 按 名单 找, 再 glob 兜底."""
        for d in MODEL_DIRS:
            p = os.path.expanduser(d) if d.startswith("~") else d
            if os.path.isfile(os.path.join(p, "config.json")):
                return p
        pats = [
            os.path.join(_H, ".ours/models/*/Qwen3-TTS-*"),
            os.path.join(_H, ".ours/models/*/Qwen3-TTS*"),
            os.path.join(_H, ".ours/models/mlm/Qwen3-TTS*"),
        ]
        for pat in pats:
            hits = sorted(glob.glob(os.path.expanduser(pat)))
            for p in hits:
                if os.path.isfile(os.path.join(p, "config.json")):
                    return p
        return ""

    def available(self):
        """权重 在 不 在 (不 触发 加载)."""
        if self.model is not None:
            return True
        return bool(self.find_model_dir())

    def state(self):
        return {"available": self.available(), "loaded": self.model is not None,
                "model_dir": self.model_dir or self.find_model_dir(),
                "sample_rate": SR, "backend": self.backend}

    # ------------------------------------------------------------ 加载
    def load(self):
        """懒 加载 (2GB 权重, 第一次 ~2s)."""
        if self.model is not None:
            return self.model
        d = self.find_model_dir()
        if not d:
            raise FileNotFoundError(
                "Qwen3-TTS 权重 不 在本机 (找过: %s)" % ", ".join(MODEL_DIRS))
        from mlx_audio.tts.utils import load_model as _load
        t0 = time.time()
        self.model = _load(d)
        self.model_dir = d
        print("[qwen3-tts] 权重 加载 完成 %.1fs: %s" % (time.time() - t0, d),
              file=sys.stderr, flush=True)
        return self.model

    def unload(self):
        self.model = None

    # ------------------------------------------------------------ ASR
    def transcribe(self, wav_path, language=None):
        """参考 音频 没 给 ref_text -> 本机 faster-whisper 转 一 遍."""
        try:
            if self._asr is None:
                import cap_engines as C
                self._asr = C.ASREngine()
            r = self._asr.transcribe(wav_path, language=language)
            return (r or {}).get("text") or ""
        except Exception as e:
            print("[qwen3-tts] 参考 音频 转写 失败: %s" % e, file=sys.stderr, flush=True)
            return ""

    # ------------------------------------------------------------ 音色 清单
    def catalog(self):
        """给 /v1/audio/voices 用: 神经 TTS 自己 没有 具名 音色, 但 能 克隆."""
        d = self.model_dir or self.find_model_dir()
        rows = [{
            "id": "clone",
            "label": "神经声纹克隆 (上传 参考 音频)",
            "display": "声纹克隆",
            "lang": "multi",
            "backend": self.backend,
            "kind": "clone",
        }]
        if d:
            rows[0]["label"] += " · %s" % os.path.basename(d)
        return rows

    def speakers(self):
        """Base 模型 没有 具名 说话人 —— 返回 空 列表 (诚实)."""
        if self.model is None:
            return []
        try:
            return list(self.model.get_supported_speakers() or [])
        except Exception:
            return []

    # ------------------------------------------------------------ 合成
    def speak(self, text, voice=None, lang="zh", speed=1.0, out_path=None,
              ref_audio=None, ref_text=None, ref_text_lang=None, temperature=None):
        """出 wav 路径.

        ref_audio: 参考 音频 的 文件 路径 (给 出来 了 就 克隆).
        ref_text : 参考 音频 的 逐字 文本 (克隆 的 关键 —— 没 给 就 本机 ASR 转).
        voice    : Base 模型 无 具名 音色, 这里 只 记 不 用 (跟 闭源 一致: 忽略).
        speed    : 模型 本身 不 支持 变速 -> WSOLA 时间 伸缩 (音高 不变).
        """
        out_path = out_path or "/tmp/mx_qwen3tts_%d.wav" % int(time.time() * 1000)
        model = self.load()
        lc = lang_code(lang)
        self.last_lang_code = lc
        self.last_cloned = False
        self.last_ref_samples = 0
        self.last_voice_used = ""
        self.last_substituted = ""
        self.last_seconds = 0.0

        kwargs = {"text": text, "lang_code": lc}
        if temperature is not None:
            # 采样 温度 —— 默认 取 模型 自带 的 generation_config (0.9).
            # 给 0 (或 很小) 就 是 贪心 解码, 输出 可 复现 (测 试 用).
            kwargs["temperature"] = float(temperature)
        if ref_audio:
            rt = (ref_text or "").strip()
            if not rt:
                rt = self.transcribe(ref_audio, language=ref_text_lang)
            if not rt:
                print("[qwen3-tts] 没 参考 文本, 退 普通 合成", file=sys.stderr, flush=True)
            else:
                kwargs["ref_audio"] = ref_audio
                kwargs["ref_text"] = rt
                self.last_cloned = True

        t0 = time.time()
        results = list(model.generate(**kwargs))
        if not results:
            raise RuntimeError("Qwen3-TTS 没 出 音频")

        import numpy as np
        chunks = []
        for r in results:
            a = np.asarray(r.audio, dtype=np.float32)
            if a.ndim > 1:
                a = a.mean(axis=-1)
            chunks.append(a)
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = np.nan_to_num(audio).astype(np.float32)

        # speed: 模型 不 支持 -> WSOLA 时间 伸缩 (alpha<1 = 更短 = 更快)
        sp = float(speed or 1.0)
        if abs(sp - 1.0) > 0.01 and len(audio) > 2048:
            try:
                from voice_convert import wsola_time_stretch
                audio = wsola_time_stretch(audio, SR, 1.0 / max(sp, 0.25))
            except Exception as e:
                print("[qwen3-tts] 变速 失败: %s" % e, file=sys.stderr, flush=True)

        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.99:
            audio = audio * (0.99 / peak)

        from cap_engines import write_wav
        write_wav(out_path, audio, SR)
        self.last_seconds = round(len(audio) / float(SR), 2)
        print("[qwen3-tts] %.1fs 生成 %s (clone=%s lang=%s) %.2fs 音频"
              % (time.time() - t0, os.path.basename(out_path),
                 self.last_cloned, lc, self.last_seconds),
              file=sys.stderr, flush=True)
        return out_path

