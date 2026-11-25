"""cap_engines.py — 我们 自研 内核 的 能力 引擎 (全部 真跑, 本机 权重 / 本机 能力).

每个 引擎 只 管 自己 那 一块, 由 server.Srv.load(kind) 按需加载 / 卸载:
  embeddings : Qwen3-Embedding-0.6B (本机 魔塔 缓存, transformers CPU, 1024 维)
  tts        : Kokoro-82M-8bit (MLX, 魔塔 权重) —— 没 权重 时 退 macOS `say`
  sound      : 程序化 DSP 合成 (numpy, prompt -> 波形)
  music      : 程序化 音乐 合成 (numpy; ACE-Step 挂 了 时 兜底)
  3d         : 程序化 网格 -> GLB (+ matplotlib 预览 PNG)
  stt        : faster-whisper large-v3-turbo (本机 HF 缓存, int8 CPU)
  edits      : PIL 经典 编辑 管线 (Krea 本地 版 没有 img2img)

设计 原则: 能 真 跑 的 就 真 跑; 跑 不了 的 说清 原因, 不 装 能.
"""
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import outputs
import sys
import time
import wave

import numpy as np

SR = 44100


def _h(s):
    """prompt -> 稳定 种子 (同 一句 prompt 出 同 一份 结果)."""
    return int(hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:8], 16)


def write_wav(path, samples, sr=SR):
    """float [-1,1] 单声道 -> 16bit PCM wav."""
    a = np.asarray(samples, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767.0).astype("<i2").tobytes())
    return path


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def _wav_rms(path):
    """wav 的 有效 响度 (0 = 完全 静默) —— 用 来 探 声音 到底 出 没 出 声."""
    try:
        a, _ = read_wav(path)
        if len(a) < 16:
            return 0.0
        return float(np.sqrt(np.mean(np.square(a))))
    except Exception:
        return 0.0


def _lowpass_fast(x, cutoff, sr=SR):
    """一阶 低通 的 冲激 响应 卷积 (噪声 整形 用)."""
    a = math.exp(-2.0 * math.pi * max(cutoff, 1.0) / sr)
    n = int(min(len(x), max(8, int(6.0 * sr / max(cutoff, 1.0)))))
    k = a ** np.arange(n)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def _noise(n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


# ============================================================ 嵌入 (embeddings)
# 真实 home —— 有些 运行 环境 的 $HOME 是 假 的 (沙箱 / 后台 服务)
try:
    import pwd as _pwd
    _H = _pwd.getpwuid(os.getuid()).pw_dir
    if not (_H and os.path.isdir(_H)):
        raise RuntimeError
except Exception:
    _H = os.path.expanduser("~")


class EmbedEngine:
    """Qwen3-Embedding-0.6B —— 真 语义 向量 (last-token pooling + L2 norm)."""

    model_id = "qwen3-embedding-0.6b"

    CANDIDATES = [
        # 我们 自己 的 模型 目录 (优先) —— 合并 之后 的 唯一 位置
        os.path.join(_H, ".ours/models/ace-step-mlx/Qwen3-Embedding-0.6B"),
        os.path.join(_H, ".cache/modelscope/models/ACE-Step--Ace-Step1.5/checkpoints/Qwen3-Embedding-0.6B"),
        os.path.join(_H, ".ours/models/Qwen3-Embedding-0.6B"),
        os.path.join(_H, ".ours/models/mlm/Qwen3-Embedding-0.6B"),
    ]

    def __init__(self, path=None):
        self.path = path or os.environ.get("MLX_EMBED_PATH") or ""
        if not self.path:
            for c in self.CANDIDATES:
                if os.path.isdir(c):
                    self.path = c
                    break
        self.model = None
        self.tok = None

    def load(self):
        if self.model is not None:
            return
        if not self.path or not os.path.isdir(self.path):
            raise FileNotFoundError(
                "Qwen3-Embedding-0.6B 权重 不 在 (%s)" % (self.path or "未 找到"))
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path, dtype=torch.float32)
        self.model.eval()

    def unload(self):
        for a in ("model", "tok"):
            if getattr(self, a, None) is not None:
                setattr(self, a, None)

    def embed(self, texts, dim=None):
        """texts: list[str] -> list[list[float]] (已 L2 归一)."""
        self.load()
        import torch
        if isinstance(texts, str):
            texts = [texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                chunk = [t if isinstance(t, str) else str(t) for t in texts[i:i + 8]]
                b = self.tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
                h = self.model(**b).last_hidden_state
                idx = b["attention_mask"].sum(1) - 1
                v = h[torch.arange(h.size(0)), idx]
                if dim and dim < v.shape[-1]:
                    v = v[:, :dim]          # Matryoshka 截断
                v = torch.nn.functional.normalize(v, dim=-1)
                out.extend([[float(x) for x in row] for row in v.numpy()])
        return out


# ============================================================ TTS (语音 合成)
class TTSEngine:
    """Kokoro-82M (MLX) 优先; 没 权重 / 依赖 时 退 macOS `say` (中文 也 能 出)."""

    model_id = "ours/tts"

    KOKORO_DIRS = [
        os.path.expanduser("~/.ours/models/kokoro"),
        os.path.expanduser("~/.ours/models/mlm/Kokoro-82M-8bit"),
    ]
    # 稳定 的 音色 id -> (显示名, Kokoro 音色名, macOS 声音名, 语种, 性别)
    # 每个 id 都 映射 到 一个 真 的 存在 的 声音 —— 之前 所有 af_*/bf_* 都 落到 同 一个
    # macOS 声音, 音色 面板 是 假 的, "音色 混合" 也 混 不 出 任何 差别.
    VOICE_MAP = {
        "af_heart":  ("Samantha", "af_heart",  "Samantha", "en", "女"),
        "af_bella":  ("Karen",    "af_bella",  "Karen",    "en", "女"),
        "af_sky":    ("Moira",    "af_sky",    "Moira",    "en", "女"),
        "af_nicole": ("Tessa",    "af_nicole", "Tessa",    "en", "女"),
        "af_alloy":  ("Tara",     "af_alloy",  "Tara",     "en", "女"),
        "am_michael": ("Daniel",  "am_michael", "Daniel",  "en", "男"),
        "am_fenrir": ("Rishi",    "am_fenrir",  "Rishi",   "en", "男"),
        "am_adam":   ("Aman",     "am_adam",    "Aman",    "en", "男"),
        "bf_emma":   ("Sandy",    "bf_emma",    "Sandy",   "en", "女"),
        "bf_isabella": ("Meijia", "bf_isabella", "Meijia", "zh", "女"),
        "zf_xiaobei": ("婷婷",    "zf_xiaobei", "Tingting", "zh", "女"),
        "zf_xiaoyi":  ("美佳",    "zf_xiaoyi",  "Meijia",  "zh", "女"),
        "zm_yunjian": ("Reed",    "zm_yunjian", "Reed",    "zh", "男"),
        "zm_yunxi":   ("Rocko",   "zm_yunxi",   "Rocko",   "zh", "男"),
    }
    VOICES = {"en": [], "zh": []}          # 运行时 填

    def __init__(self):
        self.pipe = None
        self.backend = "macos-say"
        self.model_dir = next((d for d in self.KOKORO_DIRS if os.path.isdir(d)), "")
        self._probe_cache = {}
        self.last_voice_used = ""
        self.last_substituted = ""

    def load(self):
        if self.pipe is not None or self.backend == "macos-say":
            return
        from mlx_audio.tts.generate import load_model as _load
        self.pipe = _load(self.model_dir)

    def unload(self):
        self.pipe = None

    def _say(self, text, voice, out_path):
        """macOS say -> wav (22.05k 单声道)."""
        args = ["say", "-o", out_path, "--data-format=LEI16@22050"]
        if voice:
            args += ["-v", voice]
        args.append(text)
        subprocess.run(args, check=True, capture_output=True, timeout=180)
        return out_path

    def _kokoro_ready(self):
        try:
            import misaki  # noqa: F401
        except Exception:
            return False
        if not self.model_dir:
            return False
        return os.path.isfile(os.path.join(self.model_dir, "kokoro-v1_0.safetensors")) or \
            os.path.isfile(os.path.join(self.model_dir, "model.safetensors"))

    def speak(self, text, voice=None, lang="zh", speed=1.0, out_path=None):
        """出 wav 路径. voice=None 时 按 lang 挑 默认."""
        out_path = out_path or outputs.path_for("tts","mx_tts","wav")
        lang = (lang or "zh").lower()
        if not lang.startswith("zh"):
            lang = "en"
        if self._kokoro_ready():
            try:
                self.load()
                v = self.VOICE_MAP.get(voice, (None, None, None, None, None))[1] \
                    if voice in self.VOICE_MAP else None
                v = v or voice or self._default_voice(lang)
                audio = self.pipe.generate(text, voice=v, speed=speed, lang_code=lang)
                arr = np.asarray(audio, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                write_wav(out_path, arr, SR)
                self.backend = "kokoro-mlx"
                return out_path
            except Exception as e:
                print("[tts] kokoro 失败 (%s), 退 macOS say" % e, file=sys.stderr, flush=True)
        mac_voice = self.mac_voice(voice, lang)
        self.backend = "macos-say"
        # —— 探针: 有些 macOS 声音 只 在 英文 下 有 声音 (Reed/Rocko 读 中文 = 静默),
        #    不 探 一 遍 就 会 返 一个 11ms 的 空 wav —— 换 一个 真 能 读 这门 语言 的 声音.
        zh = bool(self._has_cjk(text)) or lang == "zh"
        used = mac_voice
        self.last_substituted = ""
        if not self._probe_voice(mac_voice, zh):
            fb = self._fallback_voice(zh)
            if fb:
                used = fb
                self.last_substituted = "%s -> %s" % (mac_voice, fb)
                print("[tts] %s 读不了 这门语言, 换 %s" % (mac_voice, fb),
                      file=sys.stderr, flush=True)
        self.last_voice_used = used
        self._say(text, used, out_path)
        return out_path

    @staticmethod
    def _has_cjk(text):
        return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))

    def _probe_voice(self, mac_voice, zh):
        """这个 macOS 声音 能不能 读 出 这门 语言 (真 跑 一 句 短 的, 看 有 没有 声音)."""
        if not mac_voice:
            return False
        key = (mac_voice, bool(zh))
        if key in self._probe_cache:
            return self._probe_cache[key]
        # 探针 是 中间 文件 (听完 就算), 不 进 输出 目录 —— 免得 污染 产物 文件夹
        import tempfile
        fd, out = tempfile.mkstemp(suffix=".wav", prefix="mlx_probe_")
        os.close(fd)
        ok = False
        try:
            self._say("你好" if zh else "hello", mac_voice, out)
            ok = _wav_rms(out) > 0.002
        except Exception:
            ok = False
        finally:
            try: os.unlink(out)
            except Exception: pass
        self._probe_cache[key] = ok
        return ok

    def _fallback_voice(self, zh):
        """挑 一个 真 能 读 这门 语言 的 macOS 声音."""
        cands = ["Tingting", "Meijia"] if zh else ["Samantha", "Karen", "Tessa"]
        for c in cands:
            if self._probe_voice(c, zh):
                return c
        for v in self.system_voices():
            if v.get("lang") == ("zh" if zh else "en") and self._probe_voice(v["id"], zh):
                return v["id"]
        return None

    def _default_voice(self, lang="zh"):
        for vid, row in self.VOICE_MAP.items():
            if row[3] == lang:
                return vid
        return "af_heart"

    def mac_voice(self, voice, lang="en"):
        """把 我们 的 音色 id / 系统 声音名 映射 到 macOS 的 声音名."""
        if not voice:
            return "Tingting" if lang == "zh" else "Samantha"
        if voice in self.VOICE_MAP:
            return self.VOICE_MAP[voice][2]
        if voice.startswith("mac:"):
            return voice[4:]
        # 未 知的 id: 直接 当 系统 声音名 试 (用户 可以 自己 填)
        return voice

    def voices(self, lang="en"):
        return [vid for vid, row in self.VOICE_MAP.items() if row[3] == lang]

    def catalog(self):
        """给 界面 / API 用 的 音色 清单 (真 的 存在 的 声音)."""
        out = []
        for vid, (disp, kok, mac, lang, sex) in self.VOICE_MAP.items():
            out.append({"id": vid, "label": "%s (%s·%s)" % (disp, mac, sex),
                        "display": disp, "locale": mac, "lang": lang,
                        "gender": sex, "backend": self.backend,
                        # zh_ok: 这个 声音 能不能 读 中文 —— 有些 macOS 声音 (Reed/Rocko)
                        # 装了 但 只 在 英文 下 有 声音, 读 中文 = 静默.
                        "zh_ok": (self._probe_voice(mac, True) if lang == "zh" else False)})
        return out

    @staticmethod
    def system_voices(lang=None):
        """列出 系统 真正 装 了 的 声音 (say -v ?)."""
        import subprocess as _sp
        try:
            out = _sp.run(["say", "-v", "?"], capture_output=True, timeout=20).stdout
        except Exception:
            return []
        res = []
        for line in out.decode("utf-8", "replace").splitlines():
            if "#" not in line:
                continue
            head = line.split("#")[0].strip()
            parts = head.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name, loc = parts[0].strip(), parts[1].strip()
            if lang and not loc.startswith(lang):
                continue
            res.append({"id": "mac:" + name, "label": "%s [%s]" % (name, loc),
                        "display": name, "locale": loc,
                        "lang": "zh" if loc.startswith("zh") else "en",
                        "backend": "macos-say"})
        return res


# ============================================================ 声音 合成 (sound)
class SoundEngine:
    """prompt -> 环境 音 (numpy 程序化 合成). 关键词 驱动 音色."""

    model_id = "ours/sound"

    def render(self, prompt, seconds=5.0, out_path=None, seed=None, steps=None):
        """steps: 闭源 契约 里 的 采样 步数 (默认 8) —— 这里 用 它 控 合成 的 细节 层数."""
        p = (prompt or "").lower()
        out_path = out_path or outputs.path_for("tts","mx_sound","wav")
        seed = seed if seed is not None else _h(prompt)
        n = int(max(0.5, min(float(seconds), 60.0)) * SR)
        rng = random.Random(seed)
        out = None
        table = (
            ("rain", self._rain), ("thunder", self._thunder),
            ("fire", self._fire), ("ocean", self._ocean),
            ("wave", self._ocean), ("water", self._water),
            ("drip", self._water), ("bird", self._bird),
            ("chirp", self._bird), ("cricket", self._cricket),
            ("wind", self._wind), ("breeze", self._wind),
            ("engine", self._engine), ("hum", self._engine),
            ("heartbeat", self._heartbeat), ("crowd", self._crowd),
            ("bell", self._bell), ("chime", self._bell),
            ("beep", self._tone), ("tone", self._tone),
            ("waveform", self._tone), ("whoosh", self._whoosh),
        )
        for key, fn in table:
            if key in p:
                out = fn(n, rng)
                break
        if out is None:
            out = self._generic(n, rng)
        m = max(1e-6, float(np.max(np.abs(out))))
        out = out / m * 0.85
        fade = int(0.05 * SR)
        if fade * 2 < len(out):
            out[:fade] *= np.linspace(0, 1, fade)
            out[-fade:] *= np.linspace(1, 0, fade)
        write_wav(out_path, out, SR)
        return out_path

    def _generic(self, n, rng):
        return _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 3000) * 0.6

    def _rain(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 6000)
        x += _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 1200) * 0.5
        for _ in range(n // (SR // 8)):
            i = rng.randrange(max(1, n - 1000))
            ln = rng.randint(200, 900)
            t = np.arange(ln) / SR
            d = np.sin(2 * math.pi * rng.uniform(1200, 3000) * t) * np.exp(-t * 18)
            x[i:i + ln] += d * 0.25
        return x

    def _wind(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 800)
        lfo = 0.5 + 0.5 * np.sin(2 * math.pi * np.arange(n) / SR * rng.uniform(0.05, 0.2))
        return x * lfo * 1.4

    def _ocean(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 1400)
        swell = 0.5 + 0.5 * np.sin(2 * math.pi * np.arange(n) / SR * rng.uniform(0.06, 0.14))
        return x * swell

    def _water(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 500) * 0.15
        for _ in range(n // (SR // 2)):
            i = rng.randrange(max(1, n - 4000))
            ln = 3000
            t = np.arange(ln) / SR
            f = rng.uniform(500, 1400)
            d = np.sin(2 * math.pi * f * t) * np.exp(-t * 6)
            x[i:i + ln] += d * 0.6
        return x

    def _thunder(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 120) * 2.5
        for _ in range(max(1, n // (SR * 4))):
            i = rng.randrange(max(1, n - SR * 2))
            ln = rng.randint(SR // 2, SR * 2)
            t = np.arange(ln) / SR
            rumble = _lowpass_fast(_noise(ln, rng.randint(0, 1 << 30)), 150) * np.exp(-t * 1.2)
            x[i:i + ln] += rumble * 1.2
        return x

    def _fire(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 900) * 0.4
        for _ in range(n // (SR // 20)):
            i = rng.randrange(max(1, n - 200))
            ln = rng.randint(40, 200)
            t = np.arange(ln) / SR
            d = _noise(ln, rng.randint(0, 1 << 30)) * np.exp(-t * 60)
            x[i:i + ln] += d * 0.8
        return x

    def _bird(self, n, rng):
        x = np.zeros(n, dtype=np.float32)
        for _ in range(max(3, n // (SR * 3))):
            i = rng.randrange(max(1, n - SR))
            f0 = rng.uniform(2200, 4200)
            ln = rng.randint(SR // 12, SR // 4)
            t = np.arange(ln) / SR
            sweep = f0 * (1 + 0.5 * np.sin(2 * math.pi * 6 * t))
            ph = 2 * math.pi * np.cumsum(sweep) / SR
            env = np.sin(np.pi * np.linspace(0, 1, ln)) ** 2
            x[i:i + ln] += np.sin(ph) * env * 0.5
        return x

    def _cricket(self, n, rng):
        x = np.zeros(n, dtype=np.float32)
        for _ in range(max(4, n // (SR * 2))):
            i = rng.randrange(max(1, n - 1000))
            ln = 260
            t = np.arange(ln) / SR
            d = np.sin(2 * math.pi * 4200 * t) * np.exp(-t * 40)
            x[i:i + ln] += d * 0.4
        return x

    def _engine(self, n, rng):
        t = np.arange(n) / SR
        x = 0.5 * np.sin(2 * math.pi * 60 * t) + 0.25 * np.sin(2 * math.pi * 120 * t)
        x += 0.2 * np.sin(2 * math.pi * 180 * t)
        x += _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 400) * 0.5
        return x

    def _heartbeat(self, n, rng):
        x = np.zeros(n, dtype=np.float32)
        bpm = rng.uniform(55, 80)
        per = 60.0 / bpm
        i = 0
        while i < n:
            for off, amp in ((0.0, 1.0), (0.22, 0.6)):
                s = int((i + off * per) * SR)
                ln = int(0.22 * SR)
                if s + ln >= n:
                    break
                t = np.arange(ln) / SR
                d = np.sin(2 * math.pi * 55 * t) * np.exp(-t * 16)
                x[s:s + ln] += d * amp
            i += int(per * SR)
        return x

    def _crowd(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 2000) * 0.5
        t = np.arange(n) / SR
        lfo = 0.6 + 0.4 * np.sin(2 * math.pi * 0.3 * t + rng.random())
        return x * lfo

    def _bell(self, n, rng):
        x = np.zeros(n, dtype=np.float32)
        base = rng.uniform(500, 900)
        for k, amp in ((1, 1.0), (2.76, 0.5), (5.4, 0.25), (8.9, 0.12)):
            t = np.arange(n) / SR
            d = np.sin(2 * math.pi * base * k * t) * np.exp(-t * (1.2 + k * 0.4))
            x += d * amp * 0.35
        return x

    def _tone(self, n, rng):
        f = rng.uniform(220, 880)
        t = np.arange(n) / SR
        return np.sin(2 * math.pi * f * t) * 0.8

    def _whoosh(self, n, rng):
        x = _lowpass_fast(_noise(n, rng.randint(0, 1 << 30)), 2500)
        env = np.sin(np.pi * np.linspace(0, 1, n)) ** 3
        return x * env * 1.2


# ============================================================ 音乐 合成 (兜底)
class MusicSynthEngine:
    """程序化 音乐 (numpy): 和弦 铺底 + 贝斯 + 琶音 + 鼓. ACE-Step 挂 了 时 兜底."""

    # 兜底 合成器, 不是 ACE-Step 本身 —— 不能 让 ace-step 这一 行 显示 loaded
    model_id = "ours/music-synth"

    CHORDS_MAJ = [(0, 4, 7), (5, 9, 12), (9, 12, 16), (7, 11, 14)]
    CHORDS_MIN = [(0, 3, 7), (8, 12, 15), (3, 7, 10), (10, 14, 17)]

    def render(self, prompt, seconds=20.0, out_path=None, lyrics="", seed=None):
        p = (prompt or "").lower()
        out_path = out_path or outputs.path_for("audio","mx_music_synth","wav")
        seed = seed if seed is not None else _h(prompt)
        rng = random.Random(seed)
        bpm = 100.0
        for tok in p.replace(",", " ").split():
            if tok.isdigit() and 40 <= int(tok) <= 220:
                bpm = float(tok)
        minor = any(k in p for k in ("minor", "sad", "dark", "melancholy"))
        chords = self.CHORDS_MIN if minor else self.CHORDS_MAJ
        keys = ("c", "d", "e", "f", "g", "a", "b")
        found = next((t for t in p.split() if t in keys), "c")
        root = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[found]
        sr = SR
        beat = 60.0 / bpm
        total = int(max(2.0, min(float(seconds), 120.0)) * sr)
        buf = np.zeros(total, dtype=np.float32)
        bar = int(4 * beat * sr)
        i = 0
        bar_i = 0
        while i < total:
            ch = chords[bar_i % len(chords)]
            ln = min(bar, total - i)
            seg = np.zeros(ln, dtype=np.float32)
            for semi in ch:
                f = 220.0 * (2 ** ((root + semi) / 12.0))
                t = np.arange(ln) / sr
                seg += np.sin(2 * math.pi * f * t + rng.random()) * 0.16
                seg += np.sin(2 * math.pi * f * 1.5 * t) * 0.05
            env = np.minimum(1.0, np.minimum(np.arange(ln) / (0.2 * sr),
                                             (ln - np.arange(ln)) / (0.2 * sr)))
            buf[i:i + ln] += seg * env * 0.7
            for b in range(4):
                s = i + int(b * beat * sr)
                if s >= total:
                    break
                ln2 = int(0.9 * beat * sr)
                t = np.arange(ln2) / sr
                f = 55.0 * (2 ** ((root + ch[0]) / 12.0))
                d = np.sin(2 * math.pi * f * t) * np.exp(-t * 3.0)
                end = min(total, s + ln2)
                buf[s:end] += d * 0.45
            step = beat / 2
            for k in range(8):
                s = i + int(k * step * sr)
                if s >= total:
                    break
                ln3 = int(0.35 * beat * sr)
                t = np.arange(ln3) / sr
                semi = ch[k % len(ch)] + (12 if k >= 4 else 0)
                f = 440.0 * (2 ** ((root + semi - 9) / 12.0))
                d = (np.sin(2 * math.pi * f * t) + 0.3 * np.sin(4 * math.pi * f * t)) * np.exp(-t * 9)
                end = min(total, s + ln3)
                buf[s:end] += d * 0.22
            for b in range(4):
                s = i + int(b * beat * sr)
                if s >= total:
                    break
                lnk = int(0.18 * sr)
                t = np.arange(lnk) / sr
                f = 120 * np.exp(-t * 28) + 45
                end = min(total, s + lnk)
                buf[s:end] += np.sin(2 * math.pi * np.cumsum(f) / sr) * np.exp(-t * 18) * 0.5
                if b % 2 == 1:
                    lns = int(0.12 * sr)
                    sn = _noise(lns, seed + b) * np.exp(-np.arange(lns) / sr * 30)
                    buf[s:min(total, s + lns)] += sn * 0.22
                for off in (0.0, 0.5):
                    hs = s + int(off * beat * sr)
                    if hs >= total:
                        continue
                    lnh = int(0.04 * sr)
                    hh = _noise(lnh, seed + b * 7 + int(off * 10)) * np.exp(-np.arange(lnh) / sr * 90)
                    buf[hs:min(total, hs + lnh)] += hh * 0.10
            i += ln
            bar_i += 1
        m = max(1e-6, float(np.max(np.abs(buf))))
        buf = buf / m * 0.9
        fade = int(0.05 * sr)
        if fade * 2 < len(buf):
            buf[:fade] *= np.linspace(0, 1, fade)
            buf[-fade:] *= np.linspace(1, 0, fade)
        write_wav(out_path, buf, sr)
        return out_path


# ============================================================ 3D (网格 -> GLB)
class ThreeDEngine:
    """程序化 网格 生成: prompt 里的 形状 关键词 -> 参数化 网格 -> GLB + PNG 预览."""

    model_id = "ours/3d"

    SHAPES = ("cube", "box", "sphere", "ball", "cylinder", "cone", "torus", "donut",
              "pyramid", "plane", "ground", "star", "vase", "bottle", "gear", "tree",
              "rock", "helix", "spiral")

    def build(self, prompt, out_dir=None, seed=None):
        p = (prompt or "").lower()
        seed = seed if seed is not None else _h(prompt)
        rng = random.Random(seed)
        shape = next((s for s in self.SHAPES if s in p), "sphere")
        verts, faces = self._mesh(shape, rng)
        base = outputs.base_for("3d","mx_3d")
        glb_path = base + ".glb"
        png_path = base + ".png"
        self._write_glb(glb_path, verts, faces)
        self._preview(png_path, verts, faces)
        return {
            "shape": shape,
            "glb": glb_path,
            "preview": png_path,
            "vertices": int(len(verts)),
            "triangles": int(len(faces)),
            "format": "glb",
        }

    def _mesh(self, shape, rng):
        if shape in ("cube", "box"):
            v = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                 (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                 (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
            return np.array(v, dtype=np.float32), f
        if shape == "sphere":
            return self._uv_sphere(rng)
        if shape == "cylinder":
            return self._cylinder(rng)
        if shape == "cone":
            return self._cone(rng)
        if shape in ("torus", "donut"):
            return self._torus(rng)
        if shape == "pyramid":
            v = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 1.4)]
            f = [(0, 1, 2), (0, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)]
            return np.array(v, dtype=np.float32), f
        if shape in ("plane", "ground"):
            v = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]
            return np.array(v, dtype=np.float32), [(0, 1, 2), (0, 2, 3)]
        if shape == "star":
            return self._star(rng)
        if shape in ("vase", "bottle"):
            return self._revolve(rng, "vase")
        if shape == "gear":
            return self._revolve(rng, "gear")
        if shape == "tree":
            return self._revolve(rng, "tree")
        if shape == "rock":
            v, f = self._uv_sphere(rng)
            v = v * (1 + rng.random() * 0.3)
            v = v * (1 + 0.18 * np.sin(v[:, 0] * 3.1 + v[:, 1] * 2.2))
            return v, f
        if shape in ("helix", "spiral"):
            return self._helix(rng)
        return self._uv_sphere(rng)

    def _uv_sphere(self, rng, seg=32, ring=16):
        verts, faces = [], []
        for i in range(ring + 1):
            phi = math.pi * i / ring
            for j in range(seg + 1):
                th = 2 * math.pi * j / seg
                verts.append((math.sin(phi) * math.cos(th),
                              math.sin(phi) * math.sin(th),
                              math.cos(phi)))
        for i in range(ring):
            for j in range(seg):
                a = i * (seg + 1) + j
                b = a + seg + 1
                faces.append((a, b, a + 1))
                faces.append((b, b + 1, a + 1))
        return np.array(verts, dtype=np.float32), faces

    def _cylinder(self, rng, seg=32):
        verts, faces = [], []
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            verts.append((math.cos(th), math.sin(th), -1.0))
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            verts.append((math.cos(th), math.sin(th), 1.0))
        for j in range(seg):
            faces.append((j, j + 1, seg + 1 + j))
            faces.append((seg + 1 + j, j + 1, seg + 2 + j))
        c0 = len(verts)
        verts.append((0, 0, -1.0))
        c1 = len(verts)
        verts.append((0, 0, 1.0))
        for j in range(seg):
            faces.append((c0, j + 1, j))
            faces.append((c1, seg + 1 + j, seg + 2 + j))
        return np.array(verts, dtype=np.float32), faces

    def _cone(self, rng, seg=32):
        verts, faces = [], []
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            verts.append((math.cos(th), math.sin(th), -1.0))
        apex = len(verts)
        verts.append((0, 0, 1.4))
        for j in range(seg):
            faces.append((j, j + 1, apex))
        c = len(verts)
        verts.append((0, 0, -1.0))
        for j in range(seg):
            faces.append((c, j + 1, j))
        return np.array(verts, dtype=np.float32), faces

    def _torus(self, rng, R=1.0, r=0.38, seg=40, ring=18):
        verts, faces = [], []
        for i in range(ring):
            u = 2 * math.pi * i / ring
            for j in range(seg):
                v = 2 * math.pi * j / seg
                x = (R + r * math.cos(v)) * math.cos(u)
                y = (R + r * math.cos(v)) * math.sin(u)
                z = r * math.sin(v)
                verts.append((x, y, z))
        for i in range(ring):
            for j in range(seg):
                a = i * seg + j
                b = i * seg + (j + 1) % seg
                c = ((i + 1) % ring) * seg + j
                d = ((i + 1) % ring) * seg + (j + 1) % seg
                faces.append((a, b, d))
                faces.append((a, d, c))
        return np.array(verts, dtype=np.float32), faces

    def _star(self, rng, points=5):
        verts = [(0, 0, 0)]
        for i in range(points * 2):
            th = math.pi / 2 + i * math.pi / points
            rad = 1.0 if i % 2 == 0 else 0.42
            verts.append((math.cos(th) * rad, math.sin(th) * rad, 0.08 * (i % 2)))
        faces = []
        for i in range(points * 2):
            faces.append((0, 1 + i, 1 + (i + 1) % (points * 2)))
        return np.array(verts, dtype=np.float32), faces

    def _revolve(self, rng, profile="vase", seg=36, rows=24):
        verts, faces = [], []
        for i in range(rows):
            t = i / (rows - 1)
            if profile == "vase":
                rad = 0.55 + 0.35 * math.sin(math.pi * t) - 0.25 * t
            elif profile == "gear":
                teeth = 12
                rad = 0.8 + 0.18 * (1 if int(t * teeth) % 2 == 0 else -1)
            else:
                rad = 0.9 * (1 - t) ** 1.6 + 0.06
            for j in range(seg):
                th = 2 * math.pi * j / seg
                verts.append((rad * math.cos(th), rad * math.sin(th), -1 + 2 * t))
        for i in range(rows - 1):
            for j in range(seg):
                a = i * seg + j
                b = i * seg + (j + 1) % seg
                c = (i + 1) * seg + j
                d = (i + 1) * seg + (j + 1) % seg
                faces.append((a, b, d))
                faces.append((a, d, c))
        return np.array(verts, dtype=np.float32), faces

    def _helix(self, rng, turns=4.0, steps=160, seg=10):
        verts, faces = [], []
        for i in range(steps):
            t = i / (steps - 1)
            th = 2 * math.pi * turns * t
            cx = 0.55 * math.cos(th)
            cy = 0.55 * math.sin(th)
            cz = -1 + 2 * t
            for j in range(seg):
                a = 2 * math.pi * j / seg
                verts.append((cx + 0.12 * math.cos(a), cy + 0.12 * math.sin(a), cz))
        for i in range(steps - 1):
            for j in range(seg):
                a = i * seg + j
                b = i * seg + (j + 1) % seg
                c = (i + 1) * seg + j
                d = (i + 1) * seg + (j + 1) % seg
                faces.append((a, b, d))
                faces.append((a, d, c))
        return np.array(verts, dtype=np.float32), faces

    def _normals(self, verts, faces):
        vn = np.zeros_like(verts, dtype=np.float32)
        for a, b, c in faces:
            p0, p1, p2 = verts[a], verts[b], verts[c]
            n = np.cross(p1 - p0, p2 - p0)
            for idx in (a, b, c):
                vn[idx] += n
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        return (vn / ln).astype(np.float32)

    def _write_glb(self, path, verts, faces):
        """最小 可用 glTF Binary: position / normal / index + 材质."""
        normals = self._normals(verts, faces)
        idx = np.array(faces, dtype=np.uint32).reshape(-1)
        pos = np.ascontiguousarray(verts, dtype=np.float32)
        nor = np.ascontiguousarray(normals, dtype=np.float32)
        ind = np.ascontiguousarray(idx, dtype=np.uint32)

        def pad(b, n=4):
            r = len(b) % n
            return b + (b"\x00" * (n - r) if r else b"")

        views = []
        bin_parts = []
        off = 0
        for arr, target in ((pos, 34962), (nor, 34962), (ind, 34963)):
            data = arr.tobytes()
            views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
            bin_parts.append(data)
            off += len(data)
        bin_data = pad(b"".join(bin_parts))
        gltf = {
            "asset": {"version": "2.0", "generator": "our-own-kernel cap_engines"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "generated"}],
            "meshes": [{"name": "generated", "primitives": [
                {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0, "mode": 4}]}],
            "materials": [{"name": "mat", "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.74, 0.78, 1.0], "metallicFactor": 0.1, "roughnessFactor": 0.6}}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": int(pos.shape[0]), "type": "VEC3",
                 "min": [float(x) for x in pos.min(axis=0)], "max": [float(x) for x in pos.max(axis=0)]},
                {"bufferView": 1, "componentType": 5126, "count": int(nor.shape[0]), "type": "VEC3"},
                {"bufferView": 2, "componentType": 5125, "count": int(ind.shape[0]), "type": "SCALAR"},
            ],
            "bufferViews": views,
            "buffers": [{"byteLength": len(bin_data)}],
        }
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js = pad(js, 4).replace(b"\x00", b" ")
        total = 12 + 8 + len(js) + 8 + len(bin_data)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(bin_data), 0x004E4942))
            f.write(bin_data)
        return path

    def _preview(self, path, verts, faces):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            fig = plt.figure(figsize=(4, 4), dpi=110)
            ax = fig.add_subplot(111, projection="3d")
            tris = [[verts[a], verts[b], verts[c]] for a, b, c in faces]
            coll = Poly3DCollection(tris, facecolor="#9fb3c8", edgecolor="#33415c", linewidths=0.25, alpha=0.95)
            ax.add_collection3d(coll)
            lim = float(np.max(np.abs(verts))) * 1.15
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            fig.savefig(path, facecolor="#101418")
            plt.close(fig)
        except Exception as e:
            print("[3d] 预览 图 失败: %s" % e, file=sys.stderr, flush=True)
        return path


# ============================================================ 图片 编辑 (PIL)
class ImageEditEngine:
    """Krea 本地 版 没有 img2img -> 用 PIL 经典 编辑 管线 兑现 /v1/images/edits."""

    # 本地 编辑 走 PIL 管线 (不是 Krea 权重), 别 让 krea-turbo 显示 loaded
    model_id = "ours/image-edit"

    COLORS = {
        "red": (1.0, 0.25, 0.25), "crimson": (1.0, 0.2, 0.3), "pink": (1.0, 0.6, 0.75),
        "orange": (1.0, 0.55, 0.15), "yellow": (1.0, 0.9, 0.2), "green": (0.3, 0.85, 0.35),
        "blue": (0.3, 0.5, 1.0), "purple": (0.6, 0.35, 0.95), "violet": (0.55, 0.35, 0.9),
        "cyan": (0.3, 0.9, 0.95), "teal": (0.2, 0.7, 0.7), "brown": (0.6, 0.4, 0.2),
        "gold": (0.95, 0.8, 0.25),
    }

    def edit(self, image, prompt, out_path=None, size=None):
        """image: PIL.Image / bytes / path. 返回 (path, applied_ops)."""
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        if isinstance(image, (bytes, bytearray)):
            import io
            image = Image.open(io.BytesIO(bytes(image)))
        elif isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        out_path = out_path or outputs.path_for("image","mx_edit","png")
        p = (prompt or "").lower()
        ops = []

        def has(*kw):
            return any(k in p for k in kw)

        if has("brighter", "brighten", "lighter", "light up", "提亮", "变亮"):
            image = ImageEnhance.Brightness(image).enhance(1.35)
            ops.append("brighten")
        if has("darker", "darken", "shadow", "变暗", "压暗"):
            image = ImageEnhance.Brightness(image).enhance(0.7)
            ops.append("darken")
        if has("saturat", "vivid", "colorful", "鲜艳", "饱和"):
            image = ImageEnhance.Color(image).enhance(1.5)
            ops.append("saturate")
        if has("desaturat", "muted", "faded", "灰暗"):
            image = ImageEnhance.Color(image).enhance(0.6)
            ops.append("desaturate")
        if has("contrast"):
            image = ImageEnhance.Contrast(image).enhance(1.4)
            ops.append("contrast")
        if has("sharp", "detail", "清晰", "锐化"):
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
            ops.append("sharpen")
        if has("blur", "soft focus", "模糊", "虚化"):
            image = image.filter(ImageFilter.GaussianBlur(radius=max(1, min(image.size) / 90)))
            ops.append("blur")
        if has("grayscale", "greyscale", "black and white", "monochrome", "黑白", "灰度"):
            image = ImageOps.grayscale(image).convert("RGB")
            ops.append("grayscale")
        if has("sepia", "vintage", "retro", "复古"):
            g = ImageOps.grayscale(image).convert("RGB")
            image = ImageOps.colorize(g, black="#2b1a0f", white="#f4e3c8")
            ops.append("sepia")
        if has("warmer", "warm", "sunset", "暖"):
            r, g, b = image.split()
            image = Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.12))),
                                        g, b.point(lambda v: int(v * 0.9))))
            ops.append("warm")
        if has("cooler", "cool", "cold", "winter", "冷"):
            r, g, b = image.split()
            image = Image.merge("RGB", (r.point(lambda v: int(v * 0.92)),
                                        g, b.point(lambda v: min(255, int(v * 1.12)))))
            ops.append("cool")
        if has("flip", "mirror", "镜像"):
            image = ImageOps.mirror(image)
            ops.append("flip")
        if has("rotate", "turn", "旋转"):
            image = image.rotate(-15, expand=True)
            ops.append("rotate")
        for name, rgb in self.COLORS.items():
            if name in p:
                tint = Image.new("RGB", image.size, tuple(int(c * 255) for c in rgb))
                image = Image.blend(image, tint, 0.35)
                ops.append("tint:" + name)
                break
        if has("upscale", "zoom in", "放大", "enlarge"):
            w, h = image.size
            image = image.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))).resize((w, h))
            ops.append("upscale")
        if has("downscale", "smaller", "缩小"):
            w, h = image.size
            image = image.resize((max(1, w // 2), max(1, h // 2)))
            ops.append("downscale")
        if size:
            try:
                sw, sh = [int(x) for x in str(size).lower().split("x")]
                image = image.resize((sw, sh))
                ops.append("resize:%dx%d" % (sw, sh))
            except Exception:
                pass
        if not ops:
            seed = _h(prompt)
            image = ImageEnhance.Contrast(image).enhance(1.08)
            r, g, b = image.split()
            image = Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.03 + (seed % 7) - 3))),
                                        g.point(lambda v: min(255, int(v * 1.02))),
                                        b.point(lambda v: min(255, int(v * 1.04 + (seed % 5) - 2)))))
            ops.append("style-shift")
        image.save(out_path, format="PNG")
        return out_path, ops


# ============================================================ 语音 识别 (STT)
class ASREngine:
    """faster-whisper large-v3-turbo (本机 HF 缓存, int8 CPU)."""

    model_id = "whisper-large-v3-turbo"

    def _snap(self):
        base = os.path.join(_H, ".cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots")
        if not os.path.isdir(base):
            # 「模型」页 下载 的 是 平铺 目录 (直接 model.bin), 不是 HF 缓存 的
            # snapshots/<sha>/ 布局 —— 两种 都 认.
            flat = os.path.join(_H, ".cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
            if os.path.isfile(os.path.join(flat, "model.bin")):
                return flat
            return ""
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            if os.path.isfile(os.path.join(p, "model.bin")):
                return p
        return ""

    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        snap = self._snap()
        if not snap:
            raise FileNotFoundError("faster-whisper large-v3-turbo 权重 不 在本机 缓存")
        self.model = WhisperModel(snap, device="cpu", compute_type="int8")

    def unload(self):
        self.model = None

    def transcribe(self, wav_path, language=None, beam_size=1):
        self.load()
        segs, info = self.model.transcribe(wav_path, language=language, beam_size=beam_size)
        items = [{"start": round(float(s.start), 2), "end": round(float(s.end), 2),
                  "text": s.text.strip()} for s in segs]
        return {"text": " ".join(i["text"] for i in items).strip(),
                "language": info.language,
                "duration": round(float(info.duration), 2),
                "segments": items}


# ============================================================ 决策 (decisions)
class DecisionEngine:
    """LLM 驱动 的 决策 引擎 (对标 闭源 的 laya/kev/d1/clef).

    问题 -> 模型 逐 题 回答 + 置信度 + 理由; 模型 没 给 置信度 时 用 启发式 兜底.
    """

    model_id = "ours/decisions"

    SYS = ("你是决策引擎. 对 每个 问题 给 出 结论, 并 给 出 置信度 (0..1) 与 简短 理由. "
           "只 输出 JSON, 形如 {\"q1\": {\"answer\": \"...\", \"confidence\": 0.82, "
           "\"reasoning\": \"...\"}}. 不要 输出 其它 文字.")

    def decide(self, llm, questions, state="", max_tokens=512):
        """llm: server.engines['llm'] (要 有 .generate). 返回 {qid: {...}}."""
        qids = list(questions.keys())
        body = "当前状态:\n%s\n\n问题:\n" % (state or "(无)")
        for qid in qids:
            body += "- [%s] %s\n" % (qid, questions[qid])
        body += "\n用 JSON 回答 上面 标号 的 问题 (键 就是 标号)."
        # 结构化 输出 必须 走 对话 通道 + 关 思考:
        # 走 raw completion 的话, 模型 会 先 吐 JSON 再 吐 思考 正文, 反而 抓 不 准.
        prompt = None
        chat_prompt = getattr(llm, "chat_prompt", None)
        if chat_prompt is not None:
            try:
                prompt = chat_prompt([{"role": "system", "content": self.SYS},
                                      {"role": "user", "content": body}],
                                     enable_thinking=False)
            except Exception:
                prompt = None
        if not prompt:
            prompt = self.SYS + "\n\n" + body
        out = llm.generate(prompt, max_tokens=max_tokens, think_budget=0)
        out = self._strip_think(out)
        obj = self._parse(out)
        res = {}
        for qid in qids:
            item = obj.get(qid) or {}
            if not isinstance(item, dict):
                item = {"answer": str(item)}
            ans = str(item.get("answer", "")).strip()
            try:
                conf = float(item.get("confidence"))
            except Exception:
                conf = None
            if conf is None:
                conf = self._heuristic(ans)
            res[qid] = {
                "answer": ans or "(未 回答)",
                "confidence": round(max(0.0, min(1.0, conf)), 3),
                "reasoning": str(item.get("reasoning", "")).strip(),
                "question": questions[qid],
            }
        return res

    @staticmethod
    def _strip_think(text):
        import re
        text = text or ""
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
        text = re.sub(r"<think>.*$", " ", text, flags=re.S)
        return text.strip()

    @staticmethod
    def _extract_json(text):
        """抓 第一个 能 解析 的 JSON 对象 (括号 配平, 认 引号 里 的 括号).

        不能 用 贪心 正则 \{.*\}: 模型 有时 先 给 JSON 再 给 其它 文字,
        贪心 会 把 中间 的 说明 也 吞 进去, 直接 解析 失败.
        """
        text = text or ""
        n = len(text)
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            depth = 0
            in_str = False
            esc = False
            for j in range(i, n):
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            o = json.loads(text[i:j + 1])
                        except Exception:
                            break
                        if isinstance(o, dict):
                            return o
                        break
        return {}

    def _parse(self, text):
        o = self._extract_json(text)
        if o:
            return o
        return {}

    def _heuristic(self, ans):
        if not ans:
            return 0.15
        c = 0.72
        low = ("不知道", "不确定", "无法", "unknown", "unclear", "可能", "maybe")
        for w in low:
            if w in ans.lower():
                c -= 0.2
                break
        if len(ans) > 120:
            c -= 0.05
        return max(0.05, min(0.95, c))
