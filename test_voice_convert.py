"""test_voice_convert.py — 音色迁移 的 自检 (可 独立 运行).

    HOME="$HOME" ~/.venvs/mtx/bin/python test_voice_convert.py

检查 四 件事:
  1. 基频 对齐: 输出 的 主导 基频 跟 参考 一致 (±8%);
  2. 共振峰 迁移: 输出 的 F1/F2/F3 落到 参考 的 位置 (误差 小);
  3. 时长 不变;
  4. 音节 节奏 保留 (短时能量 包络 相关).
"""

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_convert as vc  # noqa: E402

SR = 22050
TXT = "Hello there, this is a test of the voice conversion system."


def rd(p):
    with wave.open(p, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def formants(x, sr=SR):
    e, _ = vc._formant_env(x, sr)
    f = np.fft.rfftfreq(512, 1.0 / sr)
    out = []
    for lo, hi in ((250, 900), (800, 2400), (2000, 3600)):
        m = (f >= lo) & (f <= hi)
        idx = np.where(m)[0]
        out.append(float(f[idx[np.argmax(e[idx])]]))
    return out


def energy_env(x, sr=SR, hop=256):
    n = len(x) // hop
    e = np.array([float(np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2))) for i in range(n)])
    return e - e.min()


def main():
    import cap_engines
    eng = cap_engines.TTSEngine()
    eng.load()
    pairs = [("af_heart", "am_michael"), ("af_bella", "am_fenrir"),
             ("af_sky", "am_adam"), ("am_michael", "am_fenrir")]
    fails = 0
    for sv, rv in pairs:
        src, sr = rd(eng.speak(TXT, voice=sv, lang="en"))
        ref, rsr = rd(eng.speak(TXT, voice=rv, lang="en"))
        out, osr = vc.convert_voice(src, sr, ref, rsr)
        f0s = vc._med_f0(src, sr)
        f0r = vc._med_f0(ref, rsr)
        f0o = vc._med_f0(out, osr)
        fr = formants(ref, rsr)
        fo = formants(out, osr)
        ferr = [abs(fo[i] - fr[i]) for i in range(3)]
        a = energy_env(src, sr)
        b = energy_env(out, osr)
        m = min(len(a), len(b))
        corr = float(np.corrcoef(a[:m], b[:m])[0, 1]) if m > 4 else 0.0
        ok_f0 = abs(f0o - f0r) / max(1.0, f0r) < 0.08
        ok_f = max(ferr) < 120.0
        ok_dur = abs(len(out) / osr - len(src) / sr) < 0.05
        ok_env = corr > 0.75
        good = ok_f0 and ok_f and ok_dur and ok_env
        fails += 0 if good else 1
        print("%-9s -> %-9s F0 %.0f->%.0f (目标 %.0f) | F %s -> %s (误差 %s Hz) | "
              "时长 %.2f/%.2f | 节奏相关 %.2f | %s" % (
                  sv, rv, f0s, f0o, f0r,
                  ["%.0f" % v for v in formants(src, sr)],
                  ["%.0f" % v for v in fo], ["%.0f" % v for v in ferr],
                  len(out) / osr, len(src) / sr, corr,
                  "OK" if good else "FAIL"))
    print("---- %d/%d 通过" % (len(pairs) - fails, len(pairs)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
