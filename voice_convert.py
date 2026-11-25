"""voice_convert.py — 真 的 音色 迁移 (PSOLA 保共振峰变调 + 倒谱包络匹配).

之前 的 ref_audio 实现 有 两个 致命 问题:
  1. "WSOLA 时间伸缩 + 重采样" —— 重采样 会 把 **共振峰 一起 拉低**, 听着 像 变了 个 小孩,
     根本 不是 音色 迁移;
  2. "长时平均频谱 (LTAS) 包络 匹配" —— LTAS 里 混 着 基频 梳状 结构, 用 它 算 均衡 曲线
     会 和 谐波 打架, 结果 比 不处理 还 差.

这一 版:
  1. **YIN 基频跟踪** (累积均值归一化差分) —— 比 自相关 稳, 不 乱 跳 八度;
  2. **真 PSOLA 变调** —— 合成 脉冲 均匀 排布 在 新 周期 上, 但 每个 脉冲 都 取 自 原始
     波形 的 对应 位置, 所以 **共振峰 (音色) 完全 不动**, 只有 基频 变, 时长 也 不变;
  3. **倒谱共振峰包络匹配** —— 用 倒谱 滤掉 基频 梳状 结构, 只 留下 "声道 共振峰",
     用 它 和 参考 音频 的 共振峰 求 差 → 静态 均衡 滤波器 → 真正 的 音色 (音色 轮廓) 迁移.
"""

import numpy as np

_EPS = 1e-9


def _yin_f0_frame(x, sr, fmin=60.0, fmax=400.0, thr=0.15):
    """单 帧 的 基频 (YIN: 累积均值归一化 差分 函数). 返回 0 表示 静音/无声."""
    W = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(W - 1, int(sr / fmin))
    if tau_max <= tau_min:
        return 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * W)))
    X = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft)[:tau_max + 1]
    p = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    r0 = float(p[W])
    head = p[:tau_max + 1]                       # sum_{j<tau} x[j]^2
    tail = r0 - p[W - np.arange(tau_max + 1)]     # sum_{j>=W-tau} x[j]^2
    d = 2.0 * r0 - head - tail - 2.0 * ac
    d = np.maximum(d, 0.0)
    cm = np.ones(len(d))
    cs = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, len(d)) / np.maximum(cs, _EPS)
    cand = None
    for tau in range(tau_min, tau_max + 1):
        if cm[tau] < thr:
            while tau + 1 <= tau_max and cm[tau + 1] < cm[tau]:
                tau += 1
            cand = tau
            break
    if cand is None:
        cand = int(np.argmin(cm[tau_min:tau_max + 1])) + tau_min
    if cand <= 0 or cand >= len(cm):
        return 0.0
    shift = 0.0
    if 0 < cand < len(cm) - 1:
        a, b, c = cm[cand - 1], cm[cand], cm[cand + 1]
        den = a - 2.0 * b + c
        if abs(den) > 1e-12:
            shift = 0.5 * (a - c) / den
    tau = cand + shift
    if tau <= 0:
        return 0.0
    f = sr / tau
    return float(f) if fmin * 0.9 <= f <= fmax * 1.1 else 0.0


def _f0_track(x, sr, hop=0.010, win=0.040):
    """逐帧 基频 轨迹: 返回 (f0 数组, 每帧 中心 的 样本 位置)."""
    hl = max(1, int(sr * hop))
    fl = max(8, int(sr * win))
    n = len(x)
    f0s, pos = [], []
    i = 0
    while i + fl <= n:
        seg = x[i:i + fl].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg * seg)))
        f = 0.0
        if rms > 2e-4:
            f = _yin_f0_frame(seg - seg.mean(), sr)
        f0s.append(f)
        pos.append(i + fl / 2.0)
        i += hl
    if not f0s:
        return np.zeros(0), np.zeros(0)
    return np.asarray(f0s), np.asarray(pos)


def _global_f0(x, sr, fmin=60.0, fmax=400.0):
    """**主导 基频** (倒谱 法): 长时平均频谱 的 倒谱 峰 —— 比 逐帧 自相关 稳 得多.

    macOS `say` 的 激励 很 宽带 (谐波 梳 不 清晰), 逐帧 自相关 很容易 跳 八度
    (60 ↔ 120 ↔ 240); 倒谱 峰 在 这种 信号 上 稳 得 多.
    """
    try:
        S = _lts_mag(x, sr)
        L = np.log(S + 1e-9)
        c = np.fft.irfft(L)
        qmin = max(2, int(sr / fmax))
        qmax = min(len(c) - 2, int(sr / fmin))
        if qmax <= qmin + 1:
            return 0.0
        k = int(np.argmax(c[qmin:qmax])) + qmin
        if 0 < k < len(c) - 1:
            a, b, cc = c[k - 1], c[k], c[k + 1]
            den = a - 2 * b + cc
            sh = 0.5 * (a - cc) / den if abs(den) > 1e-18 else 0.0
        else:
            sh = 0.0
        return float(sr / max(1.0, (k + sh)))
    except Exception:
        return 0.0


def _lts_mag(x, sr, nfft=4096):
    """长时平均幅度频谱 (只 统计 能量 够 的 帧)."""
    hop = nfft // 4
    win = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, max(1, len(x) - nfft + 1), hop):
        s = x[i:i + nfft].astype(np.float64)
        if len(s) < nfft:
            break
        if np.sqrt(np.mean(s * s)) < 0.01:
            continue
        acc += np.abs(np.fft.rfft(s * win))
        cnt += 1
    if cnt == 0:
        s = np.asarray(x[:nfft], dtype=np.float64)
        return np.abs(np.fft.rfft(s * win))
    return acc / cnt


def _clean_f0(f0, sr, ref=None):
    """填 空洞 + 中值滤波 + 八度 纠错 (自相关 常见 的 2x/0.5x 跳变)."""
    f = np.asarray(f0, dtype=np.float64).copy()
    if len(f) == 0:
        return f
    voiced = f > 0
    if not voiced.any():
        return f
    idx = np.where(voiced)[0]
    # 线性插值 补 无声 洞
    f[~voiced] = np.interp(np.where(~voiced)[0], idx, f[idx])
    # 八度 纠错: 以 **全局 主导 基频** (倒谱 估 的, 稳) 为 基准, 把 每帧 归到 最近 的 八度
    lg = np.log2(np.maximum(f, 1e-6))
    if ref is None or ref <= 0:
        hist, edges = np.histogram(lg, bins=48, range=(3.0, 9.0))
        pk = int(np.argmax(hist))
        ref = 2.0 ** (0.5 * (edges[pk] + edges[pk + 1]))
    k = np.clip(np.round(np.log2(ref) - lg), -1, 1)
    f = f * (2.0 ** k)
    # 中值 滤波 (窗口 5) 去 抖动
    pad = np.pad(f, 2, mode="edge")
    f = np.median(np.stack([pad[i:i + len(f)] for i in range(5)]), axis=0)
    return f


def _pitch_marks(x, sr, f0_raw, f0c, pos):
    """音高脉冲 (epochs): 每个 周期 一个, 且 间距 要 严格 等 于 当前 周期.

    做法: 先 用 YIN 基频 得到 周期 曲线, 再 **顺着 信号 走** ——
    每 步 预测 下一个 脉冲 位置 (p + T), 在 附近 窗口 里 用 与 上一 周期 的
    归一化 互相关 挑 最 像 的 那个. 这样 脉冲 间距 不会 忽大忽小
    (之前 那版 "局部 峰值" 会 漏 掉 一半 脉冲, 间距 直接 翻倍).
    """
    n = len(x)
    if len(f0c) == 0 or n < 512:
        return np.zeros(0, dtype=np.int64)
    med = float(np.median(f0c[f0c > 0])) if (f0c > 0).any() else 120.0
    T_samp = sr / f0c
    # 周期 曲线 插值 到 样本
    T_at = np.interp(np.arange(n), pos, T_samp)
    # 有声 段 (原来 有基频 的 帧)
    voiced = (f0_raw > 0)
    if not voiced.any():
        return np.zeros(0, dtype=np.int64)
    vf = np.where(voiced)[0]
    # 起点: 第一 个 有声 帧 里 幅度 最大 的 点
    a = max(0, int(pos[vf[0]] - 0.020 * sr))
    end = int(min(n, pos[vf[-1]] + 0.5 * sr / med))
    seg = x[a:end]
    if len(seg) < 8:
        return np.zeros(0, dtype=np.int64)
    p = a + int(np.argmax(np.abs(seg)))
    marks = [p]
    guard = 0
    while guard < 200000:
        guard += 1
        T = float(T_at[min(p, n - 1)])
        if T < 4:
            break
        exp = p + T
        if exp >= n - 2:
            break
        w = int(max(2, T * 0.33))
        lo = max(1, int(exp - w))
        hi = min(n - 2, int(exp + w))
        if hi <= lo:
            break
        # 模板: 上一 周期
        hp = int(T / 2)
        tmpl = x[max(0, p - hp):max(0, p - hp) + 2 * hp]
        if len(tmpl) < 8:
            break
        tn = float(np.linalg.norm(tmpl)) + 1e-9
        best = -1
        best_sc = -1e9
        for q in range(lo, hi + 1):
            seg2 = x[max(0, q - hp):max(0, q - hp) + 2 * hp]
            if len(seg2) < len(tmpl):
                continue
            sc = float(np.dot(seg2, tmpl)) / ((np.linalg.norm(seg2) + 1e-9) * tn)
            if sc > best_sc:
                best_sc = sc
                best = q
        if best < 0:
            best = lo + int(np.argmax(np.abs(x[lo:hi + 1])))
        if best <= marks[-1]:
            break
        marks.append(best)
        p = best
    return np.asarray(marks, dtype=np.int64)


def psola_pitch_shift(x, sr, ratio):
    """真 PSOLA 变调: **共振峰 不动**, 只 改 基频, 时长 不变.

    TD-PSOLA: 合成 脉冲 均匀 排布 在 新 周期 T' = T/ratio 上, 但 每个 脉冲 的 波形
    都 取 自 源 信号 里 **最接近 的 那 个 分析脉冲** —— 所以 源 的 共振峰 (音色)
    原封不动 地 被 "拉长/压短" 成 新 的 基频, 这 正是 变调 不 变 音色 的 关键.
    (之前 用 的 "WSOLA + 重采样" 会 把 共振峰 一起 缩放, 变调 后 听着 完全 不像 原 人.)
    """
    x = np.asarray(x, dtype=np.float64)
    if abs(ratio - 1.0) < 5e-3 or len(x) < 2048:
        return x.astype(np.float32), sr
    f0, pos = _f0_track(x, sr)
    f0c = _clean_f0(f0, sr, _global_f0(x, sr))
    marks = _pitch_marks(x, sr, f0, f0c, pos)
    if len(marks) < 4:
        idx = np.clip(np.arange(len(x)) * float(ratio), 0, len(x) - 1)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32), sr
    T_at = np.interp(marks, pos, sr / np.maximum(f0c, 1e-6))
    M = len(marks)
    # 按 "静音 断点" 分 段: 段 与 段 之间 是 清辅音/停顿, 不 参与 合成
    segs = []
    st = 0
    for i in range(1, M):
        if marks[i] - marks[i - 1] > 2.5 * float(T_at[i - 1]):
            segs.append((st, i - 1))
            st = i
    segs.append((st, M - 1))
    pad = 4096
    out = np.zeros(len(x) + pad, dtype=np.float64)
    wsum = np.zeros_like(out)
    for (a, b) in segs:
        t = float(marks[a])
        limit = float(marks[b]) + float(T_at[b])
        k = 0
        while t < limit and k < 400000:
            k += 1
            j = int(np.searchsorted(marks, t))
            if j >= M:
                break
            if j > a and abs(marks[j] - t) >= abs(marks[j - 1] - t):
                j -= 1
            j = int(np.clip(j, a, b))
            Tsrc = float(T_at[j])
            Tnew = Tsrc / float(ratio)
            half = max(2, int(round(Tnew * 0.5)))
            s0 = int(marks[j]) - half
            e0 = int(marks[j]) + half
            o0 = int(round(t)) - half
            if s0 >= 0 and e0 <= len(x) and o0 >= 0 and o0 + (e0 - s0) <= len(out):
                seg = x[s0:e0]
                win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(len(seg)) / max(1, len(seg) - 1))
                out[o0:o0 + len(seg)] += seg * win
                wsum[o0:o0 + len(seg)] += win
            t += Tnew
    good = wsum > 1e-6
    out[good] /= wsum[good]
    out[~good] = 0.0
    y = out[:len(x)].copy()
    # 没 被 合成 的 区域 (清辅音 / 静音) —— 保留 原始 波形, 别 把 辅音 吃掉
    keep = ~good[:len(x)]
    y[keep] = x[keep]
    return y.astype(np.float32), sr


def _median_f0(x, sr, fmin=60.0, fmax=400.0, frame=0.04, hop=0.01):
    """旧 接口 (逐帧 自相关 估 基频) —— 保留 兼容, 新 代码 用 _f0_track."""
    n = len(x)
    fl = int(sr * frame)
    hl = int(sr * hop)
    if n < fl:
        return np.zeros(0), np.zeros(0)
    lo = max(2, int(sr / fmax))
    hi = min(fl - 1, int(sr / fmin))
    if hi <= lo:
        return np.zeros(0), np.zeros(0)
    f0s, cents = [], []
    for i in range(0, n - fl + 1, hl):
        seg = x[i:i + fl]
        seg = seg - seg.mean()
        if np.sqrt(np.mean(seg * seg)) < 1e-5:
            continue
        ac = np.correlate(seg, seg, mode="full")[fl - 1:]
        if len(ac) <= hi:
            continue
        k = int(np.argmax(ac[lo:hi])) + lo
        if ac[k] > 0.25 * ac[0]:
            f0s.append(sr / float(k))
            cents.append(i + fl / 2.0)
    return np.asarray(f0s), np.asarray(cents)


def wsola_time_stretch(x, sr, alpha, frame=1024, delta=256):
    """WSOLA 时间 伸缩: alpha>1 更长, alpha<1 更短. 音高 不变."""
    if alpha <= 0:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    out_len = int(n * alpha)
    hop = frame // 2
    win = np.hanning(frame).astype(np.float32)
    out = np.zeros(out_len + frame, dtype=np.float32)
    wsum = np.zeros_like(out)
    ana = 0
    syn = 0
    tmpl = x[hop:hop + hop].astype(np.float32)
    while syn + frame < out_len + frame and ana + frame + delta < n:
        lo = max(0, ana - delta)
        hi = min(n - frame - hop, ana + delta)
        best = ana
        best_sc = -1e9
        for cand in range(lo, hi + 1, 16):
            seg = x[cand:cand + hop]
            if len(seg) < hop:
                break
            sc = float(np.dot(seg, tmpl)) / (np.linalg.norm(seg) + 1e-9)
            if sc > best_sc:
                best_sc = sc
                best = cand
        seg = x[best:best + frame].astype(np.float32)
        if len(seg) < frame:
            break
        out[syn:syn + frame] += seg * win
        wsum[syn:syn + frame] += win
        tmpl = x[best + hop:best + hop + hop].astype(np.float32)
        syn += hop
        ana += hop
    good = wsum > 1e-6
    out[good] = out[good] / wsum[good]
    out[~good] = 0.0
    return out[:out_len]


def _ltas(x, sr, nfft=1024):
    """长时平均幅度频谱 (线性) —— 兼容 旧 接口."""
    hop = nfft // 4
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    f0, _ = _median_f0(x, sr)
    thr = float(np.median(f0)) * 0.9 if len(f0) else 0.0
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    win = np.hanning(nfft).astype(np.float32)
    for i in range(0, len(x) - nfft + 1, hop):
        seg = x[i:i + nfft]
        if thr > 0:
            sf0, _ = _median_f0(seg, sr)
            if len(sf0) == 0 or float(np.median(sf0)) < thr * 0.8:
                continue
        acc += np.abs(np.fft.rfft(seg * win))
        cnt += 1
    if cnt == 0:
        return np.abs(np.fft.rfft(x[:nfft] * win)), 0
    return acc / cnt, cnt


def _voiced_logspec(x, sr, nfft=512):
    """有声 帧 的 平均 对数幅度 频谱 (dB) —— 只 统计 有声 帧, 静音 不 污染."""
    hop = nfft // 4
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    win = np.hanning(nfft)
    acc = None
    cnt = 0
    f0, fpos = _f0_track(x, sr)
    f0c = _clean_f0(f0, sr)
    for i in range(0, len(x) - nfft + 1, hop):
        seg = x[i:i + nfft].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg * seg)))
        if rms < 3e-4:
            continue
        c = i + nfft / 2.0
        j = int(np.argmin(np.abs(fpos - c)))
        if len(f0c) == 0 or f0c[j] <= 0:
            continue
        S = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(seg * win)), 1e-7))
        acc = S if acc is None else acc + S
        cnt += 1
    if cnt == 0:
        seg = x[:nfft].astype(np.float64)
        return 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(seg * win)), 1e-7)), 0
    return acc / cnt, cnt


def _formant_env(x, sr, nfft=512):
    """倒谱 平滑 的 **共振峰包络** (dB): 把 基频 梳状 结构 滤掉, 只 留 "声道 形状".

    这是 音色 (timbre) 的 客观 度量: 两个 声音 共振峰 包络 越 接近, 音色 越 像.
    """
    L, cnt = _voiced_logspec(x, sr, nfft)
    c = np.fft.irfft(L, nfft)
    # 倒谱 倒置率 峰 在 T0 (基频 周期) 附近 —— 那 是 "脉冲 串", 不是 声道
    med = _global_f0(x, sr) or 120.0
    T0 = sr / max(1.0, med)
    qc = int(min(0.45 * T0, 0.0025 * sr))
    qc = max(8, min(qc, nfft // 2 - 2))
    c[qc + 1:nfft - qc - 1] = 0.0
    env = np.fft.rfft(c).real          # c 长度 = nfft → 输出 nfft//2+1 个 频点
    # 去掉 整体 电平 (响度 另 外 对齐) —— 只 留 "形状"
    fq = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (fq >= 300.0) & (fq <= 4000.0)
    lvl = float(np.mean(env[band])) if band.any() else float(np.mean(env))
    return env - lvl, sr


def _apply_env_eq(x, sr, g_db, nfft_src=512, max_db=12.0, lo=90.0, hi=7500.0):
    """把 包络 差 (dB) 变成 **零相位 静态 均衡** —— 直接 在 整段 信号 的 频谱 上 相乘."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(1024, n))))
    f_src = np.fft.rfftfreq(nfft_src, 1.0 / sr)
    f_dst = np.fft.rfftfreq(nfft, 1.0 / sr)
    g = np.interp(np.log(np.maximum(f_dst, 1e-6)),
                  np.log(np.maximum(f_src, 1e-6)), g_db)
    g = np.clip(g, -max_db, max_db)
    # 只 在 人耳 / 语音 有效 频段 内 补偿, 两端 平滑 归零
    g[f_dst < lo] = 0.0
    g[f_dst > hi] = 0.0
    ramp = 0.0025 * sr
    band = (f_dst >= lo) & (f_dst < lo + ramp)
    g[band] *= (f_dst[band] - lo) / ramp
    band2 = (f_dst > hi - ramp) & (f_dst <= hi)
    g[band2] *= (hi - f_dst[band2]) / ramp
    gain = 10.0 ** (g / 20.0)          # 长度 = nfft//2+1, 正好 是 rfft 的 输出 长度
    Y = np.fft.rfft(x.astype(np.float64), nfft) * gain
    y = np.fft.irfft(Y, nfft)[:n]
    return y.astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def _med_f0(x, sr):
    return float(_global_f0(x, sr))


def convert_voice(syn, syn_sr, ref, ref_sr):
    """把 syn 的 音色 迁到 ref 上. 返回 (wav 数组, 采样率).

    流程: PSOLA 变调 (基频 → 参考 的 基频, 共振峰 不动) → 倒谱共振峰包络匹配
    (把 声道 形状 拉向 参考) → 响度 对齐.
    """
    syn = np.asarray(syn, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if len(ref) < 256 or len(syn) < 256:
        return syn, syn_sr
    if ref_sr != syn_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(syn_sr), int(ref_sr))
        ref = resample_poly(ref, int(ref_sr) // g, int(syn_sr) // g).astype(np.float32)
        ref_sr = syn_sr
    fs = _med_f0(syn, syn_sr)
    fr = _med_f0(ref, syn_sr)
    if fs <= 0 or fr <= 0:
        return syn, syn_sr
    ratio = float(np.clip(fr / fs, 0.55, 1.8))
    out = syn
    if abs(ratio - 1.0) > 0.02:
        out, _ = psola_pitch_shift(out, syn_sr, ratio)
        got = _med_f0(out, syn_sr)
        if got > 0:
            err = fr / got
            if abs(err - 1.0) > 0.02:
                out, _ = psola_pitch_shift(out, syn_sr, float(np.clip(err, 0.9, 1.1)))
    try:
        env_out, _ = _formant_env(out, syn_sr)
        env_ref, _ = _formant_env(ref, syn_sr)
        out = _apply_env_eq(out, syn_sr, env_ref - env_out)
    except Exception:
        pass
    rs, rr = _rms(out), _rms(ref)
    if rs > 1e-6 and rr > 1e-6:
        out = out * float(np.clip(rr / rs, 0.25, 4.0))
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out.astype(np.float32), syn_sr
