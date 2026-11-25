"""出歌 时长 估算 —— 「自动时长」用 的 那 个 脑子.

为什么 需要: 出歌 引擎 的 时长 是 **输入** (要 多少 秒 就 渲染 多少 秒),
所以 手动 拖 滑块 有 两个 毛病:
  · 拖 短 了 -> 歌词 唱 不完, 结尾 被 硬切;
  · 拖 长 了 -> 词 唱 完 了 后面 全 是 器乐 垫子, 白 等 一 倍 时间.
这个 模块 按 **歌词 本身** 算出 「这首歌 大概要 多少 秒」:
  · 认 段落 标记 ([intro] [verse] [chorus] [bridge] [outro] ... 也 认 "Verse:")
  · 每段 按 文字 量 算 唱 的 时间 (中文 按 字, 英文 按 词)
  · 每行 + 换气 停顿, 每段 + 过渡, 每段 有 最小 时长
  · 纯音乐 (没 歌词 / 勾了 纯音乐) 按 段落 数 × 器乐 段 时长
  · 结果 向上 取整 到 5 秒, 再 clamp 到 [min_s, max_s]
max_s 必须 是 **引擎 真 上限** —— 不能 悄悄 截断 (截断了 歌 就 少 一截).

返回 (秒数, 依据 dict). 依据 里 有 段落 数 / 行数 / 字数 / 词数 —— 界面 要 拿 它
把 「为什么 是 这个 时长」 告诉 用户.
"""
import re

MIN_S = 10.0
DEFAULT_S = 60.0

# 段落 标记 -> (类别, 固定秒数 或 None)
#   类别: "vocal" 按 文字 算 / "intro" "outro" 器乐 段 / "inst" 纯器乐 段
_TAGS = {
    "intro": ("intro", 6.0), "verse": ("vocal", None), "pre-chorus": ("vocal", None),
    "prechorus": ("vocal", None), "pre chorus": ("vocal", None),
    "chorus": ("vocal", None), "hook": ("vocal", None), "refrain": ("vocal", None),
    "bridge": ("vocal", None), "verse2": ("vocal", None),
    "outro": ("outro", 6.0), "inst": ("inst", 8.0), "instrumental": ("inst", 8.0),
    "interlude": ("inst", 8.0), "break": ("inst", 6.0), "drop": ("inst", 8.0),
}
# 中文 常见 段落 写法 (有些 人 写 "[副歌]" 而不是 "[chorus]")
_TAGS_ZH = {
    "前奏": ("intro", 6.0), "主歌": ("vocal", None), "副歌": ("vocal", None),
    "预副歌": ("vocal", None), "桥段": ("vocal", None), "尾奏": ("outro", 6.0),
    "间奏": ("inst", 8.0), "器乐": ("inst", 8.0), "间奏段": ("inst", 8.0),
}

# 段落 标记 只 认 两种 写法:
#   带 括号:  [verse] / [Chorus 1] / （副歌）
#   带 冒号:  Verse: / 副歌:
# 而且 括号 里 的 词 必须 在 已知 名单 里 —— 否则 一 句 歌词 "[夜色沉进车窗]"
# 会被 误 当 成 段落 标记.
_SEC_BRACKET_RE = re.compile(r"^\s*[\[\(（]\s*([^\]\)）]{1,20}?)\s*[\]\)）]\s*$")
_SEC_COLON_RE = re.compile(r"^\s*([A-Za-z一-鿿][A-Za-z0-9 一-鿿_-]{0,18}?)\s*[:：]\s*$")
# 唱 的 速度 (含 一点 拖腔) —— 中文流行 约 3.5~4 字/秒, 英文 约 1.8~2 词/秒
ZH_CHARS_PER_SEC = 3.6
EN_WORDS_PER_SEC = 1.8
# 每 行 换气, 每 段 过渡
LINE_BREATH_S = 0.6
SECTION_GAP_S = 1.0
MIN_SECTION_S = 5.0
# 中日文 字符
_CJK = re.compile(r"[一-鿿぀-ヿ]")
# 英文 单词
_ENWORD = re.compile(r"[A-Za-z']+")


def _split_sections(lyrics):
    """歌词 -> [(段落标记 或 None, [行, ...]), ...]"""
    out = []
    cur_tag, cur = None, []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        tag = None
        m = _SEC_BRACKET_RE.match(line)
        if m:
            cand = m.group(1).strip().lower()
            if _known_tag(cand):
                tag = cand
        if tag is None:
            m = _SEC_COLON_RE.match(line)
            if m:
                cand = m.group(1).strip().lower()
                if _known_tag(cand):
                    tag = cand
        if tag is not None:
            if cur_tag is not None or cur:
                out.append((cur_tag, cur))
            cur_tag, cur = tag, []
            continue
        cur.append(line)
    if cur_tag is not None or cur:
        out.append((cur_tag, cur))
    return out


def _known_tag(t):
    """这个 词 像 不 像 段落 标记"""
    if not t:
        return False
    if t in _TAGS or t in _TAGS_ZH:
        return True
    base = re.sub(r"[\s_-]*\d+$", "", t).strip()
    return base in _TAGS or base in _TAGS_ZH


def _tag_kind(tag):
    """段落 标记 -> (类别, 固定秒数 或 None)"""
    if not tag:
        return ("vocal", None)
    t = tag.strip().lower()
    if t in _TAGS:
        return _TAGS[t]
    if t in _TAGS_ZH:
        return _TAGS_ZH[t]
    # "verse 2" / "chorus 1" 这种
    base = re.sub(r"[\s_-]*\d+$", "", t).strip()
    if base in _TAGS:
        return _TAGS[base]
    if base in _TAGS_ZH:
        return _TAGS_ZH[base]
    return ("vocal", None)


def _text_stats(lines):
    """一行 文字 的 唱 时长 + 字数/词数"""
    text = " ".join(l for l in lines if l)
    cjk = len(_CJK.findall(text))
    words = len(_ENWORD.findall(text))
    sec = cjk / ZH_CHARS_PER_SEC + words / EN_WORDS_PER_SEC
    voiced = sum(1 for l in lines if l.strip())
    sec += voiced * LINE_BREATH_S
    return sec, cjk, words


def estimate(lyrics="", instrumental=False, bpm=None, min_s=MIN_S, max_s=240.0,
             default_s=DEFAULT_S):
    """算 一首歌 该 多长.

    返回 (seconds, 依据). lyrics 空 且 非 纯音乐 -> 用 default_s.
    """
    lyrics = (lyrics or "").strip()
    if not lyrics and not instrumental:
        return float(default_s), {"sections": 0, "lines": 0, "chars": 0, "words": 0,
                                 "lang": "-", "basis": "没 歌词, 用 默认"}
    sections = _split_sections(lyrics)
    if instrumental:
        # 纯音乐: 每 段 一 个 器乐 块
        n = max(1, len(sections))
        total = n * 8.0
        base = {"sections": n, "lines": 0, "chars": 0, "words": 0, "lang": "inst",
                "basis": "纯音乐 · %d 段 × 8 秒" % n}
    else:
        total = 0.0
        n_lines = 0
        n_cjk = 0
        n_words = 0
        for tag, lines in sections:
            kind, fixed = _tag_kind(tag)
            voiced = [l for l in lines if l.strip()]
            if kind == "inst":
                sec = fixed
            elif kind in ("intro", "outro") and not voiced:
                sec = fixed
            elif not voiced:
                continue          # 空 段落 (只有 标记) 不 算
            else:
                sec, c, w = _text_stats(voiced)
                sec = max(MIN_SECTION_S, sec)
                n_lines += len(voiced)
                n_cjk += c
                n_words += w
            total += sec + SECTION_GAP_S
        if total <= 0:
            return float(default_s), {"sections": len(sections), "lines": 0, "chars": 0,
                                      "words": 0, "lang": "-", "basis": "没 歌词, 用 默认"}
        lang = "zh" if n_cjk >= max(1, n_words) else "en"
        base = {"sections": len(sections), "lines": n_lines, "chars": n_cjk,
                "words": n_words, "lang": lang,
                "basis": "%d 段 · %d 行 · %d 字" % (len(sections), n_lines, n_cjk)}
    # 向上 取整 到 5 秒 —— 别 给 出 「97.3 秒」 这种 假 精确
    total = float(int(total + 4.999) // 5 * 5)
    if total < min_s:
        total = float(min_s)
        base["capped"] = "太 短, 按 最小 %d 秒" % int(min_s)
    if total > max_s:
        total = float(max_s)
        base["capped"] = "超过 引擎 上限 %d 秒, 已 按 上限" % int(max_s)
    base["seconds"] = total
    return total, base


def explain(base):
    """依据 dict -> 一句 人话 (界面 显示 用)"""
    if not base:
        return ""
    bits = [b for b in (base.get("basis"),) if b]
    if base.get("capped"):
        bits.append(base["capped"])
    return " · ".join(bits)
