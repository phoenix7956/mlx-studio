"""ui.py — 我们 自研 内核 的 内置 控制台 ("MLX Serve Studio" 复刻).

布局 / 配色 / 文案 照 闭源 的 Web 控制台 (MECHANICS: 220px 侧栏 / 44px 顶栏 /
8 张 指标 卡 / #007aff 强调色), 但 数据 全部 打 我们 自己的 接口.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def _find_html():
    """控制台 HTML 在 哪 —— 三 种 装法 都 要 找得到:
       ① 源码 / App bundle:  <HERE>/ui/console.html
       ② pip 装 (wheel):     <prefix>/share/mlx-studio/ui/console.html
       ③ 开发 态 直跑:        <repo>/ui/console.html
    """
    cands = [
        os.path.join(HERE, "ui", "console.html"),
        os.path.join(os.path.dirname(HERE), "ui", "console.html"),
        os.path.join(sys.prefix, "share", "mlx-studio", "ui", "console.html"),
        os.path.join("/usr/local/share/mlx-studio/ui/console.html"),
        os.path.join("/opt/homebrew/share/mlx-studio/ui/console.html"),
    ]
    for c in cands:
        try:
            if os.path.isfile(c): return c
        except Exception:
            pass
    return cands[0]

HTML = _find_html()

FALLBACK = """<!doctype html><meta charset="utf-8"><title>MLX Serve Studio</title>
<body style="font-family:system-ui;padding:40px">控制台 文件 缺失: ui/console.html</body>"""


def console_html():
    try:
        with io.open(HTML, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return FALLBACK
