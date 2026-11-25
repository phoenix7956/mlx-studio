"""魔搭 (modelscope) 下载器 —— 国内 最快, 优先 走 这 条.

为什么 单独 一个 文件:
  server.py 已经 很 大 了, 魔搭 下载 的 细节 (看门狗 / 重试 / 找 python 环境)
  单独 放 这 里, server.py 只 调 一 层.

为什么 魔搭 比 HF 快:
  国内 直连 huggingface.co 常常 卡死, 走 hf-mirror 才 稳; 魔搭 是 国内 的,
  本来 就 在 国内 骨干 网上, 拉 几 十个 G 的 权重 明显 快 一 截.
  魔搭 上 基本 都 有 —— 连 mlx-community 的 量化 版 都 镜像 了.
"""

import os
import subprocess
import sys
import threading
import time


def has_ms(py):
    """这个 python 装 了 modelscope 吗."""
    try:
        r = subprocess.run([py, "-c", "import modelscope"],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def _candidates():
    home = os.path.expanduser("~")
    return [
        os.path.join(home, ".venvs", "mlxstudio", "bin", "python"),
        os.path.join(home, ".venvs", "mtx", "bin", "python"),
        os.path.join(home, ".venvs", "h3m3", "bin", "python"),
        os.path.join(home, ".venvs", "h3", "bin", "python"),
    ]


def _dir_size(d):
    total = 0
    try:
        for dp, dn, fn in os.walk(d):
            for f in fn:
                try:
                    total += os.path.getsize(os.path.join(dp, f))
                except Exception:
                    pass
    except Exception:
        pass
    return total


def ms_download(repo, target, cb=None, tries=2):
    """从 魔搭 拉 一个 仓库 到 target 目录.

    返回 (ok: bool, msg: str). cb 是 进度 回调 (给 前端 显示).
    """
    target = os.path.expanduser(target)
    os.makedirs(target, exist_ok=True)
    code = (
        "import sys\n"
        "from modelscope import snapshot_download\n"
        "print(snapshot_download(sys.argv[1], local_dir=sys.argv[2]))\n"
    )
    py = sys.executable
    if not has_ms(py):
        for c in _candidates():
            if os.path.exists(c) and has_ms(c):
                py = c
                break
    import queue
    last = ""
    for attempt in range(1, max(1, tries) + 1):
        try:
            p = subprocess.Popen([py, "-c", code, repo, target],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        except Exception as e:
            return False, "起 魔搭 下载 进程 失败: %s" % e
        q = queue.Queue()

        def _reader(_p=p, _q=q):
            try:
                for line in _p.stdout:
                    _q.put(line)
            except Exception:
                pass

        threading.Thread(target=_reader, daemon=True).start()
        _last_size = -1
        _stall = 0
        while p.poll() is None:
            try:
                line = q.get(timeout=5)
                line = (line or "").strip()
                if line:
                    last = line
                    if cb and ("%" in line or "Downloading" in line or "it/s" in line):
                        cb(line[-110:])
                continue
            except queue.Empty:
                pass
            # 看门狗: 2 分钟 没 多 一个 字节 -> 掐 掉 重来 (断点 续传)
            _sz = _dir_size(target)
            if _sz > _last_size:
                _last_size = _sz
                _stall = 0
            else:
                _stall += 1
                if _stall >= 24:
                    print("[download] 魔搭 卡住 2 分钟, 重启 (第 %d/%d 次)"
                          % (attempt, tries), file=sys.stderr, flush=True)
                    try:
                        p.kill()
                    except Exception:
                        pass
                    break
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        if p.returncode == 0:
            return True, "魔搭 下载 完成 -> %s" % target
        if attempt >= max(1, tries):
            break
        print("[download] 魔搭 第 %d 次 没 成 (rc=%s), 再试"
              % (attempt, p.returncode), file=sys.stderr, flush=True)
        if cb:
            cb("魔搭 不顺, 重试 %d/%d" % (attempt + 1, tries))
        time.sleep(3)
    return False, (last[-300:] or "魔搭 下载 失败")


def has_repo(repo):
    """魔搭 上 有 没有 这个 repo —— 用来 决定 走 魔搭 还是 走 HF."""
    try:
        import json
        import urllib.request
        url = "https://modelscope.cn/api/v1/models/%s" % repo
        req = urllib.request.Request(url, headers={"User-Agent": "MLX-Studio"})
        with urllib.request.urlopen(req, timeout=12) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        return bool((d.get("Data") or {}).get("Name"))
    except Exception:
        return False
