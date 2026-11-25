#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""我们 自研 的 出歌 引擎 (纯 MLX) —— 不 再 依赖 闭源 MLX-Serve.

之前 秒级 出歌 是 打 闭源 /Applications/MLX-Serve.app 的 /v1/audio/music-generations.
那个 App 一 删, 出歌 就 只 剩 PyTorch 那 条 14 分钟/首 的 兜底. 现在 我们 自己 跑:

  ACE-Step 1.5 (MIT)     ~/.ours/ace-step-engine/   权重 ~/.ours/models/ace-step-mlx/
  MiniMax Music3 4bit    mlx-audio (MIT)             权重 ~/.ours/models/MiniMax-Music3-4bit/
  两 个 权重 都 是 走 魔搭 (modelscope) 拉 的 —— 不 走 huggingface.

协议: JSON-lines —— stdin 收 请求, stdout 出 事件 (ready / phase / done / error).

内存 规矩 (跟 用户 要 的 一致):
  · 出歌 引擎 常驻, 连续 出歌 不 重载 (快)
  · 30 分钟 没 用 -> 自动 卸 进程 (放 掉 ~12G), 下次 用 再 起 (~22 秒)
  · 一次 只 挂 一个 出歌 引擎 —— 换 模型 = 停 旧的, 起 新的
"""
import json
import os
import queue
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


HOME_DIR = _home()
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.environ.get("ACE_MLX_ENGINE_DIR", os.path.join(HOME_DIR, ".ours/ace-step-engine"))
PYTHON = os.environ.get("ACE_MLX_PYTHON", os.path.join(HOME_DIR, ".venvs/ace/bin/python"))
CHECKPOINTS = os.environ.get("ACE_MLX_CHECKPOINTS", os.path.join(HOME_DIR, ".ours/models/ace-step-mlx"))
SONGS_DIR = os.environ.get("ACE_MLX_SONGS", os.path.join(HOME_DIR, ".ours/ace-songs"))
PROJECT_DIR = os.path.join(ENGINE_DIR, "project")

# —— MiniMax Music3 ——
MM3_DIR = os.environ.get("MM3_MODEL_DIR", os.path.join(HOME_DIR, ".ours/models/MiniMax-Music3-4bit"))
MM3_PYTHON = os.environ.get("MM3_PYTHON", os.path.join(HOME_DIR, ".venvs/mtx/bin/python"))
def _mm3_service_path():
    """出歌 引擎 (Music3 服务端) 在 哪 —— 源码/bundle 在 旁边, pip 装 在 share 里."""
    cands=[os.environ.get("MM3_SERVICE") or "",
           os.path.join(SELF_DIR,"engines/minimax_music3_service.py"),
           os.path.join(os.path.dirname(SELF_DIR),"engines/minimax_music3_service.py")]
    for c in (os.path.join(sys.prefix,"share","mlx-studio","engines/minimax_music3_service.py"),
              "/usr/local/share/mlx-studio/engines/minimax_music3_service.py",
              "/opt/homebrew/share/mlx-studio/engines/minimax_music3_service.py",
              os.path.expanduser("~/.ours/mlx_studio/engines/minimax_music3_service.py")):
        cands.append(c)
    for c in cands:
        if c and os.path.isfile(c): return c
    return cands[1]

MM3_SERVICE = _mm3_service_path()
MM3_SONGS_DIR = os.environ.get("MM3_SONGS", os.path.join(HOME_DIR, ".ours/music3-songs"))

# 30 分钟 没 用 就 自动 卸载 (跟 用户 要 的 "模型 30 分钟 没 用 就 自动 卸载" 一致)
IDLE_UNLOAD_SECONDS = int(os.environ.get("ACE_MLX_IDLE_SECONDS", "1800"))
# 起 引擎 要 多久 (模型 加载 ~22 秒 + 首次 编译)
START_TIMEOUT = float(os.environ.get("ACE_MLX_START_TIMEOUT", "300"))


# ————————————————————————————————————————————————————————————
# 引擎 登记表 —— 一个 id 一个 跑法
# ————————————————————————————————————————————————————————————
def _ace_available():
    return (os.path.isfile(os.path.join(PROJECT_DIR, "checkpoints", "acestep-v15-turbo", "model.safetensors"))
            and os.path.isdir(os.path.join(ENGINE_DIR, "acestep_mlx"))
            and os.path.isfile(PYTHON)
            and os.path.isfile(os.path.join(CHECKPOINTS, ".acestep-mlx",
                                            "acestep-5Hz-lm-0.6B-148d8ea0225bdab342ee1ae3a354275ccd60ca80.ok")))


def _ace_suno_available(dit="acestep-suno-e1"):
    """自训 LoRA 合并出来的权重（MLX 不支持运行时插 LoRA，只能整份换权重）。"""
    return (os.path.isfile(os.path.join(PROJECT_DIR, "checkpoints", dit, "model.safetensors"))
            and os.path.isdir(os.path.join(ENGINE_DIR, "acestep_mlx"))
            and os.path.isfile(PYTHON)
            and os.path.isfile(os.path.join(CHECKPOINTS, ".acestep-mlx",
                                            "acestep-5Hz-lm-0.6B-148d8ea0225bdab342ee1ae3a354275ccd60ca80.ok")))


def _mm3_available():
    return (os.path.isfile(os.path.join(MM3_DIR, "config.json"))
            and os.path.isfile(MM3_PYTHON)
            and os.path.isfile(MM3_SERVICE))


ENGINES = {
    "ours/ace-step-v15-turbo-mlx": {
        "id": "ours/ace-step-v15-turbo-mlx",
        "name": "ACE-Step 1.5 XL Turbo (MLX 自研)",
        "kind": "acestep",
        "python": lambda: PYTHON,
        "cwd": lambda: ENGINE_DIR,
        "env": lambda: {"PYTHONPATH": ENGINE_DIR, "ACESTEP_PROJECT": PROJECT_DIR},
        "cmd": lambda: [PYTHON, "-m", "acestep_mlx.engine_service",
                        "--project", PROJECT_DIR, "--out", SONGS_DIR, "--preset", "mini",
                        "--max-duration", str(MAX_DURATION)],
        "songs": lambda: SONGS_DIR,
        "available": _ace_available,
    },
    "ours/ace-step-suno-mlx": {
        "id": "ours/ace-step-suno-mlx",
        "name": "ACE-Step 1.5 XL Turbo + Suno 自训 LoRA (epoch1, 本机训练)",
        "kind": "acestep",
        "python": lambda: PYTHON,
        "cwd": lambda: ENGINE_DIR,
        "env": lambda: {"PYTHONPATH": ENGINE_DIR, "ACESTEP_PROJECT": PROJECT_DIR,
                        "ACE_DIT_MODEL": "acestep-suno-e1"},
        "cmd": lambda: [PYTHON, "-m", "acestep_mlx.engine_service",
                        "--project", PROJECT_DIR, "--out", SONGS_DIR, "--preset", "mini",
                        "--max-duration", str(MAX_DURATION)],
        "songs": lambda: SONGS_DIR,
        "available": lambda: _ace_suno_available("acestep-suno-e1"),
    },
    "ours/ace-step-suno-e2-mlx": {
        "id": "ours/ace-step-suno-e2-mlx",
        "name": "ACE-Step 1.5 XL Turbo + Suno 自训 LoRA (epoch2 对照)",
        "kind": "acestep",
        "python": lambda: PYTHON,
        "cwd": lambda: ENGINE_DIR,
        "env": lambda: {"PYTHONPATH": ENGINE_DIR, "ACESTEP_PROJECT": PROJECT_DIR,
                        "ACE_DIT_MODEL": "acestep-suno-e2"},
        "cmd": lambda: [PYTHON, "-m", "acestep_mlx.engine_service",
                        "--project", PROJECT_DIR, "--out", SONGS_DIR, "--preset", "mini",
                        "--max-duration", str(MAX_DURATION)],
        "songs": lambda: SONGS_DIR,
        "available": lambda: _ace_suno_available("acestep-suno-e2"),
    },
    "ours/minimax-music3-4bit": {
        "id": "ours/minimax-music3-4bit",
        "name": "MiniMax Music3 4bit (MLX 自研)",
        "kind": "minimax",
        "python": lambda: MM3_PYTHON,
        "cwd": lambda: SELF_DIR,
        "env": lambda: {},
        "cmd": lambda: [MM3_PYTHON, MM3_SERVICE, "--model", MM3_DIR, "--out", MM3_SONGS_DIR],
        "songs": lambda: MM3_SONGS_DIR,
        "available": _mm3_available,
    },
}

DEFAULT_MODEL_ID = os.environ.get("MUSIC_MLX_DEFAULT", "ours/ace-step-v15-turbo-mlx")

# 一 首 歌 最长 多少 秒 —— 引擎 的 硬 上限.
# 配方表 (recipe.py) 的 mini preset 已经 抬 到 480 (本机 128G M5 Max 实测 能 出),
# 这里 只是 "环境变量 再 覆盖 一 层" —— 没 设 环境变量 时 也 是 480, 不 会 掉 回 240.
MAX_DURATION = float(os.environ.get("ACE_MAX_DURATION", "480") or 480)


def max_duration(model_id=None):
    """引擎 真 上限 (秒). 界面 滑块 要 按 这个 画 —— 别 画 到 600 再 被 引擎 截断."""
    sp = _spec(model_id)
    if sp and sp.get("kind") == "minimax":
        return MAX_DURATION
    return MAX_DURATION

# 兼容 老 代码 用 的 常量
MODEL_ID = DEFAULT_MODEL_ID
MODEL_NAME = ENGINES[DEFAULT_MODEL_ID]["name"]

_lock = threading.RLock()
_proc = None
_events = None          # queue.Queue —— 引擎 事件
_active = None          # 当前 跑 着 的 引擎 id
_state = {"running": False, "load_s": None, "started_at": None,
          "last_used": None, "last_error": None, "last_song": None}
_idle_timer = None


def _log(*a):
    print("[music-mlx]", *a, file=os.sys.stderr, flush=True)


def _spec(model_id=None):
    """拿 引擎 配置 —— 认 短 名 / 全 id, 认 不 出 就 回 默认."""
    mid = (model_id or "").strip()
    if mid in ENGINES:
        return ENGINES[mid]
    # "minimax-music3" / "ace-step" 这种 短 名
    for k, v in ENGINES.items():
        tail = k.split("/")[-1].lower()
        if mid and (tail == mid.lower() or tail.startswith(mid.lower() + "-")
                    or mid.lower() in tail):
            return v
    return ENGINES.get(DEFAULT_MODEL_ID)


def available(model_id=None):
    """引擎 能不能 起来 (代码 / 权重 / 解释器 都 在).

    不 给 model_id -> 只要 有 任一 个 引擎 可用 就 True.
    """
    if model_id:
        sp = _spec(model_id)
        ok = bool(sp) and sp["available"]()
        if not ok:
            _state["last_error"] = "引擎 组件 不 全 (代码/权重/解释器)"
        return ok
    for sp in ENGINES.values():
        try:
            if sp["available"]():
                return True
        except Exception:
            continue
    _state["last_error"] = "引擎 组件 不 全 (代码/权重/解释器)"
    return False


def running():
    """引擎 进程 现在 在 不 在."""
    with _lock:
        return bool(_proc and _proc.poll() is None)


def status():
    """给 /v1/music/health 用 的 状态."""
    with _lock:
        act = _active
    sp = ENGINES.get(act) if act else None
    return {
        "available": available(),
        "running": running(),
        "model": act or DEFAULT_MODEL_ID,
        "name": (sp["name"] if sp else ENGINES[DEFAULT_MODEL_ID]["name"]),
        "load_s": _state.get("load_s"),
        "started_at": _state.get("started_at"),
        "last_used": _state.get("last_used"),
        "last_song": _state.get("last_song"),
        "last_error": _state.get("last_error"),
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "engine": ("ours-" + sp["kind"]) if sp else "ours-acestep-mlx",
        "models": [{"id": k, "name": v["name"], "available": v["available"]()}
                   for k, v in ENGINES.items()],
    }


def _reset_idle_timer():
    """重置 空闲 计时 —— 30 分钟 没 用 就 停 进程."""
    global _idle_timer
    with _lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        if IDLE_UNLOAD_SECONDS > 0:
            _idle_timer = threading.Timer(IDLE_UNLOAD_SECONDS, _idle_unload)
            _idle_timer.daemon = True
            _idle_timer.start()


def _idle_unload():
    """空闲 到 点 -> 停 引擎 进程 (放 内存)."""
    with _lock:
        if _proc is None:
            return
        idle = time.time() - (_state.get("last_used") or 0)
        if idle < IDLE_UNLOAD_SECONDS:
            return
    _log("空闲 %d 秒, 自动 卸载 出歌 引擎" % int(idle))
    stop()


def _reader():
    """读 引擎 的 stdout (JSON-lines), 把 事件 塞 进 队列."""
    global _proc
    proc = _proc
    try:
        for line in proc.stdout:
            line = (line or "").strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            _events.put(ev)
    except Exception:
        pass
    finally:
        if proc is not None and proc.poll() is not None:
            _events.put({"event": "error", "id": None,
                         "message": "引擎 进程 退出 (rc=%s)" % proc.returncode})


def start(model_id=None, timeout=None):
    """起 引擎 (加载 模型). 已经 在 跑 就 直接 返回 True.

    换 模型 = 停 掉 现在 这个, 起 新的 (内存 只 留 一个).
    """
    global _proc, _events, _active
    sp = _spec(model_id)
    if sp is None:
        _state["last_error"] = "不 认识 的 出歌 模型: %s" % model_id
        return False
    with _lock:
        if _proc and _proc.poll() is None:
            if _active == sp["id"]:
                return True
            _log("换 模型:", _active, "->", sp["id"], "(先 停 旧的)")
            # 必须 先 停 旧 进程 —— 不 停 它 就 变 成 孤儿, 白 占 十几 G 内存
            _stop_locked()
        if not sp["available"]():
            _state["last_error"] = "%s 组件 不 全 (代码/权重/解释器)" % sp["name"]
            return False
        os.makedirs(sp["songs"](), exist_ok=True)
        env = dict(os.environ)
        env.update(sp["env"]())
        cmd = sp["cmd"]()
        _log("起 引擎:", " ".join(cmd))
        _events = queue.Queue()
        _proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  cwd=sp["cwd"](), env=env)
        _active = sp["id"]
        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    deadline = time.time() + (timeout or START_TIMEOUT)
    while time.time() < deadline:
        try:
            ev = _events.get(timeout=1.0)
        except queue.Empty:
            if not running():
                break
            continue
        if ev.get("event") == "ready":
            with _lock:
                _state.update({"running": True, "load_s": ev.get("load_s"),
                               "started_at": time.time(), "last_error": None})
            _log("引擎 就绪 (%s, 加载 %s 秒)" % (sp["id"], ev.get("load_s")))
            _reset_idle_timer()
            return True
        if ev.get("event") == "error":
            with _lock:
                _state["last_error"] = ev.get("message")
            _log("引擎 起 失败:", ev.get("message"))
            stop()
            return False
    with _lock:
        _state["last_error"] = "引擎 启动 超时"
    stop()
    return False


def stop():
    """停 引擎 进程 (卸载 出歌 模型)."""
    with _lock:
        return _stop_locked()


def _stop_locked():
    """停 引擎 进程 —— 调用 方 已 持 锁."""
    global _proc, _idle_timer, _active
    with _lock:
        proc = _proc
        _proc = None
        _active = None
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        _state.update({"running": False, "load_s": None})
    if proc is None:
        return True
    try:
        proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
        proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=12)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _log("引擎 已 停")
    return True


def _compose_prompt(prompt, bpm=None, keyscale=None, language=None):
    """引擎 从 提示词 里 解析 BPM / 调式 —— 我们 把 UI 上 选 的 并 进去."""
    p = (prompt or "").strip()
    if bpm:
        try:
            bpm = int(bpm)
            if not any(s in p.lower() for s in ("bpm", "beats per minute")):
                p = "%s, %d bpm" % (p, bpm)
        except Exception:
            pass
    if keyscale:
        k = str(keyscale).strip()
        if k and k.lower() not in p.lower():
            p = "%s, key %s" % (p, k)
    return p


def generate(prompt, lyrics="", duration=60.0, instrumental=False,
             bpm=None, keyscale=None, language=None, model_id=None, timeout=1800,
             seed=None):
    """出 一首歌. 返回 (路径 或 None, 说明).

    引擎 挂 了 / 权重 不 在 -> 返回 (None, 原因), 由 上层 决定 降级.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "提示词 是 空 的"
    sp = _spec(model_id)
    if sp is None:
        return None, "不 认识 的 出歌 模型: %s" % model_id
    if not sp["available"]():
        return None, "%s 不 在 (权重/代码/解释器 缺)" % sp["name"]

    # 要 换 模型 -> 先 停 现在 这个
    with _lock:
        cur = _active
    if cur != sp["id"]:
        if running():
            stop()
        if not start(sp["id"]):
            return None, "出歌 引擎 起 不 来: %s" % (_state.get("last_error") or "未知")
    elif not running():
        if not start(sp["id"]):
            return None, "出歌 引擎 起 不 来: %s" % (_state.get("last_error") or "未知")

    options = {"duration": float(duration or 60.0), "format": "mp3"}
    if instrumental:
        options["vocals"] = False
    if lyrics and lyrics.strip():
        options["lyrics"] = lyrics.strip()
    if seed is not None:
        # 固定 seed -> 同一首歌可复现（引擎侧同时播种 5Hz LM 的全局随机源）
        options["seed"] = int(seed)
    req = {"id": uuid.uuid4().hex, "cmd": "generate",
           "prompt": _compose_prompt(prompt, bpm, keyscale, language),
           "options": options}
    with _lock:
        try:
            _proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            _proc.stdin.flush()
        except Exception as e:
            return None, "引擎 写入 失败: %s" % e

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev = _events.get(timeout=2.0)
        except queue.Empty:
            if not running():
                return None, "引擎 进程 死 了"
            continue
        e = ev.get("event")
        if e == "done":
            song = ev.get("song") or {}
            d = song.get("dir") or ""
            audio = song.get("audio") or ""
            path = os.path.join(d, audio) if d and audio else ""
            with _lock:
                _state["last_used"] = time.time()
                _state["last_song"] = path
            _reset_idle_timer()
            if not path or not os.path.isfile(path):
                return None, "引擎 没 给 出 音频"
            return path, "%s (渲染 %s 秒)" % (sp["name"], song.get("render_s"))
        if e == "error":
            with _lock:
                _state["last_error"] = ev.get("message")
            return None, "引擎 报错: %s" % (ev.get("message") or "未知")
    return None, "出歌 超时"
