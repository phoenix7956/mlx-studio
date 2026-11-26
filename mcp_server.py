#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLX Studio —— MCP server (stdio).

让 agent (Codex / Claude Code / Cursor / ...) 直接 驱动 **本机** 的 MLX Studio:
出图 / 出片 / 出歌 / 查模型 / 加载卸载模型.

—— 它 是 一 个 **薄壳**: 所有 活 都 交给 本机 的 Studio 内核 (HTTP, 端口 11251) 干,
    本身 不 加载 任何 模型, 不 占 显存. 装 完 MLX Studio, agent 就 能 直接 挂上.

用法:
    python mcp_server.py                 # MCP stdio 模式 (agent 拉起 它)
    python mcp_server.py --check         # 自检: 连 不 连得上 内核, 有 哪些 工具
    python mcp_server.py --worker <job>  # 内部: 单 个 异步 任务 的 执行 进程

环境变量:
    MLX_STUDIO_URL   内核 地址 (默认 http://127.0.0.1:11251)
    MLX_MCP_JOB_DIR  异步 任务 的 落地 目录 (默认 ~/.ours/mcp_jobs)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = (os.environ.get("MLX_STUDIO_URL") or "http://127.0.0.1:11251").rstrip("/")
JOB_DIR = os.environ.get("MLX_MCP_JOB_DIR") or os.path.expanduser("~/.ours/mcp_jobs")
VERSION = "1.0"


# ---------------------------------------------------------------- HTTP 薄壳


class StudioDown(RuntimeError):
    """内核 没 起 / 连 不 上."""


def _http(method: str, path: str, body=None, timeout: float = 60.0):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            err = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            err = {}
        msg = ""
        if isinstance(err, dict):
            inner = err.get("error")
            msg = inner.get("message") if isinstance(inner, dict) else (inner or err.get("message") or "")
        raise RuntimeError("%s %s -> HTTP %s: %s" % (method, path, e.code, msg or raw[:200]))
    except urllib.error.URLError as e:
        raise StudioDown("连不上 MLX Studio 内核 (%s): %s —— 先 双击 打开「MLX Serve Studio.app」"
                         % (BASE, e.reason))
    except OSError as e:
        raise StudioDown("连不上 MLX Studio 内核 (%s): %s" % (BASE, e))
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return {"raw": raw[:2048].decode("utf-8", "replace")}


def _get(path, timeout=60.0):
    return _http("GET", path, None, timeout)


def _post(path, body, timeout=1800.0):
    return _http("POST", path, body or {}, timeout)


# ---------------------------------------------------------------- 异步 任务
# 出图 要 1~7 分钟, 出片 更 久 —— 绝 不能 让 agent 的 一次 工具 调用 干等.
# 所以 提交 完 立刻 回 一个 job_id, agent 隔 一会儿 用 job_status 查.
# 图片/音乐 任务 跑 在 **独立 子进程** 里 (worker 模式) —— agent 关了 也不 影响 它 跑完.

_JOBS = {}
_LOCK = threading.Lock()


def _job_path(job_id: str) -> str:
    return os.path.join(JOB_DIR, job_id + ".json")


def _write_job(job: dict) -> None:
    try:
        os.makedirs(JOB_DIR, exist_ok=True)
        tmp = _job_path(job["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _job_path(job["id"]))
    except Exception:
        pass


def _read_job(job_id: str):
    try:
        with open(_job_path(job_id), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _new_job(kind: str, params: dict) -> dict:
    job_id = "%s_%d" % (kind, int(time.time() * 1000))
    job = {"id": job_id, "kind": kind, "status": "running",
           "params": params, "started": time.time()}
    with _LOCK:
        _JOBS[job_id] = job
    _write_job(job)
    return job


def _spawn_worker(job: dict) -> None:
    """把 图片/音乐 这 类 耗时 的 活 丢 给 独立 子进程 —— agent 退出 也 不 会 打断."""
    try:
        os.makedirs(JOB_DIR, exist_ok=True)
        log = open(os.path.join(JOB_DIR, job["id"] + ".log"), "ab")
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
                          _job_path(job["id"])],
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
    except Exception as e:
        job["status"] = "error"
        job["error"] = "起 任务 进程 失败: %s" % e
        _write_job(job)


def _worker_main(job_file: str) -> int:
    """独立 进程: 跑 完 一个 任务, 把 结果 写 回 job 文件."""
    try:
        with open(job_file, encoding="utf-8") as f:
            job = json.load(f)
    except Exception:
        return 1
    kind = job.get("kind")
    p = job.get("params") or {}
    try:
        if kind == "image":
            r = _post("/v1/images/generations", {
                "prompt": p.get("prompt", ""),
                "size": p.get("size", "1344x768"),
                "model": p.get("model") or None,
                "steps": p.get("steps", 0),
                "seed": p.get("seed", 0),
            }, timeout=3600)
            job["result"] = _strip_b64(r)
        elif kind == "music":
            r = _post("/v1/music/generate", {
                "prompt": p.get("prompt", ""),
                "duration_seconds": p.get("duration_seconds", 60),
                "lyrics": p.get("lyrics", ""),
                "lang": p.get("lang", "en"),
            }, timeout=3600)
            job["result"] = _strip_b64(r)
        else:
            raise RuntimeError("未知 任务 类型: %s" % kind)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    job["finished"] = time.time()
    _write_job(job)
    return 0


def _strip_b64(obj):
    """把 base64 图片/音频 从 返回 里 摘 掉 —— 那 玩意儿 会 把 agent 的 上下文 撑爆.
    只 留 文件 路径."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("b64_json", "b64", "audio_base64"):
                continue
            out[k] = _strip_b64(v)
        return out
    if isinstance(obj, list):
        return [_strip_b64(x) for x in obj]
    return obj


def _job_view(job: dict) -> dict:
    v = {"id": job.get("id"), "kind": job.get("kind"), "status": job.get("status")}
    if job.get("error"):
        v["error"] = job["error"]
    if job.get("result") is not None:
        v["result"] = job["result"]
    if job.get("started"):
        v["elapsed_sec"] = round((job.get("finished") or time.time()) - job["started"], 1)
    return v


# ---------------------------------------------------------------- 工具 实现


def _tool_status() -> dict:
    try:
        h = _get("/health", timeout=8)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "base_url": BASE,
        "node": h.get("node"),
        "loaded_engines": h.get("loaded"),
        "idle_unload_sec": h.get("idle_s"),
        "route": h.get("route"),
        "free_gib": h.get("free_gib"),
        "note": "模型 闲置 %s 秒 会 自动 卸载 (连续 生成 不 打断)" % h.get("idle_s"),
    }


def _guess_cap(model: str) -> str:
    m = (model or "").lower()
    if any(k in m for k in ("krea", "z-image", "zimage", "moss", "flux", "image", "img")):
        return "image"
    if any(k in m for k in ("h3", "wan", "video", "ltx")):
        return "video"
    if any(k in m for k in ("music", "ace", "song")):
        return "music"
    return ""


def _tool_list_models(capability: str = "") -> dict:
    r = _get("/v1/models", timeout=30)
    data = r.get("data") or []
    if capability:
        want = capability.lower()
        data = [m for m in data
                if want in [str(c).lower() for c in (m.get("capabilities") or [])]]
    out = []
    for m in data:
        out.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "capabilities": m.get("capabilities"),
            "state": m.get("state"),
            "loaded": m.get("loaded"),
            "size": m.get("size"),
            "bytes_on_disk": m.get("bytes_on_disk"),
        })
    return {"ok": True, "count": len(out), "models": out}


def _tool_load_model(model: str, capability: str = "") -> dict:
    if not model:
        raise RuntimeError("要 模型 id (先 用 list_models 看)")
    cap = capability or _guess_cap(model)
    r = _post("/v1/load-model", {"model": model, "capability": cap}, timeout=3600)
    return {"ok": True, "model": model, "capability": cap, "result": _strip_b64(r)}


def _tool_unload_model(model: str = "") -> dict:
    r = _post("/v1/unload-model", {"model": model} if model else {}, timeout=600)
    return {"ok": True, "result": _strip_b64(r)}


def _tool_list_nodes() -> dict:
    r = _get("/v1/video/nodes", timeout=20)
    return {"ok": True, "nodes": r.get("nodes"),
            "note": "镜头 可以 出 在 远端机, 合并 永远 在 本机"}


def _tool_list_output_images() -> dict:
    r = _get("/v1/video/images", timeout=30)
    return {"ok": True, "result": _strip_b64(r)}


def _tool_list_output_audio() -> dict:
    r = _get("/v1/audio/files", timeout=30)
    return {"ok": True, "result": _strip_b64(r)}


def _tool_list_outputs() -> dict:
    r = _get("/v1/outputs", timeout=60)
    return {"ok": True, "root": r.get("root"), "total": r.get("total"),
            "kinds": r.get("kinds")}


def _tool_submit_image(prompt: str, size: str = "1344x768", model: str = "",
                       steps: int = 0, seed: int = 0) -> dict:
    if not (prompt or "").strip():
        raise RuntimeError("要 提示词")
    job = _new_job("image", {"prompt": prompt, "size": size, "model": model,
                              "steps": steps, "seed": seed})
    _spawn_worker(job)
    return {"ok": True, "job_id": job["id"], "status": "running",
            "note": "出图 要 1~7 分钟 —— 用 job_status 查, 别 反复 提交"}


def _tool_submit_music(prompt: str, duration_seconds: int = 60,
                        lyrics: str = "", lang: str = "en") -> dict:
    if not (prompt or "").strip():
        raise RuntimeError("要 提示词 (曲风/风格/情绪)")
    job = _new_job("music", {"prompt": prompt, "duration_seconds": duration_seconds,
                              "lyrics": lyrics, "lang": lang})
    _spawn_worker(job)
    return {"ok": True, "job_id": job["id"], "status": "running"}


def _tool_submit_video_batch(items: list, model: str = "h3-4bit", seconds: int = 5,
                              steps: int = 8, width: int = 832, height: int = 480,
                              seed: int = 0) -> dict:
    """items: [{"prompt": "...", "first": "<图路径 或 base64>", "last": "<图路径 或 base64>"}]
    首帧/尾帧 用 list_output_images 给 的 路径; 尾帧 可 省略 (= 图生视频)."""
    items = [i for i in (items or []) if isinstance(i, dict) and (i.get("prompt") or "").strip()]
    if not items:
        raise RuntimeError("清单 是 空 的 —— 每个 镜头 至少 要 提示词")
    r = _post("/v1/video/batch", {
        "items": items, "model": model, "seconds": seconds, "steps": steps,
        "width": width, "height": height, "seed": seed,
    }, timeout=120)
    jid = r.get("job_id") or r.get("id")
    if not jid:
        return {"ok": False, "error": "内核 没 给 job_id", "raw": _strip_b64(r)}
    return {"ok": True, "job_id": jid, "status": r.get("status", "running"),
            "total": len(items), "out_dir": r.get("out_dir"),
            "note": "用 job_status 查 进度"}


def _tool_submit_mv(frames: list, prompts: list, seconds: int = 5,
                    music: str = "", node: str = "local") -> dict:
    """frames: 关键帧 路径 列表 (>= 2); prompts: 每个 镜头 的 提示词;
    music: 配乐 路径 (list_output_audio 给 的); node: local / remote."""
    frames = [f for f in (frames or []) if f]
    if len(frames) < 2:
        raise RuntimeError("至少 要 2 张 关键帧")
    r = _post("/v1/video/mv", {
        "frames": frames, "prompts": prompts or [], "seconds": seconds,
        "music": music or "", "node": node or "local",
    }, timeout=120)
    jid = r.get("job_id") or r.get("id")
    if not jid:
        return {"ok": False, "error": "内核 没 给 job_id", "raw": _strip_b64(r)}
    return {"ok": True, "job_id": jid, "status": r.get("status", "running"),
            "node": node, "note": "用 job_status 查 进度"}


def _tool_job_status(job_id: str) -> dict:
    if not job_id:
        raise RuntimeError("要 job_id")
    # 本地 (图片/音乐) 任务 —— 读 落地 文件
    job = _read_job(job_id)
    if job:
        return {"ok": True, **_job_view(job)}
    # 视频批次 / MV —— 任务 在 内核 进程 里
    if job_id.startswith("vbatch_"):
        return {"ok": True, **_strip_b64(_get("/v1/video/batch?id=" + job_id, timeout=30))}
    if job_id.startswith("vmv_"):
        return {"ok": True, **_strip_b64(_get("/v1/video/mv?id=" + job_id, timeout=30))}
    return {"ok": False, "error": "找 不 到 任务 %s" % job_id}


# ---------------------------------------------------------------- MCP 装配

try:                                        # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                         # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = (
    "MLX Studio —— 本机 (Apple Silicon) 的 创作 工作台: 出图 / 出片 / 出歌 / 配音 / 模型 管理.\n"
    "\n"
    "规矩:\n"
    "1. 出图 / 出片 都 是 **异步** 的 —— submit 完 拿 job_id, 再 用 job_status 查.\n"
    "   不要 一个 镜头 提交 一次 就 等 着; 一次 提交 整个 清单.\n"
    "2. 首尾帧 / 配乐 用 list_output_images / list_output_audio 给 的 路径 (别 自己 编 路径).\n"
    "3. 模型 闲置 30 分钟 会 自动 卸载; 生成 连续 跑 不会 被打断.\n"
    "4. 出片 的 镜头 可以 派 给 远端机 (node='remote'), 合并 永远 在 本机.\n"
    "5. 产物 落在 ~/Documents/MLX Studio 输出/ 下.\n"
)


# 工具 清单 —— 一 张 表, 顺便 拿 来 数 个数 (SDK 的 list_tools 是 异步 的, 别 在 自检 里 await).
TOOLS = [
    ("studio_status", _tool_status,
     "MLX Studio 内核 状态: 在不在 / 哪些 引擎 挂着 / 出片 走 本机 还是 远端 / 剩 多少 内存. "
     "**任何 别的 工具 报 连不上 的时候, 先 调 这个**."),
    ("list_models", _tool_list_models,
     "列出 所有 模型: id / 能力 (image,video,audio,chat) / 有没有 权重 (state=ready|absent) / "
     "在不在 已 加载 (loaded). 装 完 先 看 这个, 缺 权重 的 去 界面「模型」页 下载."),
    ("load_model", _tool_load_model,
     "加载 一个 模型 (占 显存). capability: image / video / audio / chat. "
     "不 填 就 按 模型 id 自动 判断."),
    ("unload_model", _tool_unload_model,
     "卸载 模型 (释放 显存). 不 填 model 就 全 卸. "
     "模型 闲置 30 分钟 也 会 自动 卸载, 一般 不用 手 动."),
    ("list_nodes", _tool_list_nodes,
     "出片 节点: 本机 / 远端机, 以及 谁 在 跑. node='remote' 时 镜头 在 远端 出, 产物 拉 回 本机 合并."),
    ("list_output_images", _tool_list_output_images,
     "列出 已 生成 的 图片 (出图 产物 + 首尾帧 素材 池). 挑 首尾帧 用 这个 的 路径."),
    ("list_output_audio", _tool_list_output_audio,
     "列出 已 生成 的 音频 (出歌 产物). MV 配乐 从 这里 挑."),
    ("list_outputs", _tool_list_outputs,
     "输出 目录 统计: 总 文件数 + 各类 (音频/图片/视频/语音/3D) 的 位置 与 占用."),
    ("submit_image", _tool_submit_image,
     "出图. 返 job_id, 隔 一会儿 用 job_status 查. "
     "size 形如 '1344x768'; 4MP 出图 要 几分钟."),
    ("submit_music", _tool_submit_music,
     "出歌 (ACE-Step). 返 job_id, 用 job_status 查. "
     "duration_seconds 是 时长 (引擎 有 上限); lyrics 可 空."),
    ("submit_video_batch", _tool_submit_video_batch,
     "批量 出片 (文生视频 / 图生视频 / 首尾帧). items 形如 "
     "[{'prompt':'...', 'first':'<图路径>', 'last':'<图路径>'}]; "
     "last 可 省略 (= 只有 首帧), first 也 可 省略 (= 文生视频). 返 job_id."),
    ("submit_mv", _tool_submit_mv,
     "出 MV: 关键帧 (>=2 张) + 每个 镜头 的 提示词 (+ 可选 配乐). "
     "frames 用 list_output_images 的 路径; music 用 list_output_audio 的 路径. 返 job_id."),
    ("job_status", _tool_job_status,
     "查 任务 进度: 出图 / 出歌 / 批量出片 / MV 都 用 它. 传 上一步 拿到的 job_id."),
]


def build_server():
    mcp = _Server(name="mlx-studio", version=VERSION, instructions=INSTRUCTIONS)
    for _name, _fn, _desc in TOOLS:
        mcp.tool(name=_name, description=_desc)(_fn)
    return mcp


# ---------------------------------------------------------------- 入口


def _selfcheck() -> int:
    print("MLX Studio MCP —— 自检")
    print("  内核地址: %s" % BASE)
    s = _tool_status()
    if not s.get("ok"):
        print("  ❌ %s" % s.get("error"))
        return 1
    print("  ✅ 内核 在线 —— node=%s  路由=%s  空闲内存=%sG"
          % (s.get("node"), s.get("route"), s.get("free_gib")))
    try:
        m = _tool_list_models()
        print("  ✅ 模型 %s 个 (ready %s 个)"
              % (m.get("count"), len([x for x in (m.get("models") or []) if x.get("state") == "ready"])))
    except Exception as e:
        print("  ⚠️ 模型 清单 拿不到: %s" % e)
    print("  ✅ 工具 %s 个: %s" % (len(TOOLS), ", ".join(t[0] for t in TOOLS)))
    return 0


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "--worker":
        return _worker_main(argv[2])
    if len(argv) > 1 and argv[1] == "--check":
        return _selfcheck()
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
