#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video_mv.py — MV 工作台 (关键帧 序列 -> 一 个 镜头 一个 镜头 -> 拼 成 MV -> 配乐).

为什么 是 这 样
    一 个 首尾帧 片段 只 有 几 秒, 单看 没 用 —— 它 的 用处 是 "一 串 关键帧 之间 的 过 场".

        关键帧:  图1 -> 图2 -> 图3 -> 图4
        镜头 1:  图1 当 首帧, 图2 当 尾帧
        镜头 2:  图2 当 首帧, 图3 当 尾帧      <- 首尾 相接, 连贯
        镜头 3:  图3 当 首帧, 图4 当 尾帧
        MV:      镜头1 + 镜头2 + 镜头3 (+ 配乐)

分工 (用户 定 的)
    远端机: 只 出 镜头 (它 机器 大, 有 权重, 出片 快)
    本机  : 做 合并 —— 拼接 (concat) + 配乐 (mux) 一律 在 本机 跑

    所 以 镜头 在 远端机 出 的 时候, 会 先 scp 回 本机, 再 在 本机 拼.
    反过来 (远端机 拼完 再 传 回 本机) 是 白 传 一 轮 —— 不 这么 做.

端点 (挂 在 server.py 上)
    POST /v1/video/mv        —— 提交, 后台 跑, 返回 job_id
    GET  /v1/video/mv?id=    —— 查 进度 (log / done / total / shots / mv)
    GET  /v1/audio/files     —— 本机 出歌 产物 (给 配乐 下拉 用)
    GET  /v1/video/nodes     —— 可 用 的 出片 节点 (本机 / 远端机, 带 连通 性)
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import threading
import time

try:
    import outputs
except Exception:                                   # pragma: no cover
    outputs = None


try:
    import remote_node
except Exception:                                   # pragma: no cover
    remote_node = None


def _node():
    """返回 (user, host) —— 开源版 不 写死 任何 机器 的 地址."""
    if remote_node is not None:
        return remote_node.user(), remote_node.host()
    return (os.environ.get("REMOTE_SSH_USER", ""),
            os.environ.get("REMOTE_SSH_HOST", ""))


NODE_USER, NODE_HOST = _node()
NODE_PORT = os.environ.get("REMOTE_PORT", os.environ.get("NODE_PORT", "11251"))

_lock = threading.Lock()
_jobs = {}
_node_cache = {"at": 0, "data": None}


def _root():
    return outputs.ROOT if outputs else os.path.expanduser("~/Documents/MLX Studio 输出")


# ------------------------------------------------------------------ ffmpeg
def ffmpeg_bin():
    found = shutil.which("ffmpeg")
    if found:
        return found
    home = os.path.expanduser("~")
    for c in (os.path.join(home, ".ours", "bin", "ffmpeg"),
              os.path.join(home, ".ours", "ffmpeg", "ffmpeg"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "ffmpeg"),
              "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    return None


def _b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _resolve(path):
    """相对 路径 -> 绝对 路径 (只 允许 在 输出 目录 里)."""
    if not path:
        return None
    if path.startswith("file://"):
        path = path[7:]
    root = _root()
    if os.path.isabs(path):
        return path if os.path.isfile(path) else None
    p = os.path.normpath(os.path.join(root, str(path).lstrip("/")))
    if not p.startswith(os.path.normpath(root)):
        return None
    return p if os.path.isfile(p) else None


def _materialize(frames, work_dir):
    """关键帧 可以 是 输出 目录 里 的 相对 路径, 也 可以 是 data URL (本机 选 的 图).
    data URL 落 一 张 临时 png, 后面 一 样 用 文件 走."""
    out = []
    os.makedirs(work_dir, exist_ok=True)
    for i, f in enumerate(frames or []):
        if not f:
            continue
        if isinstance(f, str) and f.startswith("data:"):
            try:
                raw = f.split(",", 1)[1]
                ext = ".png"
                if "jpeg" in f[:40]:
                    ext = ".jpg"
                p = os.path.join(work_dir, "kf_%02d%s" % (i, ext))
                with open(p, "wb") as fh:
                    fh.write(base64.b64decode(raw))
                out.append(p)
                continue
            except Exception:
                continue
        p = _resolve(f)
        if p:
            out.append(p)
    return out


def list_audio(limit=300):
    """本机 出歌 产物 清单 —— 给 MV 配乐 用 (配乐 一律 在 本机 混)."""
    d = os.path.join(_root(), "音频")
    out = []
    try:
        names = sorted(os.listdir(d))
    except Exception:
        names = []
    for n in reversed(names):
        if not n.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".flac")):
            continue
        p = os.path.join(d, n)
        try:
            st = os.stat(p)
        except Exception:
            continue
        rel = outputs.rel(p) if outputs else n
        out.append({"name": n, "path": p, "rel": rel, "url": "/out/" + rel,
                    "bytes": st.st_size, "mtime": st.st_mtime})
        if len(out) >= limit:
            break
    return {"ok": True, "items": out, "count": len(out)}


# ------------------------------------------------------------------ 节点
def _my_host():
    try:
        return socket.gethostname()
    except Exception:
        return ""


def _i_am_node():
    """我 就是 远端 这 台 机器 吗 (那 时候 不 该 再 出现 「远端机」 这个 节点)."""
    if not NODE_HOST:
        return False
    mine = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((NODE_HOST, 80))
        mine.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        mine.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    theirs = set()
    try:
        theirs.add(socket.gethostbyname(NODE_HOST))
    except Exception:
        pass
    try:
        theirs.update(socket.gethostbyname_ex(NODE_HOST)[2])
    except Exception:
        pass
    theirs.add(NODE_HOST)
    return bool(mine & theirs)


def _ssh(cmd, timeout=60, stdin=None):
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=no",
            "%s@%s" % (NODE_USER, NODE_HOST), cmd]
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    except Exception as e:
        return 255, "", str(e)
    return (p.returncode,
            (p.stdout or b"").decode("utf-8", "ignore"),
            (p.stderr or b"").decode("utf-8", "ignore"))


def node_reachable(timeout=8):
    rc, out, _ = _ssh("echo ok", timeout=timeout)
    return rc == 0 and "ok" in out


def nodes(force=False):
    """可用 的 出片 节点 —— 镜头 可以 在 本机 出, 也 可以 在 远端机 出 (合并 永远 在 本机)."""
    now = time.time()
    if not force and _node_cache["data"] and now - _node_cache["at"] < 30:
        return _node_cache["data"]
    me = _my_host()
    data = {"ok": True, "host": me,
            "nodes": [{"id": "local", "name": "本机", "host": me,
                       "note": "镜头 在 这台 机器 上 出", "ok": True}]}
    if NODE_HOST and not _i_am_node():
        data["nodes"].append({
            "id": "remote", "name": "远端机", "host": NODE_HOST,
            "note": "镜头 在 远端机 出, 产物 拉 回 本机 合并",
            "ok": node_reachable()})
    _node_cache["at"] = now
    _node_cache["data"] = data
    return data


# ------------------------------------------------------------------ 拼接
def concat(clips, out_path):
    """把 一 串 镜头 拼 成 一个 mp4 (参数 一 致, 直接 拼 流, 不 重 编)."""
    ffm = ffmpeg_bin()
    if not ffm:
        raise RuntimeError("没 ffmpeg —— 拼 不 成 MV")
    lst = out_path + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for c in clips:
            f.write("file '%s'\n" % c.replace("'", "'\\''"))
    cmd = [ffm, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-c", "copy", out_path]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=3600)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or b"").decode("utf-8", "ignore")[-300:])
    finally:
        try:
            os.unlink(lst)
        except Exception:
            pass
    return out_path


def mux_music(mv_path, music, out_path):
    """给 MV 配上 音乐 (音乐 短 就 循环, 长 就 截到 视频 长度)."""
    ffm = ffmpeg_bin()
    if not ffm or not music or not os.path.isfile(music):
        return mv_path
    cmd = [ffm, "-y", "-v", "error", "-i", mv_path, "-stream_loop", "-1", "-i", music,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=1800)
    except Exception:
        return mv_path
    if p.returncode != 0 or not os.path.isfile(out_path) \
            or os.path.getsize(out_path) < 1000:
        return mv_path
    return out_path


def strip_audio(path):
    """去掉 音轨 (画面 不 重 编 —— 直接 拷 视频 流)."""
    ffm = ffmpeg_bin()
    if not ffm or not path or not os.path.isfile(path):
        return None
    out = path[:-4] + "_无声.mp4" if path.lower().endswith(".mp4") else path + "_无声.mp4"
    cmd = [ffm, "-y", "-v", "error", "-i", path, "-map", "0:v:0", "-c:v", "copy",
           "-an", out]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=900)
    except Exception:
        return None
    if p.returncode != 0 or not os.path.isfile(out) or os.path.getsize(out) < 1000:
        return None
    try:
        os.replace(out, path)          # 原地 换掉 —— 界面 拿到 的 还是 同 一个 路径
    except Exception:
        return out
    return path


# ------------------------------------------------------------------ 镜头 (本机)
def _shot_local(srv, prompt, first, last, seconds, width, height, steps, seed, out_dir):
    q = {"prompt": [prompt], "sec": ["%.2f" % float(seconds)],
         "steps": [str(steps)], "seed": [str(int(seed))],
         "width": [str(width)], "height": [str(height)],
         "first_frame_image": [_b64(first)]}
    if last:
        q["last_frame_image"] = [_b64(last)]
    if out_dir:
        q["out_dir"] = [out_dir]
    r = srv._do_video(q)
    url = (r or {}).get("url") or ""
    p = url.split("file://")[-1] if url else ""
    if not p or not os.path.isfile(p):
        raise RuntimeError("镜头 生成 失败")
    return p, (r or {}).get("sec")


# ------------------------------------------------------------------ 镜头 (远端机)
def _scp_pull(remote_path, local_dir):
    """从 远端机 把 产物 拉 回 本机."""
    dst = os.path.join(local_dir, os.path.basename(remote_path))
    # 新 版 scp 走 SFTP (不 经过 远端 shell) —— 路径 里 的 空格 / 中文 直接 传 就 行,
    # 加 引号 反而 会 被 当成 路径 里 的 字. 老 版 (-O, 走 shell) 才 需要 引号.
    tries = [["-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
              "-o", "StrictHostKeyChecking=no",
              "%s@%s:%s" % (NODE_USER, NODE_HOST, remote_path), dst],
             ["-O", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
              "-o", "StrictHostKeyChecking=no",
              "%s@%s:%s" % (NODE_USER, NODE_HOST,
                            "'%s'" % remote_path.replace("'", "'\\''")), dst]]
    err = ""
    for extra in tries:
        argv = ["scp"] + extra
        try:
            p = subprocess.run(argv, capture_output=True, timeout=1800)
        except Exception as e:
            err = str(e)
            continue
        if p.returncode == 0 and os.path.isfile(dst):
            return dst
        err = (p.stderr or b"").decode("utf-8", "ignore")[-200:]
    raise RuntimeError("拉 回来 失败: %s" % err)


def _shot_remote_node(items, seconds, width, height, steps, seed, local_dir,
                     on_log=None):
    """把 整 个 批次 交 给 远端机 出, 产物 拉 回 本机.

    items: [{"prompt":..., "first": <b64>, "last": <b64>}]
    返回: [{"remote_path":..., "local_path":..., "prompt":...}]
    """
    payload = json.dumps({"items": items, "seconds": seconds, "steps": steps,
                          "width": width, "height": height, "seed": seed})
    cmd = ("curl -sS -m 60 -X POST -H 'Content-Type: application/json' "
           "--data-binary @- http://127.0.0.1:%s/v1/video/batch" % NODE_PORT)
    rc, out, err = _ssh(cmd, timeout=180, stdin=payload.encode("utf-8"))
    if rc != 0:
        raise RuntimeError("连不上 远端机: %s" % (err or out)[-200:])
    try:
        r = json.loads(out.strip() or "{}")
    except Exception:
        raise RuntimeError("远端机 回 的 不 是 JSON: %s" % out[:200])
    if not r.get("ok"):
        raise RuntimeError("远端机 拒收: %s" % (r.get("error") or out[:200]))
    job_id = r.get("job_id")
    if not job_id:
        raise RuntimeError("远端机 没 给 批次 号")
    if on_log:
        on_log("▶ 远端机 开始 出 %d 个 镜头 (批次 %s)" % (len(items), job_id))

    last_done = -1
    deadline = time.time() + 4 * 3600
    st = {}
    while True:
        time.sleep(6)
        if time.time() > deadline:
            raise RuntimeError("等 远端机 等 太 久, 放弃 了")
        cmd = ("curl -sS -m 20 'http://127.0.0.1:%s/v1/video/batch?id=%s'"
               % (NODE_PORT, job_id))
        rc, out, err = _ssh(cmd, timeout=60)
        if rc != 0:
            continue
        try:
            st = json.loads(out.strip() or "{}")
        except Exception:
            continue
        done = st.get("done") or 0
        if done != last_done and on_log:
            last_done = done
            on_log("… 远端机 出 到 %d/%d" % (done, st.get("total") or len(items)))
        if st.get("status") != "running":
            break
    items_out = st.get("items") or []
    got = []
    for it in items_out:
        rp = it.get("path") or ""
        if not it.get("ok") or not rp:
            continue
        lp = _scp_pull(rp, local_dir)
        got.append({"remote_path": rp, "local_path": lp,
                    "prompt": it.get("prompt", ""), "sec": it.get("sec")})
    return got


# ------------------------------------------------------------------ 任务
def submit(srv, frames, prompts, seconds=5, width=832, height=480, steps=8,
           seed=42, music=None, node="local", model="h3-4bit"):
    """提交 一个 MV 任务 —— 后台 跑, 用 status 查 进度.

    frames:  关键帧 路径 列表 (至少 2 张) —— 相邻 两 张 = 一个 镜头
    prompts: 每个 镜头 的 提示词 (不够 就 套用 最后 一 条)
    music:   配乐 路径 (可 空) —— 本机 出歌 产物
    node:    "local" (本机 出镜头) / "remote" (远端机 出镜头, 拉 回 本机)
    """
    root = _root()
    out_dir = os.path.join(root, "视频", "MV%s" % time.strftime("%m%d%H%M"))
    frames = _materialize(frames, os.path.join(out_dir, "关键帧"))
    if len(frames) < 2:
        return {"ok": False, "error": "至少 要 2 张 关键帧"}
    prompts = [str(p).strip() for p in (prompts or []) if str(p).strip()]
    if not prompts:
        return {"ok": False, "error": "写 点 提示词"}
    music_path = _resolve(music) if music else None
    if music and not music_path:
        return {"ok": False, "error": "找 不 到 那 首 配乐"}
    node = (node or "local").lower()
    if node not in ("local", "remote"):
        node = "local"
    shots = len(frames) - 1
    job_id = "vmv_%d" % int(time.time() * 1000)
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "running", "done": 0,
                         "total": shots + 2, "shots": [], "log": [], "node": node,
                         "started": time.time(), "out_dir": out_dir,
                         "shots_total": shots, "music": music or ""}

    def _log(msg):
        with _lock:
            j = _jobs.get(job_id)
            if j:
                j["log"].append(msg)

    def _run():
        sid = None
        try:
            os.makedirs(out_dir, exist_ok=True)
            shot_dir = os.path.join(out_dir, "镜头")
            os.makedirs(shot_dir, exist_ok=True)
            clips = []
            if node == "remote":
                items = []
                for i in range(shots):
                    prompt = prompts[min(i, len(prompts) - 1)]
                    items.append({"prompt": prompt,
                                  "first": _b64(frames[i]),
                                  "last": _b64(frames[i + 1])})
                got = _shot_remote_node(items, seconds, width, height, steps, seed,
                                       shot_dir, on_log=_log)
                if not got:
                    raise RuntimeError("远端机 没 出 出 镜头")
                for k, g in enumerate(got):
                    clips.append(g["local_path"])
                    with _lock:
                        j = _jobs.get(job_id)
                        if j:
                            j["done"] = k + 1
                            j["shots"].append({"idx": k, "ok": True,
                                               "url": "/out/" + outputs.rel(g["local_path"]),
                                               "path": g["local_path"],
                                               "prompt": g.get("prompt", ""),
                                               "node": "remote",
                                               "first": "/out/" + outputs.rel(frames[k]),
                                               "last": "/out/" + outputs.rel(frames[k + 1])})
                            j["log"].append("✅ 镜头 %d/%d (远端机 出, 已 拉 回 本机)"
                                            % (k + 1, len(got)))
            else:
                try:
                    sid = srv.open_session("video")
                except Exception:
                    sid = None
                for i in range(shots):
                    prompt = prompts[min(i, len(prompts) - 1)]
                    _log("▶ 镜头 %d/%d (本机 出)" % (i + 1, shots))
                    p, sec = _shot_local(srv, prompt, frames[i], frames[i + 1],
                                         seconds, width, height, steps,
                                         seed + i, out_dir)
                    clips.append(p)
                    with _lock:
                        j = _jobs.get(job_id)
                        if j:
                            j["done"] = i + 1
                            j["shots"].append({"idx": i, "ok": True,
                                               "url": "/out/" + outputs.rel(p),
                                               "path": p, "prompt": prompt,
                                               "node": "local", "sec": sec,
                                               "first": "/out/" + outputs.rel(frames[i]),
                                               "last": "/out/" + outputs.rel(frames[i + 1])})
                            j["log"].append("✅ 镜头 %d/%d" % (i + 1, shots))
            # —— 合并 一律 在 本机 ——
            with _lock:
                j = _jobs.get(job_id)
                if j:
                    j["done"] = shots + 1
                    j["log"].append("▶ 本机 合并 (拼接 %d 个 镜头)" % len(clips))
            mv_path = os.path.join(out_dir, "MV_%s.mp4" % time.strftime("%H%M%S"))
            concat(clips, mv_path)
            if music_path:
                with_music = os.path.join(out_dir,
                                          "MV_配乐_%s.mp4" % time.strftime("%H%M%S"))
                mv_path = mux_music(mv_path, music_path, with_music)
            try:
                srv.record_library("video", mv_path, model,
                                   "MV · %d 个 镜头" % len(clips))
            except Exception:
                pass
            with _lock:
                j = _jobs.get(job_id)
                if j:
                    j["done"] = shots + 2
                    j["mv"] = {"url": "/out/" + outputs.rel(mv_path), "path": mv_path}
                    j["log"].append("✅ MV 完成 (%s)" % os.path.basename(mv_path))
        except Exception as e:
            _log("❌ %s" % e)
        finally:
            if sid:
                try:
                    # 只 关 会话, 不 卸 引擎 —— 置闲 30min 后 自动 卸 (下 一 次 出片 不用 重 载)
                    srv.close_session(sid, unload=False)
                except Exception:
                    pass
            with _lock:
                j = _jobs.get(job_id)
                if j:
                    j["status"] = "succeeded" if j.get("mv") else "failed"
                    j["ended"] = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id, "shots": shots, "node": node,
            "out_dir": out_dir}


def status(job_id):
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            return {"ok": False, "error": "没 这个 任务"}
        return {"ok": True, **j}
