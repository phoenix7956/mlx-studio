#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video_batch.py — 批量 首尾帧 出片 (并 进 我们 自研 内核).

为什么 要 有 这 个:
  单片 出片 要 7 分钟 (4bit DiT 25G 常驻 + 124 帧), 一 个 一个 点 太 慢.
  实际 干 活 是: 先 批量 出 图 (Krea-2-Turbo), 再 把 图 当 首帧/尾帧 串 成 片子.
  所以 批量 的 输入 是 **图 的 配对**, 不是 一 个 个 手 点.

端点:
  POST /v1/video/batch      —— 提交 清单, 后台 跑, 返回 job_id
  GET  /v1/video/batch?id=  —— 查 进度 (log / done / total / items)

跟 批量 出歌 (music_workbench.batch) 一模一样 的 模式 —— 前端 有 现成 模板.

关键 点:
  · 出片 是 单飞 的 (一次 只 能 一 片), 所以 批量 必须 **串行** —— 不 并发.
  · 首尾帧 走 base64 (POST body, 不是 URL —— URL 塞不下 图).
  · 整 批 共 用 一个 session —— 模型 全程 常驻, 不 每 片 重 载 (重 载 要 好 几分钟).
"""
import base64
import os
import threading
import time

import outputs

_jobs = {}
_lock = threading.Lock()


def _decode_frame(v):
    """首/尾帧: base64 / data URL / 本地 路径 -> base64 (给 内核 用).

    本地 路径 只 允许 落在 输出 目录 里 —— 防止 有人 让 内核 读 任意 文件.
    """
    if not v or not isinstance(v, str):
        return None
    if v.startswith("data:"):
        _head, _sep, raw = v.partition(",")
        return raw or None
    root = os.path.abspath(outputs.ROOT)
    if v.startswith("/"):
        p = os.path.abspath(v)
    elif "/" in v and len(v) < 4096:
        p = os.path.abspath(os.path.join(root, v))
    else:
        return v.strip() or None
    if not (p == root or p.startswith(root + os.sep)):
        return None
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def pair_images(paths, mode="adjacent"):
    """把 一 组 图 按 配对 规则 拆 成 (首帧, 尾帧) 的 任务 列表.

    mode:
      adjacent  —— 两 两 相邻 配 (第1张->第2张, 第3张->第4张)  ← 默认
      chain     —— 首尾 相接 (第1张->第2张, 第2张->第3张, ...)  画面 连续
      single    —— 每 张 只 当 首帧 (尾帧 让 模型 自己 收)
    """
    ps = [p for p in (paths or []) if p]
    if mode == "single":
        return [(p, None) for p in ps]
    if mode == "chain":
        # 第1->2, 2->3, 3->4 ...
        return [(ps[i], ps[i + 1]) for i in range(len(ps) - 1)]
    # adjacent: 奇数 的话 最后 一张 自己 当 首帧
    out = []
    i = 0
    while i < len(ps):
        if i + 1 < len(ps):
            out.append((ps[i], ps[i + 1]))
            i += 2
        else:
            out.append((ps[i], None))
            i += 1
    return out


def batch(srv, items, out_dir=None, model="h3-4bit", seconds=5, steps=8,
          width=832, height=480, seed=0, prompt_default=""):
    """批量 首尾帧 出片 —— 后台 跑, 返回 job_id. 进度 用 batch_status 查.

    items: [{"prompt": "...", "first": <图>, "last": <图>}, ...]
      图 可以 是 base64 / data URL / 输出 目录 里 的 相对 路径.
    """
    items = [i for i in (items or []) if isinstance(i, dict)]
    if not items:
        return {"ok": False, "error": "清单 是 空 的"}
    out_dir = out_dir or os.path.join(outputs.ROOT, "视频",
                                      "批次%s" % time.strftime("%m%d%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    job_id = "vbatch_%d" % int(time.time() * 1000)
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "running", "done": 0,
                         "total": len(items), "out_dir": out_dir,
                         "items": [], "log": [], "started": time.time()}

    def _one(it, idx):
        prompt = (it.get("prompt") or prompt_default or "").strip()
        first = _decode_frame(it.get("first"))
        last = _decode_frame(it.get("last"))
        if not prompt:
            return {"idx": idx, "ok": False, "error": "没 提示词"}
        if not first:
            return {"idx": idx, "ok": False, "error": "没 首帧 图"}
        q = {"prompt": [prompt], "sec": [str(seconds)], "steps": [str(steps)],
             "seed": [str(seed + idx)], "width": [str(width)],
             "height": [str(height)],
             "first_frame_image": [first]}
        if out_dir:
            q["out_dir"] = [out_dir]
        if last:
            q["last_frame_image"] = [last]
        r = srv._do_video(q)
        url = (r or {}).get("url") or ""
        vp = url.split("file://")[-1]
        try:
            srv.record_library("video", vp, model, prompt)
        except Exception:
            pass
        return {"idx": idx, "ok": bool(url), "url": url,
                "path": vp, "sec": (r or {}).get("sec"),
                "prompt": prompt}

    def _run():
        sid = None
        try:
            # 开 一个 session —— 出片 期间 模型 别 被 置闲 计时器 卸 了
            try:
                sid = srv.open_session("video")
            except Exception:
                sid = None
            for i, it in enumerate(items):
                with _lock:
                    j = _jobs.get(job_id)
                    if j:
                        j["log"].append("▶ %d/%d %s" % (i + 1, len(items),
                                                       (it.get("prompt") or "")[:40]))
                try:
                    res = _one(it, i)
                except Exception as e:
                    res = {"idx": i, "ok": False, "error": str(e)}
                with _lock:
                    j = _jobs.get(job_id)
                    if j:
                        j["items"].append(res)
                        j["done"] = len(j["items"])
                        if res.get("ok"):
                            j["log"].append("✅ %d/%d (%s 秒)" % (i + 1, len(items),
                                                               res.get("sec")))
                        else:
                            j["log"].append("❌ %d/%d %s" % (i + 1, len(items),
                                                               res.get("error")))
            if sid:
                try:
                    # 只 关 会话, 不 卸 引擎 —— 下一 批 不用 重 载 25G (置闲 30min 后 自动 卸)
                    srv.close_session(sid, unload=False)
                except Exception:
                    pass
        except Exception as e:
            with _lock:
                j = _jobs.get(job_id)
                if j:
                    j["log"].append("批次 异常: %s" % e)
        finally:
            with _lock:
                j = _jobs.get(job_id)
                if j:
                    j["status"] = "succeeded" if j["done"] == j["total"] else "failed"
                    j["ended"] = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id, "total": len(items), "out_dir": out_dir}


def batch_status(job_id):
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            return {"ok": False, "error": "没 这个 批次"}
        return {"ok": True, **j}


def list_images(limit=200):
    """出图 产物 清单 —— 给 界面 做 批量 首尾帧 的 选图 池."""
    d = os.path.join(outputs.ROOT, "图片")
    out = []
    try:
        names = sorted(os.listdir(d))
    except Exception:
        names = []
    for n in reversed(names):
        if not n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        p = os.path.join(d, n)
        try:
            st = os.stat(p)
        except Exception:
            continue
        out.append({"name": n, "path": p,
                    "url": "/out/" + outputs.rel(p),
                    "rel": outputs.rel(p),
                    "bytes": st.st_size, "mtime": st.st_mtime})
        if len(out) >= limit:
            break
    return {"ok": True, "items": out, "count": len(out)}
