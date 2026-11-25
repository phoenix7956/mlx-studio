#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniMax Music3 引擎 —— 我们 自研 的 第二个 出歌 模型 (纯 MLX, 走 魔搭 权重).

跟 ACE-Step 引擎 用 同一套 JSON-lines 协议:
    stdin  收 {"id":.., "cmd":"generate", "prompt":"..", "options":{..}}
    stdout 出 {"event":"ready"|"phase"|"done"|"error", ..}

为了 不 让 库 自己 的 print / tqdm 污染 协议, 启动 时 把 fd 1 指向 stderr,
另 存 一 份 fd 专门 写 协议.
"""
import argparse
import datetime
import json
import os
import sys
import time
import uuid

# —— 协议 fd: 原始 stdout 留 一 份, fd1 改 指 stderr ——
_PROTO_FD = os.dup(1)
os.dup2(2, 1)


def emit(**kw):
    """写 一 条 协议 事件."""
    try:
        os.write(_PROTO_FD, (json.dumps(kw, ensure_ascii=False) + "\n").encode("utf-8"))
    except Exception:
        pass


def song_dir(out_dir, req_id):
    now = datetime.datetime.now()
    d = os.path.join(out_dir, "%s-%s" % (now.strftime("%Y%m%d-%H%M%S"), req_id[:8]))
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    try:
        from mlx_audio.music import load
        from mlx_audio import audio_io
        import numpy as np
    except Exception as e:
        emit(event="error", message="依赖 缺失: %s" % e)
        return

    try:
        model = load(args.model)
    except Exception as e:
        emit(event="error", message="模型 加载 失败: %s" % e)
        return

    load_s = time.time() - t0
    emit(event="ready", load_s=round(load_s, 1))
    print("[minimax-music3] 加载 %0.1fs" % load_s, file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = (line or "").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "generate":
            continue

        rid = req.get("id") or uuid.uuid4().hex
        prompt = (req.get("prompt") or "").strip()
        opts = req.get("options") or {}

        try:
            dur = float(opts.get("duration") or 60.0)
        except Exception:
            dur = 60.0
        dur = max(1.0, min(360.0, dur))

        # lyrics 是 必填 的 —— 纯 器乐 用 [instrumental]
        lyrics = (opts.get("lyrics") or "").strip()
        if not lyrics:
            if opts.get("vocals") is False:
                lyrics = "[instrumental]"
            else:
                emit(event="error", id=rid, message="缺 歌词 (或 勾 选 器乐)")
                continue

        fmt = (opts.get("format") or "mp3").lower()
        steps = int(opts.get("steps") or 30)
        steps = max(1, min(30, steps))
        seed = int(opts.get("seed") or 0)

        d = song_dir(args.out, rid)
        emit(event="phase", id=rid, phase="render")

        try:
            audio = None
            sr = 44100
            for res in model.generate(text=prompt, lyrics=lyrics, duration=dur,
                                      steps=steps, seed=seed):
                if getattr(res, "audio", None) is not None:
                    audio = np.asarray(res.audio, dtype="float32")
                    sr = int(getattr(res, "sample_rate", sr) or sr)
            if audio is None:
                raise RuntimeError("模型 没 出 音频")
        except Exception as e:
            emit(event="error", id=rid, message="生成 失败: %s" % e)
            continue

        name = "%s.%s" % (rid, "mp3" if fmt == "mp3" else "wav")
        path = os.path.join(d, name)
        try:
            audio_io.write(path, audio, sr, format=fmt)
        except Exception:
            audio_io.write(os.path.join(d, "song.wav"), audio, sr, format="wav")
            path = os.path.join(d, "song.wav")

        with open(os.path.join(d, "song.json"), "w") as f:
            json.dump({"prompt": prompt, "lyrics": lyrics, "duration": dur,
                       "engine": "minimax-music3", "audio": os.path.basename(path),
                       "sample_rate": sr}, f, ensure_ascii=False, indent=2)

        emit(event="done", id=rid,
             song={"dir": d, "audio": os.path.basename(path), "duration": dur,
                   "engine": "minimax-music3"})


if __name__ == "__main__":
    main()
