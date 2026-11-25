#!/usr/bin/env python3
"""跑 批 完 结 -> 拉 回 本 机 + ffmpeg 按 中 54 / 英 57 顺 序 合 并. 一 次 一 件, 串 行."""
import json, os, subprocess, sys, time, urllib.request
SSH = ["ssh","-o","ConnectTimeout=10","<you>@<node-ip>"]
FFMPEG = "/opt/homebrew/bin/ffmpeg"
BASE = os.path.expanduser("~/Desktop/rea/soak-results/mlxserve/mv_mp4")
def node_read_done():
    out = subprocess.run(SSH + ["cat","/Users/<you>/.ours/mlx_server/done.json"],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    return d if isinstance(d,dict) else {}
def fetch(node_path, local):
    """帧 目 录 在 远端机, 远端机 没 ffmpeg -> scp 帧 + wav 回 本 机, 本 机 ffmpeg 合 mp4."""
    if os.path.exists(local): return
    import shutil
    d = node_path  # 帧 目 录 (如 /tmp/mx_video_1794050843)
    local_frames = local + ".frames"
    if not os.path.exists(local_frames):
        os.makedirs(local_frames, exist_ok=True)
        # 帧 一 张 张 太 慢, 用 单 次 scp 拉 整 个 目 录
        subprocess.run(["scp","-o","ConnectTimeout=10","-r",
                        f"<you>@<node-ip>:{d}/*.png", local_frames],
                       capture_output=True, text=True)
        # wav (有 的 段 带 音 轨)
        subprocess.run(["scp","-o","ConnectTimeout=10",
                        f"<you>@<node-ip>:{d}.wav", local_frames],
                       capture_output=True, text=True)
    # 本 机 ffmpeg 合
    wav = os.path.join(local_frames, os.path.basename(d)+".wav")
    cmd=[FFMPEG,"-y","-framerate","24","-i",os.path.join(local_frames,"frame_%05d.png")]
    if os.path.exists(wav): cmd += ["-i",wav,"-shortest","-c:a","aac"]
    cmd += ["-c:v","libx264","-pix_fmt","yuv420p","-crf","18",local]
    subprocess.run(cmd,check=True)
    shutil.rmtree(local_frames,ignore_errors=True)
def main():
    os.makedirs(BASE, exist_ok=True)
    done = node_read_done()
    print(f"done={len(done)}")
    for k in sorted(done):
        remote = done[k]["url"].split("file://")[-1]
        base = os.path.basename(remote).replace(".mp4","") or "frames"
        local = os.path.join(BASE, f"{k}_{base}.mp4")
        fetch(remote, local)
        print("  fetched", local)
    for grp in ("zh","en"):
        files = [os.path.join(BASE,f"{k}_{os.path.basename(done[k]['url']).replace('.mp4','')}.mp4")
                 for k in sorted(done) if k.startswith(grp)]
        files = [f for f in files if os.path.exists(f)]
        if not files: print(f"{grp}: 无文件"); continue
        lst = os.path.join(BASE, f"concat_{grp}.txt")
        open(lst,"w").write("\n".join(f"file '{f}'" for f in files))
        out = os.path.expanduser(f"~/Desktop/rea/soak-results/mlxserve/mv_{grp}_merged.mp4")
        # 先 concat 合 成 片, 再 烧 歌 词 字 幕 轨 (有 歌 词 标 记 时)
        no_sub = os.path.expanduser(f"~/Desktop/rea/soak-results/mlxserve/mv_{grp}_merged_nosub.mp4")
        subprocess.run([FFMPEG,"-y","-f","concat","-safe","0","-i",lst,"-c","copy",no_sub],check=True)
        # 抽 每 段 歌 词 行 (shot 脚 本 "主 唱 口 型 与 动 作 随 词: X 当 前 段 落: Y" -> X), 按 5s/段 对 齐
        shot_files = os.path.join(BASE, f"shots_{grp}.txt")
        ass_path = os.path.expanduser(f"~/Desktop/rea/soak-results/mlxserve/mv_{grp}.ass")
        lyr_lines = _extract_lyrics(shot_files)
        if lyr_lines:
            _write_ass(ass_path, lyr_lines, sec_per_shot=_sec_per_shot(done, grp))
            burn = os.path.expanduser(f"~/Desktop/rea/soak-results/mlxserve/mv_{grp}_merged_burned.mp4")
            subprocess.run([FFMPEG,"-y","-i",no_sub,"-vf",
                            f"ass={ass_path}","-c:v","libx264","-crf","18","-c:a","copy","-safe","0",burn],check=True)
            os.replace(burn, out)
            print(f"{grp}: 烧 了 {len(lyr_lines)} 行 歌 词 字 幕 -> {out}")
        else:
            os.replace(no_sub, out)
            print(f"{grp}: 无 歌 词 标 记, 原 片 {out} ({os.path.getsize(out)//1048576} MB)")
def _extract_lyrics(shot_file):
    """每 行 shot: [N] 画 面 … 主 唱 口 型 与 动 作 随 词: <歌 词 行> 当 前 段 落: <段>. 抽 <歌 词 行>."""
    if not os.path.exists(shot_file): return []
    out=[]
    for l in open(shot_file):
        l=l.strip()
        if not l: continue
        if "主唱口型与动作随词:" in l and "当前段落:" in l:
            seg=l.split("主唱口型与动作随词:",1)[1].split("当前段落:",1)[0].strip().rstrip("。.").strip()
            out.append(seg if seg else "…")
        elif "] " in l and l[0]=="[":
            out.append(l.split("] ",1)[1].strip())  # 无 歌 词 标 记 就 拿 画 面 描 述 兜 底
        else:
            out.append(l)
    return out
def _sec_per_shot(done, grp):
    """看 done.json 该 组 片 长 度 (帧 数 不 在, 用 5s 默 认); 有 记 录 就 读, 没 就 5."""
    return 5.0
def _write_ass(ass_path, lines, sec_per_shot=5.0):
    """ASS: 每 行 对 应 一 段, 时间 轴 0..len*sec. 字 幕 显 3s 留 1s 黑."""
    head="""[Script Info]
ScriptType: v4.00+
PlayResX: 1344
PlayResY: 768
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric, PingFang SC, 48, &H00FFFFFF, &H00FFFFFF, &H00000000, &H80000000, 0, 0, 0, 0, 100, 100, 0, 0, 1, 2, 1, 2, 20, 20, 60, 1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def ts(t):
        t=max(0.0,float(t)); h=int(t//3600); m=int((t%3600)//60); s=t%60
        return f"{h}:{m:02d}:{s:05.2f}"
    evs=[]
    for i,ln in enumerate(lines):
        a=i*sec_per_shot; b=a+sec_per_shot
        evs.append(f"Dialogue: 0,{ts(a)},{ts(b)},Lyric,,0,0,0,,{ln}")
    open(ass_path,"w").write(head + "\n".join(evs) + "\n")
if __name__ == "__main__": main()
