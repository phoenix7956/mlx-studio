#!/usr/bin/env python3
"""GUI 批量出片 启 动 层. 接 收 一 个 JSON 参 数 (shell), 做 3 件 事:
1. 写 远端机 batch_params.json (跑 批 读 它, 不 框 死 111 段)
2. 启 远端机 后 台 run_full.py (nohup)
3. 起 本 机 watch_batch.sh (跑 批 结 束 自 动 拉 回 + ffmpeg 合 并)
引 号/ssh 全 在 这 里, GUI 的 Swift 只 传 参 数 不 碰 shell."""
import json, os, subprocess, sys
import remote_node


def main():
    远端机 = remote_node.target()
    if not 远端机:
        print("没 配 「第二台机」—— 设 REMOTE_SSH_TARGET (或写 ~/.ours/node.json) 再 跑", flush=True)
        sys.exit(2)
    SSH = ["ssh", "-o", "ConnectTimeout=10", 远端机]
    P = {"shots": "/tmp/mv_zh_shots.txt,/tmp/mv_en_shots.txt", "sec": 5, "steps": 12, "resume": True}
    if len(sys.argv) > 1:
        try:
            P.update(json.loads(sys.argv[1]))
        except Exception as e:
            print("参数 解 析 失 败, 用 默 认:", e, flush=True)
    shots = P["shots"] if isinstance(P["shots"], str) else ",".join(P["shots"])
    node_json = json.dumps({"shots": [s.strip() for s in shots.split(",") if s.strip()],
                           "sec": P["sec"], "steps": P["steps"], "resume": P["resume"]})
    # 1. 写 远端机 batch_params.json (单 引 号 包 echo, json 里 只 有 双 引 号, 不 冲 突)
    subprocess.run(SSH + ["mkdir -p ~/.ours/mlx_server && echo '" + node_json + "' > ~/.ours/mlx_server/batch_params.json"],
                   capture_output=True, text=True, shell=False)
    # 2. 启 后 台 run_full.py (续 跑 读 done.json, 跳 已 出)
    subprocess.run(SSH + ["nohup python3 ~/.ours/mlx_server/run_full.py > /tmp/full_mv.log 2>&1 &"],
                   capture_output=True, text=True, shell=False)
    # 3. 起 本 机 watch_batch.sh (盯 跑 批 结 束 -> pull_and_merge 拉 回 合 并 -> AUTO_DONE)
    _WATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_batch.sh")
    subprocess.Popen(["zsh", _WATCH], start_new_session=True)
    print(f"批量出片 已 启 动: shots={shots} sec={P['sec']} steps={P['steps']} 续 跑 ={P['resume']}", flush=True)


if __name__ == "__main__":
    main()
