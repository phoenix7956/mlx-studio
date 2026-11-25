"""mlx-studio —— 命令行 入口 (pip / uv / pipx 装完 直接 用).

    uv tool install mlx-studio
    mlx-studio                 # 起 内核 + 控制台 (http://127.0.0.1:11251)
    mlx-studio --open          # 起 好 自动 开 浏览器
    mlx-studio --port 11252    # 换 端口

桌面 App (MLX Serve Studio.app) 用 的 是 同一 份 内核 代码 —— 只是 换 个 壳.
"""
import argparse
import os
import sys
import threading
import time
import urllib.request


def _wait_and_open(host, port, timeout=40):
    """等 内核 起来, 然后 (可选) 开 浏览器."""
    url = "http://%s:%d/" % (host, port)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3).read(1)
            print("\n[mlx-studio] 控制台: %s" % url, flush=True)
            return
        except Exception:
            time.sleep(1)
    print("\n[mlx-studio] %d 秒 内 没 起来 —— 看 上面 的 日志" % timeout,
          file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(
        prog="mlx-studio",
        description="MLX Studio —— 本机 多模态 推理 内核 (出图/出歌/对话/语音) + 控制台")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MLX_STUDIO_PORT", "11251")))
    ap.add_argument("--host", default=os.environ.get("MLX_STUDIO_HOST", "127.0.0.1"))
    ap.add_argument("--open", action="store_true", help="起 好 后 自动 开 浏览器")
    ap.add_argument("--model-dir", action="append", default=[],
                    help="额外 的 模型目录 (可 多次 给)")
    ap.add_argument("--idle-evict-secs", type=int, default=1800,
                    help="闲置 多少 秒 自动 卸载 模型 (0=关)")
    args, rest = ap.parse_known_args()

    # 独立 装 的 时候, 本机 就是 全能 内核 (不 把 视频/图片 代理 给 别的 机器)
    os.environ.setdefault("MINEPROXY_NODE", "local")

    # 交给 内核 自己 的 参数 解析
    sys.argv = ["mlx-studio", "--port", str(args.port), "--host", args.host,
                 "--idle-evict-secs", str(args.idle_evict_secs)]
    for d in args.model_dir:
        sys.argv += ["--model-dir", d]
    sys.argv += rest

    try:
        import server  # noqa: F401
    except Exception as e:
        print("[mlx-studio] 内核 import 失败: %s" % e, file=sys.stderr, flush=True)
        return 1

    if args.open:
        threading.Thread(target=_wait_and_open,
                         args=(args.host, args.port), daemon=True).start()
    print("[mlx-studio] 内核 启动中 … http://%s:%d" % (args.host, args.port),
          flush=True)
    server.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
