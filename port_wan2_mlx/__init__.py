"""Wan2.2 TI2V-5B —— 我们 自己 的 MLX port (基于 mlx-video, MIT).

为什么 有 这 一 层:
  mlx-video 的 generate_video() 是 命令行 一次性 的 —— 每次 调用 都 重新 加载
  DiT / T5 / VAE. 我们 是 桌面端 常驻 服务, 批量 出片 时 一条 一条 生,
  每次 重载 5.4G DiT 根本 没法 用.

所以 这 层 做 三件事:
  ① DiT 常驻 (缓存 load_wan_model) —— 出片 之间 不 重载.
  ② T5 / VAE 临时 (不缓存) —— T5 被 mlx-video 从 bf16 升成 float32, 一次 就
     22.7G, 常驻 太 肥; VAE 只 在 编码/解码 那 一小会儿 用.
  ③ 把 mlx-video 写 mp4 的 动作 截 下来, 直接 拿 帧 数组 回 我们 引擎 契约.

内核 源码 (mlx_video/) 是 上游 的, 原样 vendor 进来, 没 改 一个 字.
"""
from .pipeline import Wan2Pipeline, Wan2Result

__all__ = ["Wan2Pipeline", "Wan2Result"]
