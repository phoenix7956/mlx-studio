# vendor —— 引擎 源码

| 目录 | 是什么 | 上游 | 许可 |
|---|---|---|---|
| `minimax_h3_mlx_src/` | MiniMax-H3 的 MLX 移植 (出片 / 视频) | https://github.com/PipeNetwork/minimax-h3-mlx | Apache-2.0 (`LICENSE` + `NOTICE` 在 本目录) |
| `krea2_engine_src/` | Krea-2-Turbo 的 MLX 移植 (出图). VAE 复用 mflux (MIT) | https://github.com/avlp12/krea2_alis_mlx | Krea 2 Community License (`LICENSE` + `NOTICE` 在 本目录) |

装 完 之后 这 两 个 目录 会被 install.sh 复制 到 `~/.ours/vendor/`, 并写 进
`~/.ours/mlx_server/env.sh` (H3M_PORT_DIR / H3M_KREA_DIR).

两个目录都 **不含权重**. 权重 全部 由 「模型」 页 下载, 各自 遵循 模型 作者 的
许可, 跟 这 份 代码 的 许可 无关.

改动 说明 在 各自 目录 的 `NOTICE` 里 (Apache-2.0 §4b 要求 标注 改动 的 文件;
Krea 2 Community License 要求 携带 协议 原文 并 标注 改动).
