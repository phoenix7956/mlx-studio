# 第三方组件与许可

MLX Studio 主体是 **Apache-2.0** 许可（见根目录 `LICENSE`），**允许商业使用**。
下面 这些 是 仓库 里 附带 的 第三方 代码，各自 独立 授权，**不 继承** 本项目 的 Apache-2.0。

每一项 的 完整 协议 原文 都 放在 对应 目录 里。

## 代码

| 目录 | 组件 | 上游 | 许可 | 原文 |
|---|---|---|---|---|
| `ace-step-engine/` | ACE-Step 出歌引擎 | [ACEStep/ACE-Step](https://github.com/ACEStep/ACE-Step) | MIT | `ace-step-engine/LICENSE` |
| `ace-step-engine/` | ACE-Step MLX 移植 | Fabrizio Salmi | MIT | `ace-step-engine/LICENSE.acestep-mlx` |
| `ace-step-engine/acestep/third_parts/nano-vllm/` | nano-vllm | Xingkai Yu | MIT | `ace-step-engine/acestep/third_parts/nano-vllm/LICENSE` |
| `vendor/minimax-h3_mlx_src/` | MiniMax-H3 的 MLX 移植 | [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx) | Apache-2.0 | `vendor/minimax-h3_mlx_src/LICENSE` |
| `vendor/krea2_engine_src/` | Krea-2-Turbo 的 MLX 引擎 | [avlp12/krea2_alis_mlx](https://github.com/avlp12/krea2_alis_mlx) | Krea 2 Community License | `vendor/krea2_engine_src/LICENSE` |
| `vendor_src/mlx_video/` | mlx-video | [Prince Canuma/mlx-video](https://github.com/PrinceCanuma/mlx-video) | MIT | `vendor_src/mlx_video/LICENSE` |

我们 对 这些 组件 做 的 改动，一处 一处 列 在 各自 目录 的 `NOTICE` 里
（Apache-2.0 §4(b) 要求 标注 改动 的 文件）。

### Krea 2 社区许可 的 特别 说明

`vendor/krea2_engine_src/` 是 Krea 2 模型的 修改衍生品，受 **Krea 2 Community License
Agreement** 约束（不是开源协议）。分发 它 必须 满足 该协议 的 要求：

- 携带 协议 原文 —— `vendor/krea2_engine_src/LICENSE`
- 标注 改动 —— `vendor/krea2_engine_src/NOTICE`
- 保留 内容过滤 —— 引擎 自带 `safety.py`（纯 MLX 的 NSFW 分类器），**默认 开启**

该协议 还 对 商业用途 有 营收门槛，对 权重 的 分发 有 附加条款。详见 协议 原文。

### MiniMax-H3 权重协议

`vendor/minimax-h3_mlx_src/` 里 的 **代码** 是 Apache-2.0（第三方 独立实现）。
**权重** 是 MiniMax 的 物料，适用 **MiniMax H3 Community License**
（原文 一并 附 在 `vendor/minimax-h3_mlx_src/MINIMAX-H3-COMMUNITY-LICENSE.txt`）。
仓库 里 **不含 任何 权重**；用户 下载 权重 时 自己 接受 该协议。

## 权重（不由本仓库分发）

本仓库 **不含 任何 模型权重**。所有 权重 由 用户 在 界面「模型」页 下载，
各自 遵循 模型作者 的 许可，与 本项目 无关。

| 模型 | 权重许可 |
|---|---|
| Qwen3 系列 / Qwen3-Embedding / Qwen3-TTS | Apache-2.0 |
| Z-Image-Turbo | Apache-2.0 |
| MOSS-Image-SFT | Apache-2.0 |
| Wan2.2 TI2V-5B / FastWan2.2 | Apache-2.0 |
| ACE-Step 1.5 | MIT |
| Whisper (faster-whisper) | MIT |
| Chatterbox TTS | MIT |
| VoxCPM2 / CosyVoice 2 / OmniVoice / IndexTTS-2 MLX 量化 | Apache-2.0 |
| MiniMax-Music3 | MiniMax-Music3 Community License（MIT 式 + 商业条款） |
| MiniMax-H3 | MiniMax H3 Community License（营收门槛 / 署名 / 地区限制） |
| Krea-2-Turbo | Krea 2 Community License |
| Gemma 3 | Gemma Terms of Use |
| Llama 3.1 | Llama 3.1 Community License |

界面「模型」页 会 显示 每个 权重 的 许可 类型；`license_free` 为真 的 是
Apache-2.0 / MIT / 本项目内置，可以 用 「只看自由许可」 一键 过滤。

## ffmpeg

仓库 **不 分发** 任何 ffmpeg 二进制。ffmpeg 是 LGPL/GPL，随 仓库 分发 会 让
整个 项目 被迫 改成 GPL。安装脚本 从 pip 依赖 `imageio-ffmpeg` 里 取 一份，
在 `~/.ours/bin/ffmpeg` 写 一个 转发壳 —— 那是 每个 用户 自己 装 的，不 算 分发。

