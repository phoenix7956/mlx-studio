# MLX Studio

[![Release](https://img.shields.io/github/v/release/phoenix7956/mlx-studio?label=%E6%9C%80%E6%96%B0%E7%89%88)](https://github.com/phoenix7956/mlx-studio/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-9cf2d0)

**跑在你自己 Mac 上的 AI 创作工作室。** 图像、视频、音乐、语音、对话，一个桌面 App 全包，
模型权重全部本地跑，不上传任何数据。

独立实现，不依赖任何闭源 App。MIT 许可。

---

## 它能干什么

12 个页面，一个 App：

| 页 | 能做什么 |
|---|---|
| **模型** | 看本机装了什么模型、下载、删除。每一行 带 许可 徽章，可一键「只看自由许可」。模型跑起来后在项目页加载 / 卸载 |
| **图像** | 文生图、图生图、SDEdit、参考图加权、RGBA。Krea-2-Turbo / Z-Image-Turbo / MOSS-Image-SFT |
| **视频** | 文生视频、图生视频、首尾帧。H3（带声音）/ Wan2.2。批量首尾帧跑批 |
| **音乐** | 出歌工作台：ACE-Step 本地出歌（30 秒 ≈ 2.3 秒）、风格标签、歌词、批量、广场、作品库 |
| **音频** | 语音合成、音效、语音克隆 |
| **配音** | 音色库、克隆新音色（自动转写 + 出试听样音）、批量配音、贴回视频、保留原声 |
| **对话** | 本地 LLM 聊天，带思考开关和流式输出 |
| **资料库** | 产物管理 |
| **监控** | 引擎状态、显存、卸载 |
| **API** | 所有能力都有 HTTP 端点 |
| **设置 / 代码启动器** | 参数、跑批脚本 |

产物统一落在 `~/Documents/MLX Studio 输出/{音频,图片,视频,语音,3D}/`。

## 安装

要求：**macOS + Apple Silicon**（M 系列芯片）。

### 方式一：下载安装包（推荐）

去 [Releases](https://github.com/phoenix7956/mlx-studio/releases/latest) 下 `.dmg`，拖进「应用程序」。

### 方式二：从源码装

```bash
git clone https://github.com/phoenix7956/mlx-studio.git
cd mlx-studio
./install.sh
```

装完：

- 内核跑在 `http://127.0.0.1:11251`，开机自启（launchd `com.ours.mlxstudio`）
- 桌面 App 装到 `/Applications/MLX Serve Studio.app`
- **模型权重不随安装包分发** —— 打开「模型」页点「下载」，或者用界面选好参数自己下

常用参数：

```bash
./install.sh --no-agent   # 不装开机自启，只跑一次
./install.sh --no-ace     # 不装出歌引擎的大环境（省 ~2GB）
./install.sh --app        # 只装桌面 App
./install.sh --port 11251 # 换端口
./clean.sh                # 清临时产物和旧缓存
```

## 模型权重与许可

**这个仓库不包含任何模型权重。** 引擎代码和权重是两回事：

- **引擎代码**在仓库里，许可见下面的「许可」一节。
- **模型权重**由你在「模型」页下载，**各自遵循模型作者自己的许可**，跟本项目无关。
  你下载 MiniMax-H3 就接受 MiniMax 的社区许可，下载 Krea-2 就接受 Krea 的社区许可，
  下载 Wan2.2 / Z-Image / MOSS-Image 这些 Apache-2.0 的就按 Apache-2.0 用。

本项目不对你用哪个模型、怎么商用负责，也不替你接受任何协议 —— 那是你和模型作者之间的事。

界面「模型」页会显示每个权重的许可类型：

- **✓ Apache-2.0 / MIT / 内置** —— 自由许可，随便用，商用也可以
- **⚠ 社区许可** —— 有附加条款（营收门槛 / 署名 / 地区限制等），商用前自己看原文

点右上角「只看自由许可」可以把 ⚠ 的那些筛掉。**默认全部显示**，只是自由许可的排在前面。

下载默认走 **魔搭**（国内最快），hf-mirror 自动兜底。

## HTTP API

内核是一套标准 HTTP 服务，界面用的就是它，脚本也能直接调。

```bash
curl http://127.0.0.1:11251/v1/models   # 模型列表 + 装载态
curl http://127.0.0.1:11251/health
```

主要端点：

| 能力 | 端点 |
|---|---|
| 图像生成 | `POST /v1/images/generations` |
| 视频生成 | `POST /v1/videos/generations` |
| 音乐生成 | `POST /v1/audio/music-generations`、`POST /v1/music/generate` |
| 语音合成 | `POST /v1/audio/speech` |
| 音色克隆 / 配音 | `POST /v1/voices/clone`、`POST /v1/voices/dub` |
| 对话 | `POST /v1/chat/completions` |

完整参数面见 `api_full.py`，契约回归见 `smoke_contract.py`（37 条）。

## 目录结构

```
server.py            HTTP 服务：路由分发、模型装载态、指标、媒体 SSE
api_full.py          所有 /v1/* 端点的参数面
cap_engines.py       能力引擎（向量 / TTS / 音效 / 音乐 / 3D / 图像 / 视频 / ASR …）
image_engine.py      Krea-2-Turbo 封装
image_advanced.py    潜变量混合、真 CFG、Otsu 羽化、色彩迁移
music_workbench.py   出歌工作台服务端
music_engine.py      ACE-Step 出歌
voice_studio.py      配音工作台（音色库 / 克隆 / 批量）
llm_engine.py        本地 LLM 封装
voice_convert.py     音色迁移（YIN + PSOLA + 倒谱共振峰包络）
engine.py            推理胶水
ui/console.html      界面前端（单文件）

desktop/             macOS 桌面 App 源码（Swift + AppKit + WKWebView）
vendor/              第三方引擎源码（各自带自己的 LICENSE / NOTICE）
vendor_src/          mlx-video（MIT）
ace-step-engine/     ACE-Step 出歌引擎（MIT）
port_minimax_h3_mlx/ H3 引擎补丁
port_wan2_mlx/       Wan2.2 引擎
install.sh           一键安装
desktop/package.sh   打成 DMG / ZIP 安装包
docs/                内部运维文档
smoke_*.py           回归测试
```

## 许可

本项目 **MIT**（见 `LICENSE`）。

第三方代码各自独立授权，见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)：

| 组件 | 许可 |
|---|---|
| `vendor/minimax-h3_mlx_src/` | Apache-2.0（上游 PipeNetwork/minimax-h3-mlx） |
| `vendor/krea2_engine_src/` | Krea 2 Community License（上游 avlp12/krea2_alis_mlx） |
| `vendor_src/mlx_video/` | MIT（Prince Canuma） |
| `ace-step-engine/` | MIT（ACEStep / Fabrizio Salmi）+ nano-vllm 的 MIT |

`vendor/` 下的两个目录带自己的 `LICENSE` 和 `NOTICE`，改动过的文件在里面列明了。

## 独立实现声明

MLX Studio 是从零写的 clean-room 实现。仓库里**没有**任何闭源 App 的代码、资源或二进制，
运行时也不读取闭源 App 的任何文件。界面的功能与交互是按公开可观察的行为重新实现的。

## 开发

```bash
python3 -m venv ~/.venvs/mlxstudio
~/.venvs/mlxstudio/bin/pip install -r requirements.txt
~/.venvs/mlxstudio/bin/python server.py --port 11251
```

回归：

```bash
~/.venvs/mlxstudio/bin/python smoke_contract.py    http://127.0.0.1:11251
~/.venvs/mlxstudio/bin/python smoke_capabilities.py http://127.0.0.1:11251
```

