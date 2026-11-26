# 参与开发

先读 [`docs/开源说明.md`](docs/开源说明.md) —— 仓库 边界、密钥 怎么 配、第二台机 怎么 接。

## 环境

只 支持 **macOS + Apple Silicon**（Metal / MLX，没有 CUDA 路径）。

```bash
git clone https://github.com/phoenix7956/mlx-studio.git
cd mlx-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./install.sh --no-agent
```

模型 权重 **不 随仓库 分发**。装完 去 界面「模型」页 点「下载」，默认 走 **魔搭**。

## 提交 前 —— 三 件事 必须 过

```bash
python3 tools/check_catalog.py
bash -n install.sh && bash -n clean.sh
python3 -m compileall -q .
```

CI 跑 的 就是 这 三 件 + **泄露扫描**。

下面 这些 **绝对 不许 进 仓库**：

- 真实 的 API key / 令牌 / 证书
- 某台 机器 的 用户名 或 内网 IP（用 `~` 和 占位 符）
- 你 自己 的 作品 / 产物

一旦 push 上去，密钥 **必须 当场 轮换**；历史 也要 重写
（`git filter-repo`，见 `docs/开源说明.md` 第八节）。

## 代码 约定

| 位置 | 放 什么 |
|---|---|
| `server.py` / `api_full.py` | HTTP 路由、核心 调度 |
| `engine.py` / `image_engine.py` / `music_workbench.py` | 各 能力 的 引擎 |
| `ui/console.html` | 界面（单 文件 前端，中 / EN 双语） |
| `model_catalog.json` | **模型页 的 唯一 数据源** |
| `tools/` | 下载 / 校验 / 辅助 脚本 |
| `vendor/` `vendor_src/` `ace-step-engine/` | 第三方 源码，**别 改** |

加 新模型：往 `model_catalog.json` 加 一条，在 `tools/` 放 下载 脚本。权重 不 进 仓库。

## 风格

- 界面 文案：中文 / English 两 套 都 要 改（`ui/console.html` 里 的 `L` 表）
- 注释 用 中文，和 周边 一致
- 别 引 新 依赖 —— `requirements.txt` 越短 越好装
- 别 把 「废弃」留 着：删 掉 的 代码 / 目录 一起 删

## 提 Issue

带上 macOS 版本、芯片型号、相关 报错，
以及 `curl http://127.0.0.1:11251/status` 的 输出。别 贴 密钥。
