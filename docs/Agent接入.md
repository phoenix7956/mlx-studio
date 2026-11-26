# 接入 Agent（MCP）

MLX Studio 自带一个 **MCP server**。装完，Codex / Claude Code / Cursor / Gemini CLI /
OpenCode 就能直接调本机的出图、出片、出歌 —— 不用用户手抄配置。

## 它是怎么工作的

```
agent --stdio-->  mlx-mcp  --HTTP-->  Studio 内核 (127.0.0.1:11251)  -->  MLX 引擎
                   (薄壳)                (真正 干活 的)
```

MCP server 是个**薄壳**：自己不加载模型、不占显存、不做推理，只把 agent 的调用
转成 Studio 内核的 HTTP 请求。所以：

- 装 Studio 和装 MCP 是两件事，但 `install.sh` 一起做掉
- agent 挂了不影响内核；内核挂了 MCP 会明确报错，不会静默失败
- 引擎实际跑在 Studio 那一侧，显存还是 Studio 占

## 安装后会自动做什么

`install.sh` 跑完会：

1. 装 `mcp` SDK 到 Studio 的 venv
2. 写启动器 `~/.ours/bin/mlx-mcp`
3. 把 `mcp_server.py` 装到 `~/.ours/mlx_server/`
4. 扫本机**已经装了**的 agent，把 `mlx-studio` 加进去（只改已存在的配置，
   不给没装的 agent 乱建目录；每次改都先备份 `.bak-<时间戳>`）

> `mcp` 装在**独立的 venv**（`~/.venvs/mlx-mcp`）里，不进 Studio 的 venv。
> mcp 会带自己的依赖（pydantic / httpx / starlette…），装进 Studio 那个 venv，
> pip 有可能顺手升级那些包、把正在跑的内核弄出毛病。独立 venv = 零污染。
> 启动器 `~/.ours/bin/mlx-mcp` 会自己挑那个带 mcp sdk 的 python。

| Agent | 配置位置 | 写法 |
|---|---|---|
| Codex | `~/.codex/config.toml` | `[mcp_servers.mlx-studio]` |
| Claude Code | `~/.claude.json` | `mcpServers["mlx-studio"]` |
| Cursor | `~/.cursor/mcp.json` | `mcpServers["mlx-studio"]` |
| Gemini CLI | `~/.gemini/settings.json` | `mcpServers["mlx-studio"]` |
| OpenCode | `~/.config/opencode/config.json` | `mcp["mlx-studio"]` |

**重开 agent 才生效**（Codex / Claude Code 退出再进）。

## 工具清单

| 工具 | 参数 | 说明 |
|---|---|---|
| `studio_status` | — | 内核在不在、哪些引擎挂着、出片走本机还是远端、剩多少内存 |
| `list_models` | `capability?` | 模型 id / 能力 / 有没有权重（`state=ready\|absent`）/ 在不在已加载 |
| `load_model` | `model`, `capability?` | 加载模型（占显存） |
| `unload_model` | `model?` | 卸载模型；不填就全卸 |
| `list_nodes` | — | 出片节点（本机 / 远端机） |
| `list_output_images` | — | 已生成的图片 —— 挑首尾帧从这里取路径 |
| `list_output_audio` | — | 已生成的音频 —— MV 配乐从这里挑 |
| `list_outputs` | — | 输出目录统计 |
| `submit_image` | `prompt`, `size?`, `model?`, `steps?`, `seed?` | 出图 |
| `submit_music` | `prompt`, `duration_seconds?`, `lyrics?`, `lang?` | 出歌（ACE-Step） |
| `submit_video_batch` | `items`, `model?`, `seconds?`, `steps?`, `width?`, `height?`, `seed?` | 批量出片（文生 / 图生 / 首尾帧） |
| `submit_mv` | `frames`, `prompts`, `seconds?`, `music?`, `node?` | 出 MV |
| `job_status` | `job_id` | 查任务进度 |

## 为什么要异步

出图要 1~7 分钟，出片更久。同步工具调用会把 agent 卡死，所以：

- `submit_*` **立刻**返回 `job_id`，不干等
- `job_status` 查进度
- 图片 / 音乐 任务跑在**独立子进程**里（`mcp_server.py --worker`），
  **agent 中途退出也不会打断生成**
- 视频批次 / MV 的任务本来就住在内核进程里，更不怕

`submit_video_batch` 的 `items` 格式：

```json
[
  {"prompt": "镜头一 的 描述", "first": "<图路径>", "last": "<图路径>"},
  {"prompt": "镜头二 的 描述", "first": "<图路径>"}
]
```

- 省略 `last` = 图生视频；两个都省略 = 文生视频
- 路径从 `list_output_images` 拿（内核只认输出目录里的路径）

## 不会把 agent 上下文撑爆

出图接口返回 base64，4MP 图能到几 MB。**MCP 这一层把 base64 全摘掉**，只回文件路径
和秒数。agent 拿到的是路径，不是几 MB 的字符串。

## 手动接 / 摘掉

```bash
python3 tools/register_mcp.py            # 注册（幂等）
python3 tools/register_mcp.py --check    # 看状态
python3 tools/register_mcp.py --print    # 打印手抄版配置
python3 tools/register_mcp.py --remove   # 摘掉
```

指向别的机器（内网里的另一台 Studio）：

```bash
MLX_STUDIO_URL=http://192.168.x.x:11251 ~/.ours/bin/mlx-mcp
```

## 自检

```bash
~/.ours/bin/mlx-mcp --check
```

输出内核状态、模型数量、工具数量。连不上内核会直接告诉你「先双击打开
MLX Serve Studio.app」，不会静默失败。

## 许可

MCP server 本身是本项目代码，跟仓库同许可（MIT）。
它**只调用**模型，不分发也不附带任何权重 —— 权重还是各自作者按自己的许可发。
用 MiniMax-H3 做商业产品，界面要显著标注「MiniMax H3」（见
`vendor/minimax-h3_mlx_src/MINIMAX-H3-COMMUNITY-LICENSE.txt`）。

