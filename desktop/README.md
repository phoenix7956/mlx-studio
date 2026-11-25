# MLX Serve Studio —— 桌面端

**双击 `MLX Serve Studio.app` 就 用，不需要浏览器。**

## 它是什么

原生 macOS 窗口（纯 AppKit + WKWebView），里面装的是逆向复刻的那份 Studio 控制台
（`../实现代码/ui/console.html`，与网页版同一份文件，界面像素一致）。

## 内核没跑怎么办

启动时会探 `http://127.0.0.1:11251/health`；内核没在跑就显示兜底页，
点「启动内核」自动拉起 `../实现代码/run_local_dev.sh <port>`，起来后界面自动亮起。

## 菜单（内核）

| 菜单项 | 快捷键 | 作用 |
|---|---|---|
| 重新加载 界面 | ⌘R | 重载控制台 |
| 刷新 内核 状态 |  | 立刻再探一次健康 |
| 启动 内核 |  | 手动拉起内核 |
| 在 浏览器 打开 |  | 备用（正常不用） |
| 导出 界面 截图 | ⌘S | 存 PNG |

## 命令行参数（自测用）

构建产物在 `~/.ours/build/MLX Serve Studio.app`（**不放桌面** —— 桌面上每多一个 `.app`，
Launchpad 就多一个同名图标；装好的那一个在 `/Applications`）。

```bash
A="$HOME/.ours/build/MLX Serve Studio.app/Contents/MacOS/MLXServeStudio"
"$A" --port 11299      # 换端口
"$A" --url "/#library" # 直达 某个 页面
"$A" --snapshot a.png  # 载入完导图后退出
"$A" --autostart       # 启动即拉起内核
"$A" --fresh          # 一次性数据存储(不读本地状态)

# 调试口：在 WebView 里跑 JS 并把结果打到 stderr
"$A" --eval "JSON.stringify({n:state.models.length})"
"$A" --eval-delay 6 --eval-result "__RESULT" --eval "<异步脚本>"
```

`--eval-delay` 是等页面稳的秒数；`--eval-result <全局变量名>` 是让页面里的异步脚本
把结果写进那个全局变量，App 轮询取回后退出 —— 用来做真正的异步测量。

## 重新构建

```bash
./build.sh     # swiftc -O … + codesign --force --deep -s -
```

> 这台机器的 CommandLineTools 没有 `libSwiftUIMacros.dylib`，SwiftUI 的
> `@State/@StateObject/@EnvironmentObject` 宏编译不过，所以本壳是纯 AppKit 手写的。

## 放到 远端机 上（<you>@<node-ip>）

- 桌面放一个 **符号链接** `MLX Serve Studio.app` → `~/.ours/build/MLX Serve Studio.app`，
  双击即开（要 的是 那 一 份 唯一 的 app，别 在 桌面 再 放 实体 副本）。
- 内核**不跑在桌面那份代码上**：macOS 上 launchd 读 `~/Desktop` 会被挡住（`open()` 挂死），
  实际跑 `~/.ours/mlx-serve-studio/`，由 `com.ours.mlxserve.studio` LaunchAgent 常驻，
  端口 `127.0.0.1:11251`、`node=remote`。
- 桌面交付目录是唯一真源：`启动 MLX Serve Studio.command` 每次先 rsync 到 `~/.ours` 再开 App。

细节见 `../远端机部署说明.md`。

---

# 安装包（Studio 整套，单机可装）

把桌面端 Studio 打成能直接拖进 `/Applications` 的 macOS 安装包。

## 产物

| 文件 | 说明 |
|---|---|
| `dist/MLX-Serve-Studio-1.1.dmg` | 拖拽安装版（2.4 MB），内含 App + 安装说明 + 一键安装脚本 |
| `dist/MLX-Serve-Studio-1.1.zip` | 同样内容的 zip，给不方便挂 DMG 的人 |

DMG 挂载后内容：

```
MLX Serve Studio.app/     ← 原生壳（AppKit + WKWebView）+ 内核代码 + 图标
安装说明.txt
安装到本机.command        ← 双击就装到 /Applications 并打开
```

## 装

1. 打开 DMG
2. 把 `MLX Serve Studio.app` 拖进「应用程序」
3. 第一次打开会被 Gatekeeper 拦（没 Apple 开发者证书，只有 ad-hoc 签名）：
   - 右键点 App → 「打开」→「打开」，或者
   - `xattr -cr "/Applications/MLX Serve Studio.app"`

## 三种用法

```bash
# A) 本机完整：装好双击，底栏显示「内核 … ok」就能用
#    内核没起来时，点中间「启动内核」按钮（App 自带 run_local_dev.sh）

# B) 指向局域网里已有内核的机器（那台有权重，这台只当界面）
open -a "/Applications/MLX Serve Studio.app" --args --server <node-ip>:11251

# C) 命令行参数
--port 11251            # 内核端口
--server host[:port]    # 内核地址（默认 127.0.0.1）
--url /#music           # 启动就停在音乐页
--snapshot out.png      # 截图后退出（调试）
```

## 依赖（App 自带内核代码，但不带模型权重）

- **Apple Silicon**（MLX 只支持苹果芯片）
- Python 环境带 mlx：`python3 -m venv ~/.venvs/mtx && ~/.venvs/mtx/bin/pip install mlx numpy`
- 模型权重（不在 App 内）：
  - 出歌 ACE-Step Turbo → `~/.mlx-serve/models/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit`
  - 对话 Qwen3 → `~/.ours/models/…`
  - 出图 Krea → `~/.ours/models/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8`
- 出歌的 MLX 引擎宿主跑在闭源 `/Applications/MLX-Serve.app` 里；没有它会退回 ACE-Step 兜底（慢很多）

> 装到一台没权重/没内核的 Mac 上，App 会老老实实停在「内核没在跑」那页，不会假装能用。

## 重新构建

```bash
./build_app.sh   # swiftc -O + 图标 + 内置 kernel + ad-hoc 签名
./package.sh     # 打 DMG + ZIP 到 dist/
```
