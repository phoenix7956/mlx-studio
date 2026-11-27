// MLX Serve Studio —— 桌面端 (AppKit + WKWebView)
// 壳子 只 做 三 件事: 保证 内核 在跑 / 把 Studio 控制台 装进 原生 窗口 / 给 菜单 与 快捷键.
// 界面 本体 仍是 那份 逆向 复刻 的 前端 —— 与 浏览器 里 看到 的 完全 一致.
// 用 纯 AppKit (不用 SwiftUI 宏): 这 台 机器 的 CommandLineTools 没有 SwiftUIMacros 插件.
import AppKit
import WebKit
import UniformTypeIdentifiers

// MARK: - 内核 状态
final class Kernel {
    static let shared = Kernel()
    private(set) var port: Int
    var portOverride: Int? {
        didSet { if let p = portOverride { port = p } }
    }
    let root: String
    private(set) var version: String? = nil
    private(set) var up = false
    // 内核 可以 在 本机 (127.0.0.1), 也 可以 是 局域网 里 另一 台 机器
    var host: String = "127.0.0.1"
    var hostOverride: String? {
        didSet { if let h = hostOverride { applyHost(h) } }
    }
    var isRemote: Bool { host != "127.0.0.1" && host != "localhost" }
    var base: String { "http://\(host):\(port)" }
    private func applyHost(_ spec: String) {
        // "192.168.1.5" / "192.168.1.5:11251" / "mac.local:11251"
        if let c = spec.lastIndex(of: ":"), let p = Int(spec[spec.index(after: c)...]) {
            port = p
            host = String(spec[..<c])
        } else {
            host = spec
        }
    }

    init(port: Int = 11251) {
        self.port = port
        // 找 内核 代码 目录: 先看 同包 上级, 再看 默认 路径
        var cands: [String] = []
        let here = Bundle.main.bundleURL.deletingLastPathComponent().path
        cands.append(here)
        let up = (here as NSString).deletingLastPathComponent
        cands.append(up)
        cands.append("\(up)/实现代码")          // 交付 包 里 自带 的 内核 源码
        cands.append("\(up)/../实现代码")
        // 安装 包 里 内置 的 内核 (app 自己 带 着 server.py)
        if let res = Bundle.main.resourceURL {
            cands.append(res.appendingPathComponent("kernel").path)
            cands.append(res.appendingPathComponent("实现代码").path)
        }
        // install.sh 装 完 的 内核 (普通 用户 走 的 是 这 条)
        cands.append("\(NSHomeDirectory())/.ours/mlx_server")
        // 开发 机器 上 的 源码 目录
        cands.append("\(NSHomeDirectory())/Desktop/rea/ours/mlx_server")
        self.root = cands.first { FileManager.default.fileExists(atPath: "\($0)/server.py") }
            ?? "\(NSHomeDirectory())/.ours/mlx_server"
    }

    func check(_ done: @escaping (Bool) -> Void) {
        var req = URLRequest(url: URL(string: "\(base)/health")!)
        req.timeoutInterval = 2
        req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        // 后端 是 HTTP/1.0 BaseHTTPServer, 连接 用完 就 关 —— 必须 每次 用 新 会话, 否则 复用 死 连接 会 假 失败
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 2
        cfg.httpShouldSetCookies = false
        cfg.urlCache = nil
        cfg.httpAdditionalHeaders = ["Cache-Control": "no-cache"]
        URLSession(configuration: cfg).dataTask(with: req) { data, resp, err in
            var ok = false
            var ver: String? = nil
            if let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
                ok = true
                if let data = data,
                   let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    ver = (j["version"] as? String) ?? (j["status"] as? String) ?? "ok"
                } else {
                    ver = "ok"
                }
            } else {
                let code = (resp as? HTTPURLResponse)?.statusCode
                let msg = err.map { "\($0)" } ?? ("http " + (code.map { String($0) } ?? "nil"))
                FileHandle.standardError.write(("健康检查 失败: \(msg)\n").data(using: .utf8)!)
                // 兜底: 裸 socket 再 探 一次 —— 绕开 CFNetwork 对 HTTP/1.0 断包 的 苛刻
                DispatchQueue.global(qos: .utility).async {
                    let v = self.rawHealth()
                    DispatchQueue.main.async {
                        self.up = v != nil; self.version = v
                        done(v != nil)
                    }
                }
                return
            }
            DispatchQueue.main.async {
                self.up = ok; self.version = ver
                done(ok)
            }
        }.resume()
    }

    // 裸 socket 探活: 不 依赖 HTTP 栈 —— 某些 服务 (HTTP/1.0 + 不 带 Content-Length)
    // 会 让 CFNetwork 直接 报 -1005, 但 原始 socket 拿 得到 响应.
    func rawHealth() -> String? {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        // 连接 超时: 先 设 非阻塞, 再 select
        let fl = fcntl(fd, F_GETFL, 0)
        _ = fcntl(fd, F_SETFL, fl | O_NONBLOCK)
        let cres = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                connect(fd, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if cres != 0 {
            if errno != EINPROGRESS { return nil }
            var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
            let n = poll(&pfd, 1, 2000)
            if n <= 0 { return nil }
            var err: Int32 = 0
            var len = socklen_t(MemoryLayout<Int32>.size)
            _ = withUnsafeMutablePointer(to: &err) { p in
                getsockopt(fd, SOL_SOCKET, SO_ERROR, p, &len)
            }
            if err != 0 { return nil }
        }
        let req = "GET /health HTTP/1.0\r\nHost: \(host)\r\nConnection: close\r\n\r\n"
        _ = req.withCString { write(fd, $0, strlen($0)) }
        var buf = [UInt8](repeating: 0, count: 8192)
        var acc = Data()
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            let n = read(fd, &buf, buf.count)
            if n > 0 { acc.append(contentsOf: buf[0..<n]); if acc.count > 1 << 20 { break } }
            else if n == 0 { break }
            else {
                var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
                if poll(&pfd, 1, 200) <= 0 { break }
            }
        }
        guard let text = String(data: acc, encoding: .utf8) else { return nil }
        guard let sep = text.range(of: "\r\n\r\n") else { return nil }
        let head = String(text[text.startIndex..<sep.lowerBound])
        guard head.contains("200") else { return nil }
        let body = String(text[sep.upperBound...])
        if let j = try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any] {
            return (j["version"] as? String) ?? (j["status"] as? String) ?? "ok"
        }
        return body.isEmpty ? "ok" : body.prefix(40).description
    }

    // 内核 没 在 跑 就 用 官方 脚本 拉 起来
    func start() {
        let script = "\(root)/run_local_dev.sh"
        guard FileManager.default.fileExists(atPath: script) else {
            NSSound.beep()
            let a = NSAlert()
            a.messageText = "找 不 到 内核 启动 脚本"
            a.informativeText = script
            a.runModal()
            return
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = [script, String(port)]
        var env = ProcessInfo.processInfo.environment
        env["MLX_NODE_ROLE"] = "local"
        // 脚本 里 用 $HOME 找 venv / 模型 —— 必须 给 真实 用户 目录, 否则 继承 到 奇怪 的 HOME 会 起 不 来
        env["HOME"] = NSHomeDirectory()
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        proc.environment = env
        proc.standardOutput = FileHandle.standardError
        proc.standardError = FileHandle.standardError
        do { try proc.run() } catch {
            let a = NSAlert(); a.messageText = "拉 不 起 内核"; a.informativeText = error.localizedDescription
            a.runModal(); return
        }
        AppDelegate.shared?.setStatus("正在 拉起 内核 …", ok: false)
        var tries = 0
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { t in
            tries += 1
            self.check { ok in
                if ok { t.invalidate(); AppDelegate.shared?.bringUp() }
                else if tries > 120 { t.invalidate(); AppDelegate.shared?.setStatus("内核 60 秒 内 没 起来", ok: false) }
            }
        }
    }
}

// MARK: - 状态条
final class StatusBar: NSView {
    private let dot = NSView()
    private let label = NSTextField(labelWithString: "连接 中 …")
    private let reloadBtn = NSButton(title: "重新加载", target: nil, action: nil)
    private let browserBtn = NSButton(title: "浏览器打开", target: nil, action: nil)
    private let outBtn = NSButton(title: "输出文件夹", target: nil, action: nil)
    var onReload: (() -> Void)?
    var onBrowser: (() -> Void)?
    var onOutputs: (() -> Void)?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        dot.wantsLayer = true
        dot.layer?.backgroundColor = NSColor.systemGray.cgColor
        dot.layer?.cornerRadius = 4
        label.font = .systemFont(ofSize: 11)
        label.textColor = .secondaryLabelColor
        reloadBtn.target = self; reloadBtn.action = #selector(reload)
        browserBtn.target = self; browserBtn.action = #selector(openBrowser)
        outBtn.target = self; outBtn.action = #selector(openOutputs)
        for b in [reloadBtn, browserBtn, outBtn] { b.bezelStyle = .inline; b.controlSize = .small; b.isBordered = true }
        addSubview(dot); addSubview(label); addSubview(reloadBtn); addSubview(browserBtn); addSubview(outBtn)
    }
    required init?(coder: NSCoder) { fatalError() }

    func setStatus(_ text: String, ok: Bool) {
        label.stringValue = text
        dot.layer?.backgroundColor = (ok ? NSColor.systemGreen : NSColor.systemRed).cgColor
    }
    override func layout() {
        super.layout()
        let h = bounds.height
        dot.frame = NSRect(x: 12, y: (h - 8) / 2, width: 8, height: 8)
        label.frame = NSRect(x: 26, y: (h - 14) / 2, width: 460, height: 14)
        let bw: CGFloat = 78
        browserBtn.frame = NSRect(x: bounds.width - bw - 12, y: (h - 22) / 2, width: bw, height: 22)
        reloadBtn.frame = NSRect(x: bounds.width - bw * 2 - 24, y: (h - 22) / 2, width: bw, height: 22)
        let ow: CGFloat = 88
        outBtn.frame = NSRect(x: bounds.width - bw * 2 - 24 - ow - 12, y: (h - 22) / 2, width: ow, height: 22)
    }
    @objc private func reload() { onReload?() }
    @objc private func openBrowser() { onBrowser?() }
    @objc private func openOutputs() { onOutputs?() }
}

// MARK: - 没 内核 时 的 兜底 视图
final class DownView: NSView {
    var onStart: (() -> Void)?
    private let title = NSTextField(labelWithString: "内核 没 在 跑")
    private let sub = NSTextField(labelWithString: "")
    private let btn = NSButton(title: "启动 内核", target: nil, action: nil)
    private let spinner = NSProgressIndicator()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        title.font = .systemFont(ofSize: 22, weight: .semibold)
        title.alignment = .center
        sub.font = .systemFont(ofSize: 11)
        sub.textColor = .secondaryLabelColor
        sub.alignment = .center
        btn.target = self; btn.action = #selector(start)
        btn.bezelStyle = .rounded
        spinner.style = .spinning; spinner.isHidden = true
        addSubview(title); addSubview(sub); addSubview(btn); addSubview(spinner)
    }
    required init?(coder: NSCoder) { fatalError() }
    private var remote = false
    func setInfo(_ s: String) { sub.stringValue = s }
    func setRemote(_ r: Bool) {
        remote = r
        btn.isHidden = r
        if r { title.stringValue = "连 不 上 内核" }
    }
    func setBusy(_ b: Bool) {
        btn.isHidden = b || remote
        spinner.isHidden = !b
        if b { spinner.startAnimation(nil) } else { spinner.stopAnimation(nil) }
    }
    override func layout() {
        super.layout()
        let cx = bounds.midX
        title.frame = NSRect(x: 0, y: bounds.midY + 30, width: bounds.width, height: 30)
        sub.frame = NSRect(x: 0, y: bounds.midY, width: bounds.width, height: 16)
        btn.frame = NSRect(x: cx - 60, y: bounds.midY - 46, width: 120, height: 30)
        spinner.frame = NSRect(x: cx - 8, y: bounds.midY - 44, width: 16, height: 16)
    }
    @objc private func start() { onStart?() }
}

// MARK: - 主程序
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    static private(set) var shared: AppDelegate?
    var window: NSWindow!
    var web: WKWebView!
    var statusBar: StatusBar!
    var downView: DownView!
    var timer: Timer!
    var pendingShot: String? = nil
    var webLoaded = false
    var checking = false
    var startPath: String? = nil
    var evalJS: String? = nil
    var evalDelay: Double = 5.0
    var evalResultKey: String? = nil
    var evalTries = 0
    var startMenuItem: NSMenuItem? = nil

    func applicationDidFinishLaunching(_ n: Notification) {
        AppDelegate.shared = self
        NSApp.setActivationPolicy(.regular)
        buildMenu()
        buildWindow()
        web.navigationDelegate = self
        // 命令行: --snapshot <路径>  导出 截图 后 退出 ; --port <端口>
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count { pendingShot = args[i + 1] }
        if let i = args.firstIndex(of: "--url"), i + 1 < args.count { startPath = args[i + 1] }
        if let i = args.firstIndex(of: "--eval"), i + 1 < args.count { evalJS = args[i + 1] }
        if let i = args.firstIndex(of: "--eval-delay"), i + 1 < args.count,
           let d = Double(args[i + 1]) { evalDelay = d }
        if let i = args.firstIndex(of: "--eval-result"), i + 1 < args.count { evalResultKey = args[i + 1] }
        if let i = args.firstIndex(of: "--port"), i + 1 < args.count,
           let p = Int(args[i + 1]) { Kernel.shared.portOverride = p }
        // --server <host[:port]> —— 指向 局域网 里 另一 台 有 权重 的 内核
        if let i = args.firstIndex(of: "--server"), i + 1 < args.count {
            let spec = args[i + 1]
            if spec.hasPrefix("http://") {
                Kernel.shared.hostOverride = String(spec.dropFirst(7))
            } else {
                Kernel.shared.hostOverride = spec
            }
        }
        // 参数 解析 发生 在 buildWindow 之后 —— 这里 补 一次 远程 状态
        downView.setRemote(Kernel.shared.isRemote)
        startMenuItem?.isEnabled = !Kernel.shared.isRemote
        poll()
        if args.contains("--autostart") && !Kernel.shared.isRemote { Kernel.shared.start() }
        if pendingShot != nil && evalJS == nil {
            // 兜底: 界面 没 加载 出来 (内核 没 跑) 时, 也 要 能 出 图
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
                guard let s = self, let path = s.pendingShot else { return }
                s.pendingShot = nil
                s.snapshot(to: path)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { exit(0) }
            }
        }
        // 探针: 不 等 didFinish (那个 回调 经常 不 来), 直接 反复 跑 eval 那 段 JS,
        // 直到 它 把 window.__MLX_EVAL_RESULT__ 填 上.
        if evalJS != nil {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
                self?.probeEval()
            }
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    func buildWindow() {
        let rect = NSRect(x: 0, y: 0, width: 1440, height: 900)
        window = NSWindow(contentRect: rect, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "MLX Serve Studio"
        window.setFrameAutosaveName("MLXServeStudio")
        window.center()
        window.minSize = NSSize(width: 1000, height: 640)

        let cfg = WKWebViewConfiguration()
        // --fresh: 用 一次性 数据存储 (不 读 本地 localStorage) —— 截 干净 首页 用
        cfg.websiteDataStore = CommandLine.arguments.contains("--fresh") ? .nonPersistent() : .default()
        cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")
        web = WKWebView(frame: .zero, configuration: cfg)
        web.allowsBackForwardNavigationGestures = true
        web.setValue(false, forKey: "drawsBackground")
        // 没 这个, 界面 里 "选择图片..." / "选择音频文件..." / "录制音频..." 全 都 点 不 动
        // (WKWebView 默认 不 给 <input type="file"> 弹 原生 选 文件 面板)
        web.uiDelegate = self

        statusBar = StatusBar(frame: .zero)
        statusBar.onReload = { self.reloadWeb() }
        statusBar.onBrowser = { if let u = URL(string: Kernel.shared.base + "/") { NSWorkspace.shared.open(u) } }
        statusBar.onOutputs = { self.openOutputFolder() }

        downView = DownView(frame: .zero)
        downView.onStart = { Kernel.shared.start(); self.downView.setBusy(true) }
        downView.setRemote(Kernel.shared.isRemote)

        let content = ContainerView(frame: rect)
        content.delegate = self
        content.addSubview(downView)
        content.addSubview(web)
        content.addSubview(statusBar)
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
    }

    func layoutSubviews() {
        guard let c = window.contentView else { return }
        let barH: CGFloat = 30
        statusBar.frame = NSRect(x: 0, y: 0, width: c.bounds.width, height: barH)
        let top = NSRect(x: 0, y: barH, width: c.bounds.width, height: c.bounds.height - barH)
        web.frame = top
        downView.frame = top
    }

    // MARK: 导出 界面 截图 (WebView 画面 + 底部 状态条)
    func snapshot(to path: String) {
        guard let content = window.contentView else { return }
        let barH = statusBar.bounds.height
        // 内核 没 起来 时, 画面 是 兜底 页 —— 直接 渲染 窗口, 不 走 WebView 快照
        if web.isHidden || !webLoaded {
            writePNG(contentView: content, to: path)
            return
        }
        let size = NSSize(width: web.bounds.width, height: web.bounds.height + barH)
        web.takeSnapshot(with: nil) { [weak self] snap, err in
            guard let self = self, let snap = snap else {
                FileHandle.standardError.write(("截图 失败: \(err?.localizedDescription ?? "未知")\n").data(using: .utf8)!)
                exit(2)
            }
            let canvas = NSImage(size: size)
            canvas.lockFocus()
            NSColor.windowBackgroundColor.setFill()
            NSRect(origin: .zero, size: size).fill()
            snap.draw(in: NSRect(x: 0, y: barH, width: size.width, height: size.height - barH))
            if let rep = self.statusBar.bitmapImageRepForCachingDisplay(in: self.statusBar.bounds) {
                self.statusBar.cacheDisplay(in: self.statusBar.bounds, to: rep)
                rep.draw(in: NSRect(x: 0, y: 0, width: size.width, height: barH))
            }
            canvas.unlockFocus()
            let out = NSBitmapImageRep(data: canvas.tiffRepresentation!)!
            if let png = out.representation(using: .png, properties: [:]) {
                try? png.write(to: URL(fileURLWithPath: path))
                FileHandle.standardError.write(("已 导出 \(path)\n").data(using: .utf8)!)
            }
        }
    }

    // 直接 把 窗口 内容 渲染 成 PNG
    func writePNG(contentView: NSView, to path: String) {
        guard let rep = contentView.bitmapImageRepForCachingDisplay(in: contentView.bounds) else { return }
        contentView.cacheDisplay(in: contentView.bounds, to: rep)
        if let png = rep.representation(using: .png, properties: [:]) {
            try? png.write(to: URL(fileURLWithPath: path))
            FileHandle.standardError.write(("已 导出 \(path)\n").data(using: .utf8)!)
        }
    }

    func poll() {
        // 上一 次 探活 还 没 回来 就 不 再 发 —— 避免 并发 请求 把 内核 队列 打爆
        if checking { return }
        checking = true
        Kernel.shared.check { [weak self] ok in
            guard let s = self else { return }
            s.checking = false
            if ok {
                s.setStatus("内核 \(Kernel.shared.base) · \(Kernel.shared.version ?? "ok")", ok: true)
                s.downView.isHidden = true
                s.web.isHidden = false
                if !s.webLoaded { s.webLoaded = true; s.reloadWeb() }
            } else {
                s.setStatus("内核 没 在 跑 · \(Kernel.shared.base)", ok: false)
                s.web.isHidden = true
                s.downView.isHidden = false
                s.downView.setInfo(Kernel.shared.isRemote
                    ? "地址 \(Kernel.shared.base)"
                    : "地址 \(Kernel.shared.base)  ·  代码 \(Kernel.shared.root)")
                s.downView.setBusy(false)
            }
        }
        // 定时器 只 建 一次! (之前 在 poll() 里 新建 → 每 轮 多 一个 定时器 → 数量 指数 增长,
        //  并发 探活 越 越多 → 把 内核 的 listen 队列 打爆 → 状态条 全 红)
        if timer == nil {
            timer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in self.poll() }
        }
    }

    @objc func reloadWeb() {
        guard let u = URL(string: Kernel.shared.base + (startPath ?? "/")) else { return }
        webLoaded = true
        web.load(URLRequest(url: u))
    }

    // 输出 目录 —— 一键 到 访达. 路径 向 内核 要 (可能 被 MLX_OUT_DIR 改过),
    // 拿 不 到 就 退回 默认 的 ~/Documents/MLX Studio 输出
    func openOutputFolder() {
        let fallback = NSHomeDirectory() + "/Documents/MLX Studio 输出"
        guard let u = URL(string: "\(Kernel.shared.base)/v1/outputs") else {
            NSWorkspace.shared.open(URL(fileURLWithPath: fallback)); return
        }
        var req = URLRequest(url: u)
        req.timeoutInterval = 4
        req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        // 后端 是 HTTP/1.0 BaseHTTPServer, 连接 用完 就 关 —— 必须 用 新 会话
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 4
        cfg.httpShouldSetCookies = false
        cfg.urlCache = nil
        cfg.httpAdditionalHeaders = ["Cache-Control": "no-cache"]
        URLSession(configuration: cfg).dataTask(with: req) { data, _, _ in
            var root: String? = nil
            if let data = data,
               let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                root = j["root"] as? String
            }
            DispatchQueue.main.async {
                let p = (root?.isEmpty == false ? root! : fallback)
                NSWorkspace.shared.open(URL(fileURLWithPath: p))
            }
        }.resume()
    }

    func setStatus(_ t: String, ok: Bool) { statusBar.setStatus(t, ok: ok) }

    // 内核 起来 之后: 收起 兜底 页, 把 界面 亮 出来
    func bringUp() {
        downView.isHidden = true
        web.isHidden = false
        if !webLoaded { reloadWeb() }
    }

    // MARK: 菜单
    func buildMenu() {
        let main = NSMenu()
        main.delegate = self
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "关于 MLX Serve Studio", action: #selector(about), keyEquivalent: "")
            .target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "隐藏", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(withTitle: "隐藏 其他", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)

        let kernelItem = NSMenuItem()
        let kMenu = NSMenu(title: "内核")
        kMenu.addItem(withTitle: "重新加载 界面", action: #selector(reloadWeb), keyEquivalent: "r").target = self
        kMenu.addItem(withTitle: "刷新 内核 状态", action: #selector(refresh), keyEquivalent: "").target = self
        kMenu.addItem(.separator())
        let startItem = NSMenuItem(title: "启动 内核", action: #selector(startKernel), keyEquivalent: "")
        startItem.target = self
        kMenu.addItem(startItem)
        startMenuItem = startItem
        kMenu.addItem(withTitle: "在 浏览器 打开", action: #selector(openBrowser), keyEquivalent: "").target = self
        kMenu.addItem(.separator())
        kMenu.addItem(withTitle: "导出 界面 截图", action: #selector(exportShot), keyEquivalent: "s").target = self
        kernelItem.submenu = kMenu
        main.addItem(kernelItem)

        let winItem = NSMenuItem()
        let wMenu = NSMenu(title: "窗口")
        wMenu.addItem(withTitle: "最小化", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        wMenu.addItem(withTitle: "缩放", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        wMenu.addItem(.separator())
        wMenu.addItem(withTitle: "前置 全部 窗口", action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
        winItem.submenu = wMenu
        main.addItem(winItem)

        NSApp.mainMenu = main
    }

    @objc func about() {
        let a = NSAlert()
        a.messageText = "MLX Serve Studio"
        a.informativeText = "本地 内核 \(Kernel.shared.base)\n代码 \(Kernel.shared.root)\n界面 = 逆向 复刻 的 Studio 控制台"
        a.runModal()
    }
    @objc func refresh() { poll() }
    @objc func startKernel() { Kernel.shared.start() }
    @objc func openBrowser() { if let u = URL(string: Kernel.shared.base + "/") { NSWorkspace.shared.open(u) } }
    @objc func exportShot() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "mlx-serve-studio.png"
        panel.allowedContentTypes = [.png]
        guard panel.runModal() == .OK else { return }
        snapshot(to: panel.url!.path)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) { timer?.invalidate() }

    // --eval 跑 完 (或者 没 跑 出 结果) 时, 如果 还 挂着 --snapshot, 就 顺带 截 图 再 退出.
    // 否则 只能 靠 启动 后 5 秒 的 兜底 定时器 —— 那 时 页面 常常 还没 切 到 目标 状态.
    func finishWithShotIfNeeded(_ extra: String) {
        guard let path = pendingShot else { exit(0) }
        pendingShot = nil
        // 让 页面 把 最后 一 帧 画 完
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let s = self else { exit(0) }
            s.snapshot(to: path)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { exit(0) }
        }
        _ = extra
    }

    // 探针 轮询 —— 页面 里 反复 跑 同一 段 JS, 直到 它 返回 非空 字符串.
    func probeEval() {
        guard let js = evalJS, let w = web else { exit(0) }
        evalTries += 1
        if evalTries > 150 {
            FileHandle.standardError.write("EVAL -> TIMEOUT (页面 一直 没 满足 条件)\n".data(using: .utf8)!)
            exit(2)
        }
        w.evaluateJavaScript(js) { _, _ in }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            w.evaluateJavaScript("(function(){ try { return window.__MLX_EVAL_RESULT__ || ''; } catch(e) { return ''; } })()") { v, _ in
                let s = (v as? String) ?? ""
                if !s.isEmpty {
                    FileHandle.standardError.write(("EVAL -> " + s + "\n").data(using: .utf8)!)
                    self.finishWithShotIfNeeded(s)
                    return
                }
                self.probeEval()
            }
        }
    }
}

extension AppDelegate: WKNavigationDelegate {
    // 轮询 页面 里 的 结果 变量 —— 页面 的 异步 脚本 跑 完 就 取 回 走
    func pollEvalResult(_ w: WKWebView, key: String, deadline: Date) {
        if Date() > deadline {
            FileHandle.standardError.write("EVAL -> TIMEOUT (没有拿到 \(key))\n".data(using: .utf8)!)
            exit(2)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
            w.evaluateJavaScript("window.\(key) || ''") { v, _ in
                let s = (v as? String) ?? ""
                if !s.isEmpty {
                    FileHandle.standardError.write(("EVAL -> " + s + "\n").data(using: .utf8)!)
                    self?.finishWithShotIfNeeded(s)
                }
                _ = self
                self?.pollEvalResult(w, key: key, deadline: deadline)
            }
        }
    }

    func webView(_ w: WKWebView, didFinish nav: WKNavigation!) {
        if let js = evalJS {
            // 探针 模式: 统一 由 probeEval() 驱动 —— didFinish 这个 回调 经常 不 来,
            // 等 它 会 让 --eval 卡 住 不 退出.
            _ = js
            return
            /*
            // 调试 口: 把 页面 里 的 真实 状态 打 出来.
            // --eval-result <全局变量名>: 页面 里 的 异步 脚本 跑 完 后 把 结果
            // 写 进 那个 全局 变量, 这里 轮询 取 回 —— 可以 写 真正 的 异步 测量.
            DispatchQueue.main.asyncAfter(deadline: .now() + evalDelay) { [weak self] in
                w.evaluateJavaScript(js) { _, _ in }
                guard let key = self?.evalResultKey, !key.isEmpty else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        w.evaluateJavaScript(js) { v, e in
                            let out = (v.map { String(describing: $0) } ?? "ERR " + (e?.localizedDescription ?? "?"))
                            FileHandle.standardError.write(("EVAL -> " + out + "\n").data(using: .utf8)!)
                            self?.finishWithShotIfNeeded(out)
                        }
                    }
                    return
                }
                self?.pollEvalResult(w, key: key, deadline: Date().addingTimeInterval(120))
            }
            return
            */
        }
        guard let path = pendingShot else { return }
        pendingShot = nil
        // 等 页面 把 模型 / 指标 拉 完 再 截图 (不然 会 截 到 「模型数 0」)
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            self?.snapshot(to: path)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { exit(0) }
        }
    }
}

// MARK: - 原生 文件 / 麦克风 —— 界面 里 的 "选择图片..." / "录制音频..." 靠 这个
//
// 没 有 这 个 委托, WKWebView 收 到 <input type="file"> 的 点击 时 **什么 都 不 会 发生**
// (不 弹 面板, 不 报错) —— 表现 就是 "按 了 没 反应".
// 同理, getUserMedia (录制音频) 也 需要 显式 授权, 否则 静默 失败.
extension AppDelegate: WKUIDelegate {

    // MARK: 选 文件 / 选 目录
    func webView(_ webView: WKWebView,
                 runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        FileHandle.standardError.write(("弹 选 文件 面板 (多选=\(parameters.allowsMultipleSelection))\n").data(using: .utf8)!)
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.canCreateDirectories = false
        panel.resolvesAliases = true
        panel.title = "选择文件"
        // 用 begin (独立 模态 面板) —— 不 依赖 窗口 是否 key, 一定 能 弹 出来
        panel.begin { resp in
            if resp == .OK, !panel.urls.isEmpty {
                FileHandle.standardError.write(("选了 \(panel.urls.count) 个文件\n").data(using: .utf8)!)
                completionHandler(panel.urls)
            } else {
                FileHandle.standardError.write(("选文件 取消\n").data(using: .utf8)!)
                completionHandler(nil)
            }
        }
    }

    // MARK: 麦克风 / 摄像头 —— "录制音频..." 用 getUserMedia
    func webView(_ webView: WKWebView,
                 requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo,
                 type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        FileHandle.standardError.write(("麦克风/摄像头 授权 -> 放行\n").data(using: .utf8)!)
        decisionHandler(.grant)
    }

    // MARK: 页面 里 target=_blank 的 链接 —— 丢 给 系统 浏览器
    func webView(_ webView: WKWebView,
                 createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let u = navigationAction.request.url { NSWorkspace.shared.open(u) }
        return nil
    }
}

// 窗口 尺寸 变化 时 重排 子视图
final class ContainerView: NSView {
    weak var delegate: AppDelegate?
    override func layout() {
        super.layout()
        delegate?.layoutSubviews()
    }
}

// 入口 (不用 SwiftUI 的 @main: 顶层 代码 需要 以 main.swift 命名 且 不能 -parse-as-library)
// 强制 App 语言 = 简体中文. AppKit 的 选 文件 面板 (侧栏 Recents/Applications/...) 和
// WebKit 的 右键 菜单 (Open Image in New Window/...) 都 按 **App 的 语言** 取字符串,
// 不是 按 系统 语言 —— bundle 里 没有 zh-Hans.lproj 就 全部 退回 英文.
// 这里 再 钉 一次 AppleLanguages, 保证 就算 换 了 台 没 装 lproj 的 机器 也 是 中文.
UserDefaults.standard.register(defaults: ["AppleLanguages": ["zh-Hans"]])
UserDefaults.standard.set(["zh-Hans"], forKey: "AppleLanguages")

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
