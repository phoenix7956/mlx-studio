"""我们 自己 的 mlx_server — 彻底 不 依 靠 闭源 MLX-Serve / ComfyUI.
第二台机 (远端 节点) 出 视 频/图; 本 机 出 歌/专辑 图/主控.
模 型 不 驻 留: 置 闲 30min 自 动 卸 或 手 动 /unload; 批 量 跑 批 不 逐 段 卸.
内 存 守 门: 可 用 <15G 拒 绝 起 视 频. 出 片 排 他 (一 次 一 片, 串 行).
/v1/* = 壳 兼 容 层 (MLX-Serve Studio 壳 直 接 调 这 组 端 点), 打 我 们 自 己 的 内 核."""
import os, sys, time, json, threading, argparse, subprocess, queue, re
import hashlib, base64
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import api_full
import outputs
try:
    import mlx.core as mx
except Exception:
    mx = None

IDLE_S = 30*60
NODE = os.environ.get("MLX_NODE_ROLE", os.environ.get("MINEPROXY_NODE","remote"))
# NODE = local (本 机) / remote (第二台机). 旧 名 MINEPROXY_NODE 仍 认.
ROUTE = {"video":"remote","image":"remote","music":"local"}

# 生成 产物 的 文件名 前缀 / 后缀 -> 素材库 的 类型
LIB_PREFIXES = ("mx_img_", "mx_v1_img_", "mx_v1img_", "mx_edit_", "mx_music_",
                "mx_music_synth_", "mx_sound_", "mx_tts_", "mx_video_", "mx_3d_",
                # outputs.KIND_PREFIX 里 的 那 几个 —— 漏 了 的 话, 自研 出歌 (mx_audio_)
                # 生成 完 卡片 不 进 资料库
                "mx_audio_", "mx_out_")
LIB_KIND = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
            ".wav": "audio", ".mp3": "audio", ".m4a": "audio", ".aac": "audio", ".flac": "audio",
            ".mp4": "video", ".mov": "video", ".webm": "video",
            ".glb": "3d", ".gltf": "3d"}
ACE_MUSIC = os.environ.get("ACE_MUSIC_URL","http://127.0.0.1:7861")
MIN_FREE_GIB = 15

# 引擎 的 大致 权重 占用 (GiB) —— 装 之前 先 看 内存 够 不够
ENGINE_WEIGHT_GIB = {"video": 25, "image": 15, "llm": 6}
# 装 一个 之前 要 先 卸 掉 冲突 的 引擎 —— 两个 大 引擎 同时 在 内存 里 就 会 撞爆
ENGINE_CONFLICT = {"video": ("image",), "image": ("video",), "llm": ("video", "image")}

# 路由 -> 会 用 到 的 引擎. 请求 进行 中 的 引擎 不能 被 置闲 卸载 (否则 会 跑到 一半 被 卸).
PATH_KINDS = {
    "/v1/chat/completions": ("llm",), "/v1/completions": ("llm",),
    "/v1/messages": ("llm",), "/v1/responses": ("llm",),
    "/v1/responses/compact": ("llm",),
    "/v1/embeddings": ("embeddings",),
    "/v1/images/generations": ("image",), "/v1/images/edits": ("image", "edits", "tts"),
    "/v1/audio/speech": ("tts", "tts_neural"),
    "/v1/audio/music-generations": ("music",),
    "/v1/audio/sound-generations": ("sound",),
    "/v1/audio/transcriptions": ("stt",),
    "/v1/video/generations": ("video",),
    "/v1/3d/generations": ("3d",),
    "/v1/decisions": ("llm", "decisions"), "/v1/systemone": ("llm", "decisions"),
    "/api/chat": ("llm",), "/api/generate": ("llm",),
    "/api/embed": ("embeddings",), "/api/embeddings": ("embeddings",),
}

class MemGuardError(RuntimeError):
    """内存 不够 / 引擎 冲突, 拒绝 加载 (防 内存 崩, 别 让 机器 被 拖死)."""

ACE_LABEL = "com.ours.acestep"
ACE_PLIST = os.path.expanduser("~/Library/LaunchAgents/com.ours.acestep.plist")

def _ace_running():
    """出歌 服务 (ACE-Step, :7861) 在 不 在."""
    try:
        import urllib.request
        urllib.request.urlopen(ACE_MUSIC+"/health", timeout=4).read()
        return True
    except Exception:
        return False

def _ace_start(timeout=120):
    """起 出歌 服务 (它 挂着 音乐 模型 权重 ~23G 显存)."""
    if _ace_running(): return True
    try:
        import subprocess
        uid=os.getuid()
        # 先 把 job bootstrap 回来 (如果 之前 被 bootout 过), 再 start
        if os.path.exists(ACE_PLIST):
            subprocess.run(["launchctl","bootstrap","gui/%d"%uid,ACE_PLIST],
                           capture_output=True,timeout=30)
        subprocess.run(["launchctl","start",ACE_LABEL],capture_output=True,timeout=30)
    except Exception:
        return False
    t0=time.time()
    while time.time()-t0<timeout:
        if _ace_running(): return True
        time.sleep(2)
    return False

def _ace_stop():
    """停 出歌 服务 —— 释放 它 占 着 的 ~23G 显存.

    注意: 那个 plist 是 keepalive 的, 光 launchctl stop 会被 立刻 拉 回来,
    所以 这里 用 bootout 把 job 摘掉 —— 要 用 的 时候 再 bootstrap 回来.
    """
    try:
        import subprocess
        uid=os.getuid()
        subprocess.run(["launchctl","bootout","gui/%d/%s"%(uid,ACE_LABEL)],
                       capture_output=True,timeout=30)
        subprocess.run(["launchctl","stop",ACE_LABEL],capture_output=True,timeout=30)
    except Exception:
        return False
    t0=time.time()
    while time.time()-t0<30:
        if not _ace_running(): return True
        time.sleep(1)
    return not _ace_running()

def _engine_cn(kind):
    """引擎 key -> 人 看 的 名字 (给 报错 用)."""
    return {"video":"视频 模型", "image":"图像 模型", "llm":"大模型",
            "tts":"语音 模型", "tts_neural":"神经 语音 模型",
            "embeddings":"向量 模型", "sound":"音效 模型", "3d":"3D 模型",
            "stt":"转写 模型", "decisions":"决策 模型", "music":"音乐 模型",
            "voice":"配音 模型"}.get(kind, kind)


def _is_wan_model(mid):
    """模型 id -> 是 不是 Wan2.2 那 条 线 (Wan2 引擎 跟 H3 引擎 不一样)."""
    return bool(mid) and str(mid).lower().startswith("wan2")


class Result:
    def __init__(self): self.e=threading.Event(); self.v=None
    def set_result(self,v): self.v=v; self.e.set()
    def wait(self,timeout): 
        if self.e.wait(timeout): return self.v
        return {"error":"超 时"}

# 视频 排 他: 出片 时 DiT 常驻 ~25G, 别的 大 引擎 装 不 上.
# 闭源 也是 串行 (一次 出片 会 挡 住 chat 几 分钟), 我们 用 等 + 明确 报错, 不 是 死锁.
VIDEO_WAIT_MAX = 600

# ============================================================ WebSocket (WS /v1/responses)
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_accept_key(key):
    """RFC6455 的 Sec-WebSocket-Accept."""
    return base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()).decode()


def _b64_file(path):
    """读 文件 -> base64 (给 前端 直接 播 / 下载)."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return ""


def ws_read_frame(rfile):
    """读 一 个 WS 帧 -> (opcode, payload bytes). opcode: 1 文本 / 8 关闭 / 2 二进制."""
    hdr = rfile.read(2)
    if len(hdr) < 2:
        return None, None
    b0, b1 = hdr[0], hdr[1]
    op = b0 & 0x0F
    masked = bool(b1 & 0x80)
    ln = b1 & 0x7F
    if ln == 126:
        ln = int.from_bytes(rfile.read(2), "big")
    elif ln == 127:
        ln = int.from_bytes(rfile.read(8), "big")
    mask = rfile.read(4) if masked else None
    data = rfile.read(ln) if ln else b""
    if mask:
        data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
    return op, data


def ws_write(wfile, payload, opcode=1):
    """写 一 个 WS 帧 (server -> client 不 加 掩码)."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    n = len(data)
    hdr = bytes([0x80 | opcode])
    if n < 126:
        hdr += bytes([n])
    elif n < 65536:
        hdr += bytes([126]) + n.to_bytes(2, "big")
    else:
        hdr += bytes([127]) + n.to_bytes(8, "big")
    wfile.write(hdr + data)
    wfile.flush()

def _flush():
    if mx is None: return
    for a in ("metal_flush","flush"):
        f=getattr(mx,a,None)
        if callable(f):
            try: f(); return
            except Exception: pass
    try: mx.metal.flush()
    except Exception: pass

def free_gib():
    try:
        o=subprocess.check_output(["vm_stat"],text=True)
        def g(k):
            for ln in o.splitlines():
                if k in ln:
                    try: return int(ln.split(":")[1].strip().rstrip("."))*16384/1073741824
                    except Exception: return 0
            return 0
        return g("free")+g("inactive")
    except Exception:
        return None

_gpu_cache = {"t": 0.0, "v": None}

def gpu_util_pct():
    """真 GPU 占用 % (Apple Silicon: ioreg IOAccelerator 的 "Device Utilization %").

    之前 这 个 指标 是 用 "100 - 空闲内存" 假 出来 的 —— 空闲 100G 的 机 上 会 显示
    负数 (-13.6%). 现在 读 真 值, 读 不 到 就 回退 0.
    """
    now = time.time()
    if _gpu_cache["v"] is not None and now - _gpu_cache["t"] < 1.5:
        return _gpu_cache["v"]
    v = 0.0
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=2).stdout
        import re as _re
        m = _re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            v = float(m.group(1))
    except Exception:
        v = 0.0
    _gpu_cache["t"] = now
    _gpu_cache["v"] = v
    return v

class Srv:
    # 原版 服务 参数 (MECHANICS §13) —— 我们 内核 自己 的 默认 值
    MIN_FREE_GIB = MIN_FREE_GIB
    os_reserve_gib = 4          # --os-reserve-gib (原版 默认 = RAM/8, 钳 2..8)
    wired_margin_gib = 8        # --wired-margin-gib (原版 默认 8, 取值 2..32)
    idle_evict_secs = IDLE_S    # --idle-evict-secs (我们 默认 30min, 原版 默认 关)
    max_resident_models = 3     # --max-resident-models
    prefix_cache_mem = 0        # --prefix-cache-mem (0 = 关)
    prefix_cache_disk = 0       # --prefix-cache-disk (0 = 关)
    ctx_size = 32768            # --ctx-size
    max_tokens = 100            # --max-tokens (原版 serve 默认)
    api_key = ""                # --api-key (空 = 不 鉴权)
    metrics_on = False          # --metrics
    def __init__(self):
        # RLock: load() 持 锁 时 会 调 unload() (装 LLM 前 先 卸 视频/图像),
        # 普通 Lock 会 自己 死锁 —— 表现 是 请求 永久 挂起, LLM 永远 装 不 上.
        self.lock=threading.RLock(); self.engines={}; self._last={}; self._stop=False
        # 正在 用 的 引擎 (请求 进行 中) —— 置闲 卸载 不能 把 它 卸 了
        self._inflight={}; self._inflight_lock=threading.Lock()
        self.video_lock=threading.RLock()   # RLock: _do_video 内部 也 会 拿 (老 /video 路由 已 拿 过)
        self._video_busy=False          # 出片 中 (视频 引擎 持有 ~25G, 别的 引擎 装 不 上)
        self._image_busy=False         # 出图 中 (老 标志)
        self._image_running=0          # 正在 出图 的 请求 数 (换 模型 时 看 有没有 别人 在 跑)
        self.sessions={}; self.session_lock=threading.Lock()
        self.gen_count=0; self.gen_err=0; self.gen_bytes=0
        self.peak_rss=0; self.metrics_lock=threading.Lock()
        # 原版 Responses 存储 + 请求 计数 (MECHANICS §6 指标 源)
        self.responses_store={}; self.resp_lock=threading.Lock()
        self.req_ok=0; self.req_err=0; self.running=0
        self.prompt_tokens=0; self.generation_tokens=0
        self.start_time=int(time.time())
        self.model_dirs=[]; self._scanned=[]
        # 用户 「选择 的 那 个」 模型 (出图 / 出片 各 一个) —— 生成 时 不带 model
        # 参数 就 用 它, 别 随便 挑 一个 在位 的.
        self._image_mid_pref=None; self._video_mid_pref=None
        self.library=[]; self.lib_lock=threading.Lock()
        # 上 一次 出片 的 实测 速度 (界面 估 预计 耗时 用)
        self.last_video_sec=0.0; self.last_video_steps=0
        self.last_video_w=0; self.last_video_h=0; self.last_video_frames=0
        # 出片 速度 模型: t_step ≈ a·rows + b·rows²   (rows = 打包 序列 行数)
        #   · rows 越大 越 接近 平方 (注意力 是 平方 的), 所以 用 二次项 拟合
        #   · 界面 用 它 换算 「换个 画布/帧数/步数 要 多久」
        self.last_video_rows=0; self.last_video_sec_per_step=0.0
        self.video_fit_a=0.0; self.video_fit_b=0.0
        self.video_fit_n=0
        self.video_samples=[]            # [(rows, sec_per_step), ...] 最近 8 次
        # —— 资料库 索引 的 落盘 位置 (在 内核 目录, **不** 在 ~/Documents) ——
        # macOS 的 坑: 内核 是 launchd 起的 (没有 GUI 会话), 它 去 枚举 ~/Documents
        # 会被 系统 卡 住 (opendir 阻塞 —— 没 权限 时 等 一个 永远 不来 的 授权),
        # 表现 就是 「点 资料库 → 整个 界面 卡死」. 所以:
        #   · 索引 落 在 ~/.ours/mlx_server/library.json —— 读 写 都 不碰 ~/Documents
        #   · 磁盘 扫描 放 后台 线程, 带 超时; 卡 住 就 放弃, 绝不 拖 死 请求
        self._lib_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"library.json")
        self._lib_disk=[]; self._lib_scan_state="idle"   # idle / running / done / blocked
        self._lib_scan_t=0.0
        self._load_library()
        self.decode_tok_s=0.0; self.prefill_tok_s=0.0; self.ttft_ms=0.0; self._ttft_n=0
    def models_resp(self):
        """给 api_full 用: 模型 清单 (catalog + --model-dir 扫描 + 在位 状态)."""
        # 扫描 出来 的 模型 要 一并 返回 —— 漏 掉 的 话 --model-dir / 「重扫」 都 白 做
        return _models_resp(self.model_dirs, self._scanned)

    def _video_mid_default(self):
        """没 指定 视频 模型 时 用 哪个 —— 跟 界面 「视频 模型」 默认 一致."""
        try:
            cur=getattr(self,"_video_mid_pref",None)
            if cur: return cur
        except Exception:
            pass
        return "h3-4bit"

    def _image_mid_default(self):
        """没 指定 出图 模型 时 用 哪个 —— 界面 选 的 那 个, 否则 第一个 在位 的."""
        try:
            cur=getattr(self,"_image_mid_pref",None)
            if cur: return cur
        except Exception:
            pass
        for m in self.models_resp().get("models",[]):
            if m.get("cap")=="image" and m.get("present"):
                return m.get("id")
        return "krea-turbo"

    def scan_model_dirs(self):
        """重扫 --model-dir (POST /v1/models/rescan 用)."""
        self._scanned = _scan_model_dirs(self.model_dirs)
        return self._scanned
    def default_model(self):
        """默认 模型: 第一个 在位 的 llm 模型."""
        for m in self.models_resp().get("models",[]):
            if m.get("cap")=="llm" and m.get("present"): return m.get("id")
        return ""
    def free_gib(self):
        return free_gib()
    def store_response(self,rid,body):
        with self.resp_lock: self.responses_store[rid]=body
    def get_response(self,rid):
        with self.resp_lock: return self.responses_store.get(rid)
    def delete_response(self,rid):
        with self.resp_lock: return self.responses_store.pop(rid,None) is not None
    def rss_gib(self):
        try:
            import subprocess as _sp
            out=_sp.check_output(["ps","-o","rss=","-p",str(os.getpid())],text=True).strip()
            return int(out)*1024/1073741824
        except Exception:
            return 0.0
    def mlx_mem(self):
        """MLX 真实 的 显存 占用 (active / cache) —— 不是 ps 的 RSS."""
        a=c=0
        try:
            import mlx.core as _mx
            a=int(_mx.get_active_memory()); c=int(_mx.get_cache_memory())
        except Exception:
            pass
        return a,c
    def metrics_snapshot(self):
        with self.metrics_lock:
            cnt,er,byt=self.gen_count,self.gen_err,self.gen_bytes
        fr=free_gib()
        mla,mlc=self.mlx_mem()
        return {"node":NODE,"engines":list(self.engines),"free_gib":round(fr,1) if fr is not None else None,
                "gen_count":cnt,"gen_err":er,"gen_bytes":byt,"rss_gib":round(self.rss_gib(),1),"idle_s":getattr(self,"idle_evict_secs",IDLE_S),
                # 原版 指标 名 (MECHANICS §6) —— 壳 的 监控 卡 也 认 这 套
                "prompt_tokens":self.prompt_tokens,"generation_tokens":self.generation_tokens,
                "req_ok":self.req_ok,"req_err":self.req_err,"running":self.running,
                "gpu_utilization_pct":round(gpu_util_pct(),1),
                "mlx_active_bytes":mla,"mlx_cache_bytes":mlc,
                "prefix_cache_hits":0,"prefix_cache_queries":0,"start_time":self.start_time,
                # 控制台 监控 卡 用 的 指标 (名 与 闭源 面板 对齐)
                "decode":round(self.decode_tok_s,1),
                "prefill":round(self.prefill_tok_s,1),
                "running":self.running,
                "ttft":round(self.ttft_ms),
                "cache":0.0,
                "gpu":round(gpu_util_pct(),1),
                "mem":round(self.rss_gib()*1024),
                "gen":self.generation_tokens,
                # 同一批 值 的 另一套 键名 (壳 的 监控 卡 按 这些 键 取 值)
                "decode_tps":round(self.decode_tok_s,1),
                "decode_tok_s":round(self.decode_tok_s,1),
                "prefill_tps":round(self.prefill_tok_s,1),
                "prefill_tok_s":round(self.prefill_tok_s,1),
                "running_requests":self.running,
                "ttft_ms":round(self.ttft_ms),
                "time_to_first_token_ms":round(self.ttft_ms),
                "cache_hit_rate":0.0,
                "prefix_cache_hit_rate":0.0,
                "gpu_utilization":round(gpu_util_pct(),1),
                "gpu_percent":round(gpu_util_pct(),1),
                "process_mem_mb":round(self.rss_gib()*1024),
                "process_memory_mb":round(self.rss_gib()*1024),
                "generated_total":self.generation_tokens,
                "generated_tokens":self.generation_tokens,
                "updated":time.strftime("%H:%M:%S")}
    def _video_windows(self,q,kf_imgs,anchors,ref_imgs=None):
        """出片 (支持 **链式窗口**): chain_windows>1 时, 逐 个 窗口 生成,
        上一 个 窗口 的 末帧 作为 下一 个 窗口 的 首帧 条件 —— 画面 连续, 不 断片.
        (闭源 的 H3 chain_windows=1-6 就 是 这个 意思: 长 片 由 窗口 串 起来.)"""
        import numpy as np
        eng=self.engines["video"]
        prompt=q.get("prompt",[""])[0]
        # 每个 窗口 一 条 提示词 (ComfyUI Continuum 的 "Sequence Prompt" 语义).
        # 不 给 就 全部 窗口 用 同 一 条 —— 那 样 只是 把 一段 片子 重复 N 遍, 不是 长片.
        _cp=q.get("chain_prompts") or []
        chain_prompts=[str(x).strip() for x in _cp if str(x).strip()]
        sec=float(q.get("sec",["5"])[0])
        steps=int(q.get("steps",["8"])[0])
        seed=int(q.get("seed",["0"])[0])
        h_=q.get("height",[None])[0]; w_=q.get("width",[None])[0]
        windows=max(1,min(6,int(q.get("chain_windows",["1"])[0] or 1)))
        if windows==1:
            return eng.generate(prompt,sec,steps,seed,
                                height=int(h_) if h_ else None,
                                width=int(w_) if w_ else None,
                                images=kf_imgs or None,
                                keyframe_anchors=tuple(anchors),
                                ref_images=ref_imgs or None)
        per=max(1.0,sec/windows)
        vids=[]; auds=[]; sr=None; fps=None
        cur=list(kf_imgs) if kf_imgs else []
        cur_anchors=tuple(anchors)
        if "last" in cur_anchors:
            keep=[i for i,a in zip(kf_imgs or [],anchors) if a!="last"]
            cur=keep
            cur_anchors=tuple(a for a in cur_anchors if a!="last")
        for wi in range(windows):
            # 窗口 多了 就 用 最后 一 条 兜底 (跟 ComfyUI Continuum 一样: 不足 的 补 尾条)
            if chain_prompts:
                wp = chain_prompts[wi] if wi < len(chain_prompts) else chain_prompts[-1]
            else:
                wp = prompt
            r=eng.generate(wp,per,steps,seed+wi*1000,
                           height=int(h_) if h_ else None,
                           width=int(w_) if w_ else None,
                           images=cur or None,
                           keyframe_anchors=cur_anchors,
                           ref_images=ref_imgs or None)
            vids.append(r.video); auds.append(r.audio); sr=r.sample_rate; fps=r.fps
            if wi<windows-1:
                from PIL import Image
                cur=[Image.fromarray(np.ascontiguousarray(r.video[-1]))]
                cur_anchors=("first",)
        video=np.concatenate(vids,axis=0)
        audio=np.concatenate(auds,axis=1)
        class _R: pass
        o=_R(); o.video=video; o.audio=audio; o.sample_rate=sr; o.fps=fps
        return o
    def _do_video(self,q,done=None):
        sid=q.get("session",[None])[0]
        hold=sid is not None and self.session_active(sid)
        # 单飞 + 排他: 一次 只 出 一 片; 出片 期间 标记 busy,
        # 别的 引擎 (LLM) 装 不 上 时 会 等 + 明确 报错, 不 是 死锁.
        got=self.video_lock.acquire(timeout=VIDEO_WAIT_MAX)
        if not got:
            raise RuntimeError("已 有 出片 在 跑, 一次 只 出 一 片, 稍 后 再试")
        self._video_busy=True
        try:
            # 视频 模型 (h3-4bit / wan2-ti2v-5b) —— 决定 用 哪个 引擎
            mid=(q.get("model") or [None])[0]
            self.load("video", mid)
            t0=time.time()
            # 闭源 契约 的 扩展 参数: 画布 / 首尾 关键帧
            h_=q.get("height",[None])[0]; w_=q.get("width",[None])[0]
            kf_imgs=[]; anchors=[]
            for key,anc in (("first_frame_image","first"),("last_frame_image","last")):
                v=q.get(key,[None])[0]
                if v:
                    try:
                        from PIL import Image
                        import io as _io, base64 as _b64
                        raw=v.split(",",1)[-1] if "," in v else v
                        im=Image.open(_io.BytesIO(_b64.b64decode(raw))).convert("RGB")
                        kf_imgs.append(im); anchors.append(anc)
                    except Exception as e:
                        print("[video] %s 读 不 了: %s" % (key,e),file=sys.stderr,flush=True)
            # 参考图 (H3 官方 reference 模式): 只 进 文本塔, 不 当 关键帧 ——
            # 提示词 里 用 <Picture N> 指代 (官方 文案 写 @图片N)。官方 上限: 图片 9 / 视频 3 / 音频 3。
            ref_imgs=[]
            for v in (q.get("ref_images") or [])[:9]:
                if not v: continue
                try:
                    from PIL import Image
                    import io as _io, base64 as _b64
                    raw=v.split(",",1)[-1] if "," in v else v
                    ref_imgs.append(Image.open(_io.BytesIO(_b64.b64decode(raw))).convert("RGB"))
                except Exception as e:
                    print("[video] 参考图 读 不 了: %s" % e,file=sys.stderr,flush=True)
            res=self._video_windows(q,kf_imgs,tuple(anchors),ref_imgs or None)
            # —— 记 一次 实测 速度 (界面 用 它 估 预计 耗时) ——
            try:
                _st=int((q.get("steps") or ["0"])[0] or 0)
                _nf=int((q.get("num_frames") or ["0"])[0] or 0)
                self.last_video_sec=round(time.time()-t0,1)
                self.last_video_steps=_st
                self.last_video_w=int(w_ or 0); self.last_video_h=int(h_ or 0)
                self.last_video_frames=_nf
                # 真实 画布 / 帧数 / 行数 —— 引擎 知道, 别 猜
                _vw=int(getattr(res,"video").shape[2]) if getattr(res,"video",None) is not None else 0
                _vh=int(getattr(res,"video").shape[1]) if getattr(res,"video",None) is not None else 0
                if _vw and _vh:
                    self.last_video_w=_vw; self.last_video_h=_vh
                _vf=int(getattr(res,"video").shape[0]) if getattr(res,"video",None) is not None else 0
                if _vf:
                    self.last_video_frames=_vf
                _rows=int(getattr(res,"sequence_length",0) or 0)
                _sps=float(getattr(res,"seconds_per_step",0) or 0)
                self.last_video_rows=_rows; self.last_video_sec_per_step=round(_sps,3)
                if _rows>0 and _sps>0:
                    self.video_samples.append((float(_rows),float(_sps)))
                    self._fit_video_speed()
                print("[video] 实测: %.1f 秒 (%s 步, %sx%s, %s 帧, %s 行, %.1f 秒/步)" % (
                    self.last_video_sec,_st,self.last_video_w,self.last_video_h,
                    self.last_video_frames,_rows,_sps),file=sys.stderr,flush=True)
            except Exception:
                pass
            # 批量 出片 时 产物 落 批次 目录 (out_dir 传 进来) —— 不然 一堆 片 混 在 视频 根目录
            _odir=q.get("out_dir",[None])[0]
            if _odir:
                try:
                    os.makedirs(_odir, exist_ok=True)
                    out=outputs.path_for("video","mx_video","",sub=os.path.basename(_odir.rstrip("/")))
                except Exception:
                    out=outputs.base_for("video","mx_video")
            else:
                out=outputs.base_for("video","mx_video")
            try:
                from minimax_h3_mlx.media import save_mp4
                save_mp4(out+".mp4",res.video,res.fps,res.audio,res.sample_rate)
                fr=f"file://{out}.mp4"
            except Exception:
                from minimax_h3_mlx.media import save_frames,save_wav
                save_frames(out,res.video); save_wav(out+".wav",res.audio,res.sample_rate); fr=f"file://{out}"
        finally:
            self._video_busy=False
            self.video_lock.release()
        # 出完不卸 — 只由 置 闲 计 时 器 (30min) 或 手 动 /unload 决 定 卸
        self.touch("video")
        with self.metrics_lock:
            self.gen_count+=1; self.gen_bytes+=int(res.video.shape[0])
        # 可选回传: local_out=<dir> 时, 把产物 scp 到调用方本机, 返回本机真实路径
        local_out=q.get("local_out",[None])[0]
        if local_out and NODE=="remote":
            import subprocess as _sp
            import remote_node
            _sp.run(["scp","-o","ConnectTimeout=10",f"{remote_node.target()}:{out}.mp4",
                    os.path.expanduser(local_out)], capture_output=True, timeout=300)
            local_path=os.path.join(os.path.expanduser(local_out), os.path.basename(out+".mp4"))
            if os.path.exists(local_path):
                fr=f"file://{local_path}"
        return {"url":fr,"sec":round(time.time()-t0,1),"frames":int(res.video.shape[0]),"unloaded":False,"session":sid}
    def _fit_video_speed(self):
        """用 最近 几次 实测 拟合 「每步 耗时」 与 「序列 行数」 的 关系.

        t_step ≈ a·rows + b·rows²
        —— 线性 项 是 MLP/归一化 的 开销, 平方 项 是 注意力 的 开销.
        样本 够 (>=2) 就 最小二乘; 只有 1 个 就 当 纯 线性.
        """
        try:
            s=[(float(r),float(t)) for (r,t) in self.video_samples
               if r and t and r>0 and t>0]
            if not s: return
            s=s[-8:]
            self.video_samples=s
            self.video_fit_n=len(s)
            if len(s)==1:
                self.video_fit_a=s[0][1]/s[0][0]; self.video_fit_b=0.0
                return
            S1=S2=S3=T1=T2=0.0
            for r,t in s:
                r2=r*r
                S1+=r2; S2+=r*r2; S3+=r2*r2
                T1+=r2*t; T2+=r2*r*t
            det=S1*S3-S2*S2
            if abs(det)<1e-9:
                self.video_fit_a=sum(t/r for r,t in s)/len(s); self.video_fit_b=0.0
                return
            a=(T1*S3-T2*S2)/det
            b=(S1*T2-S2*T1)/det
            if a<0 and b>0: a=0.0
            if b<0: b=0.0
            self.video_fit_a=a; self.video_fit_b=b
        except Exception:
            pass
    def open_session(self, kind="video"):
        import uuid
        sid=uuid.uuid4().hex[:12]
        with self.session_lock:
            self.sessions[sid]={"kind":kind,"last":time.time(),"keep_gib":40 if kind=="video" else 10}
        return sid
    def close_session(self, sid, unload=True):
        """关 会话.

        unload=False —— 只 关 会话, **不 卸 引擎**: 让 置闲 计时器 按 30min 规则 自己 卸.
        (批量 / MV 跑 完 别 马上 卸 —— 下 一 批 还得 重 载 25G 权重, 慢 好几 分钟.)
        """
        with self.session_lock:
            s=self.sessions.pop(sid,None)
        if s:
            if unload:
                self.unload(s["kind"]); print(f"[mlx_server] session {sid} 关, 卸 {s['kind']}",file=sys.stderr,flush=True)
            else:
                self.touch(s["kind"])
                print(f"[mlx_server] session {sid} 关 (引擎 留 着, 置闲 {getattr(self,'idle_evict_secs',IDLE_S)}s 后 自动 卸)",file=sys.stderr,flush=True)
        return s is not None
    def session_active(self, sid):
        with self.session_lock:
            s=self.sessions.get(sid)
            return s and time.time()-s["last"]<30*60
    def keepalive(self, sid):
        with self.session_lock:
            s=self.sessions.get(sid)
            if s: s["last"]=time.time()
    def load(self,kind,mid=None):
        with self.lock:
            # —— 出图 有 两 套 引擎 (Krea-2-Turbo / mflux 的 Z-Image、MOSS), 按 mid 选 ——
            # 换 模型 必须 先 卸 旧 的: 两 份 权重 加 起来 会 撞爆 内存.
            if kind=="image":
                mid=mid or self._image_mid_default()
                cur=self.engines.get("image")
                if cur is not None and getattr(cur,"model_id",None)==mid:
                    self.touch(kind); return
                if cur is not None:
                    # 注意: 不能 用 _engine_in_use("image") —— /v1/images/generations
                    # 自己 就 把 image 标成 inflight 了, 换 模型 时 会 撞上 自己.
                    if self._image_running>0:
                        raise MemGuardError(
                            "出图 引擎 还 在 用（%s），占着内存。先 等 它 出完，"
                            "再 换 模型。" % _engine_cn("image"))
                    self.unload("image")
            # —— 视频 有 两个 引擎 (H3 / Wan2.2), 按 mid 选 ——
            # 换 模型 时 必须 先 卸 旧 的: 两个 DiT 加 起来 会 撞爆 内存.
            # (不能 走 下面 「kind in self.engines 就 直接 return」 那 条 ——
            #  不然 已经 装 着 H3 的 时候 点 「加载 Wan2」 会 啥 也 没 发生.)
            if kind=="video":
                mid=mid or self._video_mid_default()
                cur=self.engines.get("video")
                if cur is not None and getattr(cur,"model_id",None)==mid:
                    self.touch(kind); return
                if cur is not None:
                    if self._engine_in_use("video"):
                        raise MemGuardError(
                            "出片 引擎 还 在 用（%s），占着内存。先 等 它 出完，"
                            "再 换 模型。" % _engine_cn("video"))
                    self.unload("video")
            elif kind=="voice":
                # 配音 工作台 的 TTS 引擎 —— 跟 出图/出片 一样, 按 mid 选;
                # 换 引擎 先 卸 旧 的 (两个 TTS 权重 同时 挂 着 白 占 内存).
                import voice_studio as VS
                st=VS.get_studio()
                eid=""
                if mid:
                    eid=str(mid)
                    if eid.startswith("voice-"): eid=eid[6:]
                cur=getattr(st,"_model_id","")
                if cur and eid and cur!=eid:
                    if self._engine_in_use("voice"):
                        raise MemGuardError(
                            "配音 引擎 还 在 用（%s），占着内存。先 等 它 出完，"
                            "再 换 引擎。" % _engine_cn("voice"))
                    st.unload()
                self.engines["voice"]=st
                if eid:
                    try:
                        st._load_engine(eid)
                    except Exception as e:
                        raise MemGuardError("配音 引擎 %s 加载 失败: %s" % (mid, e))
                self.touch(kind); return
            elif kind in self.engines:
                self.touch(kind); return
            # ① 冲突 的 引擎: 装 之前 先 卸 (视频 DiT ~25G + 图像 ~15G = 撞爆)
            for c in ENGINE_CONFLICT.get(kind, ()):
                if c not in self.engines:
                    continue
                if kind=="llm" and c=="video":
                    # 装 LLM: 等 视频 出完 (老 行为), 不 直接 拒绝
                    self._wait_video_idle(VIDEO_WAIT_MAX)
                    if c in self.engines: self.unload(c)
                    continue
                if self._engine_in_use(c):
                    raise MemGuardError(
                        "%s 还 在 用（%s），占着内存。先 在「模型」页 卸载 它，再 装 这个 模型。"
                        % (_engine_cn(c), _engine_cn(c)))
                self.unload(c)
            # ② 内存 闸门: 不够 就 明确 拒绝, 别 硬 装 把 机器 拖死
            self._mem_guard(kind)
            if kind=="video":
                if _is_wan_model(mid):
                    from engine import Wan2Engine
                    e=Wan2Engine(model_id=mid); e.load()
                else:
                    from engine import Engine
                    e=Engine(); e.model_id=mid; e.load()
                self.engines["video"]=e
            elif kind=="image":
                # 按 model id 选 引擎: mflux 那 两个 (Z-Image / MOSS) 走 mflux,
                # 其余 (Krea-2-Turbo) 走 我们 自研 的 krea2_engine.
                import image_engine_mflux as _MF
                if str(mid or "") in _MF.known_models():
                    e=_MF.MFluxImageEngine(model_id=mid); e.load()
                else:
                    from image_engine import ImageEngine
                    e=ImageEngine(); e.model_id=mid or e.model_id; e.load()
                self.engines["image"]=e
            elif kind=="llm":
                # 先卸其它引擎再 load LLM, 防峰值内存撞爆 (视频 4bit DiT ~33G + LLM ~5G)
                # 视频 正在 出片 时 不能 硬卸 (生成 线程 还 持 有 权重) —— 等 它 出完
                self._wait_video_idle(VIDEO_WAIT_MAX)
                for _k in ("video","image"):
                    if _k in self.engines: self.unload(_k)
                # 加载 的 是 「用户 指定 的 那 一个」 模型 —— 不 是 写死 的 默认 那个.
                # 不 给 mid (内部 调用) 就 照旧 用 默认 的 Qwen3-4B.
                from llm_engine import LLMEngine
                _p=_model_weight_path(self,mid) if mid else None
                e=LLMEngine(path=_p) if _p else LLMEngine()
                if mid: e.model_id=mid
                e.load(); self.engines["llm"]=e
            elif kind in ("embeddings","tts","sound","3d","stt","edits","music_synth","decisions"):
                # 能力 引擎 (cap_engines): 轻量 的 直接 建, 重 的 走 各自 load()
                import cap_engines as C
                cls={"embeddings":C.EmbedEngine,"tts":C.TTSEngine,"sound":C.SoundEngine,
                     "3d":C.ThreeDEngine,"stt":C.ASREngine,"edits":C.ImageEditEngine,
                     "music_synth":C.MusicSynthEngine,"decisions":C.DecisionEngine}[kind]
                e=cls()
                if hasattr(e,"load"):
                    e.load()
                self.engines[kind]=e
            elif kind=="music":
                # 出歌 引擎 = 我们 自研 的 MLX (主, 纯 MLX 不 依赖 闭源); ACE-Step :7861 兜底
                ok=False; via="ours"; mid="ours/ace-step-v15-turbo-mlx"
                try:
                    import music_workbench as MW
                    st_=MW.our_mlx_status()
                    if st_.get("available"):
                        ok=bool(st_.get("running")); via="ours"
                    else:
                        via="ace"; mid="ace-step"
                except Exception:
                    via="ace"; mid="ace-step"
                if not ok and via!="ours":
                    import urllib.request
                    try:
                        urllib.request.urlopen(ACE_MUSIC+"/health", timeout=8).read(); ok=True
                    except Exception:
                        pass
                self.engines["music"]={"via":via,"url":ACE_MUSIC,"ok":ok,"model_id":mid}
            self._last[kind]=time.time()
    def touch(self,kind): self._last[kind]=time.time()
    def loaded_engines(self):
        """真正 占 着 权重 的 引擎 —— 有些 引擎 对象 只是 被 「查 出来」 (比如 列 音色 时
        创建 的 神经 TTS), 权重 其实 没 在 显存 里, 不能 算 「已加载」."""
        out=[]
        for k,e in list(self.engines.items()):
            if isinstance(e,dict):
                out.append(k); continue
            try:
                fn=getattr(e,"state",None); s=fn() if callable(fn) else None
                if isinstance(s,dict) and "loaded" in s and not s.get("loaded"):
                    continue
            except Exception:
                pass
            out.append(k)
        return out
    def _begin_use(self,kind):
        """请求 开始 用 这个 引擎 —— 记 计时 + 标记 在用 (置闲 卸载 会 跳过 它)."""
        with self._inflight_lock:
            self._inflight[kind]=self._inflight.get(kind,0)+1
        self.touch(kind)
    def _end_use(self,kind):
        with self._inflight_lock:
            n=self._inflight.get(kind,0)-1
            if n>0: self._inflight[kind]=n
            else: self._inflight.pop(kind,None)
        # 「距 上 次 使用」 按 请求 结束 算 —— 生成 花 了 10 分钟, 就 从 结束 那刻 起 算 闲置
        self.touch(kind)
    def _engine_in_use(self,kind):
        """这个 引擎 现在 有没有 在 跑 / 有没有 活跃 会话 (会话 里 的 引擎 不能 硬 卸)."""
        if self._inflight.get(kind,0)>0:
            return True
        if kind=="video" and self._video_busy:
            return True
        try:
            for s in (self.sessions or {}).values():
                if s.get("kind")==kind: return True
        except Exception:
            pass
        return False
    def _mem_guard(self,kind):
        """内存 闸门: 装 大 引擎 之前 先 看 可用 内存 够 不够."""
        need=ENGINE_WEIGHT_GIB.get(kind)
        if not need: return
        fr=free_gib()
        if fr is None: return
        reserve=self.os_reserve_gib or 0
        if fr>=need+reserve: return
        hog=[k for k in self.engines if k in ENGINE_WEIGHT_GIB]
        raise MemGuardError(
            "可用内存 只有 %.0fG，装 %s 需要 约 %dG（还要 给 系统 留 %dG）。%s"
            % (fr, _engine_cn(kind), need, reserve,
               ("现在 占着 内存 的 是：%s。先 在「模型」页 卸载 它，或 点「全部卸载」。"
                % "、".join(_engine_cn(k) for k in hog)) if hog
               else "先 关掉 别的 占 内存 的 程序 再 试。"))
    def record_library(self, kind, path, model="", prompt="", preview=""):
        """记 一 条 生成 产物 (给 /v1/library 和 控制台 素材库)."""
        if not path or not os.path.isfile(path): return
        item={"type":kind,"path":path,"name":os.path.basename(path),
              "model":model,"prompt":(prompt or "")[:120],
              "at":time.strftime("%H:%M:%S"),
              "url":"/out/"+outputs.rel(path),
              "bytes":os.path.getsize(path),
              "mtime":os.path.getmtime(path)}
        if preview and os.path.isfile(preview):
            item["preview"]="/out/"+outputs.rel(preview)
        with self.lib_lock:
            self.library.append(item)
            if len(self.library)>200: self.library=self.library[-200:]
        self._save_library()
    def _load_library(self):
        """从 落盘 索引 恢复 资料库 (跨 重启 保住 卡片)."""
        try:
            with open(self._lib_file,"r",encoding="utf-8") as f:
                items=json.load(f)
            if not isinstance(items,list): return
            with self.lib_lock:
                self.library=items[-200:]
        except Exception:
            pass
    def _save_library(self):
        """索引 落 盘 (原子 写) —— 放 内核 目录, 绕 开 ~/Documents 的 枚举 坑."""
        try:
            with self.lib_lock:
                items=list(self.library)[-200:]
            tmp=self._lib_file+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(items,f,ensure_ascii=False)
            os.replace(tmp,self._lib_file)
        except Exception:
            pass
    def start_library_scan(self, force=False):
        """后台 扫 产物 目录 (补 回 重启 前 生成 的 东西).

        **绝不** 在 请求 线程 里 扫 —— launchd 上下文 里 枚举 ~/Documents 可能 阻塞,
        那 会 让 「点 资料库」 直接 卡死. 这里 放 后台 线程 + 超时:
        扫 不 完 (被 系统 卡 住) 就 标记 blocked, 以后 不 再 扫, 靠 落盘 索引 撑 着.
        """
        if self._lib_scan_state=="running": return
        if self._lib_scan_state in ("blocked","root") and not force: return
        now=time.time()
        if not force and (now-self._lib_scan_t)<60: return
        self._lib_scan_t=now; self._lib_scan_state="running"
        def _work():
            try:
                # 第一 步: 分类 子目录 (安全, 不会 卡)
                items=self._scan_kind_dirs()
                self._lib_disk=items; self._lib_scan_state="done"
                self._merge_into_index(items)
                # 第二 步: 根 目录 (补充) —— 可能 被 系统 卡 住, 卡 住 也 不 影响 上面 的 结果
                try:
                    self._lib_scan_state="root"
                    extra=self.scan_library()
                    self._merge_into_index(extra)
                    self._lib_scan_state="done"
                except Exception:
                    self._lib_scan_state="done"
            except Exception:
                self._lib_scan_state="done"
        t=threading.Thread(target=_work,daemon=True); t.start()
        # 看 门狗: 根 目录 那 一 步 60 秒 还 没 回 来 -> 判定 被 系统 卡 住, 以后 不 再 试
        def _watch():
            for _ in range(120):
                time.sleep(0.5)
                if self._lib_scan_state!="root": return
            self._lib_scan_state="blocked"
            print("[mlx_server] 产物 根目录 枚举 被 系统 卡 住 —— 资料库 只 用 分类 子目录",
                  file=sys.stderr,flush=True)
        threading.Thread(target=_watch,daemon=True).start()
    def _merge_into_index(self, items):
        """扫 到 的 产物 并 进 索引 (并 落 盘)."""
        try:
            with self.lib_lock:
                have={it.get("path") for it in self.library}
                added=0
                for it in items:
                    if it.get("path") in have: continue
                    self.library.append(it); added+=1
                if added and len(self.library)>200:
                    self.library=self.library[-200:]
            if added: self._save_library()
        except Exception:
            pass
    def library_items(self):
        """给 /v1/library 用: 内存 索引 + 后台 扫 到 的 磁盘 索引 (合并, 按 时间 倒序)."""
        with self.lib_lock:
            items=list(self.library)[::-1]
        disk=list(self._lib_disk)
        seen={it.get("path") for it in items}
        for it in disk:
            if it.get("path") in seen: continue
            items.append(it)
        items=[it for it in items if it.get("path") and os.path.isfile(it["path"])]
        items.sort(key=lambda it: it.get("mtime") or 0.0,reverse=True)
        return items
    def _scan_kind_dirs(self):
        """只 列 「分类 子目录」(音频/图片/视频/语音/3D/其他) 里 的 文件.

        macOS 的 坑: 内核 由 launchd 起 (无 GUI 会话), 列 ~/Documents **根 目录** 会
        被 系统 卡 住 (opendir 阻塞 等 一个 永远 不来 的 授权) —— 表现 是 「资料库」 点 不动.
        但 分类 子目录 直接 列 是 好 的 —— 所以 索引 从 子目录 起, 根 目录 只 当 补充.
        """
        paths=[]
        for _kind,_sub in outputs.KIND_DIR.items():
            _d=os.path.join(outputs.ROOT,_sub)
            try:
                for _f in os.listdir(_d):
                    paths.append(os.path.join(_d,_f))
            except Exception:
                continue
        return self._items_from_paths(paths)
    def scan_library(self):
        """扫 输出 目录 里 已 有 的 产物, 补 回 重启 后 丢 掉 的 索引.

        内存 里 的 library 只 活 一 次 进程 —— 服务 一 重启 就 空 了, 但 产物
        文件 还在 输出 目录. 这里 按 前缀 + 后缀 把 它们
        认 出来, 让 素材库 页 跨 重启 仍然 有 东西.

        注意: **不** 扫 /tmp 了.
        · /tmp 是 系统 临时 目录 —— 里面 全 是 别的 程序 留 的 垃圾 (探针 wav / 半截 下载),
          名字 撞 上 "mx_tts_" 前缀 的 会被 当成 我们 的 产物 收进 资料库;
        · os.listdir("/tmp") + 逐 个 stat, 慢 得 离谱 —— 点 一下 资料库 再 点 别的 就 卡.
        产物 统一 在 ~/Documents/MLX Studio 输出/, 那儿 才是 唯一 真 的 产物 目录.
        """
        out=self._scan_kind_dirs()
        # —— 补充: 递归 扫 根 目录 (用户 可能 直接 丢 文件 进 根 目录) ——
        # 这 一 步 在 launchd 上下文 里 可能 被 系统 卡 住 —— 调用 方 必须 放 后台 线程.
        paths=[]
        try:
            for dp,_dn,fn in os.walk(outputs.ROOT):
                for f in fn:
                    paths.append(os.path.join(dp,f))
        except Exception:
            pass
        extra=self._items_from_paths(paths)
        have={it["path"] for it in out}
        out.extend([it for it in extra if it["path"] not in have])
        out.sort(key=lambda x:x["mtime"],reverse=True)
        return out
    def _items_from_paths(self, paths):
        out=[]
        for path in paths:
            n=os.path.basename(path)
            if not n.startswith("mx_"): continue
            if not any(n.startswith(p) for p in LIB_PREFIXES): continue
            # 探针 / 测试 文件 (mx_tts_probe_*.wav 之类) —— 不 是 产物, 不 进 资料库
            low=n.lower()
            if "probe" in low or "_test_" in low or low.startswith("mx_test"): continue
            # 3D 产物 有 两个 文件: .glb (模型) 与 .png (预览图).
            # 预览图 不是 独立 的 图像 产物 —— 挂 到 对应 的 .glb 条目 上, 不 单独 入库.
            if n.startswith("mx_3d_") and n.lower().endswith(".png"):
                continue
            try:
                if not os.path.isfile(path): continue
                st=os.stat(path)
            except Exception:
                continue
            ext=os.path.splitext(n)[1].lower()
            kind=LIB_KIND.get(ext)
            if not kind: continue
            item={"type":kind,"path":path,"name":n,"model":"","prompt":"",
                  "at":time.strftime("%H:%M:%S",time.localtime(st.st_mtime)),
                  "url":"/out/"+outputs.rel(path),"bytes":st.st_size,
                  "mtime":st.st_mtime}
            if kind == "3d":
                # 同 名 的 .png 是 这 个 网格 的 预览图
                pv = path[:-len(".glb")] + ".png" if path.lower().endswith(".glb") else None
                if pv and os.path.isfile(pv):
                    item["preview"] = "/out/" + outputs.rel(pv)
            out.append(item)
        # 按 文件名 排 是 错 的: 文件名 里 的 前缀 (mx_img_ / mx_v1img_ / mx_edit_ …) 会
        # 盖过 时间戳, 于是 "mx_v1img_" 这 批 旧 图 会 排 到 "mx_img_" 这 批 新 图 前面.
        # 素材库 是 按 时间 倒序 给 人 看 的 —— 用 mtime 排.
        out.sort(key=lambda x:x["mtime"],reverse=True)
        return out
    def record_timing(self, ttft_ms=0.0, decode_tok_s=0.0, prefill_tok_s=0.0):
        with self.metrics_lock:
            if ttft_ms: self.ttft_ms=(self.ttft_ms*self._ttft_n+ttft_ms)/(self._ttft_n+1); self._ttft_n+=1
            if decode_tok_s: self.decode_tok_s=decode_tok_s
            if prefill_tok_s: self.prefill_tok_s=prefill_tok_s
    def unload(self,kind=None):
        # 卸 = 真的 调 引 擎 的 unload (内部 del 子 组 件 + _free()), 不 是 光 pop 掉 引用.
        # 光 pop 只 让 shell 层 丢 了 engine 对象, engine 内部 的 dit/te/vv/av/pipe 还 被 自 身 属 性 持 有,
        # GC 收 不 回 Metal buffer, active 内 存 不 回 落 (本次 v 测 48G 不 回 的 真 凶).
        with self.lock:
            if kind=="video" and self._video_busy:
                # 出片 中: 生成 线程 还 持 有 权重 引用, 硬 卸 会 让 它 崩.
                # 留给 置闲 计时器 (出完 30min 后 自动 卸).
                return False
            if kind:
                e = self.engines.pop(kind, None)
                if e is not None and hasattr(e, "unload"):
                    e.unload()
            else:
                for e in list(self.engines.values()):
                    if hasattr(e, "unload"):
                        e.unload()
                self.engines.clear()
        _flush()
        return True

    def _wait_video_idle(self, max_wait):
        """等 出片 结束 (视频 排 他). 超 时 就 明确 报错, 不 是 挂死."""
        t0=time.time()
        while self._video_busy and time.time()-t0 < max_wait:
            time.sleep(0.3)
        if self._video_busy:
            raise RuntimeError("视频 生成 中 (DiT 占 ~25G), LLM 暂 不能 加载; 等 出片 结束 再试")
    def _music(self,prompt,sec,lyrics="",lang="en",seed=None,instrumental=False,steps=None):
        """胶水层 — 出歌: 打本机 ACE-Step API (ACE_MUSIC, 默认 :7861).
        两段式: POST /release_task 拿 task_id -> 轮询 POST /query_result -> GET /v1/audio?path=... 落盘.
        有 词 (lyrics 非 空) -> 不 框 死 时 长 (ACE 自 适 应), 歌 长 度 由 词 决 定.
        无 词 -> 按 sec 框 死 (audio_duration)."""
        import urllib.request, urllib.parse, time as _t2
        # 出歌 服务 可能 被 置闲 计时器 卸 了 —— 出歌 前 自 愈 (拉 起来),
        # 别 因为 服务 没 起 就 静默 退化 成 程序化 假 合成.
        if not _ace_running():
            try:
                if not _ace_start():
                    print("[mlx_server] 出歌 服务 自 愈 失败 (ACE-Step 没 起 来)",
                          file=sys.stderr,flush=True)
            except Exception as ex:
                print("[mlx_server] 出歌 服务 自 愈 异常: %s"%ex,file=sys.stderr,flush=True)
        # 出歌 引擎 要 登记 进 引擎 表 —— 不 登记 的 话, 「闲置 30 分钟 自动 卸载」 管 不 到
        # ACE-Step 进程, 它 会 一直 占 着 ~23G 显存 (用户 直接 出歌, 没 点「加载」时 就 是 这 条 路).
        try:
            self.engines["music"]={"via":"ace","url":ACE_MUSIC,"ok":True,"model_id":"ace-step"}
        except Exception:
            pass
        _t0=time.time()
        has_lyr = bool(lyrics and lyrics.strip())
        body={"prompt":prompt,"lyrics":lyrics,"vocal_language":lang if has_lyr else "en",
              "seed":-1 if seed is None else int(seed),
              "inference_steps":int(steps or 8),"guidance_scale":7.0}
        if instrumental:
            body["instrumental"]=True
            body["lyrics"]=""
        if not has_lyr:
            body["audio_duration"]=int(sec)
        def _post(path, payload):
            r=urllib.request.Request(ACE_MUSIC+path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type":"application/json"})
            return json.loads(urllib.request.urlopen(r,timeout=900).read())
        try:
            # 1) 创建 task
            d=_post("/release_task", body)
            tid=d.get("task_id")
            if not tid:
                return {"error":f"ACE /release_task 没有 task_id: {json.dumps(d)[:200]}","via":"ace"}
            # 2) 轮询 (3s 一次, 最 多 900s)
            deadline=_t0+900
            while time.time()<deadline:
                import time as _t3; _t3.sleep(3)
                qr=_post("/query_result", {"task_id_list":[tid]})
                if not isinstance(qr,list): break
                item=qr[0]
                status=int(item.get("status",0))
                if status==1:  # succeeded
                    # result 是 json 串
                    result_str=item.get("result","[]")
                    try:
                        results=json.loads(result_str) if isinstance(result_str,str) else result_str
                    except Exception:
                        results=[]
                    if results:
                        file_path=results[0].get("file","")
                        if file_path:
                            # file 可能是 本地路径 或 相对 URL (/v1/audio?path=...)
                            if file_path.startswith("/v1/audio"):
                                au=urllib.request.urlopen(ACE_MUSIC+file_path,timeout=120).read()
                            elif file_path.startswith("http"):
                                au=urllib.request.urlopen(file_path,timeout=120).read()
                            else:
                                q=urllib.parse.quote(file_path, safe="")
                                au=urllib.request.urlopen(ACE_MUSIC+f"/v1/audio?path={q}",timeout=120).read()
                            p=outputs.path_for("audio","mx_music","mp3")
                            open(p,"wb").write(au)
                            self.touch("music")   # 刚 出完 歌 -> 闲置 计时 从头 开始
                            return {"url":f"file://{p}","via":"ace","sec":round(time.time()-_t0,1)}
                    return {"error":"ACE 完成 但 没 有 音 频 路 径","via":"ace"}
                elif status==2:  # failed
                    return {"error":f"ACE 失 败: {json.dumps(item.get('result',''))[:200]}","via":"ace"}
            return {"error":"ACE 超 时 (900s)","via":"ace"}
        except Exception as e:
            return {"error":f"出 歌 失 败 (ACE {ACE_MUSIC}): {e}","via":"ace"}

    def _v1_video(self,q):
        import time as _t
        sid=q.get("session",[None])[0]
        prompt=q.get("prompt",[None])[0] or q.get("model",[None])[0] or ""
        sec=float(q.get("seconds",["5"])[0]); steps=int(q.get("steps",["8"])[0]); seed=int(q.get("seed",["0"])[0])
        return self._do_video({"prompt":[prompt],"sec":[str(sec)],"steps":[str(steps)],"seed":[str(seed)],"session":[sid]})
    def _v1_image(self,q):
        import time as _t
        prompt=q.get("prompt",[""])[0]; out=outputs.path_for("image","mx_v1_img","png")
        self.load("image")
        self.engines["image"].generate(prompt,out=out,steps=int(q.get("n_steps",["4"])[0]),seed=int(q.get("seed",["0"])[0]))
        self.touch("image"); self._j({"data":[{"url":self._out_url(out)}]})
    def _v1_music(self,q):
        import time as _t
        prompt=q.get("prompt",[""])[0]
        lyrics=q.get("lyrics",[""])[0]; lang=q.get("lang",["en"])[0]
        r=self._music(prompt,q.get("seconds",["30"])[0],lyrics=lyrics,lang=lang)
        self._j(r)
    def _v1_load(self,m):
        kind={"video":"video","image":"image","llm":"llm"}.get(m,"llm")
        self.load(kind); return {"loaded":kind,"model":m}

    def evict(self):
        # 统 一 置 闲 计 时 卸 载: 任 意 模 型 距 上 次 被 调 用 超 过 IDLE_S(默认 30min) -> 自 动 卸
        # 连 续 出 片 时 每 片 touch 一 次, 永 远 到 不 了 30min, 不 会 被 误 卸
        lim = getattr(self,"idle_evict_secs",IDLE_S)
        if not lim:
            return
        for k in list(self.engines):
            # 正在 用 的 引擎 不能 卸 (出片 中 / 有 活跃 会话) —— 只 刷新 计时, 下 轮 再 看
            if self._engine_in_use(k):
                self.touch(k); continue
            if k in self._last and time.time()-self._last[k]>lim:
                e=self.engines.pop(k,None)
                if e is not None and hasattr(e,"unload"):
                    try: e.unload()
                    except Exception: pass
                self._last.pop(k,None)
                # 出歌 引擎 是 胶水: 真 正 占 着 ~23G 显存 的 是 ACE-Step 进程, 得 一 起 停
                if k=="music":
                    # 我们 自研 的 MLX 引擎 (占 ~12G) —— 一 并 停
                    try:
                        import music_workbench as MW
                        if MW.stop_our_mlx():
                            print("[mlx_server] 置 闲: 我们 的 MLX 出歌 引擎 已 停, 内存 释放",
                                  file=sys.stderr,flush=True)
                    except Exception: pass
                    try:
                        if _ace_stop():
                            print("[mlx_server] 置 闲: 出歌 服务 (ACE-Step) 已 停, 显存 释放",
                                  file=sys.stderr,flush=True)
                    except Exception: pass
                _lim_txt = ("%dmin"%(lim//60)) if lim>=60 else ("%ds"%lim)
                print(f"[mlx_server] {_lim_txt} 置 闲, 自 动 卸 {k}",file=sys.stderr,flush=True)
        _flush()
    def _sessions_snapshot(self):
        with self.session_lock:
            return [dict(v) for v in self.sessions.values() if time.time()-v["last"]<30*60]
    def tick(self):
        while not self._stop:
            try: self.evict()
            except Exception as e: print("[evict]",e,file=sys.stderr)
            time.sleep(60)

class H(BaseHTTPRequestHandler):
    def _out_url(self, path):
        """产物 的 HTTP URL (同源 /out/) —— 网页 控制台 才 加载 得 了.

        原来 返 "file://" + path: 桌面 客户端 能 读, 但 http 页面 加载 file://
        会被 浏览器 直接 拦掉 (界面 上 一 张 空白). 闭源 干脆 不 托管 文件,
        只 给 b64_json; 我们 既然 挂了 /out/, 就 给 一个 真能 加载 的 URL.
        """
        if not path: return path
        p = str(path).split("file://")[-1]
        try:
            host = self.headers.get("Host")
        except Exception:
            host = None
        if not host:
            try: host = "%s:%s" % self.server.server_address[:2]
            except Exception: host = "127.0.0.1"
        return "http://%s/out/%s" % (host, outputs.rel(p))
    def _fix_urls(self, obj):
        """把 响应 里 的 file:// URL 换成 HTTP URL (递归)."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("url", "preview") and isinstance(v, str) and v.startswith("file://"):
                    obj[k] = self._out_url(v)
                elif isinstance(v, (dict, list)):
                    self._fix_urls(v)
        elif isinstance(obj, list):
            for x in obj:
                self._fix_urls(x)
        return obj

    srv=None
    def log_message(self,*a): pass
    def _j(self,o,c=200):
        b=json.dumps(o).encode()
        # 必须 带 Content-Length: 不带 的话 body 只能 靠 关连接 断包 (HTTP/1.0),
        # CFNetwork / URLSession 会 直接 报 -1005 "network connection was lost",
        # 非 浏览器 客户端 (桌面 App / curl --http1.0 之外 的 栈) 拿 不到 响应.
        self.send_response(c); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b))); self.end_headers()
        if self.command=="HEAD": return
        try: self.wfile.write(b)
        except Exception: pass
    def _serve_file(self, path):
        """按 范围 发 一个 文件 (Range / ETag / immutable 缓存).

        产物 文件 是 资料库 的 主体 —— 一 次 几百 MB 的 全量 传输 会 把
        WebKit 的 请求 队列 堵死 (点 别 的 页面 都 变 慢). 这里:
          1) 支持 Range: 视频/音频 只 拉 需要 的 那 一 段
          2) 带 ETag + immutable 缓存: 同一 个 文件 不用 反复 下
        """
        import mimetypes
        ctype=mimetypes.guess_type(path)[0] or "application/octet-stream"
        st=os.stat(path); size=st.st_size
        etag='"%x-%x"'%(int(st.st_mtime),size)
        inm=self.headers.get("If-None-Match")
        if inm and inm.strip()==etag:
            self.send_response(304); self.send_header("ETag",etag)
            self.send_header("Content-Length","0"); self.end_headers(); return
        start,end,status=0,size-1,200
        rng=self.headers.get("Range")
        if rng:
            m=re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if m:
                s,e=m.group(1),m.group(2)
                if s=="" and e:
                    # suffix range: 最后 n 字节
                    n=int(e); start=max(0,size-n)
                else:
                    start=int(s) if s else 0
                    end=int(e) if e else size-1
                if start>=size:
                    self.send_response(416)
                    self.send_header("Content-Range","bytes */%d"%size)
                    self.send_header("Content-Length","0"); self.end_headers(); return
                if end>=size: end=size-1
                status=206
        length=max(0,end-start+1)
        self.send_response(status); self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(length))
        self.send_header("Accept-Ranges","bytes")
        self.send_header("ETag",etag)
        self.send_header("Cache-Control","public, max-age=31536000, immutable")
        if status==206:
            self.send_header("Content-Range","bytes %d-%d/%d"%(start,end,size))
        self.end_headers()
        if self.command=="HEAD": return
        try:
            with open(path,"rb") as f:
                f.seek(start); left=length
                while left>0:
                    chunk=f.read(min(65536,left))
                    if not chunk: break
                    self.wfile.write(chunk); left-=len(chunk)
        except Exception: pass

    def _body_json(self,limit=64*1024*1024):
        # 缓存: api_full 先 读 一次, 老 路由 再 读 时 要 拿到 同一份 body
        cached=getattr(self,"_body_cache",None)
        if cached is not None: return cached
        ln=int(self.headers.get("Content-Length",0))
        raw=self.rfile.read(min(ln,limit)) if ln else b""
        try: parsed=json.loads(raw)
        except Exception: parsed={}
        self._body_cache=parsed
        return parsed
    def _body_multipart(self,limit=256*1024*1024):
        # multipart/form-data: 文本 字段 + 文件 字段 (原版 图片编辑 / 语音转写 用)
        cached=getattr(self,"_mp_cache",None)
        if cached is not None: return cached
        import re as _re
        ct=self.headers.get("Content-Type","")
        ln=int(self.headers.get("Content-Length",0))
        raw=self.rfile.read(min(ln,limit)) if ln else b""
        fields={}; files={}
        m=_re.search(r'boundary="?([^";]+)"?',ct)
        if m:
            bnd=("--"+m.group(1)).encode()
            for part in raw.split(bnd):
                if not part or part in (b"--\r\n",b"--") or part.startswith(b"--"):
                    continue
                part=part.lstrip(b"\r\n")
                if b"\r\n\r\n" not in part:
                    continue
                head,_,data=part.partition(b"\r\n\r\n")
                if data.endswith(b"\r\n"):
                    data=data[:-2]
                headers=head.decode("utf-8","replace")
                nm=_re.search(r'name="([^"]*)"',headers)
                fn=_re.search(r'filename="([^"]*)"',headers)
                if not nm:
                    continue
                if fn:
                    ctm=_re.search(r"Content-Type:\s*(\S+)",headers)
                    ent={"filename":fn.group(1),"data":data,
                         "ctype":ctm.group(1) if ctm else ""}
                    key=nm.group(1)
                    if key in files:
                        # 闭源 契约: image[] 可以 重复 (每 张 参考图 一个 part)
                        prev=files[key]
                        if isinstance(prev,list):
                            prev.append(ent)
                        else:
                            files[key]=[prev,ent]
                    else:
                        files[key]=ent
                else:
                    key=nm.group(1)
                    val=data.decode("utf-8","replace")
                    if key in fields:
                        prev=fields[key]
                        if isinstance(prev,list):
                            prev.append(val)
                        else:
                            fields[key]=[prev,val]
                    else:
                        fields[key]=val
        self._mp_cache={"fields":fields,"files":files}
        return self._mp_cache
    def _auth_ok(self):
        """--api-key: 远端 调用 要 Authorization: Bearer <key> (原版 §13).

        本机 (loopback) 的 调用 —— 桌面 App 自己 / 本机 curl —— 不 要 key,
        否则 App 一 打开 就 全是 401. 局域网 里 的 机器 才 需要 带 token.
        """
        key=getattr(self.srv,"api_key","") or ""
        if not key:
            return True
        try:
            peer=(self.client_address or ("",))[0]
        except Exception:
            peer=""
        if peer in ("127.0.0.1","::1","localhost") or str(peer).startswith("127."):
            return True
        h=self.headers.get("Authorization","") or ""
        want=("Bearer "+key).strip().lower()
        return h.strip().lower()==want
    def _sok(self,c):
        self.send_response(c); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); return c
    def do_GET(self):
        p=urlparse(self.path).path; q=parse_qs(urlparse(self.path).query)
        # 产物 文件 (/out/...) 免 key —— 图/视频 的 链接 要 能 在 浏览器 里 直接 点开,
        # 也 让 别的 agent/机器 拿 链接 就 取得到。API 端点 仍然 要 Bearer。
        if not p.startswith("/out/") and not self._auth_ok():
            self._j({"error":{"message":"unauthorized: bad or missing Authorization: Bearer key",
                               "type":"authentication_error","code":401}},401); return
        # WS /v1/responses (闭源 有 这 个 端点): 握手 + 事件 序列
        if p.rstrip("/")=="/v1/responses" and \
                "websocket" in (self.headers.get("Upgrade") or "").lower():
            self._ws_responses(); return
        # 闭源 mlx-serve 全功能面 (逆向 契约): 先 走 api_full, 没 处理 的 落回 老 路由
        try:
            if api_full.handle(self.srv,self,"GET",p,q,{}): return
        except Exception as e:
            print("[api_full GET]",e,file=sys.stderr,flush=True)
            self._j({"error":{"message":str(e),"type":"server_error","code":500}},500); return
        if p=="/v1/models/download/status":
            mid=str((q.get("model",[""])[0]) or "").strip()
            j=_dl_job(mid)
            if not j:
                self._j({"model":mid,"state":"idle","note":"","message":""}); return
            j["model"]=mid
            j["elapsed"]=round(time.time()-(j.get("t0") or time.time()))
            if j.get("state")=="running":
                _t=str(j.get("target") or "")
                if _t and os.path.isdir(_t):
                    _sz=_dir_size(_t)
                    j["note"]="已 下载 %.0f MB · %d 秒" % (_sz/1048576.0, j["elapsed"])
                    j["bytes"]=_sz
            self._j(j); return
        if p=="/v1/outputs":
            # 统一 输出 目录 —— 视频 / 图片 / 音频 / 语音 / 3D 都 落 这里.
            root=outputs.ensure_root()
            counts={}; total=0
            for kind,name in outputs.KIND_DIR.items():
                d=os.path.join(root,name); n=0; bts=0
                try:
                    for dp,_dn,fn in os.walk(d):
                        for f in fn:
                            try:
                                n+=1; bts+=os.path.getsize(os.path.join(dp,f))
                            except OSError: pass
                except Exception: pass
                counts[kind]={"dir":d,"count":n,"bytes":bts}
                total+=n
            self._j({"ok":True,"root":root,"kinds":counts,"total":total})
            return
        if p in ("/","/ui","/index.html"):
            import ui as _ui
            html=_ui.console_html()
            # 把 服务 端 的 真实 地址 注入 页面 —— 页面 在 opaque origin (file:// / 代理) 下
            # location.origin 会 是 "null", 界面 上 的 「此服务器」 就 没 地址 可 显示.
            host=self.headers.get("Host") or "127.0.0.1:%d"%PORT
            base="http://"+host
            html=html.replace("</head>",
                "<script>window.__MLX_BASE__=%s;</script></head>"
                % json.dumps(base, ensure_ascii=False), 1)
            body=html.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            # 控制台 是 单文件 界面, 改 完 要 立刻 生效 —— 不 然 WebKit 会 拿 缓存 的 旧 版
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-cache, no-store, must-revalidate")
            self.send_header("Pragma","no-cache")
            self.end_headers()
            try: self.wfile.write(body)
            except Exception: pass
            return
        if p.startswith("/out/"):
            name=unquote(p[5:])   # 子目录 名 是 中文 (音频/…), 要 解码
            if ".." in name:
                self._j({"error":{"message":"bad path","type":"invalid_request","code":400}},400); return
            # 统一 输出 目录 (~/Documents/MLX Studio 输出/<类型>/<文件>)
            path=outputs.abs_from_rel(name)
            if not path or not os.path.isfile(path):
                path=os.path.join("/tmp",os.path.basename(name))
            if not os.path.isfile(path):
                path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out",name)
            if not os.path.isfile(path):
                self._j({"error":{"message":"not found","type":"invalid_request","code":404}},404); return
            self._serve_file(path)
            return
        if p=="/health":
            fr=free_gib(); self._j({"status":"ok","node":NODE,"loaded":self.srv.loaded_engines(),"idle_s":getattr(self.srv,"idle_evict_secs",IDLE_S),"route":ROUTE,"free_gib":round(fr,1) if fr is not None else None})
        elif p=="/unload":
            self.srv.unload(); self._j({"unloaded":True,"loaded":[]})
        elif p=="/video":
            fr=free_gib()
            if fr is not None and fr<MIN_FREE_GIB:
                self._j({"error":f"可 用 {fr:.0f}G < {MIN_FREE_GIB}G, 拒 绝 起 视 频 (防 内 存 崩)"},503); return
            if not self.srv.video_lock.acquire(timeout=30):
                self._j({"error":"已 有 出 片 在 跑, 一 次 只 出 一 片, 稍 后 再 试"},409); return
            try:
                # 同 线 程 内 直 出 (Metal 不 跨 线 程), 会 话 内 不 卸
                out=self.srv._do_video(q)
            finally:
                self.srv.video_lock.release()
            self._j(self._fix_urls(out) if out else {"error":"无 结 果"}, 200 if (out and "url" in out) else 504)
        elif p=="/image":
            out=outputs.path_for("image","mx_img","png")
            self._run_image(q.get("prompt",[""])[0],out,
                int(q.get("width",["1344"])[0]),int(q.get("height",["768"])[0]),
                int(q.get("steps",["4"])[0]),int(q.get("seed",["0"])[0]),
                local_out=q.get("local_out",[None])[0])
        elif p=="/music":
            self._run_music(q.get("prompt",[""])[0],
                lyrics=q.get("lyrics",[""])[0], lang=q.get("lang",["en"])[0],
                sec=q.get("sec",["30"])[0], local_out=q.get("local_out",[None])[0])
        elif p=="/session":
            kind=q.get("kind",["video"])[0]
            sid=self.srv.open_session(kind)
            self._j({"session":sid,"kind":kind,"keep_min":30})
        elif p=="/session/close":
            sid=q.get("sid",[None])[0]
            ok=self.srv.close_session(sid) if sid else None
            self._j({"closed":bool(ok),"sid":sid})
        elif p=="/status":
            self._j(self.srv.metrics_snapshot())
        elif p=="/llm":
            self._run_llm(q.get("prompt",[""])[0])
        elif p=="/models":
            self._j(_models_resp())
        elif p=="/download":
            m=q.get("model",[None])[0]
            ok,msg=_download_model(m)
            self._j({"ok":ok,"model":m,"dest":msg if isinstance(msg,str) else str(msg),
                     "hint":"同步下载, 完成才返回"}, 200 if ok else 500)
        elif p=="/download/status":
            self._j(_models_resp())
        elif p=="/api/version":
            self._j({"version":"mlx-studio 1.0 (our-own kernel + shell)"})
        elif p=="/props":
            self._j({"memory":{"active_bytes":int((self.srv.rss_gib() or 0)*1073741824),"total_bytes":96*1073741824}})
        elif p=="/launch":
            agent=q.get("agent",["codex"])[0]
            self._j({"agent":agent,"script":f"# launch {agent} against our mlx-server"})
        # ===== /v1/* 壳 兼 容 层 (MLX-Serve 前 端 调 这 些, 打 我 们 内 核) =====
        elif p=="/v1/models":
            # 照 抄 闭 源 MLX-Serve /v1/models: capabilities 是 逗 号 串, 按 能 力 细 分
            m=_models_resp()
            def caps(c):
                m={"video":["video"],"image":["image"],"music":["audio","music"],
                   "llm":["chat","streaming","vision"]}
                return m.get(c,"gen")
            def _caps_of(it):
                c=it.get("cap","gen")
                if it.get("caps"): return it["caps"]
                m={"video":["video"],"image":["image"],"music":["audio","music"],
                   "llm":["chat","streaming","vision"]}
                return m.get(c,"gen")
            names=[{"id":it["id"],"object":"model","name":it.get("name",it["id"]),
                    "architecture":it.get("arch"),
                    "capabilities":_caps_of(it),
                    "meta":{"architecture":it.get("arch"),"engine":it.get("engine","our-server")},
                    "state":"ready" if it.get("present") else "absent",
                    "loaded":(it.get("loaded_hint") if it.get("cap")=="music"
                              else (it.get("cap") in self.srv.engines)),
                    "size":it.get("size",""),
                    "bytes_on_disk":it.get("bytes",0),
                    "context_length":it.get("ctx")} for it in m["models"]]
            self._j({"object":"list","data":names})
        elif p=="/metrics.json":
            self._j(self.srv.metrics_snapshot())
        elif p=="/metrics":
            lines=["# mlx_studio own kernel metrics (shell Monitor panel)"]
            m=self.srv.metrics_snapshot()
            lines.append(f'mlx_studio_gen_total {m["gen_count"]}')
            lines.append(f'mlx_studio_gen_err_total {m["gen_err"]}')
            lines.append(f'mlx_studio_gen_bytes {m["gen_bytes"]}')
            lines.append(f'mlx_studio_rss_bytes {int(m["rss_gib"]*1073741824)}')
            lines.append(f'mlx_studio_free_gib {m["free_gib"]}')
            for k in m["engines"]:
                lines.append(f'mlx_studio_engine_loaded{{engine="{k}"}} 1')
            body=("\n".join(lines)+"\n").encode()
            self.send_response(200); self.send_header("Content-Type","text/plain")
            self.send_header("Content-Length",str(len(body))); self.end_headers()
            self.wfile.write(body)
            return
        elif p=="/v1/responses":
            self._run_responses(q.get("input",[""])[0])
        elif p=="/v1/embeddings":
            self._j({"error":"emb 未 实 装"},501)
        elif p=="/v1/providers":
            self._j({"providers":[],"note":"our-own kernel; no upstream providers"})
        elif p=="/v1/decisions":
            self._j({"decisions":[]})
        elif p=="/v1/load-model":
            pass # load/unload 走 POST, 见 do_POST
        elif p=="/v1/unload-model":
            pass # 走 POST
        elif p=="/v1/models/rescan":
            self._j(_models_resp())
        # ===== 出歌工作台 (合并 进 Studio 的「音乐」页) =====
        elif p=="/v1/music/health":
            import music_workbench as MW
            self._j(MW.health())
        elif p=="/v1/music/square":
            # 广场 —— 我们 自己 模型 出 的 歌 (本地, 不联网)
            import music_workbench as MW
            code=q.get("code",["trending"])[0]
            source=q.get("source",["local"])[0]
            try: page=int(q.get("page",["1"])[0])
            except ValueError: page=1
            try: ps=int(q.get("page_size",["50"])[0])
            except ValueError: ps=50
            self._j(MW.square(code=code, page=page, page_size=ps, source=source))
        elif p=="/v1/music/audio":
            # 本地 成品 的 播放 —— 只 允许 受控 目录 里 的 文件
            import music_workbench as MW
            ap=MW.audio_path(q.get("path",[""])[0])
            if not ap:
                self._j({"error":{"message":"not found","type":"invalid_request","code":404}},404); return
            self._serve_file(ap); return
        elif p=="/v1/music/works":
            # 作品库 —— 本地 成品 + suno 在线 作品
            import music_workbench as MW
            self._j(MW.works(token=q.get("token",[None])[0]))
        elif p=="/v1/music/batch":
            # 批量 出歌 —— 后台 跑, 用 /v1/music/batch?id= 查 进度
            import music_workbench as MW
            jid=q.get("id",[""])[0]
            if jid:
                self._j(MW.batch_status(jid))
            else:
                self._j({"error":{"message":"缺 id","type":"invalid_request","code":400}},400)
        elif p=="/v1/video/batch":
            # 批量 首尾帧 出片 —— 后台 跑, 用 /v1/video/batch?id= 查 进度
            import video_batch as VB
            jid=q.get("id",[""])[0]
            if jid:
                self._j(VB.batch_status(jid))
            else:
                self._j({"error":{"message":"缺 id","type":"invalid_request","code":400}},400)
        elif p=="/v1/video/images":
            # 出图 产物 清单 —— 批量 首尾帧 选图 用
            import video_batch as VB
            self._j(VB.list_images())
        elif p=="/v1/video/mv":
            # MV 工作台 —— 后台 跑, 用 /v1/video/mv?id= 查 进度
            import video_mv as VM
            jid=q.get("id",[""])[0]
            if jid:
                self._j(VM.status(jid))
            else:
                self._j({"error":{"message":"缺 id","type":"invalid_request","code":400}},400)
        elif p=="/v1/video/nodes":
            # 出片 节点 (本机 / 远端) —— 镜头 在 哪 出, 合并 永远 在 本机
            import video_mv as VM
            self._j(VM.nodes(force=q.get("force",["0"])[0]=="1"))
        elif p=="/v1/audio/files":
            # 本机 出歌 产物 —— MV 配乐 下拉
            import video_mv as VM
            self._j(VM.list_audio())
        else:
            # 闭源 契约: 未知 端点 -> {"error":{"message":"Unknown endpoint","type":"not_found","code":404}}
            self._j({"error":{"message":"Unknown endpoint","type":"not_found","code":404}},404)
    def _run_image(self,prompt,out,w,h,steps,seed,local_out=None):
        t0=time.time()
        self.srv.load("image", self.srv._image_mid_pref or None)
        self.srv._image_running+=1
        try:
            self.srv.engines["image"].generate(prompt,out=out,width=w,height=h,steps=steps,seed=seed)
        finally:
            self.srv._image_running-=1
        sec=round(time.time()-t0,1)
        self.srv.touch("image")
        with self.srv.metrics_lock:
            self.srv.gen_count+=1
            self.srv.gen_bytes+=os.path.getsize(out) if os.path.exists(out) else 0
        fr=f"file://{out}"
        # 可选回传: local_out=<dir> 时, 把 png scp 到调用方本机, 返回本机真实路径
        if local_out and NODE=="remote":
            import subprocess as _sp
            import remote_node
            _sp.run(["scp","-o","ConnectTimeout=10",f"{remote_node.target()}:{out}",
                    os.path.expanduser(local_out)], capture_output=True, timeout=120)
            local_path=os.path.join(os.path.expanduser(local_out), os.path.basename(out))
            if os.path.exists(local_path):
                fr=f"file://{local_path}"
        self._j(self._fix_urls({"url":fr,"sec":sec,"unloaded":False}))
    def _run_music(self,prompt,lyrics="",lang="en",sec="30",local_out=None):
        out=self.srv._music(prompt,sec,lyrics=lyrics,lang=lang)
        self.srv.touch("music")
        with self.srv.metrics_lock:
            if out and "url" in out: self.srv.gen_count+=1
            else: self.srv.gen_err+=1
        # 可选 回 传: local_out=<dir> 时 scp 到 调 用 方 本 机
        if local_out and NODE=="remote" and out and "url" in out:
            import subprocess as _sp
            import remote_node
            _sp.run(["scp","-o","ConnectTimeout=10",f"{remote_node.target()}:{out['url'].split('file://')[-1]}",
                    os.path.expanduser(local_out)], capture_output=True, timeout=120)
            lp=os.path.join(os.path.expanduser(local_out), out['url'].split('file://')[-1])
            if os.path.exists(lp): out={"url":f"file://{lp}","sec":out.get("sec"),"lyrics":bool(lyrics)}
        self._j(self._fix_urls(out) if out else {"error":"无 结 果"}, 200 if (out and "url" in out) else 502)
    def _run_llm(self,prompt):
        self.srv.load("llm")
        out=self.srv.engines["llm"].generate(prompt)
        self.srv.touch("llm")
        self._j({"reply":out,"unloaded":False})
    def _run_responses(self,prompt):
        self.srv.load("llm")
        try:
            r=self.srv.engines["llm"].generate(prompt)
        except Exception as e:
            self._j({"error":str(e)},503); return
        self.srv.touch("llm")
        self._j({"output":[{"type":"message","content":[{"type":"output_text","text":r}]}]})
    # ---------- WS /v1/responses (闭源 有 这 个 端点) ----------
    def _ws_responses(self):
        """握手 -> 收 客户端 的 JSON -> 产 Responses 事件 序列 (同 SSE 的 事件 名)."""
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._j({"error": {"message": "WS handshake failed: missing Sec-WebSocket-Key",
                                "type": "invalid_request", "code": 400}}, 400)
            return
        try:
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", ws_accept_key(key))
            self.end_headers()
            self.wfile.flush()
        except Exception:
            pass
        print("WS /v1/responses connected", file=sys.stderr, flush=True)
        rfile, wfile = self.rfile, self.wfile
        while True:
            try:
                op, data = ws_read_frame(rfile)
            except Exception:
                break
            if op is None or op == 8:
                break
            if op == 9:                      # ping -> pong
                try:
                    ws_write(wfile, data or b"", opcode=10)
                except Exception:
                    pass
                continue
            if op != 1:
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                ws_write(wfile, json.dumps({"type": "error", "message": "invalid JSON"}))
                continue
            if not isinstance(msg, dict):
                ws_write(wfile, json.dumps({"type": "error",
                                            "message": "expected a JSON object"}))
                continue
            t = msg.get("type")
            if t == "ping":
                ws_write(wfile, json.dumps({"type": "pong"}))
                continue
            if t not in (None, "", "response.create"):
                ws_write(wfile, json.dumps({"type": "error",
                                            "message": "unsupported type %r" % t}))
                continue
            body = msg.get("response") if isinstance(msg.get("response"), dict) else msg
            prep = api_full.responses_prepare(self.srv, body)
            if not prep.get("ok"):
                ws_write(wfile, json.dumps({"type": "error",
                                            "message": prep.get("error", "bad request"),
                                            "code": prep.get("code", 400)}))
                continue
            try:
                for name, ev in api_full.responses_events(
                        self.srv, prep["eng"], prep["model"], prep["msgs"], body,
                        prep["max_tokens"], prep["tb"]):
                    ws_write(wfile, json.dumps(ev, ensure_ascii=False))
                self.srv.touch("llm")
            except Exception as e:
                try:
                    ws_write(wfile, json.dumps({"type": "error", "message": str(e)}))
                except Exception:
                    pass
                break

    def do_HEAD(self):
        # 产物 / 静态 资源 的 HEAD (Range 探活) —— 与 GET 同 一 套 头, 不 发 body
        try: self.do_GET()
        except Exception: pass

    def do_POST(self):
        # 请求 进行 中 的 引擎 要 保护 住 —— 别 在 跑到 一半 时 被 置闲 卸载 (会 把 权重 抽走, 直接 崩)
        _p=urlparse(self.path).path.rstrip("/") or "/"
        _kinds=PATH_KINDS.get(_p) or ()
        for _k in _kinds: self.srv._begin_use(_k)
        try:
            self._do_POST()
        finally:
            for _k in _kinds: self.srv._end_use(_k)
    def _do_POST(self):
        if not self._auth_ok():
            self._j({"error":{"message":"unauthorized: bad or missing Authorization: Bearer key",
                               "type":"authentication_error","code":401}},401); return
        p=urlparse(self.path).path; q=parse_qs(urlparse(self.path).query)
        # 闭源 mlx-serve 全功能面 (逆向 契约): 先 走 api_full, 没 处理 的 落回 老 路由
        self._body_cache=None
        ctype=self.headers.get("Content-Type","")
        if ctype.startswith("multipart/form-data"):
            mp=self._body_multipart()
            body=dict(mp["fields"]); body["__files__"]=mp["files"]
        else:
            body=self._body_json()
        try:
            if api_full.handle(self.srv,self,"POST",p,q,body): return
        except Exception as e:
            print("[api_full POST]",e,file=sys.stderr,flush=True)
            self._j({"error":{"message":str(e),"type":"server_error","code":500}},500); return
        if p=="/v1/images/generations":
            b=self._body_json()
            self._run_image_v1(b); return
        elif p=="/v1/images/edits":
            self._j({"error":"edit 未 实 装 (壳 用 generation 即可)"},501); return
        elif p in ("/v1/audio/music-generations","/v1/audio/sound-generations"):
            b=self._body_json()
            self._run_music_v1(b); return
        # ===== 出歌工作台 (合并 进 Studio 的「音乐」页) =====
        elif p=="/v1/music/generate":
            # 本地 出歌 —— 走 MLX (ACE-Step Turbo), 秒 级
            b=self._body_json()
            import music_workbench as MW
            self.srv.touch("music")
            r=MW.generate(b)
            if not r.get("ok"):
                self._j({"error":{"message":r.get("error") or "出歌 失败",
                                   "type":"server_error","code":500}},500); return
            # 产物 记 进 资料库, 跟 别的 引擎 一致
            try:
                self.srv.record_library("audio", r.get("path"),
                                        model=r.get("model") or "ace-step-turbo",
                                        prompt=(b.get("prompt") or "")[:120])
            except Exception:
                pass
            with self.srv.metrics_lock:
                self.srv.gen_count+=1
                try:
                    self.srv.gen_bytes+=os.path.getsize(r.get("path") or "")
                except OSError:
                    pass
            self._j(self._fix_urls({
                "ok": True, "engine": r.get("engine"), "sec": r.get("sec"),
                "title": r.get("title"), "duration": r.get("duration"),
                "duration_auto": bool(r.get("duration_auto")),
                "duration_reason": r.get("duration_reason") or "",
                "data": [{"url": self._out_url(r.get("path")),
                          "b64_json": _b64_file(r.get("path")),
                          "duration": r.get("duration")}],
            }))
            return
        elif p=="/v1/music/batch":
            # 批量 出歌 —— 后台 跑
            b=self._body_json()
            import music_workbench as MW
            self._j(MW.batch(b.get("items") or b.get("manifest") or [],
                             out_dir=b.get("out_dir"),
                             concurrency=b.get("concurrency") or 2))
            return
        elif p=="/v1/video/batch":
            # 批量 首尾帧 出片 —— 后台 跑
            b=self._body_json()
            import video_batch as VB
            self._j(VB.batch(self.srv, b.get("items") or b.get("manifest") or [],
                             out_dir=b.get("out_dir"),
                             model=b.get("model") or "h3-4bit",
                             seconds=b.get("seconds") or b.get("duration_seconds") or 5,
                             steps=b.get("steps") or 8,
                             width=b.get("width") or 832,
                             height=b.get("height") or 480,
                             seed=b.get("seed") or 0,
                             prompt_default=b.get("prompt") or ""))
            return
        elif p=="/v1/video/mv":
            # MV 工作台 —— 镜头 (本机 或 远端 出) + 本机 合并 + 配乐
            b=self._body_json()
            import video_mv as VM
            self._j(VM.submit(self.srv,
                              b.get("frames") or [],
                              b.get("prompts") or [],
                              seconds=b.get("seconds") or 5,
                              width=b.get("width") or 832,
                              height=b.get("height") or 480,
                              steps=b.get("steps") or 8,
                              seed=b.get("seed") or 0,
                              music=b.get("music") or "",
                              node=b.get("node") or "local"))
            return
        elif p=="/v1/music/duration":
            _music_duration_route(self, self._body_json())
        elif p=="/v1/outputs":
            # 统一 输出 目录 —— 视频 / 图片 / 音频 / 语音 / 3D 都 落 这里.
            b=self._body_json() if self.command=="POST" else {}
            root=outputs.ensure_root()
            if (b.get("action") or "")=="open":
                try:
                    import subprocess
                    subprocess.run(["/usr/bin/open", root], timeout=20, check=True)
                    self._j({"ok":True,"root":root}); return
                except Exception as e:
                    self._j({"ok":False,"error":str(e)},500); return
            counts={}; total=0
            for kind,name in outputs.KIND_DIR.items():
                d=os.path.join(root,name); n=0; bts=0
                try:
                    for dp,_dn,fn in os.walk(d):
                        for f in fn:
                            try:
                                n+=1; bts+=os.path.getsize(os.path.join(dp,f))
                            except OSError: pass
                except Exception: pass
                counts[kind]={"dir":d,"count":n,"bytes":bts}
                total+=n
            self._j({"ok":True,"root":root,"kinds":counts,"total":total})
            return
        elif p=="/v1/music/reveal":
            # 在 访达 中 定位 成品
            b=self._body_json()
            import music_workbench as MW
            r=MW.reveal(b.get("path"))
            self._j(r, 200 if r.get("ok") else 400)
            return
        elif p=="/v1/audio/speech":
            b=self._body_json()
            self._run_speech_v1(b); return
        elif p=="/v1/video/generations":
            b=self._body_json()
            self._run_video_v1(b); return
        elif p in ("/v1/chat/completions","/v1/completions"):
            b=self._body_json()
            self._run_chat_v1(b); return
        elif p=="/v1/responses":
            b=self._body_json()
            self._run_responses_v1(b); return
        elif p=="/v1/embeddings":
            self._j({"error":"emb 未 实 装"},501); return
        elif p=="/v1/load-model":
            # 按 能 力 单 独 加 载: body {"model":"krea-turbo","capability":"image"}
            b=self._body_json()
            mid=b.get("model","")
            cap=b.get("capability","")
            kind=_engine_kind_for_model(self.srv, mid, cap)
            # 权重 不 在 本机 的 模型 —— 明确 报错, 别 悄悄 加载 别的 模型.
            # (老 行为 会 回 {"loaded":true,"model":"qwen3-4b"}, 用户 以为 装 上 了 gemma3-12b,
            #  其实 挂 的 还是 默认 那个 —— 这 就是 「加载 模型 上 不去」 的 根因.)
            _ent=_catalog_entry(self.srv,mid)
            if _ent is not None and not _ent.get("builtin"):
                _lp=os.path.expanduser(str(_ent.get("local","") or ""))
                # 只 看 catalog.local 会 在 换台 机器 上 误报 「权重 不在」 ——
                # 同一个 模型 的 权重 可能 落在 别的 布局 (魔搭 <org>/<name>、
                # HF 缓存 <org>--<name>/snapshots/master ……). 统一 走 _model_dir
                # 的 多 布局 解析, 才 是 真 的 在 硬盘 上.
                _here = _model_dir(_ent, _lp) if _lp else None
                if not _here:
                    self._j({"error":{"message":
                        "「%s」的 权重 还 不 在 本机 —— 请 先 在「模型」页 点「下载」, 等 下载 完成 再 加载"%mid,
                        "type":"invalid_request","code":400}},400); return
            if kind=="music":
                # 出歌 引擎: 我们 自研 的 MLX (ACE-Step 1.5, 纯 MLX) 为主, ACE-Step :7861 兜底.
                # 「加载」 = 把 权重 拉 进 内存 (免 得 第一首 等 半天).
                import music_workbench as MW
                ours=MW.our_mlx_status()
                # 自研 引擎 在 就 只 装 它 —— 别 顺手 把 ACE-Step :7861 那 23G
                # 也 拉 起来 (用户 要 的 是 别 占 内存); 不 在 才 起 兜底.
                if not ours.get("available") and not MW.mlx_ready():
                    _ace_start()
                # 让 我们 的 引擎 直接 加载 用户 选 的 那 一个 (只 认 "ours/...")
                _want=(mid or "").strip()
                note=MW.load_music(_want if _want.startswith("ours/") else None)
                # 登记 「当前 真正 在 跑 的 出歌 引擎」 —— 模型页 据此 标 loaded
                try:
                    _ours=MW.our_mlx_status()
                    if _ours.get("available"):
                        self.srv.engines["music"]={"via":"ours","ok":bool(_ours.get("running")),
                                                    "model_id":_ours.get("model") or "ours/ace-step-v15-turbo-mlx"}
                    else:
                        self.srv.engines["music"]={"via":"ace","url":ACE_MUSIC,
                                                    "ok":bool(MW.mlx_loaded()),"model_id":"ace-step"}
                except Exception: pass
                try: self.srv.touch("music")
                except Exception: pass
                self._j({"loaded":MW.mlx_loaded(),
                         "model":mid or (ours.get("model") if ours.get("available")
                                         else MW.MLX_MUSIC_MODEL),
                         "capability":"music","engine":"music","note":note})
            elif kind=="tts_neural":
                # 神经 TTS (Qwen3-TTS) 是 独立 的 引擎 (tts_neural), 权重 在 api_full 那 侧
                neng=None
                try: neng=api_full.neural_tts_engine(self.srv)
                except Exception: neng=None
                if neng is None:
                    self._j({"error":{"message":"神经 TTS 引擎 不可用","type":"server_error","code":503}},503); return
                if not neng.available():
                    self._j({"error":{"message":"神经 TTS 权重 不 在 本机","type":"invalid_request","code":400}},400); return
                try: neng.load()
                except Exception as e:
                    self._j({"error":{"message":"神经 TTS 加载 失败: %s"%e,"type":"server_error","code":500}},500); return
                self.srv.touch("tts_neural")
                self._j({"loaded":True,"model":getattr(neng,"model_id",None) or mid,
                         "requested":mid,"capability":"tts","engine":"tts_neural"})
            else:
                try:
                    # mid 传 进去 —— 装 的 是 用户 点 的 那 一个 模型, 不 是 默认 那个
                    self.srv.load(kind,mid=mid); self.srv.touch(kind)
                except MemGuardError as e:
                    self._j({"error":{"message":str(e),"type":"server_error","code":503}},503); return
                except Exception as e:
                    self._j({"error":{"message":"加载 失败: %s"%e,"type":"server_error","code":500}},500); return
                # 实际 装 上 的 是 哪个 模型 (引擎 的 model_id) —— 前端 要 如实 显示,
                # 不能 让 卡片 继续 显示 一个 没 装 上 的 名字.
                eng=self.srv.engines.get(kind)
                emid=(eng.get("model_id") if isinstance(eng,dict) else getattr(eng,"model_id",None)) or mid
                # 记 下 「用户 选 的 那 个」 —— 之后 直接 调 /v1/images/generations
                # 不 带 model 时, 要 用 它 (不 是 随便 挑 一个 在位 的).
                if kind=="image": self.srv._image_mid_pref=emid
                self._j({"loaded":True,"model":emid,"requested":mid,"capability":kind,"engine":kind})
        elif p=="/v1/unload-model":
            b=self._body_json()
            mid=b.get("model","")
            cap=b.get("capability","")
            # 引擎 key 不能 只 看 能力 —— 神经 TTS (qwen3-tts-base) 和 Kokoro (ours/tts)
            # 都 是 cap=tts, 但 是 两个 引擎 (tts_neural / tts). 照 能力 卸 会 卸错:
            # 点 「卸载 qwen3-tts-base」 结果 把 Kokoro 卸 了, 神经 的 还 占 着 显存.
            kind=_engine_kind_for_model(self.srv, mid, cap)
            if kind=="music":
                # 出歌 卸载: MLX 那 份 权重 放 掉; ACE-Step :7861 是 独立 服务,
                # 一 并 停 掉 (它 常驻 ~23G 显存).
                import music_workbench as MW
                try: self.srv.unload("music")
                except Exception: pass
                note=MW.unload_music()   # 这里 会 停 我们 自研 的 MLX 引擎 + 闭源 那份
                stopped=_ace_stop()
                self._j({"unloaded":True,"model":mid or MW.MLX_MUSIC_MODEL,
                         "capability":"music","engine":"music",
                         "note":note + ("\n已 停 出歌 服务 (ACE-Step), 释放 它 占 的 ~23G 显存"
                                        if stopped else "")})
            elif kind:
                self.srv.unload(kind)
                self._j({"unloaded":True,"model":mid,"capability":kind,"engine":kind})
            else:
                self._j({"unloaded":False,"model":mid,"note":"unknown model id"})
        elif p=="/v1/unload-all":
            # 一键 卸 掉 所有 引擎 —— 大 模型 挂 着 就 占 内存, 一次 清空 最 省事
            keep=[k for k,v in (getattr(self.srv,"engines",None) or {}).items()
                  if self.srv._engine_in_use(k)]
            for k in list((getattr(self.srv,"engines",None) or {}).keys()):
                if k in keep: continue
                try: self.srv.unload(k)
                except Exception: pass
            # 出歌 服务 (ACE-Step) 也 要 停 —— 它 是 独立 进程, 不 在 引擎 字典 的 卸载 范围 内
            ace_stopped=False
            if "music" not in keep:
                try: ace_stopped=_ace_stop()
                except Exception: ace_stopped=False
            self._j({"unloaded":True,"kept":keep,
                     "ace_stopped":ace_stopped,
                     "note":("还在 用 的 没 卸: "+ "、".join(_engine_cn(k) for k in keep)) if keep else ""})
        elif p=="/v1/models/download":
            # 下载 模型 权重 —— 后台 线程 跑, 前端 轮询 /v1/models/download/status
            b=self._body_json()
            mid=str(b.get("model","") or "").strip()
            if not mid:
                self._j({"error":{"message":"missing 'model'","type":"invalid_request","code":400}},400); return
            it=_catalog_entry(self.srv,mid)
            if it is None:
                self._j({"error":{"message":"model not found: %s"%mid,"type":"invalid_request","code":404}},404); return
            pull=str(it.get("pull","") or "")
            script=str(it.get("dl_script","") or "")
            ms=str(it.get("ms","") or "")
            local=os.path.expanduser(str(it.get("local","") or ""))
            if not pull and not script and not ms:
                self._j({"error":{"message":"%s 没有 下载 源"%mid,
                                 "type":"invalid_request","code":400}},400); return
            j=_dl_job(mid)
            if j.get("state")=="running":
                self._j({"started":True,"model":mid,"running":True}); return
            _dl_set(mid,state="running",note="准备 中",message="",t0=time.time(),
                    target=local or "")
            def _dl_cb(txt):
                _dl_set(mid,note=str(txt)[-160:])
            def _dl_work():
                try:
                    ok,msg=_download_model(mid,pull,local,script,cb=_dl_cb,ms=ms)
                except Exception as e:
                    ok,msg=False,str(e)
                _dl_set(mid,state=("done" if ok else "error"),note=msg,
                        message=("" if ok else msg))
            threading.Thread(target=_dl_work,daemon=True).start()
            self._j({"started":True,"model":mid,"pull":pull or script})
        elif p=="/v1/models/delete":
            # 删 模型 (把 权重 从 磁盘 上 删掉, 真的 腾 空间).
            # 闭源 的 Mac 应用 才 有 这个 能力 (网页 控制台 里 没有);
            # 我们 把它 做成 API + 控制台 都能 用, 但 路径 必须 落在 模型 根目录 里.
            b=self._body_json()
            mid=str(b.get("model","") or "").strip()
            if not mid:
                self._j({"error":{"message":"missing 'model'","type":"invalid_request","code":400}},400); return
            if not b.get("confirm"):
                self._j({"error":{"message":"need confirm=true (删 权重 是 不可逆 的)",
                                 "type":"invalid_request","code":400}},400); return
            it=None
            for m in self.srv.models_resp().get("models",[]):
                if m.get("id")==mid: it=m; break
            if it is None:
                self._j({"error":{"message":"model not found: %s"%mid,"type":"invalid_request","code":404}},404); return
            if it.get("builtin"):
                self._j({"error":{"message":"%s 是 内置 能力, 没有 权重 可 删"%mid,
                                 "type":"invalid_request","code":400}},400); return
            raw=os.path.expanduser(str(it.get("local","") or ""))
            if not raw or not os.path.isdir(raw):
                self._j({"error":{"message":"%s 没有 本地 权重 可 删"%mid,
                                 "type":"invalid_request","code":400}},400); return
            # 只 允许 删 模型 根目录 里 的 东西 —— 防止 路径 越权
            roots=_model_roots(self.srv)
            tgt=os.path.realpath(raw)
            for r in roots:
                if tgt==r or tgt.startswith(r+os.sep): break
            else:
                self._j({"error":{"message":"拒绝 删除 不 在 模型 目录 里 的 路径: %s"%raw,
                                 "type":"invalid_request","code":400}},400); return
            # HF 快照 结构 (…/snapshots/<rev>) -> 删 整个 repo (含 blobs), 不然 腾 不 出 空间
            if os.sep+"snapshots"+os.sep in tgt:
                tgt=tgt.split(os.sep+"snapshots"+os.sep)[0]
            size=_dir_size(tgt)
            # 先 卸 引擎 (删 磁盘 前 不能 还 占 着 显存)
            kind=_engine_kind_for_model(self.srv, mid, it.get("cap",""))
            try: self.srv.unload(kind)
            except Exception: pass
            try:
                import shutil
                shutil.rmtree(tgt)
            except Exception as e:
                self._j({"error":{"message":"删除 失败: %s"%e,"type":"server_error","code":500}},500); return
            try: self.srv.scan_model_dirs()
            except Exception: pass
            self._j({"deleted":True,"model":mid,"path":tgt,"freed_bytes":size})
        else:
            self._j({"error":{"message":"Unknown endpoint","type":"not_found","code":404}},404)
    def do_DELETE(self):
        if not self._auth_ok():
            self._j({"error":{"message":"unauthorized: bad or missing Authorization: Bearer key",
                               "type":"authentication_error","code":401}},401); return
        p=urlparse(self.path).path; q=parse_qs(urlparse(self.path).query)
        try:
            if api_full.handle(self.srv,self,"DELETE",p,q,{}): return
        except Exception as e:
            print("[api_full DELETE]",e,file=sys.stderr,flush=True)
            self._j({"error":{"message":str(e),"type":"server_error","code":500}},500); return
        self._j({"error":{"message":"Unknown endpoint","type":"not_found","code":404}},404)
    def _run_speech_v1(self,b):
        # 照 抄 闭 源 /v1/audio/speech: input(文 本) -> WAV 二 进 制 流.
        # TTS 引 擎 未 拉 入 我 们 内 核 (Qwen3-TTS-12Hz 是 闭 源 壳 专 属), 直 说 不 能 出, 不 装.
        txt=(b.get("input") or b.get("prompt") or "").strip()
        if not txt:
            self._j({"error":{"message":"missing 'input' (text to speak)","type":"invalid_request","code":400}},400); return
        self._j({"error":"TTS 引 擎 未 拉 入 内 核 (闭 源 壳 专 属), 暂 只 出 曲 风 歌 /v1/audio/music-generations"},501); return
    def _run_image_v1(self,b):
        import base64
        prompt=b.get("prompt","")
        size=str(b.get("size","1344x768"))
        try: w,h=[int(x) for x in size.lower().split("x")]
        except Exception: w,h=1344,768
        out=outputs.path_for("image","mx_v1img","png")
        t0=time.time()
        mid=b.get("model") or self.srv._image_mid_pref or None
        self.srv.load("image", mid)
        _st=int(b.get("steps",0) or 0)
        self.srv._image_running+=1
        try:
            self.srv.engines["image"].generate(prompt,out=out,width=w,height=h,steps=_st,seed=int(b.get("seed",0)))
        finally:
            self.srv._image_running-=1
        sec=round(time.time()-t0,1)
        self.srv.touch("image")
        with self.srv.metrics_lock:
            self.srv.gen_count+=1
            self.srv.gen_bytes+=os.path.getsize(out) if os.path.exists(out) else 0
        b64=base64.b64encode(open(out,"rb").read()).decode() if os.path.exists(out) else ""
        self._j({"data":[{"b64_json":b64,"url":f"file://{out}"}],"sec":sec})
    def _run_music_v1(self,b):
        # 照 抄 闭 源 MLX-Serve: prompt(风 格/曲 风 描 述) 必 填, 缺 了 报 同 款 措 辞
        prompt=(b.get("prompt") or "").strip()
        if not prompt:
            self._j({"error":{"message":"missing 'prompt' (style/genre/mood description)",
                              "type":"invalid_request","param":"prompt","code":400}},400); return
        sec=str(b.get("duration_seconds",b.get("seconds",30)))
        lyrics=(b.get("lyrics") or "")
        lang=b.get("lang","en")
        out=self.srv._music(prompt,sec,lyrics=lyrics,lang=lang)
        self.srv.touch("music")
        with self.srv.metrics_lock:
            if out and "url" in out: self.srv.gen_count+=1
            else: self.srv.gen_err+=1
        r=out or {}
        if "url" in r:
            import base64
            f=r["url"].split("file://")[-1]
            b64=base64.b64encode(open(f,"rb").read()).decode() if os.path.exists(f) else ""
            r={"data":[{"b64_json":b64,"url":self._out_url(r["url"]),"duration":r.get("sec")}],"sec":r.get("sec"),"lyrics":bool(lyrics)}
        self._j(r if r else {"error":"无 结 果"}, 200 if (r and "data" in r) else 502)
    def _run_video_v1(self,b):
        prompt=b.get("prompt","")
        # 帧数 (17n+5) 优先 —— 界面 上 的 「帧数」 滑杆 就是 这个
        nf=int(b.get("num_frames",0) or 0)
        sec=float(b.get("seconds",b.get("duration_seconds",0)) or 0)
        if nf>0: sec=nf/24.0        # H3 是 24fps —— 帧数 -> 秒 要 按 24 算
        if sec<=0: sec=5.0
        steps=int(b.get("steps",8)); seed=int(b.get("seed",0))
        q={"prompt":[prompt],"sec":["%.2f"%sec],"steps":[str(steps)],"seed":[str(seed)]}
        # 视频 模型 (h3-4bit / wan2-ti2v-5b) —— 决定 用 哪个 引擎
        mid=b.get("model")
        if mid: q["model"]=[str(mid)]
        w=int(b.get("width",0) or 0); h=int(b.get("height",0) or 0)
        if w: q["width"]=[str(w)]
        if h: q["height"]=[str(h)]
        cw=int(b.get("chain_windows",0) or 0)
        if cw>1: q["chain_windows"]=[str(cw)]
        # 首帧 / 尾帧 (data URL 或 裸 base64)
        for k in ("first_frame_image","last_frame_image"):
            v=b.get(k)
            if not v: continue
            raw=v.split(",",1)[1] if isinstance(v,str) and v.startswith("data:") else v
            if raw: q[k]=[raw]
        # 参考图 (H3 reference 模式) —— 最多 9 张, 提示词 里 用 <Picture N> 指代
        refs=b.get("reference_images") or b.get("ref_images") or []
        if isinstance(refs,str): refs=[refs]
        rl=[]
        for v in list(refs)[:9]:
            if not v: continue
            raw=v.split(",",1)[1] if isinstance(v,str) and v.startswith("data:") else v
            if raw: rl.append(raw)
        if rl: q["ref_images"]=rl
        out=self.srv._do_video(q)
        # 「要不要声音」—— 不要 声音 就 把 音轨 去掉 (画面 一点 不 动)
        if out and not b.get("sound",True):
            try:
                import video_mv as VM
                u=out.get("url") or ""
                p=u.split("file://")[-1] if u else ""
                if p and os.path.isfile(p):
                    np_=VM.strip_audio(p)
                    if np_: out=dict(out); out["url"]="file://"+np_; out["sound"]=False
            except Exception:
                pass
        self.srv.touch("video")
        self._j(out if out else {"error":"无 结 果"}, 200 if (out and "url" in out) else 504)
    def _run_chat_v1(self,b):
        msgs=b.get("messages",[])
        prompt=""
        for m in reversed(msgs):
            if m.get("role")=="user":
                c=m.get("content"); prompt=c if isinstance(c,str) else " ".join(x.get("text","") for x in c if isinstance(x,dict)); break
        self.srv.load("llm")
        try:
            reply=self.srv.engines["llm"].generate(prompt or "hello")
        except Exception as e:
            self._j({"error":str(e)},503); return
        self.srv.touch("llm")
        self._j({"choices":[{"message":{"role":"assistant","content":reply}}]})
    def _run_responses_v1(self,b):
        inp=b.get("input","")
        if isinstance(inp,list):
            inp=" ".join(x.get("text","") if isinstance(x,dict) else str(x) for x in inp)
        self.srv.load("llm")
        try:
            r=self.srv.engines["llm"].generate(inp or "hello")
        except Exception as e:
            self._j({"error":str(e)},503); return
        self.srv.touch("llm")
        self._j({"output":[{"type":"message","content":[{"type":"output_text","text":r}]}]})
    def _video_body(self,q):
        sid = q.get("session",[None])[0]
        hold_model = sid is not None and self.srv.session_active(sid)
        if hold_model:
            self.srv.load("video")  # 会 话 内 复 用, 已 加 载 则 直 接 返 回
        else:
            self.srv.load("video")
        t0=time.time()
        res=self.srv.engines["video"].generate(q.get("prompt",[""])[0],float(q.get("sec",["5"])[0]),int(q.get("steps",["8"])[0]),int(q.get("seed",["0"])[0]))
        out=outputs.base_for("video","mx_video")
        try:
            from minimax_h3_mlx.media import save_mp4
            save_mp4(out+".mp4",res.video,res.fps,res.audio,res.sample_rate); url=f"file://{out}.mp4"
        except Exception:
            from minimax_h3_mlx.media import save_frames,save_wav
            save_frames(out,res.video); save_wav(out+".wav",res.audio,res.sample_rate); url=f"file://{out}"
        # 统一: 出完不自动卸, 只由闲置计时器 / 手动卸载决定
        self.srv.touch("video"); unloaded=False
        self._j({"url":url,"sec":round(time.time()-t0,1),"frames":int(res.video.shape[0]),"unloaded":unloaded,"session":sid})
# model_type -> 我们 的 能力 (照 抄 闭源 的 model_type 表, §3)
MTYPE_CAP={
    "qwen3":"llm","qwen3_next":"llm","gemma3_text":"llm","gpt_oss":"llm",
    "nemotron_h":"llm","deepseek_v4":"llm","deepseek_v41":"llm","glm5":"llm",
    "glm5_next":"llm","qwen4":"llm","mimo":"llm","krea":"image","flux":"image",
    "qwen_image":"image","minimax_h3":"video","ltx":"video","acestep":"music",
    "music3":"music","qwen3_tts":"tts","kokoro":"tts","hunyuan3d":"3d",
    "laya":"decisions","kev":"decisions","d1":"decisions","clef":"decisions",
    "qwen3_embedding":"embeddings","bert":"embeddings","gte":"embeddings",
    "bge":"embeddings","e5":"embeddings",
}

# ---- 模型 权重 目录 解析 (加载 / 下载 都要 用) ----------------------------
def _share_dir():
    """pip / uv / pipx 装 出来 的 共享 资源 目录 (模型清单 / 下载 脚本 / 出歌 引擎).

    三 种 装法 都要 找得到:
      源码 / App bundle -> <代码目录>/
      pip (wheel)      -> <prefix>/share/mlx-studio/
      手动 拷贝         -> ~/.ours/mlx_studio/
    """
    for c in (os.path.join(sys.prefix,"share","mlx-studio"),
              "/usr/local/share/mlx-studio",
              "/opt/homebrew/share/mlx-studio",
              os.path.expanduser("~/.ours/mlx_studio")):
        try:
            if os.path.isdir(c): return c
        except Exception:
            pass
    return ""

def _catalog_entry(srv, mid):
    """按 id 在 模型清单 (catalog + 扫 到 的 目录) 里 找 条目."""
    try:
        for m in (srv.models_resp() or {}).get("models",[]):
            if str(m.get("id"))==str(mid): return m
    except Exception:
        pass
    return None

def _model_weight_path(srv, mid):
    """模型 id -> 本机 权重 目录 (没 下 载 就 None).

    除了 catalog 的 local, 还 看 「下载 落 地 的 位置」——
    权重 可能 在 ~/.ours/models/mt/<org>/<repo> (魔塔 拉 的),
    也 可能 已经 被 软链 到 catalog.local."""
    it=_catalog_entry(srv,mid)
    if not it: return None
    cands=[]
    loc=str(it.get("local","") or "")
    if loc: cands.append(os.path.expanduser(loc))
    pull=str(it.get("pull","") or "")
    if pull and "/" in pull:
        cands.append(os.path.expanduser("~/.ours/models/mt/"+pull))
        cands.append(os.path.expanduser("~/.ours/models/"+pull))
        # 老 机器 上 闭源 留下 的 那 一 份 (合并 之前 的 位置, 兜底)
        cands.append(os.path.expanduser("~/.mlx-serve/models/mt/"+pull))
        cands.append(os.path.expanduser("~/.mlx-serve/models/"+pull))
    for c in cands:
        if c and os.path.isdir(c):
            # 软链 指向 的 目标 也 算
            if os.path.islink(c):
                try:
                    t=os.path.realpath(c)
                    if os.path.isdir(t): return t
                except Exception: pass
            return c
    return None

def _is_neural_tts_id(mid):
    """这个 model id 是不是 神经 TTS (Qwen3-TTS) —— 它 走 的 是 tts_neural 引擎."""
    m=str(mid or "").strip().lower()
    if not m:
        return False
    if m in ("qwen3-tts-base","ours/qwen3-tts","qwen3-tts","qwen3-tts-0.6b-base"):
        return True
    return "qwen3-tts" in m or "qwen3_tts" in m

def _engine_kind_for_model(srv, mid, cap=""):
    """模型 id -> 引擎 key (srv.engines 里 的 key). 装/卸 都 用 它 —— 必须 一致.

    1) 先 反查 已经 装 着 的 引擎 的 model_id (最准: 装 着 的 就是 这 个 引擎);
    2) 再 看 catalog 的 cap (qwen3-tts-base 的 cap 是 tts, 但 它 是 tts_neural 不是 tts);
    3) 最后 capability 映射.
    """
    try:
        for _k,_e in list((getattr(srv,"engines",None) or {}).items()):
            _mid=_e.get("model_id") if isinstance(_e,dict) else getattr(_e,"model_id",None)
            if _mid and str(_mid)==str(mid):
                return _k
    except Exception:
        pass
    cat_cap=""
    for m in _models_resp().get("models",[]):
        if m.get("id")==mid: cat_cap=m.get("cap",""); break
    cap_map={"chat":"llm","streaming":"llm","vision":"llm","image":"image",
             "audio":"tts","music":"music","video":"video","embeddings":"embeddings",
             "sound":"sound","3d":"3d","decisions":"decisions","transcription":"stt"}
    kind=cat_cap or cap_map.get(cap,"llm")
    if kind=="tts":
        # 神经 TTS (Qwen3-TTS) 与 Kokoro 都 是 cap=tts, 但是 两个 引擎
        kind="tts_neural" if _is_neural_tts_id(mid) else "tts"
    return kind

def _model_roots(srv):
    """允许 删除/管理 的 模型 根目录 (realpath)."""
    roots=[]
    for p in [os.path.expanduser("~/.ours/models"),
              os.path.expanduser("~/.ours/quant"),
              os.path.expanduser("~/.cache/modelscope/models"),
              # 老 机器 上 闭源 留下 的 那 一 份 (合并 之前 的 位置, 兜底)
              os.path.expanduser("~/.mlx-serve/models")]:
        try: roots.append(os.path.realpath(p))
        except Exception: pass
    for d in (getattr(srv,"model_dirs",None) or []):
        try: roots.append(os.path.realpath(os.path.expanduser(d)))
        except Exception: pass
    return roots

def _dir_size(path):
    """目录 总字节 (删 之前 算 给 用户 看)."""
    total=0
    try:
        for root,_d,files in os.walk(path):
            for fn in files:
                try: total+=os.lstat(os.path.join(root,fn)).st_size
                except Exception: pass
    except Exception:
        pass
    return total

def _scan_model_dirs(dirs):
    """扫 --model-dir: 找 带 config.json 的 模型 目录, 按 model_type 推 能力."""
    out=[]
    for d in (dirs or []):
        root=os.path.expanduser(d)
        if not os.path.isdir(root):
            continue
        for org in sorted(os.listdir(root)):
            op=os.path.join(root,org)
            if not os.path.isdir(op):
                continue
            for repo in sorted(os.listdir(op)):
                rp=os.path.join(op,repo)
                cfg=os.path.join(rp,"config.json")
                if not os.path.isfile(cfg):
                    continue
                try:
                    c=json.load(open(cfg))
                except Exception:
                    continue
                mt=str(c.get("model_type") or "")
                cap=MTYPE_CAP.get(mt)
                if not cap:
                    arch=(c.get("architectures") or [""])[0].lower()
                    cap="llm" if ("causallm" in arch or "llama" in arch) else None
                if not cap:
                    continue
                out.append({"id":"%s/%s"%(org,repo),"name":c.get("_name_or_path") or repo,
                            "cap":cap,"engine":"our-server /"+cap,"local":rp,
                            "ctx":int(c.get("max_position_embeddings") or 32768),
                            "arch":mt or repo,"scanned":True})
    return out

def _norm_model_name(s):
    """模型 名 归一化 —— 不同 机器 的 目录 布局 命名 不 一 样."""
    s=str(s or "").strip().lower()
    for ch in ("--","__","/","_"):
        s=s.replace(ch,"-")
    return s


def _model_dir(it, local=None):
    """模型 权重 在 哪 —— 挨个 布局 找, 找到 就 返回 目录 (没有 返回 None).

    同一 个 模型 在 不同 机器 上 的 落法 不 一样:
      ~/.ours/models/models/<org>--<name>/snapshots/main   (HF 缓存 布局)
      ~/.ours/models/<org>/<name>                          (魔搭 布局)
      ~/.ours/models/mlm/<name>                            (我们 自建 的 对话 模型)
      ~/.ours/quant/<name>                                 (视频 DiT 量化)
    catalog 里 只 写 了 其中 一 种 —— 换台 机器 就 认 不 出 了 (出图/出片 在 远端 上
    曾 显示 absent, 其实 权重 好端端 在 硬盘 上). 所以 这里 挨个 试.
    """
    loc=os.path.expanduser(local if local is not None else (it.get("local","") or ""))
    if loc and os.path.isdir(loc):
        return loc
    pull=str(it.get("pull") or "")
    if "/" not in pull:
        return None
    org,name=pull.split("/",1)
    home=os.path.expanduser("~")
    bases=[os.path.join(home,".ours/models/models"),
           os.path.join(home,".ours/models"),
           os.path.join(home,".ours/models/mlm"),
           os.path.join(home,".ours/quant")]
    cands=[]
    for b in bases:
        cands += [os.path.join(b, org+"--"+name),
                  os.path.join(b, org, name),
                  os.path.join(b, name)]
        for rev in ("main","master"):
            cands.append(os.path.join(b, org+"--"+name, "snapshots", rev))
    for c in cands:
        if os.path.isdir(c):
            return c
    # —— 兜底: 在 ~/.ours/models 里 扫 几 层, 按 名字 模糊 匹配 ——
    want=_norm_model_name(name)
    try:
        root=os.path.join(home,".ours/models")
        for dp,dn,fn in os.walk(root):
            if dp[len(root):].count(os.sep)>3:
                dn[:]=[]
                continue
            for d in dn:
                if _norm_model_name(d)==want:
                    return os.path.join(dp,d)
    except Exception:
        pass
    return None


def _models_resp(extra_dirs=None, scanned=None):
    import json
    # 目录: 先 找 内核 自带 的 catalog, 再 回退 到 共享 的 ~/.ours/mlx_server/
    here=os.path.dirname(os.path.abspath(__file__))
    _sd=_share_dir()
    cands=[os.path.join(here,"model_catalog.json"),
           os.path.expanduser("~/.ours/mlx_server/model_catalog.json")]
    if _sd: cands.append(os.path.join(_sd,"model_catalog.json"))
    base=[]
    for cat in cands:
        if os.path.exists(cat):
            try:
                base=json.load(open(cat)).get("catalog",[])
            except Exception:
                base=[]
            break
    out=[]
    seen=set()
    for it in base:
        loc=os.path.expanduser(it.get("local",""))
        cap=it.get("cap","")
        if it.get("builtin"):
            present=True
        elif cap=="music":
            # 出歌: MLX (ACE-Step Turbo) 为主 + ACE-Step :7861 兜底.
            # present = 任 一 头 可用.
            import urllib.request
            present=False
            loc=os.path.expanduser(it.get("local",""))
            if _model_dir(it, loc):
                present=True
            try:
                urllib.request.urlopen(ACE_MUSIC+"/health", timeout=4).read()
                present=True
            except Exception:
                pass
            if not present:
                try:
                    import music_workbench as MW
                    present=MW.mlx_ready()
                except Exception:
                    pass
            # 出歌 权重 真 挂 在 内存 里 没 —— (卸载 后 要 如实 报, 别 骗 人)
            try:
                import music_workbench as MW
                it2_loaded=MW.mlx_loaded()
            except Exception:
                it2_loaded=False
        else:
            present=_model_dir(it, loc) is not None
            it2_loaded=None
        it2=dict(it); it2["present"]=present
        if it2_loaded is not None: it2["loaded_hint"]=it2_loaded
        out.append(it2); seen.add(it2["id"])
    for it in (scanned or []):
        if it["id"] in seen:
            continue
        it2=dict(it); it2["present"]=True
        out.append(it2); seen.add(it2["id"])
    return {"node":NODE,"models":out}

# ---- 下载 任务 (后台 线程 跑, 前端 轮询 状态) ----------------------------
_DL_LOCK=threading.Lock()
_DL_JOBS={}      # model id -> {"state":"running|done|error","note":str,"message":str,"t0":ts}

def _dl_job(mid):
    with _DL_LOCK: return dict(_DL_JOBS.get(str(mid)) or {})

def _dl_set(mid,**kw):
    with _DL_LOCK:
        j=_DL_JOBS.setdefault(str(mid),{"state":"running","note":"","message":"","t0":time.time()})
        if "t0" not in kw: j["t0"]=j.get("t0") or time.time()
        j.update(kw)
        return dict(j)

def _resolve_pull_name(pull):
    """短名 (gemma3:12b) -> 'org/repo' —— 问 mlx-serve 自己 的 别名 表 (它 带 缓存)."""
    if not pull: return ""
    if "/" in pull: return pull
    cache=os.path.expanduser("~/.ours/mlx_server/pull_map.json")
    try:
        m=json.load(open(cache))
        if pull in m: return m[pull]
    except Exception: pass
    serve=os.path.expanduser("~/.ours/mlx-serve/bin/mlx-serve")
    if not os.path.exists(serve): return pull
    try:
        r=subprocess.run([serve,"pull","--help"],capture_output=True,text=True,timeout=60)
        out=(r.stdout or "")+"\n"+(r.stderr or "")
    except Exception:
        return pull
    m={}
    for line in out.splitlines():
        if "->" not in line: continue
        left,right=line.split("->",1)
        repo=right.strip().split()[0] if right.strip() else ""
        if not repo or "/" not in repo: continue
        # 左边 形如 "gemma3 (= gemma3:12b)" —— 两个 名字 都 记 下
        for tok in re.findall(r"[A-Za-z0-9_.:\-]+",left):
            if tok and tok not in ("=","(" ,")"): m[tok]=repo
    try:
        os.makedirs(os.path.dirname(cache),exist_ok=True)
        json.dump(m,open(cache,"w"))
    except Exception: pass
    return m.get(pull,pull)

def _hf_ready(d):
    """权重 目录 里 是不是 已经 有 一个 完整 的 模型 (config + 权重)."""
    if not d or not os.path.isdir(d): return False
    if not os.path.isfile(os.path.join(d,"config.json")): return False
    try:
        for f in os.listdir(d):
            if f.endswith(".safetensors"): return True
    except Exception:
        return False
    return False

def _hf_download(repo, target, cb=None, tries=3):
    """从 hf-mirror 拉 一个 HF 仓库 —— 国内 直连 hf.co 常常 卡死, 走 镜像 才 稳.

    必须 开 一个 子 进程 拉: huggingface_hub 在 **导入 时** 就 把 域名 定死
    (hf_hub_download 的 默认 参数 是 导入 时 求值 的), 主进程 里 改 环境变量
    对 已经 导入 过 的 库 不 生效 —— 会 直连 huggingface.co 然后 一直 超时 重试.
    下载 中途 卡死 (镜像 CDN 偶尔 会 把 大 文件 的 连接 晾 着) -> 看门狗 重启,
    hf_hub 会 从 .incomplete 断点 续传, 不会 白 拉.
    """
    target=os.path.expanduser(target)
    os.makedirs(target,exist_ok=True)
    EP="https://hf-mirror.com"
    env=dict(os.environ); env["HF_ENDPOINT"]=EP
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
    env.setdefault("HF_HUB_DISABLE_XET","1")
    env.setdefault("HF_HUB_ETAG_TIMEOUT","8")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","30")
    code=(
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], endpoint=sys.argv[3]))\n"
    )
    py=sys.executable
    if not _has_hf(py):
        # 退而 求其次: 找 一个 装了 huggingface_hub 的 环境
        for c in (os.path.expanduser("~/.venvs/mlxstudio/bin/python"),
                  os.path.expanduser("~/.venvs/mtx/bin/python"),
                  os.path.expanduser("~/.venvs/h3m3/bin/python"),
                  os.path.expanduser("~/.venvs/h3m/bin/python")):
            if os.path.exists(c) and _has_hf(c):
                py=c; break
    import queue
    last=""
    for attempt in range(1, max(1,tries)+1):
        try:
            p=subprocess.Popen([py,"-c",code,repo,target,EP],
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,env=env,bufsize=1)
        except Exception as e:
            return False,"起 下载 进程 失败: %s"%e
        q=queue.Queue()
        def _reader(_p=p,_q=q):
            try:
                for line in _p.stdout:
                    _q.put(line)
            except Exception:
                pass
        threading.Thread(target=_reader,daemon=True).start()
        _last_size=-1; _stall=0
        while p.poll() is None:
            try:
                line=q.get(timeout=5)
                line=(line or "").strip()
                if line:
                    last=line
                    if cb and ("B/s" in line or "MiB" in line or "Fetching" in line):
                        cb(line[-110:])
                continue
            except queue.Empty:
                pass
            # 看门狗: 2 分钟 没 多 一个 字节 -> 掐 掉 重来 (断点 续传)
            _sz=_dir_size(target)
            if _sz>_last_size:
                _last_size=_sz; _stall=0
            else:
                _stall+=1
                if _stall>=24:
                    print(f"[download] 卡住 2 分钟, 重启 (第 {attempt}/{tries} 次)",file=sys.stderr,flush=True)
                    try: p.kill()
                    except Exception: pass
                    break
        try: p.wait(timeout=10)
        except Exception:
            try: p.kill()
            except Exception: pass
        if p.returncode==0:
            return True,"hf-mirror 下载 完成 -> %s"%target
        if attempt>=max(1,tries):
            break
        print(f"[download] 第 {attempt} 次 没 成 (rc={p.returncode}), 再试",file=sys.stderr,flush=True)
        if cb: cb("网络 不 顺, 重试 %d/%d"%(attempt+1,tries))
        time.sleep(3)
    return False,(last[-300:] or "hf-mirror 下载 失败")

def _has_hf(py):
    try:
        r=subprocess.run([py,"-c","import huggingface_hub"],capture_output=True,timeout=30)
        return r.returncode==0
    except Exception:
        return False

def _has_ms(py):
    try:
        r=subprocess.run([py,"-c","import modelscope"],capture_output=True,timeout=30)
        return r.returncode==0
    except Exception:
        return False

def _ms_download(repo, target, cb=None, tries=2):
    """从 魔搭 (modelscope) 拉 一个 仓库 —— 国内 最快, 优先 走 这 条."""
    try:
        import msdl
    except Exception:
        return False, "缺 msdl.py"
    return msdl.ms_download(repo, target, cb=cb, tries=tries)

def _download_model(mid, pull=None, local=None, script=None, cb=None, ms=None):
    """下载 一个 模型 的 权重.

    ① 魔搭 脚本 —— 出歌 权重 (ACE-Step / Music3) 只 在 魔搭 有;
    ② 魔搭 直拉 —— 通用 路径 (国内 最快, catalog 里 的 ms 字段);
    ③ hf-mirror 直拉 —— 魔搭 没有 的 (Wan2-Fast-MLX / MOSS-Image-SFT 等);
    ④ 魔塔 (mlx-serve pull) —— 兜底 (它 认 短名, 但 国 内 直连 常常 卡住).
    下载 完 的 权重 落 在 catalog.local —— 引擎 照 那 个 路径 找.
    """
    if not pull and not script and not ms:
        return False,"这个 模型 没有 下载 源"
    here=os.path.dirname(os.path.abspath(__file__))
    # ① 魔搭 脚本
    if script:
        sp=os.path.join(here,script)
        if not os.path.exists(sp):
            _sd2=_share_dir()
            if _sd2: sp=os.path.join(_sd2,script)
        if os.path.exists(sp):
            try:
                r=subprocess.run([sys.executable,sp],capture_output=True,text=True,timeout=7200)
                if r.returncode==0: return True,"魔搭 下载 完成"
                print(f"[download] 魔搭 脚本 失败: {((r.stderr or r.stdout or '')[-200:])}",file=sys.stderr,flush=True)
            except Exception as e:
                print(f"[download] 魔搭 脚本 异常 {e}",file=sys.stderr,flush=True)
        else:
            print(f"[download] 魔搭 脚本 不存在 {sp}",file=sys.stderr,flush=True)
        # 魔搭 拉 不 下来 -> 往 下 走, 试 HF (ACE-Step / Music3 也 有 HF 镜像)
    # ② 魔搭 直拉 (国内 最快 —— 魔搭 上 基本 都 有)
    if ms:
        target=local or os.path.expanduser("~/.ours/models/pulls/"+
            "".join(c for c in ms if c.isalnum() or c in "-_."))
        if cb: cb("魔搭: %s"%ms)
        if _hf_ready(target):
            return True,"已经 在 本机 (%s)"%target
        ok,msg=_ms_download(ms,target,cb=cb)
        if ok: return True,msg
        print(f"[download] 魔搭 失败: {msg}",file=sys.stderr,flush=True)
    # ③ HF 仓库 -> hf-mirror
    repo=_resolve_pull_name(pull)
    if "/" in repo:
        target=local or os.path.expanduser("~/.ours/models/pulls/"+
            "".join(c for c in repo if c.isalnum() or c in "-_."))
        if cb: cb("目标: %s"%target)
        # 已经 在 本机 (config + 权重 都在) -> 不用 再 拉 一遍
        # (hf-mirror 校验 小 文件 时 会 长时间 超时 重试, 看着 像 卡死)
        if _hf_ready(target):
            return True,"已经 在 本机 (%s)"%target
        ok,msg=_hf_download(repo,target,cb=cb)
        if ok: return True,msg
        print(f"[download] hf-mirror 失败: {msg}",file=sys.stderr,flush=True)
        # ③ 魔塔 兜底 (权重 会 落 在 ~/.ours/models/mt/<org>/<repo>)
        serve=os.path.expanduser("~/.ours/mlx-serve/bin/mlx-serve")
        if os.path.exists(serve):
            try:
                r=subprocess.run([serve,"pull",pull],capture_output=True,text=True,timeout=600)
                if r.returncode==0:
                    _link_after_pull(repo,local)
                    return True,"魔 塔 下载 完成"
                msg=((r.stderr or r.stdout or "")[-300:] or "魔 塔 下载 失败")
            except Exception as e:
                msg=str(e)
        return False,msg
    # ④ 只 有 魔塔 认 的 名字
    serve=os.path.expanduser("~/.ours/mlx-serve/bin/mlx-serve")
    if os.path.exists(serve):
        try:
            r=subprocess.run([serve,"pull",pull],capture_output=True,text=True,timeout=600)
            if r.returncode==0:
                _link_after_pull(repo,local)
                return True,"魔 塔 下载 完成"
            return False,((r.stderr or r.stdout or "")[-300:] or "魔 塔 下载 失败")
        except Exception as e:
            return False,str(e)
    return False,"没有 可用 的 下载 方式 (需要 hf-mirror 或 魔塔)"


def _link_after_pull(repo, local):
    """魔塔 拉 完 的 权重 -> 软链 到 引擎 期望 的 位置 (local)."""
    if not repo or not local: return
    dest=os.path.expanduser("~/.ours/models/mt/"+repo)
    if not os.path.isdir(dest):
        alt=os.path.expanduser("~/.ours/models/"+repo)
        if os.path.isdir(alt): dest=alt
    if not os.path.isdir(dest):      # 老 机器 (合并 之前) 的 位置, 兜底
        alt=os.path.expanduser("~/.mlx-serve/models/mt/"+repo)
        if os.path.isdir(alt): dest=alt
    if not os.path.isdir(dest): return
    if os.path.exists(local):
        return   # 已经有 权重 了, 不 动
    try:
        os.makedirs(os.path.dirname(local),exist_ok=True)
        os.symlink(dest,local)
        print(f"[download] 软链 {local} -> {dest}",file=sys.stderr,flush=True)
    except Exception as e:
        print(f"[download] 软链 失败 {e}",file=sys.stderr,flush=True)

def main():
    # 服务 参数 面: 照 抄 闭源 mlx-serve 的 flag 表 (MECHANICS §13), 我们 内核 只 认 真正 用得 上 的
    ap=argparse.ArgumentParser(prog="mlx-server",description="我们自研 MLX 推理内核 (mlx-serve 接口兼容)")
    ap.add_argument("--port",type=int,default=11234,help="监听端口 (原版 默认 11234)")
    ap.add_argument("--host",default="127.0.0.1",help="监听地址 (原版 默认 0.0.0.0)")
    ap.add_argument("--model-dir",action="append",default=[],
                    help="模型目录 (可多次给; 目录 里 带 config.json 的 会被 扫成 模型)")
    ap.add_argument("--os-reserve-gib",type=int,default=4,help="给系统留的余量 GiB (0..64)")
    ap.add_argument("--wired-margin-gib",type=int,default=8,help="显存余量 GiB (2..32)")
    ap.add_argument("--ctx-size",type=int,default=32768,help="上下文长度")
    ap.add_argument("--max-tokens",type=int,default=100,help="请求 不给 max_tokens 时 的 默认")
    ap.add_argument("--kv-quant",default="off",help="KV 量化 (off/4/8)")
    ap.add_argument("--kv-attn-mode",default="off",help="KV attention 模式")
    ap.add_argument("--max-resident-models",type=int,default=3,help="常驻 模型 数 上限")
    ap.add_argument("--max-resident-mem",default="auto",help="常驻 内存 上限")
    ap.add_argument("--idle-evict-secs",type=int,default=IDLE_S,help="闲置 自动 卸载 秒数 (0=关)")
    ap.add_argument("--prefix-cache-mem",type=int,default=0,help="前缀 缓存 RAM 上限 (MB, 0=关)")
    ap.add_argument("--prefix-cache-disk",type=int,default=0,help="前缀 缓存 SSD 上限 (MB, 0=关)")
    ap.add_argument("--prefill-chunk",type=int,default=16384,help="prefill 分块 上限")
    ap.add_argument("--timeout",type=int,default=300,help="无 token 停顿 超时 秒 (0=不超时)")
    ap.add_argument("--api-key",default="",help="非 本机 请求 要 的 Bearer key")
    ap.add_argument("--metrics",action="store_true",help="打开 /metrics + /metrics.json")
    ap.add_argument("--log-level",default="info",help="日志 级别")
    a=ap.parse_args()
    if a.os_reserve_gib<0 or a.os_reserve_gib>64:
        print("[mlx_server] --os-reserve-gib 越界 (要 0..64), 忽略",file=sys.stderr,flush=True)
        a.os_reserve_gib=4
    if a.wired_margin_gib<2 or a.wired_margin_gib>32:
        a.wired_margin_gib=8
    srv=Srv()
    srv.os_reserve_gib=a.os_reserve_gib; srv.wired_margin_gib=a.wired_margin_gib
    srv.idle_evict_secs=a.idle_evict_secs; srv.max_resident_models=a.max_resident_models
    srv.prefix_cache_mem=a.prefix_cache_mem; srv.prefix_cache_disk=a.prefix_cache_disk
    srv.ctx_size=a.ctx_size; srv.max_tokens=a.max_tokens; srv.api_key=a.api_key
    srv.metrics_on=a.metrics; srv.host=a.host; srv.port=a.port
    srv.model_dirs=list(a.model_dir)
    srv._scanned=_scan_model_dirs(srv.model_dirs)
    srv.kv_quant=a.kv_quant; srv.timeout=a.timeout; srv.log_level=a.log_level
    srv.prefix_cache_mem=a.prefix_cache_mem; srv.prefix_cache_disk=a.prefix_cache_disk
    H.srv=srv; threading.Thread(target=H.srv.tick,daemon=True).start()
    # 资料库: 后台 扫 一次 产物 目录 (补 回 重启 前 生成 的 卡片) —— 不 阻塞 启动
    try: srv.start_library_scan()
    except Exception: pass
    print(f"[mlx_server] 自 研 多 模 态 服 务 {a.host}:{a.port} "
          f"(视 频 排 他 + 置 闲 {a.idle_evict_secs}s 自动 卸 / 可 用 <{MIN_FREE_GIB}G 拒 视 频 / "
          f"os-reserve {a.os_reserve_gib}G)",file=sys.stderr,flush=True)
    # ThreadingHTTPServer: LLM 加 载 2min+ 时 不 会 堵 住 /health 等 其 它 请 求
    class HttpSrv(ThreadingHTTPServer):
        # listen backlog 默认 只有 5: 控制台 一开 就 并行 打 好几 个 接口
        # (metrics / library / models / 图片 缩略图 …), 队列 一 满 后面 的 连接
        # 直接 被 reset —— 客户端 看到 的 是 "connection lost" (-1005),
        # 表现 就是 桌面 App 状态条 时 绿 时 红, 首帧 加载 也 会 偶发 失败.
        request_queue_size = 256
        daemon_threads = True
        allow_reuse_address = True
    HttpSrv((a.host,a.port),H).serve_forever()
def _music_duration_route(H, b):
    """「自动时长」预估 —— 内核 算, 界面 只 显示 (免得 两 套 算法 跑偏)."""
    try:
        import music_duration as MD
        import music_mlx
        mx = float(music_mlx.max_duration())
        lyrics = b.get("lyrics") or ""
        inst = bool(b.get("instrumental"))
        sec, info = MD.estimate(lyrics, instrumental=inst, max_s=mx)
        H._j({"seconds": sec, "reason": MD.explain(info), "max": mx, "detail": info})
    except Exception as e:
        H._j({"error": {"message": str(e), "type": "server_error"}}, 500)


if __name__=="__main__": main()

