"""我们 自己 的 出 歌 引 擎 — ACE-Step (本机不动, 本 地 权 重 + MLX).
权 重 拉 到 ~/.ours/models/ace-step/ 后 就 能 跑."""
import os, sys, gc, time, tempfile
MODEL_DIR = os.path.expanduser("~/.ours/models/ace-step")
class MusicEngine:
    def __init__(self):
        self.pipe=None
    def load(self):
        if self.pipe: return
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(f"ACE-Step 权 重 不 在 {MODEL_DIR}, 先 拉 下 再 试")
        # 懒 加 载 ACE-Step MLX 引 擎
        from ace_step_mlx import ACEStepPipeline
        self.pipe=ACEStepPipeline.from_pretrained(MODEL_DIR)
    def unload(self):
        if self.pipe:
            del self.pipe; self.pipe=None
            import mlx.core as mx
            for a in ("metal_flush","flush"):
                f=getattr(mx,a,None)
                if callable(f):
                    try: f(); break
                    except Exception: pass
            gc.collect()
    def generate(self, prompt, out="/tmp/mx_music.wav", sec=30, seed=None, steps=None):
        self.load()
        kw={}
        if seed is not None: kw["seed"]=int(seed)
        if steps is not None: kw["inference_steps"]=int(steps)
        wav=self.pipe.generate(prompt,duration=sec,**kw)
        wav.save(out)
        return out
