"""我们 自己 的 本 地 大 模 型 引 擎 — Apple MLX 本 体 + mlx_lm, 全 在 本 地.

内存纪律: load/unload 与 engine.Engine 一致, 必须 gc + mx.clear_cache,
否则反复 load 堆冷页 (compressor 52G 真凶)."""
import os, gc, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import _free
# Qwen3-0.8B-MLX 是 vision 重包 (mlx_lm 加载不到), 改走纯 Qwen3-0.8B-4bit
# (mlx-community 量化, 两边内核一致). 路径可被 H3M_LLM 覆盖.
# 默认 Qwen3-4B (2026-11-10 魔塔 mlx-community/Qwen3-4B-4bit 拉完, 2.1G).
# 0.8B 目录保留为 fallback: 4B 不在就回 0.8B (vision 重包加载不到, 但有 0.8B-4bit 纯包).
_DEFAULT = os.path.expanduser("~/.ours/models/mlm/Qwen3-4B-MLX")
_FALLBACK = os.path.expanduser("~/.ours/models/mlm/Qwen3-0.8B-4bit")
MODEL = os.environ.get("H3M_LLM", _DEFAULT if os.path.isdir(_DEFAULT) else _FALLBACK)


def _model_id_from_path(p):
    """权重 目录 名 -> 模型 卡 的 id (Qwen3-4B-MLX -> qwen3-4b)."""
    base = os.path.basename(os.path.normpath(p)).lower()
    for suf in ("-mlx-serve", "-mlx", "_mlx"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


MODEL_ID = _model_id_from_path(MODEL)

# 默认 思考 预算 (客户端 没 指定 reasoning_effort 时 用)
DEFAULT_THINK_BUDGET = 1024


class LLMEngine:
    # 模型 卡 的 id (装 的 是 哪 一 份 权重) —— /v1/models 的 loaded 态 按 它 算,
    # 不 能 按 "llm 引擎 在 不 在" 算 (否则 目录 里 所有 LLM 都 显示 loaded).
    model_id = MODEL_ID

    def __init__(self, path=None, model_id=None):
        self.model=None; self.tokenizer=None
        self.last_info={}          # 最近 一 次 生成 的 元信息 (phase/truncated)
        # path: 「加载 / 更换」 时 用户 指定 的 那 一 份 权重 目录 (不 给 就 用 默认 的 Qwen3-4B)
        self.path = path or MODEL
        if model_id: self.model_id = model_id
    def load(self):
        if self.model: return
        # 反复 load 堆冷页防护: 先卸干净旧引用, 再 load
        self._drop()
        # mlx_lm.load() 把绝对路径当成 HF repo 报 401/校验错 — 本地目录必须走底层 API.
        from pathlib import Path as _Path
        from mlx_lm.utils import load as _mlx_load
        mp=_Path(self.path)
        # 先探 config; 若没有就自动从 model.safetensors 推 (mlx-lm 量化包自带 mlm meta)
        cfg_path=mp/"config.json"
        if not cfg_path.exists():
            # 兜底: 写最小 Qwen3-0.6B 架构 (0.8B 用 0.6B 的 config)
            import json as _json
            cfg={"architectures":["Qwen3ForCausalLM"],
                 "hidden_size":1024,"intermediate_size":3072,
                 "num_attention_heads":16,"num_hidden_layers":36,"num_key_value_heads":8,
                 "head_dim":64,"max_position_embeddings":32768,
                 "rms_norm_eps":1e-06,"vocab_size":151936,
                 "tie_word_embeddings":True,"model_type":"qwen3",
                 "rope_theta":5000000,"mlf_config":"mlx"}
            cfg_path.write_text(_json.dumps(cfg))
            print(f"[llm_engine] 写 兜 底 config -> {cfg_path}", file=sys.stderr, flush=True)
        self.model,self.tokenizer=_mlx_load(str(mp), lazy=False)
        gc.collect()
    def _drop(self):
        if self.model is not None:
            del self.model; self.model=None
        if self.tokenizer is not None:
            del self.tokenizer; self.tokenizer=None
        _free()
    def unload(self):
        self._drop()
    # ---- 生成: 思考 预算 与 正文 预算 分 开 ----
    #
    # Qwen3 这 类 模型 会 先 吐 一大段 <think>...</think> 再 给 正文.
    # 如果 把 客户端 给 的 max_tokens 全 让 给 思考, 回答 就 是 空 的 ——
    # 这是 mlx-serve 这 类 服务 最 常见 的 "模型 不 回话" 故障.
    # 所以 我们 把 token 预算 拆 两 份 (与 原版 reasoning_effort 语义 对齐):
    #   think_budget: 思考 阶段 的 上限 (由 reasoning_effort 决定)
    #   max_tokens:   正文 阶段 的 上限 (客户端 给 的 值, 语义 不 变)
    PROBE_TOKENS = 24        # 前 N 个 token 没 <think> -> 认定 模型 不 思考
    MIN_NEW_TOKENS = 4       # 前 N 个 生成 token 禁 EOS (防 "模型 首 token 就 结束" -> 空 回答)

    def _raw_stream(self, prompt, budget, temp=0.6, top_p=0.9, top_k=0):
        """底层 生成: 逐 片段 yield, 只 管 EOS 停, 不 做 任何 拆分."""
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        eos_ids = self._eos_ids()
        s = make_sampler(
            float(temp), float(top_p), 0.0, int(top_k), 0,
            xtc_probability=0.0, xtc_threshold=0.1,
            xtc_special_tokens=eos_ids or None,
        )
        # 采样 是 随机 的 (temp 0.6): Qwen3 在 /no_think 下 有 20~30% 概率
        # 首 token 就 吐 <|im_end|> —— 整个 回答 就 是 空 的 (用户 看到 的 是 "模型 不 回话").
        # 所以 前 MIN_NEW_TOKENS 个 生成 token 把 EOS 禁掉 (vLLM 的 min_tokens 语义).
        plen = None
        try:
            plen = len(self.tokenizer.encode(prompt))
        except Exception:
            plen = None
        lp = self._no_eos_processor(plen)
        eos_set = set(eos_ids or [])
        for tok in stream_generate(self.model, self.tokenizer, prompt,
                                  max_tokens=max(1, int(budget)), sampler=s,
                                  logits_processors=([lp] if lp else None)):
            tid = tok.token
            # 用 mlx_lm 自己 的 流式 detokenizer: 它 会 把 跨 token 的 多字节 字符
            # (中文/emoji) 攒 齐 再 吐. 自己 decode([tid]) 会 把 一个 字 拆 成
            # 几个 坏 字节 -> "等" 变成 "���" (控制台 里 能 看到 豆腐块).
            piece = getattr(tok, "text", None)
            if piece is None:
                piece = ""
            is_eos = False
            if isinstance(tid, int):
                is_eos = tid in eos_set
            elif isinstance(tid, list):
                is_eos = any(t in eos_set for t in tid)
            elif isinstance(tid, str):
                piece = piece or tid
            else:
                pass
            if piece:
                yield piece
            if is_eos:
                return

    def _no_eos_processor(self, plen):
        """logits 处理器: 前 MIN_NEW_TOKENS 个 生成 token 禁掉 EOS."""
        eos_ids = self._eos_ids()
        if not eos_ids:
            return None
        try:
            import mlx.core as mx
        except Exception:
            return None
        eos_arr = mx.array(list(eos_ids))
        cache = {}

        def lp(tokens, logits):
            try:
                n = int(tokens.shape[-1])
            except Exception:
                return logits
            if plen is not None and n - plen >= self.MIN_NEW_TOKENS:
                return logits
            v = int(logits.shape[-1])
            m = cache.get("m")
            if m is None or m.shape[-1] != v:
                idx = mx.arange(v)
                m = mx.any(mx.equal(idx[:, None], eos_arr[None, :]), axis=1)
                cache["m"] = m
            neg = mx.full(logits.shape, mx.finfo(logits.dtype).min, logits.dtype)
            return mx.where(m, neg, logits)

        return lp

    def _iter(self, prompt, max_tokens, think_budget=DEFAULT_THINK_BUDGET, temp=0.6,
               top_p=0.9, top_k=0):
        """带 思考 预算 的 生成 (stream 与 generate 共用 同一 套 语义).

        think_budget > 0 : 先 给 思考 预算, </think> 出现 后 再 给 正文 max_tokens
        think_budget == 0: 单 阶段, 预算 全 给 正文
        temp == 0.0      : 贪心 解码 (摘要 这 类 需要 忠实 于 原文 的 任务 用)
        结束后 self.last_info = {"phase","truncated","tokens"}, api 层 据此 定 finish_reason.
        """
        self.load()
        max_tokens = max(1, int(max_tokens))
        think_budget = int(think_budget or 0)
        # 兜底 重试: 生成 出 的 是 空 的 -> 换 次 采样 再 来 (随机 采样, 第二次 基本 会 出 内容)
        for attempt in range(2):
            if think_budget <= 0:
                acc = ""
                for piece in self._raw_stream(prompt, max_tokens, temp=temp,
                                              top_p=top_p, top_k=top_k):
                    acc += piece
                    yield piece
                self.last_info = {"phase": "answer", "truncated": False, "tokens": len(acc)}
            else:
                total = think_budget + max_tokens
                acc = ""
                ntok = 0
                phase = "probe"
                phase_start = 0
                truncated = False
                for piece in self._raw_stream(prompt, total, temp=temp,
                                              top_p=top_p, top_k=top_k):
                    acc += piece
                    ntok += 1
                    yield piece
                    if "</think>" in acc:
                        if phase != "answer":
                            phase = "answer"
                            phase_start = ntok
                        continue
                    if phase == "probe":
                        if "<think>" in acc:
                            phase = "think"
                            phase_start = ntok
                        elif ntok >= self.PROBE_TOKENS:
                            # 模型 没 吐 <think> -> 当 它 不 思考, 后面 都 算 正文
                            phase = "answer"
                            phase_start = ntok
                    if phase == "think":
                        if ntok - phase_start >= think_budget:
                            truncated = True
                            break
                    elif phase == "answer":
                        if ntok - phase_start >= max_tokens:
                            truncated = True
                            break
                self.last_info = {"phase": phase, "truncated": truncated, "tokens": ntok}
            if acc.strip():
                return

    def generate(self, prompt, max_tokens=256, think_budget=DEFAULT_THINK_BUDGET, temp=0.6,
                 top_p=0.9, top_k=0):
        """非流式 生成: 返回 原始 文本 (含 <think> 标签), 由 api 层 拆 思考/正文."""
        return "".join(self._iter(prompt, max_tokens, think_budget, temp=temp,
                                  top_p=top_p, top_k=top_k))

    def stream(self, prompt, max_tokens=256, think_budget=DEFAULT_THINK_BUDGET, temp=0.6,
               top_p=0.9, top_k=0):
        """逐 片段 产出 (给 chat/completions 与 /v1/messages 的 SSE)."""
        for piece in self._iter(prompt, max_tokens, think_budget, temp=temp,
                                top_p=top_p, top_k=top_k):
            yield piece

    def chat_prompt(self, messages, tools=None, enable_thinking=True):
        """用 模型 自带 的 chat template 组 prompt (没有 template 时 回退 朴素 拼接)."""
        self.load()
        try:
            import json as _json
            msgs = []
            for m in messages or []:
                role = m.get("role", "user")
                content = m.get("content")
                if isinstance(content, list):
                    parts = []
                    for x in content:
                        if not isinstance(x, dict):
                            continue
                        if x.get("type") == "text":
                            parts.append(x.get("text", ""))
                        elif x.get("type") == "image_url":
                            parts.append("[image]")
                    content = " ".join(p for p in parts if p)
                msgs.append({"role": role, "content": str(content or "")})
            if tools:
                msgs = msgs + [{"role": "user",
                                "content": "可用工具(JSON): " + _json.dumps(tools, ensure_ascii=False)}]
            tpl = getattr(self.tokenizer, "apply_chat_template", None)
            if tpl is not None:
                out = tpl(msgs, tokenize=False, add_generation_prompt=True)
                if out:
                    return out
        except Exception as e:
            print("[llm_engine] chat template 失败 (%s), 回退 ChatML" % e,
                  file=sys.stderr, flush=True)
        # tokenizer 没 带 chat_template (我们 的 MLX 量化 包 常常 不带) 时:
        # Qwen 系 用 标准 ChatML, 别的 型号 仍 回退 朴素 拼接
        if self._is_chatml():
            out = []
            for m in msgs:
                out.append("<|im_start|>%s\n%s<|im_end|>\n" % (m.get("role", "user"),
                                                                 m.get("content", "")))
            # Qwen3 的 思考 标签 是 模型 自己 吐 的 (<think>...</think>).
            # 关 思考 的 官方 做法: 在 最后 一条 用户 消息 末尾 加 /no_think
            # (模型 会 吐 一个 空 的 <think></think>, 然后 直接 给 答案).
            if not enable_thinking:
                for i in range(len(out) - 1, -1, -1):
                    if "user\n" in out[i]:
                        out[i] = out[i].replace("<|im_end|>", "/no_think<|im_end|>")
                        break
            if not enable_thinking:
                # 官方 模板 的 关 思考 形态: 预 闭合 一个 空 的 思考 块.
                # 不 这样 做, 模型 会 吐 孤零零 的 </think> (没 配对 的 开 标签),
                # 正文 里 就 会 混 进 标签 碎片.
                out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            else:
                out.append("<|im_start|>assistant\n")
            return "".join(out)
        sys_txt = ""
        turns = []
        for m in msgs:
            if m.get("role") == "system":
                sys_txt += str(m.get("content", "")) + "\n"
            else:
                turns.append("%s: %s" % (m.get("role", "user"), m.get("content", "")))
        return (sys_txt.strip() + "\n" if sys_txt else "") + "\n".join(turns)

    def _is_chatml(self):
        """模型 是 不是 Qwen 系 (要 用 ChatML 模板)."""
        try:
            cfg = os.path.join(MODEL, "config.json")
            import json as _json
            c = _json.load(open(cfg))
            mt = str(c.get("model_type") or "")
            return mt.startswith("qwen")
        except Exception:
            return False

    def _eos_ids(self):
        """停止 token 集合: tokenizer 自带 的 eos + ChatML 的 <|im_end|>.

        注意: <think> / </think> 绝对 不能 当 EOS ——
        Qwen3 的 思考 块 结束 标记 后面 还有 正文, 当 EOS 会 让 回答 直接 变 空.
        """
        ids = []
        try:
            ids.extend(list(getattr(self.tokenizer, "eos_token_ids", None) or []))
        except Exception:
            pass
        for tokstr in ("<|im_end|>", "<|endoftext|>"):
            try:
                enc = self.tokenizer.encode(tokstr, add_special_tokens=False)
                if isinstance(enc, list):
                    ids.extend(enc)
                elif isinstance(enc, int):
                    ids.append(enc)
            except Exception:
                pass
        out = []
        for i in ids:
            if i not in out:
                out.append(i)
        return out

    def tokenize(self, text):
        """文本 -> token id 列表 (原版 /v1/tokenize)."""
        self.load()
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(enc, dict):
            enc = enc.get("input_ids", [])
        return list(enc)

    def detokenize(self, ids):
        """token id 列表 -> 文本 (原版 /v1/detokenize)."""
        self.load()
        return self.tokenizer.decode(list(ids))
