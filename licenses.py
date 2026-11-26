"""许可 条款 清单 —— 谁 能 商用, 门槛 在 哪, 原文 在 哪.

模型页 的 许可 徽章 只是 个 标签; 真正 的 条款 (营收门槛 / 地区限制 / 署名 要求 /
禁止 用途) 集中 在 这里 —— 一处 改, 界面 + 文档 + MCP 都 跟着 变.

分 三 类:
  free       自由许可 —— Apache-2.0 / MIT. 商用 没 门槛.
  community  社区许可 —— 能 商用, 但 有 门槛 / 限制 / 署名 要求. **商用 前 要 看.**
  builtin    本项目 自带 的 能力 (不需要 权重).

注意: 这里 记 的 是 **权重** 的 许可, 跟 我们 自己 代码 的 许可 (Apache-2.0) 是 两码事.
"""

from __future__ import annotations

import os

# 协议 原文 候选 路径 —— 相对 "运行时 根目录" (装完 是 ~/.ours/mlx_server)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [
    _HERE,
    os.path.join(_HERE, "vendor"),
    os.path.expanduser("~/.ours/mlx_server"),
    os.path.expanduser("~/.ours/mlx_server/vendor"),
    os.path.expanduser("~/.ours/vendor"),
]


def _find(rel: str):
    """在 几个 可能 的 根目录 里 找 协议 原文. 找不到 返回 None."""
    for root in _ROOTS:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------- 条款 表
def _d(**kw):
    kw.setdefault("kind", "community")
    kw.setdefault("file", None)
    kw.setdefault("terms", [])
    kw.setdefault("threshold", None)
    kw.setdefault("territory", None)
    kw.setdefault("display", None)
    kw.setdefault("url", None)
    kw.setdefault("aliases", [])
    return kw


DOCS = [
    _d(
        key="project",
        title="MLX Studio 框架代码",
        spdx="Apache-2.0",
        kind="free",
        file="LICENSE",
        url="https://github.com/phoenix7956/mlx-studio/blob/main/LICENSE",
        terms=[
            "允许商业使用：使用 / 复制 / 修改 / 合并 / 发布 / 分发 / 再授权 / 销售，商用无需额外授权或付费。",
            "比 MIT 多了**明确的专利授权**（§3），覆盖贡献者的贡献部分。",
            "唯一要求：保留版权声明 + 许可原文（§4），并随分发附带 NOTICE。",
            "商标不在授权范围内：「MLX Studio」名称与 Logo 归项目方所有。",
        ],
        aliases=["内置 (本项目 MIT)", "内置"],
    ),
    _d(
        key="minimax-h3",
        title="MiniMax H3（视频生成权重）",
        spdx="other",
        file="vendor/minimax-h3_mlx_src/MINIMAX-H3-COMMUNITY-LICENSE.txt",
        url="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
        threshold="年营收 > 2000 万美元（commercial products and services generate more than "
                  "20 million US dollars in yearly revenue）时，必须**事先**取得 MiniMax 书面授权",
        territory="适用地域 = 全球，但**排除 欧盟 / 英国 / 韩国 / 美国**。这四个地区要部署，"
                  "需另行联系 MiniMax 取得授权（§II 末段）",
        display="商业产品或服务的**界面**上必须显著显示「MiniMax H3」（§IV.2）",
        terms=[
            "**分发权重 / 衍生品给第三方时**：必须附协议副本（§III.1）、修改文件要带显著修改声明（§III.2）、"
            "必须带 NOTICE 文件写着 “MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
            "Copyright © 2026 MiniMax. All Rights Reserved.”（§III.4）。",
            "**鼓励**：显示「Powered by MiniMax H3」、给产物加 AI 生成标识、发一篇技术博客或公开说明（§III.3）。",
            "**禁止**：用 H3 或其输出 / 结果去改进任何**其它** AI 模型（§V.3）。",
            "**禁止**：在适用地域之外使用 / 复制 / 修改 / 分发 / 展示（§V.4）。",
            "提供给别人用的服务，必须用可执行的条款把使用限制绑定给每个用户，并告知其适用（§V.2）。",
            "对外提供服务还要：维护举报渠道、调查违规、必要时停用或封禁（§V.5）。",
            "输出（Outputs）的权利归使用者，MiniMax 不主张权利（§VI.4）。",
            "**免除担保**：模型与输出「AS IS」提供，无任何担保（§VII.2）。",
            "适用**香港法**，争议由**香港法院**专属管辖（§IX）。",
        ],
        aliases=["MiniMax H3 社区许可", "minimax-h3-community"],
    ),
    _d(
        key="krea-2",
        title="Krea-2-Turbo（图像生成权重）",
        spdx="other",
        file="vendor/krea2_engine_src/LICENSE",
        url="https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE",
        threshold="**公司整体**（含同一控制下的关联实体）年营收 < 100 万美元（TTM，含全部收入来源）"
                  "才可按本协议商用；达到或超过门槛 → 立即停止商用并联系 Krea 走企业授权",
        territory="全球（无地域排除）",
        display="分发时：模型名前必须加「Krea」前缀（如 “Krea 2 [你的模型名]”），"
                "并保留 Notice 文件里的归属文本（§3.1）",
        terms=[
            "**必须实现内容过滤**（§4.2）：开源 NSFW 分类器 / 商业审核 API / 人工复核 至少其一 —— "
            "我们引擎里的 safety.py 就是这个，默认开着。",
            "分发模型或衍生品时：附协议副本 + 要求每个受方受本协议约束 + 模型名加「Krea」前缀 + 保留归属声明（§3.1）。",
            "分发你的衍生品时：归属声明里要写明你改过；不能声称是 Krea 官方产品（§3.2）。",
            "**不得移除或遮蔽**任何版权声明 / 许可条款 / 归属声明（§3.3）。",
            "Krea 可**提前 30 天书面通知**终止协议（§9.2）。",
            "商标：只能用「Krea」按 §3 的方式使用，没有别的授权（§5.4）。",
        ],
        aliases=["Krea 2 社区许可", "krea2-community"],
    ),
    _d(
        key="minimax-music3",
        title="MiniMax-Music3（音乐权重）",
        spdx="other",
        url="https://huggingface.co/MiniMax/MiniMax-Music3",
        threshold="社区许可，MIT 式条款 + 商业条款（具体门槛以 MiniMax 官方协议为准）",
        terms=[
            "商用前请读 MiniMax 官方协议原文，条款与 H3 类似但独立成文。",
        ],
        aliases=["MiniMax-Music3 社区许可 (MIT 式 + 商业条款)", "MiniMax-Music3 社区许可"],
    ),
    _d(
        key="gemma-3",
        title="Gemma 3",
        spdx="other",
        url="https://ai.google.dev/gemma/terms",
        threshold="Gemma Terms of Use（非 OSI 许可，有使用政策限制）",
        terms=[
            "按 Google 的 Gemma 使用条款，不是标准开源许可。",
        ],
        aliases=["Gemma 条款"],
    ),
    _d(
        key="llama-3-1",
        title="Llama 3.1",
        spdx="other",
        url="https://www.llama.com/llama3_1/license/",
        threshold="Llama Community License（700M 月活以下有额外条款豁免；大厂另有专门授权）",
        terms=[
            "按 Meta 的 Llama Community License，不是标准开源许可。",
        ],
        aliases=["Llama 3.1 社区许可"],
    ),
    _d(
        key="bilibili",
        title="bilibili 模型使用协议",
        spdx="other",
        terms=["按 bilibili 官方模型使用协议，条款未公开完整，需自行确认。"],
        aliases=["bilibili 模型使用协议"],
    ),
    _d(
        key="ace-step",
        title="ACE-Step 1.5（出歌引擎 + 权重）",
        spdx="MIT",
        kind="free",
        file="ace-step-engine/LICENSE",
        url="https://github.com/ACEStep/ACE-Step",
        terms=["MIT 许可，商用无门槛，保留版权声明即可。"],
        aliases=["MIT"],
    ),
    _d(
        key="free-weights",
        title="自由许可权重（Apache-2.0 / MIT）",
        spdx="Apache-2.0",
        kind="free",
        terms=[
            "商用没门槛，可自由使用 / 修改 / 分发 / 销售。",
            "覆盖：Qwen3 系列（LLM / Embedding / TTS）、Z-Image-Turbo、MOSS-Image-SFT、"
            "Wan2.2 TI2V-5B / FastWan2.2、VoxCPM2、CosyVoice 2、OmniVoice（MLX 量化）。",
            "MIT 的那些：ACE-Step 1.5、Whisper（faster-whisper）、Chatterbox TTS。",
            "⚠️ IndexTTS-2 **不算** 自由许可 —— 量化 不 改变 上游 权重 的 协议，"
            "它是 bilibili 模型使用协议。",
            "唯一要求：保留版权声明与许可原文。",
        ],
    ),
    _d(
        key="mlx",
        title="Apple MLX 框架",
        spdx="MIT",
        kind="free",
        url="https://github.com/ml-explore/mlx",
        terms=["MIT 许可，商用无门槛。运行时依赖，不随本仓库分发。"],
    ),
]

_BY_KEY = {d["key"]: d for d in DOCS}


def key_for(entry: dict):
    """给 一个 模型行 / 目录条目 找 对应 的 许可 key. 找不到 返回 None."""
    if not isinstance(entry, dict):
        return None
    lic = (entry.get("license") or "").strip()
    spdx = (entry.get("license_spdx") or "").strip().lower()
    url = (entry.get("license_url") or "").strip()
    if lic:
        for d in DOCS:
            if lic in d["aliases"] or lic == d["title"]:
                return d["key"]
    if spdx in ("apache-2.0", "mit"):
        return "free-weights"
    if "minimax" in url.lower() and "h3" in url.lower():
        return "minimax-h3"
    if "krea" in url.lower():
        return "krea-2"
    if "gemma" in url.lower():
        return "gemma-3"
    if "llama" in url.lower():
        return "llama-3-1"
    return None


def path_of(key: str):
    """协议 原文 的 绝对路径. 找不到 返回 None."""
    d = _BY_KEY.get(key)
    if not d or not d.get("file"):
        return None
    return _find(d["file"])


def read(key: str, limit: int = 400000):
    """协议 原文 (文本). 找不到 返回 ''."""
    p = path_of(key)
    if not p:
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except Exception:
        return ""


def list_docs():
    """给 /v1/licenses 用 —— 全部 条款 条目 + 原文 路径."""
    out = []
    for d in DOCS:
        p = path_of(d["key"])
        out.append({
            "key": d["key"],
            "title": d["title"],
            "spdx": d["spdx"],
            "kind": d["kind"],
            "has_text": bool(p),
            "path": p or "",
            "url": d.get("url") or "",
            "terms": d["terms"],
            "threshold": d.get("threshold"),
            "territory": d.get("territory"),
            "display": d.get("display"),
            "aliases": d.get("aliases") or [],
        })
    return out


def summary(key: str):
    """给 模型行 徽章 tooltip 用 —— 一句话 说清 能不能 商用."""
    d = _BY_KEY.get(key)
    if not d:
        return ""
    if d["kind"] == "free":
        return "%s —— 自由许可，商用没门槛。" % d["title"]
    bits = ["%s —— 社区许可，商用有附加条款。" % d["title"]]
    if d.get("threshold"):
        bits.append("门槛：%s" % d["threshold"])
    if d.get("territory"):
        bits.append("地域：%s" % d["territory"])
    if d.get("display"):
        bits.append("署名：%s" % d["display"])
    return " ".join(bits)


if __name__ == "__main__":
    import json
    print(json.dumps(list_docs(), ensure_ascii=False, indent=2))
