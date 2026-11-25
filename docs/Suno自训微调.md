# 用 Suno 公开曲库微调本地 ACE-Step —— 完整操作手册

> 起因：suno.cn 的**权重扒不下来**是事实（协议扒干净了，但模型权重是闭源的）。
> 那就换个思路 —— **用行为规格 + 公开曲库自己训一个**。
>
> 全程跑在本机（M5 Max / 128G / MPS），不烧积分、不联网下载模型、不需要 suno 账号。

---

## 0. 一句话结论（先看这个）

自训**确实有用**（分布级相似度 FAD 降约 30%，是噪声地板的 4–10 倍，在没训练过的
留出曲库上同样成立），但**没学到「条件 → 风格」**——学到的是曲库平均音色。
根因是**语料**（caption 太离散，87% 只出现一次），不是训练方法。

产物权重已注册进 MLX Studio，界面直接选：

| 引擎 id | 说明 |
|---|---|
| `ours/ace-step-suno-mlx` | epoch 1（**推荐**：与 epoch 2 统计上等价，省一半训练时间）→ 权重 `acestep-suno-e1` |
| `ours/ace-step-suno-e2-mlx` | epoch 2 对照 → 权重 `acestep-suno-e2` |

---

## 1. 工具在哪

脚本全在 `实现代码/suno_train/`（随「suno 逆向」那套工具一起，本仓库不收脚本，
只收这份说明 + 注册好的引擎）。

> 本仓库只包含**引擎注册**（`music_mlx.py` 里的 `ours/ace-step-suno-mlx`），
> **不包含**训练脚本，也**不包含**训练出来的权重 —— 权重是本机产物，
> 各人按下面流程自己训。

| 脚本 | 作用 |
|---|---|
| `prepare_dataset.py` | 曲库 → 训练语料（下载音频 + 规整歌词 + VAE/DiT 预编码成 `.pt`） |
| `make_holdout.py` | 切留出集（评测只看没见过的歌，测泛化不测记忆） |
| `train_lora.py` | 训 LoRA（本机 MPS） |
| `merge_lora.py` | LoRA 合并回基座 → 完整权重目录 |
| `gen_song.py` | 出歌（指定 DiT 权重） |
| `eval_lora.py` | A/B 评测（逐首 9 项音频特征） |
| `fad_eval.py` / `fad_floor.py` / `fad_vs_corpus.py` / `fad_pairs.py` | FAD 分布级评测 + 噪声地板标定 |
| `eval_distribution.py` | 分布 z-gap |
| `corpus_caption_acoustics.py` | 语料 caption 的声学属性分析（第二轮方案的基础） |
| `local_checkpoints.py` | 把上游写死的 checkpoint 预检改成本机实际情况 |
| `probe_init.py` | 初始化探针 |
| `run_all.sh` | 切留出集 → 训 → 合并 → 评测，一条龙 |

依赖：上游 ACE-Step 训练栈（非量化 torch DiT + PEFT LoRA + Lightning Fabric），
引擎代码在 `~/.ours/ace-step-engine/`，专用 venv `~/.venvs/ace/bin/python`。

---

## 2. 全链路六步

```bash
cd 实现代码/suno_train
PY=~/.venvs/ace/bin/python

# 1. 曲库 → 训练语料（下载音频 + 规整歌词 + VAE/DiT 预编码成 .pt）
$PY prepare_dataset.py --count 1456 --min-dur 30 --max-dur 240 --no-dedup --jobs 10

# 2. 留出 100 首不参与训练（评测只看没见过的歌，测泛化不测记忆）
$PY make_holdout.py --n 100

# 3. 训 LoRA（本机 MPS）
$PY train_lora.py --tensors ~/.ours/train-suno/tensors \
                   --output ~/.ours/train-suno/lora \
                   --rank 16 --alpha 32 --lr 2e-4 --epochs 2 --bf16

# 4. 合并回基座（MLX 运行时不支持热插 LoRA，必须整份换权重）
#    epoch 1 / epoch 2 分开出两份，studio 里分别注册成两个引擎
$PY merge_lora.py --adapter ~/.ours/train-suno/lora/checkpoints/epoch_1_loss_0.2606/adapter \
                   --out ~/.ours/models/ace-step-mlx/acestep-suno-e1
$PY merge_lora.py --adapter ~/.ours/train-suno/lora/checkpoints/epoch_2_loss_0.2312/adapter \
                   --out ~/.ours/models/ace-step-mlx/acestep-suno-e2

# 5. 出歌（新引擎 ours/ace-step-suno-mlx）
$PY gen_song.py --dit acestep-suno --prompt "city pop" --duration 30 --seed 777 --out /tmp/a.wav

# 6. A/B 评测（同一批歌、同一种子，跟 Suno 原曲比 9 项特征）
$PY eval_lora.py --n 8 --models base=acestep-v15-turbo suno=acestep-suno
```

一条龙版本：`./run_all.sh 2 16 32 2e-4`（参数依次是 epochs / rank / alpha / lr）。

### 各步产物

| 步 | 产物路径 | 实测 |
|---|---|---|
| 1 | `~/.ours/train-suno/tensors/*.pt` + `manifest.json` | 1356 首（切留出后） |
| 2 | `~/.ours/train-suno/tensors-holdout/` + `留出集.jsonl` | 100 首 |
| 3 | `~/.ours/train-suno/lora/final/adapter/`（PEFT safetensors） | e1 loss 0.2606 / e2 loss 0.2312 |
| 4 | `~/.ours/models/ace-step-mlx/acestep-suno-e1/` + `-e2/`（各 4.5G 完整权重） | 192 模块已合并 |
| 6 | `~/.ours/train-suno/ab/评测结果.json` | 见第 6 节 |

---

## 3. 关键参数

### 训练（`train_lora.py`）

| 参数 | 本次取值 | 说明 |
|---|---|---|
| `--rank` / `--alpha` | 16 / 32 | scale = alpha/r = 2.0 |
| `--lr` | 2e-4 | |
| `--epochs` | 2 | |
| `--batch-size` | 1（写死） | |
| `--accum` | 4（默认） | 梯度累积 |
| `--dropout` | 0.1（默认） | |
| `--seed` | 42（默认） | |
| `--device` | mps | |
| `--bf16` | 开 | **必须开**，理由见第 4 节 |
| `--save-every` | 1 | 每轮存一个 checkpoint |

### 数据预处理（`prepare_dataset.py`）

语料链路：

```
曲库索引（suno公开曲库.jsonl：prompt / tags / 音频直链）
  → 下载音频
  → 规范成 ACE-Step 的 AudioSample
      caption = 风格 tags（模型要学的"取向"）
      lyrics  = 结构化歌词（【主歌】→ [Verse] 这种英文标记）
  → 上游 DatasetBuilder.preprocess_to_tensors()
      （VAE encode + 文本编码 + DiT encoder）
  → 每首一个 .pt + manifest.json
```

预处理完的 `.pt` 是**训练直接吃**的，不用再跑模型 —— 这是 107 分钟那段的全部意义。

---

## 4. 踩到并修掉的七个真问题

都在**引擎/上游**层解决，没在脚本里绕过去。

### 4.1 checkpoint 预检认错了 LM

**现象**：每次初始化都去 HuggingFace 白下 3.5GB。
`MAIN_MODEL_COMPONENTS` 写死 `acestep-5Hz-lm-1.7B`，本机只有 `0.6B`。

**修法**：`local_checkpoints.py` 按本机实际 LM 改写预检项 —— 不改引擎代码，也不造假目录。

### 4.2 seed 传了但不管用

**现象**：同权重同种子跑两次，音频相关性只有 0.03。
根因是 5Hz LM 写 caption 时是**全局流随机采样**，caption 一变歌就变。

**修法**：引擎加 `seed` 选项，并在 LM 调用前给 `mx.random` / `torch` 播种；
DiT 噪声走 `mx.random.key(seed)` 独立 PRNG，不受影响。

### 4.3 引擎写死 DiT 权重名

**现象**：`DIT_MODEL = "acestep-v15-turbo"` 写死，没法挂第二份权重。

**修法**：改读 `ACE_DIT_MODEL` 环境变量；`music_mlx.py` 的 `ENGINES` 注册
`ours/ace-step-suno-mlx`（`ACE_DIT_MODEL=acestep-suno-e1`）。

### 4.4 fp16 在 MPS 上溢出

**现象**：**17% 的 batch 前向 loss 是 NaN**，那些步直接被跳过。
上游 MPS 精度写死 `16-mixed`（fp16 动态范围只有 ±65504）。

**修法**：加环境变量开关 `ACESTEP_MPS_BF16=1` → `bf16-mixed`。
实测 **20 步零 NaN**，且快 27%。

### 4.5 会话一断训练就没

**现象**：跑到 step 50 进程静默消失（不是 OOM，是会话终止时进程组被杀）。

**修法**：改用 launchd 托管 —— `com.ours.suno-train`，
启动脚本 `~/.ours/train-suno/run_train.sh`，会话断了照样跑。

### 4.6 manifest 会让训练静默少样本

**现象**：`PreprocessedTensorDataset` 只读 `manifest.json`、不扫目录。
切走留出集后清单还是旧的 → 训练样本数比目录数少，且**不报错**。

**修法**：`make_holdout.py` 移文件后同步改写 manifest；
切完必须核对 `manifest 数 == 目录数 == 训练集数`。

### 4.7 某些 mp3 soundfile 解不了

**现象**：`LibsndfileError: Internal psf_fseek() failed`（文件是好的，ffmpeg 能解）。

**修法**：音频加载加 ffmpeg 兜底。补跑后 **1456/1456** 全成功。

> 附带：`torchaudio 2.11` 的 load 依赖 `torchcodec`（本机没装）→ 直接替换
> `torchaudio.load` 为 soundfile，不用改上游代码。

---

## 5. 数值校验（合并可信度）

### 5.1 合并正确性

LoRA 键名 `base_model.model.<mod>.lora_A/B`，基座是 `decoder.<mod>.weight`，
映射后 **192 个模块全部对上**（24 层 × self/cross × q/k/v/o）。

合并公式 `W' = W + (alpha/r)·B@A`，**全程 fp32 计算再回写 BF16**。

合并后逐张量比对：**恰好 192 个张量变化，其余 485 个位级一致**。

### 5.2 「空效应」标定（重要）

只训了 8 样本 × 2 epoch 的弱 LoRA，Δ 只有权重的 **0.03%**，
同种子下与基座的音频相关系数是 **0.9995**。

> 这说明这条链路对权重变化是**测得出来的** —— 后面自训效果能落到多少分是可信的，
> 不是「换个权重听不出区别」这种自欺。

### 5.3 合并产物

`~/.ours/models/ace-step-mlx/acestep-suno-e1/`（4.5G）：

```
config.json
configuration_acestep_v15.py
merge_info.json          ← base / adapter / rank / alpha / scale / merged_modules
model.safetensors        ← 合并后的完整权重
modeling_acestep_v15_turbo.py
silence_latent.pt
```

### 5.4 本机实际有哪几份权重（别搞混）

`~/.ours/models/ace-step-mlx/` 下和自训相关的目录：

| 目录 | 来源 | rank/alpha | scale | 说明 |
|---|---|---|---|---|
| `acestep-v15-turbo` | 基座（未微调） | — | — | 对照组 |
| `acestep-suno-e1` | epoch 1 checkpoint | 16/32 | 2.0 | **studio 里 `ours/ace-step-suno-mlx` 用的就是这份** |
| `acestep-suno-e2` | epoch 2 checkpoint | 16/32 | 2.0 | studio 里 `ours/ace-step-suno-e2-mlx` 用的就是这份 |
| `acestep-suno-s10` | epoch 1 | 16/32 | **1.0** | 强度减半档（`merge_lora.py --scale 1.0`） |
| `acestep-suno-s05` | epoch 1 | 16/32 | **0.5** | 强度四分之一档 |
| `acestep-suno-s025` | epoch 1 | 16/32 | **0.25** | 强度八分之一档 |
| `acestep-suno` | **`lora-smoke`**（8 样本） | 8/16 | 2.0 | ⚠️ **不是**正式训练结果，是 5.2 节「空效应」标定那一份 |

> ⚠️ 坑：`acestep-suno` 这个名字看着像"正式的"，实际是 8 样本 smoke run 的产物
> （Δ 权重 0.03%，与基座音频相关 0.9995）。**别拿它出歌**。
> 真正挂进 studio 的是 `acestep-suno-e1`。

---

## 6. 第一轮结果（2026-11-24 实测，两批独立样本）

语料 1456 首公开曲库 → 留出 100 首 → 训练 1356 首
（r=16 / alpha=32 / lr=2e-4 / 2 epoch / bf16，**10708 秒**训完，零 NaN）。
评测在**留出集**上，同种子同歌词同 caption。

| 指标 | base | e1（epoch 1） | e2（epoch 2） | 噪声地板 |
|---|---|---|---|---|
| FAD vs 留出曲库（100 首真歌） | 86.56 | **61.13** | **60.60** | 6.50 |
| FAD vs 训练曲库（1356 首真歌） | 84.96 | **62.82** | **62.79** | 7.12 |
| 逐首相似度（跟该首原曲比 9 项特征） | 35.85 | 37.02 | 35.89 | p=0.83 / 0.93（无差异） |
| 分布 z-gap（越小越像） | 0.5196 | 0.4988 | **0.4857** | — |

### 结论

1. **自训是真的有用**：分布级相似度（FAD）降了约 30%，是噪声地板的 4–10 倍，
   而且在**没训练过**的留出曲库上同样成立 → 不是背训练集，是音色整体被拉向
   Suno 曲库那一侧。
2. **但没学到「条件 → 风格」**：逐首相似度和基座**没有统计差异**
   （配对置换检验 p=0.83 / 0.93），歌与歌之间的差异反而被压平
   （dyn_range 标准差 3.12 → 2.27，真曲库 3.39）。学到的是**曲库平均音色**。
3. **第 2 轮等于没训**：FAD(e1, e2) = 0.69（低于地板），loss 从 step 100 就平台在 0.23。
4. **根因是语料不是训练**：1356 首歌对应 **1106 个 caption 唯一值**，
   **87% 的 caption 只出现一次**；风格首句分组 795 组、**中位 1 首**。
   组内声学距离比随机近 27%（风格确实带信号），但只有 28.8% 的歌落在
   ≥2 首的组里 —— 学不到可复用的条件规律。

---

## 7. 为什么第一轮会这样（对照实验）

在此之前做过一轮 **A/B 对照**（`实现代码/suno_ab.py`，报告
`实现代码/库/AB对照/对照报告.md`）：8 首公开曲库随机抽的歌，固定 seed，
本地出三版（A 纯提示词 / B 带参考音 / C 参考音 + 显式 BPM），9 项音频特征。

| 版本 | 相似度均值 | 标准差 |
|---|---|---|
| A 纯提示词 | 36.9 | 12.6 |
| B 带参考音 | 37.9 | 11.1 |
| C 参考音+BPM | 36.9 | 9.6 |

- **对得上的**：时长（按规格对齐）、响度（偏差 7–9%）、节奏密度（16–19%）、谱质心（22–29%）
- **对不上的**：低频占比（差 36–41%）、高频占比（差 42–62%）、峰均比（差 48–73%）
- **参考音频没有整体提升**（37.9 vs 36.9，差异在标准差里）—— 上一轮看着像「+6.9 分」
  是种子噪声，固定种子后不成立
- **显式 BPM 单项有效**：BPM 偏差 25% → 6%
- **母带对齐**（`suno_match.py`，网格搜低架/高架/makeup/限幅）：均值 36.9 → **39.3**（+2.3），
  8 首里 4 首最优是「别动」

> **结论：瓶颈不在提示词，也不在后处理，在生成模型本身。**
> 本地能复刻结构 / 时长 / 节奏，复刻不了音色与编曲取向 —— 这正是「Suno 权重扒不下来」
> 的实际后果。于是才走到「自己训」这条路。

> 那份分数是低维音频特征相似度，**不等于听感**；旋律、唱腔、和声这些高维差异，
> 低维特征看不见。

---

## 8. 第二轮该改什么

**改 caption，不用改训练方法。**

把条件从离散标签换成 **9 个连续声学属性**：
BPM / 响度 / 动态范围 / 谱质心 / 低频占比 / 高频占比 / 起音密度 …

分析脚本已经写好：`corpus_caption_acoustics.py`。

代价：重跑预处理 **107 分钟** + 训练 **175 分钟**。

---

## 9. 已修：出歌路径丢 seed

studio 出歌路径原本会**丢掉 seed**（每次结果都不一样）。
修完实测同 seed 两次**逐样本完全一致**（相关 1.00000000，最大差 0.000000），
换 seed 才是另一首。

---

## 10. 实测耗时（本机 M5 Max / 128G）

| 阶段 | 耗时 |
|---|---|
| 下载 1456 首音频 | **114 秒** |
| 预编码（VAE + 文本 + DiT encoder） | **107 分钟** |
| 训练（3.0 秒/样本 × 1356 样本 × 2 轮） | **≈ 2.3 小时**（10708 秒） |
| 合并回基座 | 数分钟 |

---

## 11. 复现注意事项

1. **`--bf16` 必须开** —— 否则 MPS 上 17% 的 step 是 NaN，会被静默跳过。
2. **训练必须 launchd 托管** —— 终端会话一断进程组就被杀。
3. **切完留出集要核对 manifest** —— `manifest 数 == 目录数 == 训练集数`。
4. **别用评测集当验证集** —— 留出集存在的意义就是测泛化。
5. **合并后逐张量比对** —— 应该是 192 个变、485 个位级不变。
6. **第 2 轮意义不大** —— FAD(e1, e2) = 0.69 低于噪声地板，默认只出 e1 引擎。

---

## 12. 相关文件

| 路径 | 内容 |
|---|---|
| `实现代码/suno_train/` | 全部训练脚本 |
| `实现代码/suno_ab.py` | A/B 对照（提示词 / 参考音 / BPM） |
| `实现代码/suno_match.py` | 母带对齐（网格搜低架/高架/makeup/限幅） |
| `实现代码/库/AB对照/对照报告.md` | 对照报告 |
| `实现代码/库/suno公开曲库.jsonl` | 训练语料索引（prompt / tags / 音频直链） |
| `~/.ours/train-suno/` | 工作区（tensors / lora / 留出集 / 日志 / ab 评测结果） |
| `~/.ours/models/ace-step-mlx/acestep-suno-e1/` | epoch 1 合并权重（studio 在用） |
| `~/.ours/models/ace-step-mlx/acestep-suno-e2/` | epoch 2 合并权重（对照组） |
| `~/.ours/models/ace-step-mlx/acestep-suno-s10/s05/s025/` | 同 epoch 1，scale 1.0 / 0.5 / 0.25 强度档 |
| `~/Library/LaunchAgents/com.ours.suno-train.plist` | 训练守护 |
| `mlx-studio` → `music_mlx.py` | `ours/ace-step-suno-mlx` 引擎注册 |
