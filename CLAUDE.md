# CLAUDE.md — FinVCup 2026 Turn-Taking Competition

> Project-specific facts only. Global rules at `~/.claude/CLAUDE.md` apply.
> Lightweight-memory rules at `.claude/rules/lightweight-memory.md` (symlink to `~/.claude/shared-rules/lightweight-memory.md`) auto-load every session.

## Fresh clone setup

`.claude/` is gitignored, so the symlink doesn't survive clone:

```bash
mkdir -p .claude/rules
ln -s ~/.claude/shared-rules/lightweight-memory.md .claude/rules/lightweight-memory.md
```

程序定位等请优先使用 codegraph。

## Task (第十一届信也科技杯 — 对话轮次交互建模)

给定过去 30s（音频 + ASR 文本 + 历史标签序列），预测**未来 2s（25×80ms chunk）内** 5 类事件 `C/T/BC/I/NA` 各自是否出现。Event-level 多标签（sigmoid + BCE），输出 5 列 0/1。

- **Metric**: Macro-F1（5 类 F1 等权平均，sklearn `f1_score`，提交是硬 0/1）
- **因果约束**: 只用过去信息，不读未来音频/标签
- **目标**: 前 3（线上 ≥0.7357），保底前 10（≥0.7192）。榜首极密集，#1→#10 仅差 0.028
- 完整决策契约见 `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md`

### 数据布局

| 路径 | 内容 |
|---|---|
| `data/train/audio/<id>.wav` | 整通对话，8kHz 双声道 int16，中位 ~25min |
| `data/train/text/<id>.json` | ASR：`utterances[{channel_id,start_ms,end_ms,text}]` |
| `data/train/labels/<id>.npy` | 逐 chunk 标签 0~4 = C/T/BC/I/NA |
| `data/test/audio/<id>.wav` | 30s 切片（1000 段） |
| `data/test/text/<id>.json` | ASR + `start_ms/end_ms` |
| `data/test/context/<id>.npy` | 历史 context 标签（恒 375 chunk） |

- **类分布（chunk 级）**: C=64.2% / NA=30.1% / I=4.0% / T=1.2% / BC=0.5% — 稀有类（BC/T/I）是 Macro-F1 胜负手
- `data/` 是 gitignored；`baselines/2026_finvcup_baseline/` 是官方 baseline（自带 .git，已 gitignore）

### 提交格式

`pred_test1.csv`：`segment_id,c,na,i,bc,t`（列名小写，值 0/1，顺序同 config `labels.multi_targets`）。
用户手动提交后贴回 Macro-F1 真分校准本地 CV。

### 阈值铁律（HARD-WON，2026-05-27 实测代价 −0.027）

**不要在滑窗 CV 上做激进的逐类阈值搜索。** 实测：v2 在滑窗 5-fold OOF 上把 T 阈值调到 0.64（CV 0.5921 略高），线上反而 0.6833 < cycle1 的 0.7108（cycle1 阈值更接近 0.5）。

根因：**滑窗 CV 分布 ≠ test 独立 30s 切片分布**（gap 稳定 +0.10 是铁证）。在错配分布上调出的极端阈值，搬到 test 系统性变差，越偏离 0.5 越伤。

正确做法：①阈值接近 0.5 / 只轻调；②或用**模拟 test 的 30s 切片化验证集**调阈值（见 CONTEXT Decision 4）；③**per-class-aware，不一刀切**：C 类 94% 恒正，安全于低阈值（~0.05），用 [0.35,0.65] floor 反而砍崩 C（974→348 正例，实测 −0.05）。只有 T/I/BC 这些中低频类怕激进阈值。**激进逐类阈值搜索 = 在错配分布上过拟合。**

### BC 诊断链（2026-05-27，3 cycle 实测排除便宜路线）

BC（backchannel）是 Macro-F1 瓶颈类。手工特征 + LGBM 对 BC 彻底到顶：
- context 标签 → BC F1 **0.217**
- + 廉价声学特征（energy/zcr/voicing/双声道对比）→ **0.219**（无效）
- + ASR 词汇统计（BC词频/距/通道）→ **0.201**（无效甚至略降）

**结论：BC 真需神经编码器**——Qwen 语义（理解"对方会不会插话"的时机）或 SSL 音频（细粒度韵律/onset）。词袋/粗声学抓不到"未来 2s 会不会 BC"的预测问题。
**意外收获**：文本词汇特征帮了 **T（0.54→0.58）和 I（0.44→0.49）**——文本该救 T/I，不是 BC。

### 架构铁律（2026-05-27，杀掉 Qwen3 提取后确认）

**稠密 neural embedding 不要喂 LGBM/树模型。** 实测：Qwen3-0.6B mean-pool 1024d + context 喂 LGBM = macro 0.583→0.575（全面略降）。原因：①树逐特征切分，1024d 稠密信号被稀释 ②mean-pool 丢时序。
**3-strike 诊断**：context-v2 / 廉价声学 / 词汇 / Qwen3-embed 全喂 LGBM 都撞 ~0.71 墙——**瓶颈在"喂 LGBM"这个架构，不在特征本身。** 要突破必须换架构：**embedding 喂神经小头（可微调 + 时序建模，如 transformer over 序列），不是树。** 词袋稀疏特征是 LGBM 的例外（可解释、低维）。

### 模型使用边界（HARD RULE）

- **模型总参数量 ≤ 8B**（硬上限，同时约束复赛镜像推理成本）
- **优先白名单 Qwen 系列**：Qwen2.5-Omni-3B/7B、Qwen3-4B/0.6B/0.8B（见赛题要求.md）
- 非 Qwen 公开模型（WavLM/wav2vec2/HuBERT 等）**仅在能带明显增益时才用**，需 EDA/小实验佐证

### 合规（重要）

- 用公开数据/模型需 **2026-06-10 前**向 `xinyebei@xinye.com` 报备
- **禁止**用公榜数据训练/标注/打伪标签
- 增强数据 + 生成代码须随复赛镜像（CUDA Dockerfile）一并提交
- 比赛报备/联系邮箱是对外身份 → **问用户**，不要用系统 userEmail

## 算力分工

- **本机 MBP M3 Max 128GB（MPS，PyTorch 2.7.1 + torchaudio 2.7.1，conda env `deep-research`）**: 主力——冻结编码器+特征缓存训小头
- 复赛镜像必须在 **CUDA** 上最终验证

### 模型下载 workaround（HARD-WON 2026-05-27）

`huggingface_hub` client (v1.13) 连 hf-mirror.com / huggingface.co **都 HEAD 失败**（`FileMetadataError: Distant resource does not seem to be on huggingface.co`），但 **curl 正常**（HTTP 200）。解法：

```bash
# 列文件
curl -sL "https://hf-mirror.com/api/models/<org>/<model>" | python -c "import sys,json;[print(f['rfilename']) for f in json.load(sys.stdin)['siblings']]"
# curl 直下到本地目录，用本地路径加载（绕过 hf client）
DEST=~/.cache/manual_models/<model>; mkdir -p $DEST
curl -sL -o $DEST/config.json "https://hf-mirror.com/<org>/<model>/resolve/main/config.json"  # 等
AutoModel.from_pretrained("~/.cache/manual_models/<model>")
```

ModelScope 被墙（HTTP 000）；HF/hf-mirror 可达。

### MPS 特征提取速度（实测，决定哪些编码器本机可行）

| 编码器 | MPS 速度 | 本机可行性 |
|---|---|---|
| Qwen3-0.6B（文本） | 63ms/段 | ✓（去重缓存后 ~126min） |
| **whisper-large-v3 encoder（=Qwen2-Audio 音频塔，32层1280维）** | **800-1600ms/段** | **✗ 不可行**（40通验证5h，全量45h+；fp32必需，batch化无帮 MPS 已吃满） |

- Qwen3 文本：train 1.44M 滑窗朴素=25h，但去重后唯一上下文 8.3% → 126min。**冻结编码器必须按唯一上下文去重缓存，不可每滑窗重算。**
- whisper-large-v3 音频：每窗都不同（去重率低），且单段就 ~1s，本机 MPS 无解 → 需云 GPU 或更小 whisper（base/small）。
- 模型下载的 safetensors 默认 fp16，MPS 上 conv 需 fp32 → 加载用 `dtype=torch.float32`。

## Long-task patterns (for lwm hooks)

> Hook reads each `name: regex` line below. Match → 提醒 AI 跑 /project-state check.

training: nohup.*python.*-m[[:space:]]+src\.train
training_cloud: nohup.*python.*train.*--epochs[[:space:]]+[0-9]+
infer_test: python.*-m[[:space:]]+src\.infer_test

## User private files (do not touch)

- `docs/MY-NOTE.md` — user's working scratch, do NOT delete/rename even if content looks duplicate (若存在). See `~/.claude/projects/.../memory/feedback_dont_touch_user_private_files.md`.
