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

- **本机 MBP M3 Max 128GB（MPS，PyTorch 2.7.1）**: EDA / 数据预处理 / 小规模冒烟 / 调试代码
- **云 GPU（A100/4090 级）**: 正式大规模训练（1.4M 滑窗样本 + 大编码器）
- 复赛镜像必须在 **CUDA** 上最终验证

## Long-task patterns (for lwm hooks)

> Hook reads each `name: regex` line below. Match → 提醒 AI 跑 /project-state check.

training: nohup.*python.*-m[[:space:]]+src\.train
training_cloud: nohup.*python.*train.*--epochs[[:space:]]+[0-9]+
infer_test: python.*-m[[:space:]]+src\.infer_test

## User private files (do not touch)

- `docs/MY-NOTE.md` — user's working scratch, do NOT delete/rename even if content looks duplicate (若存在). See `~/.claude/projects/.../memory/feedback_dont_touch_user_private_files.md`.
