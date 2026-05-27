# FinVCup 2026 对话轮次交互建模 — Context

> 下游 skill（writing-plans / executing-plans）把本文件当契约读。这里已定的决策不要在 plan/execute 阶段重新问。要改决策，先改这个文件。

**Created:** 2026-05-27 11:57
**Topic:** 第十一届信也科技杯 — 对话轮次交互建模（Turn-Taking 事件预测），目标前 3 保底前 10
**Related prior CONTEXT:** none（首个）

---

## 任务本质（从数据 + baseline 实测确认）

- **预测目标**：给定过去 30s（音频 + ASR 文本 + 历史标签序列），预测**未来 2s（25×80ms chunk）内**5 类事件 `C/T/BC/I/NA` 各自是否出现。
- **形式**：event-level 多标签，sigmoid + BCE，输出 5 列 0/1。
- **评价指标**：**Macro-F1**（5 类 F1 等权平均，sklearn `f1_score`，提交是硬 0/1）。
- **因果约束**：只能用过去的音频/文本/标签，不能读未来。

### 数据实测特征（决定策略的硬事实）

| 项 | 实测值 | 含义 |
|---|---|---|
| 音频 | **8kHz, 2 声道（双说话人）, int16** | 电话语音；baseline 重采样到 16k 喂 Whisper |
| 训练集 | 369 通对话，中位 ~25min/通（chunk 数中位 18759） | 滑窗（context=375, target=25, stride=5）→ 约 1.4M 训练样本 |
| 测试集 | 1000 段 × 30s，context 恒为 375 chunk（=30s） | 推理按段；test 提供历史 context 标签 |
| **类分布（chunk 级）** | C=64.2% / NA=30.1% / **I=4.0% / T=1.2% / BC=0.5%** | **胜负手**：Macro-F1 等权 → BC/T/I 三个稀有类决定排名 |
| text json | train 有 `utterances[{channel_id,start_ms,end_ms,text}]`；test 含 `start_ms/end_ms` | 双声道 ASR，带说话人 + 时间戳 |

### 排行榜目标（用户 2026-05-27 补充，86 队有分）

| 目标 | Macro-F1 | 备注 |
|---|---|---|
| #1 | **0.747489** | ListenBeyond, 32 次提交 |
| **#3（前 3 线）** | **0.73568** | limzero |
| **#10（保底线）** | **0.719176** | |
| #1→#10 跨度 | **仅 0.028** | 榜首极密集 → 稀有类增益 + 阈值调优足以挪动名次 |

**北极星**：本地 CV 对齐线上后，先破 0.7192（前 10），再冲 0.7357（前 3）。

---

## Scope

**In（初赛阶段，按优先级）：**
1. **复现并跑通官方 baseline**，拿到第一个公榜分作为对照锚点
2. **稀有类专项**：focal loss / 重采样 / **逐类阈值搜索**（Macro-F1 下最高杠杆，固定 0.5 阈值是 baseline 的明显漏分点）
3. **上下文标签序列建模深挖**：context labels 是因果性最强、最便宜的信号（test 提供 375-chunk 历史）
4. **EDA 优先 + 多方案 bake-off**：不预先锁死单一架构，用真实 CV 指标决定主线
5. **模型集成 + TTA**（单模型打磨后）

**Out（初赛不做）：**
- Qwen2.5-Omni 端到端大模型路线（推理时延 = 复赛镜像风险；本机难训）— 留作复赛备选
- 复赛 Docker 镜像正式构建（6/20-7/7 才提交，初赛先冲公榜分；但环境要可迁移到 CUDA）

**Deferred（带触发条件）：**
- **语音编码器选型 research（/research）** → 触发：EDA 跑完，确认音频模态有边际贡献、且需要在多个编码器间选型时。EDA 若显示纯上下文标签已很强，可能根本不需要这轮 research
- **Qwen2.5-Omni 端到端路线** → 触发：单模型 + 集成 CV 已到 ~0.72 且榜单要冲前 3 还差临门一脚
- **K-fold 交叉验证** → 触发：进入中后期冲分、需要 OOF 给集成喂数据
- **复赛镜像化** → 触发：初赛结束、确定晋级后

---

## Decisions

### 1. 建模主线 — DECIDED
**Choice:** EDA 优先 + 2-3 方案 bake-off，用真实 CV 指标决定主线，不预先锁死架构。
**Why:** 榜首密集（#1→#10 仅 0.028），方向选错的代价高；先量化"context label 单信号能到多少 / test context 是否泄漏模式 / 各模态边际贡献"，再投入大训练。
**Source:** user

### 2. 候选架构池（bake-off 入选项）— DECIDED
**Choice:** 至少对照 3 条：
  - (A) **可微调语音编码器** + 文本 + 上下文标签融合 — 最贴 turn-taking SOTA，单卡可训。**编码器选型受 Decision 10 约束**：优先白名单 Qwen 系列（Qwen2.5-Omni 的音频编码器）；非 Qwen 公开模型（WavLM/Chinese-wav2vec2/HuBERT）仅在 EDA/小实验证明明显增益时才用，且需 6/10 前报备
  - (B) **强化版 baseline 融合**（解冻部分层、升级文本编码器、加强上下文标签建模）— 低风险锚点
  - (C) **上下文标签序列模型单独基线**（仅 context labels → 序列模型/统计特征）— 量化"最便宜信号"的天花板
**Why:** 覆盖"语音表征 / 多模态融合 / 纯历史信号"三个正交假设；C 还能直接验证稀有类靠历史能拉到多少。
**Source:** inferred（从"EDA + bake-off"决策派生）

### 10. 模型使用边界 — DECIDED
**Choice:** **模型总参数量 ≤ 8B（硬上限）**；**优先用赛题白名单 Qwen 系列**（Qwen2.5-Omni-3B/7B、Qwen3-4B/0.6B/0.8B）；非 Qwen 的其他公开模型**仅在能带来明显增益时才采用**（需 EDA/小实验佐证），且必须 6/10 前向 xinyebei@xinye.com 报备。
**Why:** 用户定方针——合规风险与潜在增益权衡，非白名单模型门槛高；8B 上限同时约束复赛镜像推理成本。
**Source:** user（2026-05-27）

### 3. 稀有类处理 — DECIDED（几乎必做）
**Choice:** focal loss + per-label pos_weight（baseline 已有）+ **逐类阈值后处理搜索**（在 CV 上为每类独立调阈值最大化该类 F1），可选重采样/类平衡采样。
**Why:** Macro-F1 等权且 BC=0.5%/T=1.2%，固定 0.5 阈值是免费漏分；阈值搜索是最低成本最高杠杆。
**Source:** user

### 4. 验证协议 — DECIDED
**Choice:** **按 conversation 划分** valid（避免同通对话泄漏）+ **把验证样本构造成与 test 一致的"独立 30s 切片"形式**（而非 baseline 的滑窗逐 end_idx），让离线 Macro-F1 尽量逼近线上分布。
**Why:** 官方明警告公榜 ≠ 私榜、线下高分≠线上高分；test 是独立 30s 切片，验证集必须同分布，否则 CV 不可信。
**Source:** user

### 5. 公榜反馈环 — DECIDED
**Choice:** 我生成 `pred_test1.csv`（格式：`segment_id,c,na,i,bc,t`，0/1），用户手动提交后**贴回 Macro-F1 真分**；我用真分校准离线 CV、决定下一步（climb 式真分注入环）。
**Why:** 公榜不能过拟合，但里程碑节点需要真分校准 CV gap。提交延时 < 1 小时。
**Source:** user

### 6. 算力分工 — DECIDED
**Choice:** **本机 MBP M3 Max 128GB 做开发 / EDA / 数据预处理 / 小规模冒烟 / 调试代码**（MPS 够用，128GB 内存适合数据预处理）；**正式大规模训练租云 GPU（A100/4090 级）**；复赛镜像必须在 CUDA 上验证。
**Why:** 1.4M 样本 + Whisper 级编码器在 MPS 上训练慢、AMP/DDP 支持不全；复赛要交 CUDA 镜像，环境一致性要求云端最终验证。
**Source:** user

### 7. 状态/文档体系 — DECIDED
**Choice:** **清理并重写项目 `CLAUDE.md`**（去掉从 Fusion-Control 拷来的 SAC/climb/tokamak 残留，改成本音频赛题事实：任务定义、数据布局、指标、long-task patterns）；初始化 `docs/status/` 轻量 memory。
**Why:** 当前 CLAUDE.md 含大量不相关上下文，会误导后续 session（global CLAUDE.md 已警示 stale context 的代价）。
**Source:** user

### 8. 工程栈 — DECIDED（继承项目现状，无争议）
**Choice:** Python 3.12 + `uv`（`pyproject.toml`）；PyTorch 2.7.1（本机已装，MPS）；新代码走 `src/` 布局；测试放 `tests/{branch}/`；docs 用中文，CLAUDE.md/README 英文。
**Why:** 项目已有 `.python-version=3.12` + `pyproject.toml` + global CLAUDE.md 约定。
**Source:** inferred（项目现状 + global 约定）

### 9. EDA 必答问题清单 — OPEN（plan 阶段第一批任务，但问题已锁定）
**Question:** 这些是 bake-off 前必须先量化的事实，writing-plans 要把它们排成第一个 EDA 任务：
  - test 的 context 标签序列分布 vs train 末段分布——**test context 是否泄漏未来事件的强先验？**（若是，纯 context 模型可能就很强）
  - 各模态边际贡献：只用 context labels / +text / +audio 各自 CV Macro-F1
  - 稀有类（BC/T）在"未来 2s 出现"的正样本率（窗口级，不是 chunk 级）——决定重采样比例和阈值搜索范围
  - 8kHz 原生 vs 重采样 16k 对语音编码器的影响（电话域 16k 预训练模型是否反而吃亏）
  - ASR 文本质量 / 覆盖率（test 平均 8 句 vs train 355 句，长度差异大）
**Why deferred:** 这些要跑代码量化，属 plan/execute 的第一阶段产出，不是 discuss 能拍板的。

---

## Reusable Assets Found

| Path | 是什么 | 怎么用 |
|---|---|---|
| `baselines/2026_finvcup_baseline/src/data/dataset.py` | 滑窗样本构造 + wav 切片 + text context 拼接 + collate | 直接复用 `build_train_samples_multitask` / `_read_wav_slice` / `build_text_context`；验证集改造成 30s 切片形式 |
| `baselines/2026_finvcup_baseline/src/models/multimodal_baseline.py` | Whisper/CNN 音频编码 + Qwen 文本 + ContextLabelEncoder + HandcraftedFeatures + 融合头 | 方案 B 的起点；ContextLabelEncoder / HandcraftedFeatures 是现成的上下文标签建模，方案 C 可直接抽出 |
| `baselines/2026_finvcup_baseline/src/utils.py:compute_multilabel_metrics` | Macro-F1 / per-label F1 / AUC（固定 0.5 阈值） | 复用做 CV；**注意**它写死 0.5，阈值搜索要在它之上加一层 |
| `baselines/2026_finvcup_baseline/src/infer_test.py` | 测试集推理 → pred.csv（支持 `--threshold`） | 复用导出；扩展成逐类阈值 |
| `baselines/.../src/utils.py:compute_gaussian_soft_f1_sequence` | soft-F1 序列指标 | 潜在的可微 F1 训练目标（探索项） |
| `baselines/.../configs/whisper_qwen0_6b_..._5labels_competition.yaml` | baseline 配置 | 复现起点；改路径到 `data/train` |
| `data/train/`, `data/test/` | 已解压数据（369 训练通话 / 1000 测试段） | 直接用；注意 gitignored |

**注意**：`baselines/` 自带 `.git`（是 clone 进来的官方仓库）；项目 `.gitignore` 已忽略 `baselines/` 和 `data/`。

---

## Constraints（from CLAUDE.md / 约定 / 数据）

- 语言：Python 3.12，`uv` 管依赖，PyTorch（本机 MPS / 云 CUDA）
- 测试框架：pytest，文件 `test_*.py` 放 `tests/{branch}/`
- 代码风格：Black(100) + isort(black) + f-string + pathlib + 类型注解 + loguru
- docs 用中文（`UPPERCASE_WITH_UNDERSCORES.md`），CLAUDE.md/README 英文
- 提交信息结尾：`Generated-By: Claude (claude-opus-4-7) via Claude Code CLI`
- **不碰** `docs/MY-NOTE.md`（用户私有）；不提交 `.env`/secrets/data
- **比赛合规**：用公开数据/模型需 2026-06-10 前报备 xinyebei@xinye.com；**禁止用公榜数据训练/打伪标签**；增强数据 + 生成代码须随复赛镜像提交
- **数据三分类**（global §17）：raw → `data/`(gitignore)；deliverable（如 frozen baseline pred、CV split）→ tracked dir；intermediate → `data/`/`tmp/`
- **不要把 userEmail（maxthingk@fastmail.com）当成比赛/对外身份**——比赛报备/联系邮箱须问用户

---

## Open Questions（留给 writing-plans 细化）

1. EDA 必答清单（见 Decision 9）——第一个 plan 任务，跑代码量化。
2. bake-off 三方案的具体训练预算（本机冒烟 step 数 / 云端正式 epoch 数）与里程碑提交节点。
3. 逐类阈值搜索的实现位置（训练后处理脚本 vs 推理内联）与搜索粒度。
4. 验证集"30s 切片化"的具体构造（从 train 长对话采样独立 30s 段 + 对应 context/text/label，模拟 test）。
5. 云 GPU 选型与数据上传方式（1.4M 样本但 audio 是按需切片，实际传的是 369 通原始 wav + label/text）。
6. 语音编码器具体选型（中文电话域优先：如 WavLM-Base+ / Chinese-wav2vec2 / Whisper-encoder 解冻）——bake-off 内定。

---

## Exit

Ready for: `/superpowers:writing-plans`
Input to writing-plans: 本文件 + "先重写 CLAUDE.md 清理 stale context，再做 EDA 任务（Decision 9 清单），再进 bake-off"
