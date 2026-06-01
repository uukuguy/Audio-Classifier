# B4 Knowledge Layer Findings — 2026-06-01

> **状态**: 🟡 decision-history（B4 一日研究结论，决策已采纳）
> **目的**: D-13 启动后第一日 0 算力研究，找 D-1~D-12 范围外全新正交信号源
> **方法**: gemini consult + Jina/WebSearch 9 路 + arxiv 论文 deep-dive 2 篇 (opencode 30min 未出 不阻塞)
> **结论**: 锁定 3 个候选方向 + 优先级排序 + 决定 B2 不触发

## 1. 已读到的关键论文 / 趋势

### 1.1 MM-F2F (ACL 2025 Long, arxiv 2505.12654)
- 3 模态 T+A+V 低秩融合，BC F1 0.680→**0.906** (+33%) / Turn 0.739→0.811 (+10%)
- **不直接适用**：①数据集 = 面对面视频（210h, 1.5M 词标）有 visual 模态，我们没有 ②三分类 Keep/Turn/BC 跟我们 5 类不同 ③数据规模 14× 我们
- **可借鉴**：
  - HuBERT > Wav2Vec2 印证（Tab.2）— 我们 cycle 17 实测同结论
  - **低秩多模态融合 + 模态选择 + 随机模态丢弃训练 RMDT** — Tab.6 RMDT 让单模态推理鲁棒度从 0.017-0.640 提升到 0.747-0.896，**这个训练技巧可移植到我们 ctx×whisper×hubert 三源融合的 train 阶段**

### 1.2 NTPP (arxiv 2506.00975)
- 双通道生成式语音 LM，Next-Token-Pair 预测
- **完全不适用**：判别 task vs 生成 task，且需 VQVAE 量化整个 dialogue

### 1.3 Qwen2.5-Omni-7B (Alibaba)
- MMSU/MMAU/MMAR 三榜开源第一（2025-06-12）
- Thinker-Talker 架构，原生 omnimodal（含音频理解+实时对话）
- **关键**：5/29 cycle 11 的"Omni encoder=WhisperFeatureExtractor"是错的，那是 Omni-3B 的早期版。**Omni-7B 是新架构, 不是 whisper 套壳**。但仍 ≤8B 合规
- **可用方式**：作为 LLM-judge 对 BC/T 高不确定样本做专家重排（gemini 方向 1）

### 1.4 多标签长尾对比学习（arxiv 2404.08720 + 2412.00101）
- Supervised Contrastive Loss (SupCon) + 长尾 multi-label 综合 study
- alpha BCE+contrastive 推荐 0.2-0.4
- 长尾用 memory queue + label prototype + weighted attraction/repulsion
- **直接适用**：我们 BC 0.5% / T 1.2% 是教科书级长尾

### 1.5 Distribution-Balanced Loss (ECCV 2020, arxiv 2007.09654)
- 对 BCE 两个改: re-balance weights 考虑 label co-occurrence + negative-tolerant regularization 缓解 over-suppression
- **直接适用**：我们 5 类共现严重（C 与 NA 反相关 / T 与 C 后续相关），BCE 现在没有 co-occurrence handling

### 1.6 gemini consult 三个方向
- 方向 1 Qwen2-Audio 全量/LoRA Event-centric SFT — **风险高**：等于重做 LoRA whisper（D-7 已否冻结+LoRA 全路径 cap5 欠拟合）。Omni 不同 ≠ 必然不踩同坑
- 方向 2 韵律离散化 Token 注入 LLM — **新颖**：F0+能量分箱成特殊 token 喂 Qwen2.5-7B。绕开"VAP 连续向量 r<0.04"
- 方向 3 SupCon + Distribution-Balanced — **印证学术界共识**（1.4/1.5 同源）

## 2. 候选方向（D-1~D-12 范围外）

### N1: **训练 Loss 升级 — Distribution-Balanced + SupCon 双损失**

**核心**: 现 ctx-LGBM 是 BCE/单类 split，没考虑 label co-occurrence + 长尾。把神经 head（whisper-fusion-20260531-0143 已有）的训练 loss 从 BCE 升级:
- `L = BCE_DB + α · SupCon`，α∈[0.2, 0.4]
- DB-Loss: re-balance weights + neg-tolerant regularization（针对 BC 0.5% / T 1.2%）
- SupCon: BC 样本聚类，远离 NA — 强迫学到 BC 的微弱特征

**为什么正交**:
- D-1~D-12 全是"换特征 / 换 base / 换融合"，没动过 loss
- 长尾损失改造直接对症 BC / T 极少样本
- 不动现有 ctx 基座，只重训 whisper head

**风险**:
- 长尾损失可能让 C/NA 退化（D-3 阈值铁律同款）
- 必须保持 cap1 OOF eval pipeline 不动，避免又踩 cap1 cherry-pick

**期望**: +0.002~0.005 (BC/T 各涨 0.02-0.05 → Macro/5 直接折进)

**算力**: 本机 MPS 重训 whisper head 即可（whisper OOF probs 已有），~30min-1h/fold × 5fold = 2-5h

### N2: **Qwen2.5-Omni-7B audio understanding 神经预测（LLM judge 模式）**

**核心**:
- 不重训 Omni，**冻结推理模式**用 Omni 对 1000 test segment 做音频理解
- prompt: "听这段电话音频, 30s 内 5 类事件 [C/T/BC/I/NA] 未来 2s 会不会发生? 输出 5 个 yes/no"
- 取 Omni token logits 或解析输出 → 第 4 个独立信号源
- 与 SOTA orthofuse 跨源融合（per-class 正交）

**为什么正交**:
- Omni-7B ≠ whisper (D-1)，是 thinker-talker omnimodal 架构，audio understanding SOTA
- 不是 ASR 任务范式，原生对"语音事件"敏感
- 推理而非微调，绕开 LoRA cap5 欠拟合（D-7）

**风险**:
- ⚠️ 反复教训：仅当 cap1 ≥0.66 才 push（D-13 push 门）
- Omni-7B 单次推理慢（7B 模型 + 30s 音频），云上 4090 估计 5-10s/段，1000 段 = 1-3h
- Omni zero-shot 在 cycle 11 已被 probe 测过"无判别力, 全答是"（research-tree falsified 段）— **关键风险**：但那是 cycle 11 Omni-3B 早版，且 zero-shot 文本式 prompt，跟 N2 提的 logits 提取/输出解析模式不同

**期望**: 极不确定。可能 0（重蹈 cycle 11 覆辙），可能 +0.005

**算力**: 云上 1-3h 推理 + 0.5h 融合实验

### N3: **韵律离散化 Token 注入文本 LLM**

**核心**:
- 提取 30s 内每秒（或每 chunk）F0 + 能量统计 → 分箱成 32 等级特殊 token `<f0_12>`, `<pwr_5>`
- 插入 ASR 文本中间 → 喂 Qwen2.5-7B（白名单内）作 multi-label head 微调
- 绕开"VAP 连续向量 |r|<0.04"（连续信号太噪），用离散 token 让 LLM 学韵律

**为什么正交**:
- F0/pitch 在 D-4 已实测 "BC 最强分支 r=0.128"，单独融合 +0.005 — 已知有信号但未充分利用
- 离散化 + LLM 路径 D-1~D-12 没碰过
- Mandarin tone 与 F0 强相关，中文电话客服域有利

**风险**:
- 微调 7B LLM 比 N1 重，需云 GPU
- token 分箱方案需调，第一次大概率不对

**期望**: +0.003~0.007

**算力**: 云上 5fold LoRA 微调 ~6-10h

## 3. 决策

### 优先级（按 ROI/风险/算力综合排）

| 优先级 | 方向 | 何时启动 | 投入 | 期望 |
|---|---|---|---|---|
| **P1** | **N1: DB-Loss + SupCon 升级 whisper head** | **今天/明天** | 本机 2-5h | +0.002~0.005 |
| **P2** | **B3 后处理 / TTA / pseudo-label** | 明天 | 本机 0.5-1 天 | +0.001~0.005 |
| **P3** | **B1 ctx 特征工程 v3** | 6/3-6/5 | 本机 1-2 天 | +0.003~0.010 |
| P4 | N2: Omni-7B LLM judge | 视 P1-P3 结果 | 云上 1-3h | 不确定 |
| P5 | N3: 韵律 token + LLM 微调 | 仅 P1-P4 全失败 | 云上 6-10h | +0.003~0.007 |
| ❌ | B2 整通对话神经预测 | **不触发** | — | — |

### 关键决策说明

1. **B2 取消触发**（task #4 应改 cancelled）— Knowledge Layer 没发现"必须换架构"的全新方向。N1/N2/N3 都是在现 SOTA 基础上叠损失/叠新源/叠融合，不需要重做整通对话 transformer。**B2 16 天来不及做完整 5fold 的风险 > 期望收益**。
2. **N1 最高 ROI** — 0 新模型下载, 0 新数据, 现有 whisper OOF probs 直接复用, 损失改 +50 行代码。即使失败也只烧 2-5h 本机时间。
3. **N2 LLM judge 留作 P1-P3 部分失败时的 safety net**，先不投入（D-9 教训 cycle 11 Omni zero-shot 全答是，可能踩同坑）。
4. **N3 韵律 token** 投入大但思路新颖，**先等 P1-P3 + N2 结果再决定**。如果 P1 N1 成功证明"loss 升级有效"，N3 反而冗余。
5. **三轨执行顺序调整**：
   - 原 B4→B3→B1 改为 **B4 → N1 → B3 → B1**（N1 替原 B4 的 "未发现新方向就回 B1+B3"）
   - N1 是 B4 找到的新方向，**插入到 B3 前面**优先做
   - 仍是 1-2 天周期内出 N1 push 结果

## 4. 不会再追的 stale 方向（B4 已确认无价值）

- ❌ 加 video 模态（我们没视频数据，MM-F2F 数据集独有）
- ❌ 双通道生成式 SLM（NTPP 范式跟判别任务不兼容）
- ❌ Qwen2-Audio LoRA 全量微调（gemini 方向 1 = 重蹈 LoRA whisper D-7）
- ❌ context 内 4 base 再加 transformer（D-5 不正交确认）

## 5. 待办与下一步

- ✅ B4 task #1 标 completed
- 🔄 启动 N1 cycle = 写 `tools/climb/cycle_n1_dbloss_supcon.py`：基于现 whisper-fusion-20260531-0143 OOF probs 重训 head，loss 换 DB-Loss + α·SupCon
- ⚪ B3 task #2 在 N1 跑训的同时启动（本机并行无冲突）
- ⚪ task #4 (B2) 应改 cancelled — Knowledge Layer 已否

## 引用

- [MM-F2F ACL 2025](https://arxiv.org/abs/2505.12654)
- [NTPP arxiv 2506.00975](https://arxiv.org/abs/2506.00975)
- [Distribution-Balanced Loss ECCV 2020](https://arxiv.org/abs/2007.09654)
- [Long-Tail Multi-Label Contrastive 2024](https://arxiv.org/abs/2404.08720)
- [Multi-Label Contrastive Comprehensive Study 2024](https://arxiv.org/abs/2412.00101)
- [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- gemini consult 输出全文: `/tmp/b4-out/gemini.txt`
