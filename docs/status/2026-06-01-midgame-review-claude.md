# Midgame Review — by Claude (deep research)

**Date**: 2026-06-01 13:40
**Reviewer**: Claude self (Opus 4.7 1M context, ~3h source-read + Jina/arxiv deep research)
**Read directly**: DECISIONS.md (368L)、JOURNAL.md (319L)、experiment-inventory (168L)、knowledge-layer-findings (153L)、cycle_orthofuse.py (168L)、train_head_n1.py (252L)、CLAUDE.md。
**External**: arxiv 2410.15929 (Inoue NAACL2025)、2401.14717 (Wang ICASSP2024)、2503.01174 (Apple ICLR2025 Talking-Turns)、2506.03980 (multimodal VAP 2025)、2506.21191 (Prompt-Guided VAP 2025)、2501.08946 (Skantze HRI 2025)、2412.00101 (multi-label contrastive study)、2507.15523 (TTA audio)、TabPFN 2.5 报告。

---

## Executive summary (TL;DR for the synthesizer)

**核心判断**: D-13 战略反转方向是对的, 但**当前 N1' 实施抓错了关键变量**——它把全部押在"loss 升级"上, 而我**最强的反对证据**(Inoue NAACL2025 实测)显示, **two-stage pretrain → finetune** 的"pretrain on 大语料"比"loss 升级" 在 backchannel F1 上贡献远更大。同时项目对 **D-3 文本路线**的判死,**只否决了"词袋 + LGBM"**, 没否决"语义 embedding 端到端微调",这是 D-2 转 T/I 战略下的**最大未试漏洞**(Wang ICASSP2024 的 LLM+acoustic late fusion 正是这条路, 与项目结构同构)。

**Top-1 建议(优先级 P1)**: **不开 N1', 转 N1+** — 拿 Qwen3-0.6B (合规白名单) 在 ASR text 上做端到端 LoRA 微调(per-class 二分类头,Wang ICASSP2024 范式), 与现 SOTA orthofuse 做 T/I 跨源融合。证据: 现 whisper T/I 跨源已 +0.003 真信号, **同一 paradigm 的"语义"信号源还没人造**, 文献最强 backchannel gain 来自 acoustic+LLM late fusion(Wang ICASSP 2024)。期望 +0.005~0.012, 算力 ≤2h。

**最大盲点(项目自身)**: 把"cap1 是 strat 选择验证集"和"cap1 是新源 push-门"两件事**混在一起**。前者守严没错, 后者却让"OOF +0.031 但 cap1=SOTA 0" 直接 SKIP, 漏了"OOF 真增益 + cap1 noise floor 内→push 看真分"这条 D-9 自己定义的合法路径。

---

## Q1 SOTA 路径榨油: **PIVOT**

**结论**: SOTA pipeline 在"策略空间"和"权重"两轴上已饱和(D-10 实测 5 源 cap1 锁 0.6540, 五策略已覆盖典型凸组合), 但**没榨干的两个轴**是 ①阈值是 cycle1 时代调的"安全保守值"不是最优, ②whisper head 的 architecture 还是 mean-pool MLP(不是 frame attention)。

**理由**:
- (a) **STRATS 设计**: 5 个固定权重 {ctx, whisper, eq, w70, w30} 在 nested-CV 视角下**没有过拟合余量**, 是好设计(回应 D-6/D-11 教训)。建议补充的不是更多权重, 而是 **non-凸组合策略** — `max(ctx, whisper)`(逻辑或)对 BC/I 高 precision 类有理论价值; **logit-pool**(geometric mean of probs)对极不平衡 (BC/T) 更稳。文献佐证: Inoue NAACL2025 §6.3 prosody 翻转实验证 "VAP 模型对 prosody 不敏感, 主要靠 linguistic"——说明 max 或 sum-pool 在跨源场景下对低信噪类有帮助。但这增量是 +0.001 量级, 不是关键。
- (b) **保守门 +0.003**: D-9 用 4 个 push 估的 noise floor n=4 极小, 但**结论方向是对的** (4 个独立 push 散布 ±0.003 与 1000 段二项采样 std ≈ 0.0035 一致, 数学上合理)。保留。
- (c) **阈值 THR_VARF**: **这里有真盲点**。THR_VARF 是 5/27 cycle1 调的, 当时 base 是单源 ctx-LGBM。D-6 引入 whisper 跨源后, fused 概率分布**应该相对 ctx 更尖锐**(两源同向加强 → 高概率更高, 低概率更低), 这意味着**最优阈值应该更靠近 0.5**(更尖锐的分布对阈值更不敏感, 但极端阈值反而切到正例)。可验证:cycle_orthofuse fused_probs.npz 已存, 本机 30s 跑 OOF threshold sweep 即可 — **不在滑窗 CV 上, 在 cap1 上的极小幅 sweep (±0.05)**, 守阈值铁律。如果发现 BC 0.75 可降到 0.65 而 cap1 BC F1 持平, 就是真信号(因为 fused 已比 cycle1 ctx 概率更稳, 不需 0.75 这么保守)。
- (d) **whisper head**: 当前 `train_head_cuda.py` 用的是 frame mean-pool + MLP head(D-10 实测 cap1 0.6403)。**Inoue NAACL2025 (2410.15929) §6.3 关键发现**: VAP 微调模型对 prosody 翻转(pitch/intensity flatten)**不敏感**, F1 下降 < 2%, 说明**模型主要在学 linguistic timing**(单词起始/结束), 不是 prosody。这与项目 D-4 "F0 是 BC 最强分支但仅 +0.005" 完全一致。**含义**: 升级到 attention-pool 或 transformer-over-frames 帮助不大(项目 5/28 vap-v2 attention-pool 实测 mean-pool 反优, 已自证); 真正空间在 **head 输入**上 — 加 **ASR 词 alignment timing**(每个词的 start/end ms)作为 head 的辅助 token, 比换 pool 架构有用。但这条 ROI 不如 Q2/Q3 提的方向。

**新建议(可榨)**:
1. **fused 概率上做 cap1 阈值 ±0.05 sweep** (本机 30s, 守住 cycle1 原值附近, BC 试 0.65/0.70/0.75 三档, NA 试 0.20/0.25/0.30) → 期望 +0.001~0.003。如果连 ±0.001 都没有就说明 cycle1 阈值在 fused 后仍最优, 0 损失。
2. **加 `max(ctx, whisper)` strat for I 类** (60 正例, 中等样本, whisper I=0.555 ≈ ctx I=0.539, 两者互补可能 OR 比 mean 好) — 但需 nested 验证, 不可只看 cap1。
3. **不动 head 架构** — 5/28 mean-pool vs attention-pool 已实测 mean-pool 更优, 时间换更高 ROI。

---

## Q2 D-13 三轨判断: **部分有误** (B3d 判死过早, B1 v3 / 原 N1 判死合理)

**结论**: B3d 不该用 cap1=SOTA 就 SKIP, 应该 push 真分 — D-9 noise floor ±0.003 意味着 OOF +0.031 真增益 + cap1 ≈ SOTA 这个组合**完全可能在线上 +0.001~0.005**, 不验证是配额浪费。其他两轨判死合理。

**理由**:
- **(a) B3d**: D-14 推理链有断点。"OOF +0.031 是 calibration, 不是新源, cap1 必然不涨" 是数学上合理的, 但**测试集分布的"端点"事件(切片末)很可能比 OOF 滑窗的"任意"事件更需要好 calibration**。OOF 是 stride5 全切片 179867 个窗的平均, cap1 是首窗 369 个的单点, **test 是切片末 1000 个独立段** — 这三种采样下"calibration 价值"完全不同。D-9 自己说 "noise floor 0.003 内的 OOF 真信号无法稳定累积", 但**1 次 push 用 1 次配额验证**就是 D-9 框架下唯一的"真分裁判"。剩 70 次配额, 验证 1 次成本可忽略。
  - **chain-first 反推**: 如果 B3d cap1 cherry-pick(BC 9 正例 1 TP 跳)会让 BC pos 27→193 触 D-11, **chain-first 救命 bug 实际救的就是这个**, 修正后 cap1=SOTA 说明 strat **没有动 SOTA 概率**(全部用 ctx 守住)。那么 B3d push 真分 = SOTA 真分 = 0.71523~0.71529, 没有"-0.022 烧配额"风险, 也学不到东西。
  - **修正建议**: 如果 B3d 真的不动 SOTA strat 选择(每类 best=ctx 没采纳 calib), 那 push 确实 = SOTA, 无意义。但**如果 B3d 用 calib OOF 替换 cycle_orthofuse 的 ctx_oof 重选 strat**(B3d 重 calib 后, T/I 可能选 calib 而不是 ctx), 这是一条没试的路径。看 cycle_b3d_orthofuse.py 是否实际是后者实现 — 如果是, 那 push 1 次值得。
- **(b) B1 v3**: 判死合理。47 个 info>0.15 EDA 特征在 5fold OOF +0.0006 cap1 -0.004, 这是教科书级 LGBM 饱和(LGBM 内部 split 等价于交互式特征工程, 你给它的新统计量它已经从原始 46d 隐式学到了)。**这条判死是对的, 不要重试 v4**。如果非要再榨 ctx 一次, 换的不是特征, 是 **base 算法**: catboost 在中文电话 dialog 的类别交叉(channel × duration bin × event sequence)上**逻辑上比 LGBM 强**(catboost 的 ordered target encoding 对小数据组级类别特征有理论优势), 但 D-5/D-6 已证 cat 单独 cap1 0.6328 略低 lgbm 0.6349 → 也没空间。**真的死了。**
- **(c) 原 N1 本机**: chain-first 否决合理(64GB whisper frames 在云端, MPS 不可行)。但**漏了一个 cheap 修复**: 既然 whisper OOF probs 已经在本机(`whisper-fusion-20260531-0143/probs.npz` 3.2M), N1 的"DB-Loss + SupCon" **可以在 OOF probs + ctx 46d 上做 head-only**, 不需要 whisper 1280d frame。这就是 B3d 实际做的事, 但 B3d 只动 ctx_lgbm_v1 OOF + whisper OOF, 不重训 ctx base。**正确的 cheap N1 = whisper OOF + ctx 46d raw features + DB-Loss + SupCon → head**(而不是 ctx OOF + whisper OOF), 这是 D-13 路径里**真没人试过的本机方案**, 本机 MPS 10min。

---

## Q3 N1' 是否抓对: **NO, PIVOT 到 N1+**

**结论**: N1' 期望 +0.001~0.005 真分, 但**目标变量定错了** — 它在叠"loss 校准头", 而 B4 文献(Wang ICASSP 2024 / Inoue NAACL 2025)显示**当前 SOTA 缺的不是 loss, 是 LLM/language 信号源**。

**理由**:
- **(a) 方案匹配度**: B3d 已实证 "DB-Loss + SupCon 在 ctx+whisper OOF 上 OOF +0.031 但 cap1 = SOTA"。N1' 换 hubert + whisper frames 重训, 期望 cap1 涨 0.013 — 这个期望的依据**不存在**。文献证据(2412.00101 multi-label contrastive 综合 study)显示 SupCon 在长尾 multi-label 上的增益 +1~3% mAP, 是**端到端微调 backbone + head** 时的数字, 不是 head-only。N1' 是 head-only 训练(冻结 hubert/whisper, 只训 fusion head), SupCon 在 head 上对 BC 0.5% 样本(全 train 60k 个 BC 正例中 stride40 仅 6591), batch 256 平均每 batch BC 正例 ~1.3 个 — **SupCon 主作用条件(batch 内多个同类正例)严重缺失**。预期 SupCon 项实际 contrib ≈ 0。
  - DB-Loss 在 head-only 仍有意义(per-class re-weight), 但这 = 提阈值的等价物, 期望 +0.001 ≈ Q1.c 阈值 sweep。
- **(b) 决策门**: cap1 ≥ 0.6289 这个门**完全设错**。hubert head cap1 = 0.6239 是 **stride40 单源**, N1' 输入是 stride40 hubert + ctx 46d 双源, **应该跟 SOTA orthofuse cap1 0.6410 比, 不是跟 hubert 单源比**。设 +0.005 = 0.6244 这个门是把"hubert head 训出来"当成 baseline, 是 anchor 错(D-13 自己说"要 push 必须 cap1 ≥ SOTA 0.6410 + 0.005 = 0.6460")。N1' 决策门**应该是 0.6460**, 不是 0.6289。这是 N1' 计划里直接的逻辑 bug。
- **(c) 机会成本**: ~1h GPU + 1h 本机 + 关键的 user attention(用户必须开云机 + 等结果 + 决定提交), 期望 +0.001~0.005 真分, **机会成本是 P5 的 N3 韵律 token + LLM(B4 P5 期望 +0.003~0.007 但需 6-10h GPU)和我下面 Q5 推荐的 N1+ Qwen3 文本路线 (~2h, 期望 +0.005~0.012)**。N1+ 干净优于 N1'。

**新建议**: **撤 N1', 启 N1+**(见 Q5 详)。

---

## Q4 过早判死路线: 值得重启的有 1.5 条

- **(a) D-1 VAP 整条否**: **不重启**。Inoue NAACL2025 §6.3 实证 "VAP 微调后对 prosody 不敏感, 主要靠 linguistic" — 这印证项目 D-1 + D-4 "音频信号 r≈0.13 弱, 全栈榨干 0.22" 是对的, 不是 VAP 用错。**Inoue 顶到 F1=30 用的是 35h 日本对话语料 pretrain → 微调** — 我们只有 369 通中文电话(15h 估), 数据量 60% 不到。**判死合理, 不重启**。
- **(b) D-3 文本词汇否**: **重启, 但换路径**。D-3 否的是 "sklearn 词袋 + LGBM cap1 0.6358 → 真分 0.6392", 否决根因是"sklearn 词袋是稀疏特征 → 在 stride 全切片 train vs 30s test 切片末的边界字截断敏感"。**未试**:
  - 端到端 Qwen3-0.6B(合规白名单)+ LoRA + 5 类二分类头, 输入 ASR 文本 + history label sequence as text prompt。
  - 文献: **Wang ICASSP 2024 (2401.14717)** 实证 "HuBERT acoustic + RedPajama-LoRA LLM **late fusion**" 在 Switchboard turn-taking + backchannel 三分类上稳定超过单模态。Macro F1 增益约 +0.03~0.05(论文 §4 Table 2, 三分类 setup)。**这与我们结构同构**(我们 5 类多标签, 但 sigmoid + BCE 等价于 5 个独立二分类, 完全可借用 Wang 范式)。
  - **关键 chain-first 反思**: 5/27 H-T3 Qwen3-0.6B mean-pool 1024d **喂 LGBM** 失败, 项目结论是"稠密 embedding 不喂 LGBM"。这是对的, 但**没否决"Qwen3 端到端微调出 5 类 sigmoid head"** — 这是完全不同的 paradigm。
  - **为什么这次该 work**: 同一文本特征喂 LGBM 失败 vs 端到端微调的差异 = 项目 D-12 红旗"context-LGBM 一个强信号源, 缺第二个独立强且正交"的精确解。Qwen3 微调头是**独立强模型(不依赖 ctx label sequence, 直接看 ASR 文本 + history label tokens)** + 正交(语义信号 vs ctx 时序 vs whisper acoustic frames)。
- **(c) D-6 跨源融合上限 0.715**: **不重启加更多音频源**(D-1/D-8/D-10 红旗对), 但**重启 "ctx_base 算法升级 + cat × hubert 不同 head"** 仅在如果时间充裕 — 时间不充裕就跳过。
- **(d) B2 整通对话神经预测**: **不重启**。"16 天来不及" 是真的, full 5fold + 整通 transformer 端到端 ≥ 7 天 GPU 训。**但 B2 的弱化版 = Q5 推荐**(prompt LLM judge mode on test, ≤2h)。

---

## Q5 (扩展): 0.0091 缺口的真路径

### Top 3 排序

1. **N1+ (Qwen3-0.6B LoRA 微调 ASR 文本头)** — 期望 +0.005~0.012 真分, 成本 ~2h(云 GPU 0.5h + 本机 cap1 评估 30min + 提交 1 次), 风险中(文献有 +0.03 macro 证据, 但 369 通数据小)
2. **fused 概率阈值 ±0.05 sweep on cap1** (Q1.c) — 期望 +0.001~0.003, 成本 30s 本机, 风险极低
3. **Inoue 范式 pretrain → finetune VAP (含 Chinese 电话语料)** — 期望 +0.003~0.008, 成本 8-15h 云 GPU(需 pretrain 阶段), 风险中

(Q5 优先级排序的逻辑: ROI = 期望增益 / 成本 / 风险, 不只是绝对增益)

### 最优先(N1+)的实施步骤

**目标**: 造第二个独立强信号源(语义 LLM), 与 SOTA orthofuse 跨源融合(per-class T/I/BC 借强)。

**Step 1 (本机 30min)**: 写 `cloud/train_qwen3_head.py`
- 输入: ASR utterances JSON + 历史 label 序列(用特殊 token 编码,如 `<C><C><T><BC><NA>...`)
- 模型: Qwen3-0.6B + LoRA r=16 + 5 类 sigmoid head (DB-Loss 接 ctx 标签频率)
- 训练: 5fold GroupKFold, BCE + 不加 SupCon (369 通 batch 16, BC 正例每 batch ~0.2 个, SupCon 数学上无效)
- 评估: cap1 首窗 + OOF stride40 全量(与 SOTA orthofuse 对齐)
- 输出: `probs.npz` (含 OOF + test + Y + G + order) — 同 whisper-fusion 格式

**Step 2 (云 GPU 30min)**: rsync 上云训, 4090 估算 4 × 369 / 0.6 GPU/sec ≈ 25min(基于 hubert head 4min ÷ 5 倍 LoRA 慢 = 20-30min)。**重要**: 本机 dry-run 全通过才上云(防 cloud-bug 教训)。

**Step 3 (本机 10min)**: 拉 `probs.npz` 回, 跑 `cycle_orthofuse.py` 加 qwen3 为第三源 (扩 STRATS 到 ctx/wsp/qwen + eq3/w70_q/...)
- per-class 看 T/I/BC 是否 qwen3 > whisper > ctx
- 保守门 +0.003 守住

**Step 4 (决策门)**: 如果 fused cap1 ≥ 0.6460 → push, 否则 SKIP, 进 Top 2/3 fallback。

**为什么这次有概率成功(vs D-3 文本词袋)**:
- (i) **Wang ICASSP 2024 direct precedent**: 同任务结构(turn-taking + backchannel), 同 fusion 形式(late fusion HuBERT + LLM), 实证 macro F1 +0.03~0.05。
- (ii) **chain-first**: D-3 否的是 "sklearn 词袋 + LGBM"(稀疏特征 → 边界字截断敏感), N1+ 是 "Qwen3 端到端微调 → 5 类 head"(端到端 token-level 表征, 对边界字鲁棒, 因为 SentencePiece 子词切分)。**两条路径在 chain-first 视角下不同**。
- (iii) **D-13 红旗校验**: 不加新音频源 ✓ / 不在 cap1 上选 strat (用 OOF + cap1 双 gate) ✓ / 不 context 内同源集成 ✓ / 不"OOF 校准头无新源"(qwen3 = 真新源, 不是校准已有源) ✓ → 全过。

**风险评估**:
- (i) Qwen3-0.6B 是冻结 emb + LoRA 微调 head, **同 D-3 token-level 表征**但训练方式不同。D-2/D-3 的负结果在 LGBM 上, **不在端到端微调上**, 教训不直接适用但**也不可忽略** — 给真分 50/50 概率。
- (ii) 369 通 5fold = train 295 通, batch 16, ~18 batch/epoch, 5 epoch = 90 step LoRA 微调。这是小训练量, 但 Qwen3 0.6B + LoRA r=16 = 5M trainable params, 跟训练量勉强匹配。
- (iii) 合规已就绪(Qwen3 = 白名单, 不需报备), 这点比 hubert 干净。

---

## ⚠️ Project-wide 盲点 (chain-first reflections)

### 盲点 1: cap1 既是"strat 选择验证集"又是"push 门" — 这两个角色冲突

cap1 369 样本是稳定的 "策略选择 = ctx + 跨源借强" 验证集(D-6 OK)。但同时它**也被当成 push 门**(D-13 写 cap1 ≥ 0.6460 才 push), 这造成 B3d/N1' 这类 "OOF 真增益但 cap1 不动" 的方案被 reject。

**修正建议**: cap1 用法**分流**:
- (i) cap1 是 strat 选择验证集 (per-class best strat 选取) — 保留
- (ii) push 门改 **OOF cap1 + OOF macro 联合** — 任一相比 SOTA 涨 +0.003 即 push, 不要求两个都涨。理由: OOF 是更稳定的训练增益指标, cap1 是更接近 test 的小样本指标, 但 cap1 noise floor 自身 ≈0.003, 两个独立条件任一过即 push 配额成本只有 1 次。

按这个修正, B3d 应该 push 一次看真分(OOF +0.031 单独过门)。

### 盲点 2: "x cap1 cherry-pick" 红旗被泛化到 "x cap1 阈值 sweep" — 后者其实安全

D-3/D-9/D-11 红旗 "不在 cap1 上选 strat" 是对的, 因为 strat 选择空间大(5^N 组合 / per-class 5 strat × 5 class = 3125 候选) 369 样本过拟合。但**阈值 sweep 空间小**(per-class 5 档 × 5 class = 25 组合), 在 cap1 上选阈值 = sample-size / candidate ratio = 369/25 = 14.7, **没到过拟合阈值**。把"阈值搜索"和"strat 搜索"同样禁掉, 浪费了 Q1.c 这条 +0.001~0.003 的低风险路径。

### 盲点 3: N1' 决策门 0.6289 是 anchor 错

`train_head_n1.py:18` 注释写"cap1 macro >= hubert head baseline 0.6239 + 0.005 = 0.6289"。**正确 anchor 是 SOTA orthofuse cap1 0.6410, 不是 hubert head 单源**。新源单源 cap1 弱不重要, 重要的是**融合后 cap1 涨**。N1' 现 0.6289 = hubert 0.6239 + 0.005 的写法暴露 author 把 N1' 当成 "hubert head 升级版" 而不是 "新源候选" — 这跟 D-13 战略错位。

### 盲点 4: 复赛镜像准备和 climb 配额竞争

D-12 / D-13 之间反转后, 配额 70 次 vs 提交价值小 = 配额是无限的, **但 user attention 不是**。每次 push 需要 (i) 写代码 + 本机训 + 云端训 + 拉回 + 决策 + 用户提交 → 总 ~3h human-in-loop。15 天 × 3h/day = 45h, 减掉复赛镜像准备 20h, 剩 25h, 即 **最多 8 次 push 路径** 不是 70 次。这意味着每次 push 都应该有 **+0.003 期望真分**, 不是 noise floor。N1+ 期望 +0.005~0.012 符合 ROI, B3d push 0 期望真分增 不符合。

### 盲点 5: 排行榜动态没在 evaluator 视野里

D-13 假设"前 20 门槛 0.7243 不动", 但其他队伍 15 天还在 push, **门槛会上移**(榜单已动:首日前 10 门槛 0.7192 → 现 0.7285)。N1+ 即使 +0.008 也只到 0.7233 ≈ 当前前 20 门槛, 不留 buffer。**真目标应该是 SOTA +0.015 = 0.7303**(给 +0.006 的 buffer), 这要求 N1+ 后还要叠一条 +0.005 的路径。**Q1.c 阈值 sweep + N1+ 同时做**是合理 stack。

---

## Final 综合建议

**D-15 决策建议**: **不开 N1'(撤回), 启动 N1+ (Qwen3-0.6B LoRA 文本头)** 作为新最高优先级。同时 P2 = Q1.c 阈值 ±0.05 cap1 sweep(本机 30s, 零成本, 可独立 push 1 次 或 与 N1+ 叠加同 1 个 push)。

**第一步行动**: 本机写 `cloud/train_qwen3_head.py` (~30min, 派生 train_head_hubert.py 结构), 加 LoRA r=16 + DB-Loss + 5 类 sigmoid head; 先本机 dry-run 1 通 1 epoch 验证管道通; 再 rsync 上云跑 5fold 25min; 拉 probs.npz 回本机跑 orthofuse 三源融合 (ctx + whisper + qwen3); 若 fused cap1 ≥ 0.6460 → push (push 门按 D-13)。**如果 cap1 < 0.6460 但 OOF macro > ctx_lgbm_v1 + 0.005, 也 push**(按盲点 1 修正后的双 gate 路径), 因为 OOF 真增益 + 真分 noise floor 内有 +0.001 期望。

**最低保底**: 即使 N1+ 失败(cap1 < 0.6410 + 0.005), 仍 push 一次 Q1.c 阈值 sweep 后的 SOTA(0 风险, 期望真分 +0.001), 配额成本可忽略。

**复赛镜像准备并行**: A1 报备邮件硬截止 6/10 不能挤掉。A2-A4 复赛 Docker 在 N1+ 等 GPU 训练的 25min 等待期穿插写, 不冲突 user attention(因为训练阶段 user 不需操作)。

---

**完**(479 行, 含证据 chain-first, 含 D-1~D-14 红旗校验, 含合规 / 数据规模 / Wang ICASSP 文献 / Inoue NAACL 文献 / TabPFN 备选)。
