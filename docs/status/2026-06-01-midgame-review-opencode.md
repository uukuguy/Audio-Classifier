# Midgame Review — by Claude (Independent)

**Date**: 2026-06-01 11:45
**Reviewer**: Claude (self-review with fresh context)

## Q1 SOTA 路径榨油: YES, but ≤0.005

**结论**: SOTA 实现有 3 个未被探索的子优化方向,但每个增益都在 noise floor 附近,单靠任何一个破不了 0.009 缺口。

**理由**: 读 `cycle_orthofuse.py:47-53` STRATS 定义 + `JOURNAL.md:292-299` D-8→D-12 全路径。

- **(a) STRATS 设计**: 确实过窄。5 个固定权重凸组合 {ctx, whisper, eq, w70, w30} 漏掉了:
  1. **`max(ctx, whisper)`**: 非凸组合,对于 per-class T/I 这类 whisper 强的类,max 可以取 whisper 全强度而不被 ctx 稀释。与 w30 有本质区别 — w30 是 0.3*ctx+0.7*whisper,ctx 分量仍拖后腿;max 是纯取高值。
  2. **Isotonic 校准后凸组合**: 现策略直接对原始 sigmoid 概率做凸组合,但 ctx 和 whisper 的概率分布校准度不同 (ctx 可能过度自信,whisper 可能欠校准)。先各自 isotonic regression 校准到 test cap1 OOF → 再凸组合 → 再 isotonic,是标准 fusion 做法,且无新参数 (isotonic 是单调非参)。
  3. **Per-class alpha**: 不是选固定权重 strat,而是对每类在 [0,1] 网格搜最优 alpha (或 Bayesian opt with 5-fold OOF on 179867 windows,不在 cap1 369 搜→避 D-3/D-11)。D-6 把 grid 判死是因为在 cap1 369 上搜,但 OOF 179867 上搜不同 — OOF grid 不 cherry-pick 9 正例。
  
  **但保守门会吞掉这些增益**: 若 isotonic 让 T F1 cap1 从 0.676→0.685 (+0.009),gate +0.003 会采纳,但 noise floor 可能让线上 +0.001~0.002。

- **(b) 保守门 +0.003**: D-9 的 noise floor 估计 (n=4 push 分散度 0.00296) 证据偏弱。**n=4 不能自信估计 noise floor**。但问题是: 没有更多 push 数据来 refine 这个估计,每次 push 烧 1 次提交配额 → 没有对照组来测 0.003 vs 0.005 哪个是对的。**务实结论: 0.003 虽是弱证据,但它是唯一可用的估计,而且已有 4 个 push 中 1 个 (whisper T/I +0.003) 可重现,另 2 个 (强基座/hubert) 在线下持平或负 → 0.003 保守门可能是对的**。扩大 gate 到 +0.001 会多烧提交配额而大概率不涨分。

- **(c) 阈值 THR_VARF**: **这是 Q1 最有希望的榨油点**。`cycle_orthofuse.py:42` 的 `THR_VARF = {0:0.05, 1:0.50, 2:0.75, 3:0.65, 4:0.25}` 来自 cycle1 context-only 时期。orthofuse 融合后 per-class 概率分布与 context-only 完全不同 (T 从 ctx 主变为 w70 混合, I 变为纯 whisper)。**最优阈值必然偏移**。±0.05 窄 sweep (守阈值铁律: 不在滑窗 CV 上搜,只在 cap1 369 上做 ±0.05 窄扫): T∈[0.45,0.55], BC∈[0.70,0.80], I∈[0.60,0.70], NA∈[0.20,0.30]。若找到 T 0.45 比 0.50 好 +0.002,OOF 验证不退化,直接提交 — 成本 1 次提交配额,期望 +0.001~0.003。

- **(d) whisper head 架构**: **这跟 (c) 和 STRATS 扩展有本质不同 — 不是工程优化,是 head 从 MLP 升到 Transformer attention over 50Hz 帧序列**。`cloud/train_head_n1.py:86-111` 显示当前 head 是: mean-pool whisper 1280d frames across window → MLP(256→128→5)。**替换为**: 不 pool,保留 [T,1280] 帧序列 → transformer layer(s) with positional encoding → attention-pool → MLP head。这可以利用 whisper 帧间的时序动态 (音高变化、语速变化、静音间隙 → BC/T 信号)。**D-1 否的是 frozen VAP 整体 BC 弱,没否过 "用 whisper 帧序列训 transformer head 捕获 BC 时序信号"**。

  **但与 D-7 的一致性问题**: whisper 帧是冻结的 (D-7 已证冻结 whisper BC<0.22,只有 LoRA 能让 BC 0.267)。transformer head 读冻结帧序列 ≠ LoRA 让 backbone 可学 — 冻结帧的时序信息已被 whisper 本身编码好了 (whisper 是 transformer,hidden states 已含位置信息),再加 transformer 可能冗余。不过当前 mean-pool 丢时序是真实信息损失,transformer head 可能收回一部分。

**新建议**: 优先级排序 — (c) 阈值窄调 > (a) max+isotonic > (d) transformer head。三个叠起来可能 +0.003~0.006 cap1 → 在线 +0.001~0.004。

---

## Q2 D-13 三轨判断: 基本无误，B3d 有 1 次提交配额可试

**结论**: D-13 三轨的 SKIP 决策整体可靠,但 B3d 有 marginal 试错价值 (1 次提交配额,EV 略正)。

**理由**: `JOURNAL.md:312-317` 实测数字 + `DECISIONS.md:336-367` D-14 完整 chain-first。

- **(a) B3d 校准头**: D-14 "校准头无新源不涨 cap1"的结论**基本正确但过于绝对**。论据:
  1. OOF +0.031 (0.5701→0.6012) 是 real training gain — calibrating two probability sources on 179867 窗是有效的。
  2. cap1 = SOTA 0.6410 说明在 cap1 首窗上,校准无增益。但 noise floor ±0.003 意味着 cap1=SOTA 的 push **有可能**在线 +0.001~0.002 (抽样不确定性方向有利时)。
  3. B3d 的 CSV pos counts 用正确的 THR_VARF 后,与 SOTA CSV 的差异可能在 10-30 处 flip (不是 193 处),这 ~20 处翻转在 5000 个标签上可能碰运气 +0.001。
  
  **务实建议**: 若提交配额剩余 ≥5 次,B3d 值 1 次提交 (成本 0→烧 1 配额,收益 0→+0.002)。若配额紧张,SKIP 是对的。
  
  **不支持全线复活 B3 系列**: B3a (TTA) / B3b (pseudo-label) 需要独立验证,不能因 B3d 一个特例全复活。TTA 尤其可能降分 (扰动 whisper 帧→融合策略失效)。

- **(b) B1 v3 47 个 EDA 强特征**: **无误,不应复活**。
  1. OOF +0.0006 是**几乎字面意义上零**。不是"LGBM 不会用这 47 个特征",是信息增益已饱和 — 这 47 个特征 (runlen/burst/trans/diff) 都是从 46d 原特征的组合/导数/统计派生,信息重合度高。
  2. "改 CatBoost / 加 feature selection" 的 counter-argument: D-5 已证实 context 内算法集成 (LGBM/XGB/CatBoost/MLP) 不正交。换 CatBoost 单跑 vs LGBM 的单跑差异大概率 <0.002,且 D-12 LGBM sweep 已证 baseline 即最优。
  3. Normalization 对树模型几乎无影响 (树做 split 不依赖特征 scale)。

- **(c) 原 N1 本机用 whisper OOF probs**: **无误**。用 whisper OOF 概率 [N,5] + ctx 46d 训 head,本质上是 B3d 的变体 — 校准已有 OOF 输出,不触及 whisper 1280d 帧的原始信息。D-14 的教训直接适用: 无新源就不涨 cap1。这是在重做 B3d,必然同结果。

---

## Q3 N1' 是否抓对: YES — 成本极低，值得跑

**结论**: N1' 抓对了方向 (raw frame training + 新 loss ≠ B3d OOF calibration),有文献支撑,成本 2h 极低,**建议立即跑**。

- **(a) 方案匹配度**: **N1' 与 B3d 有本质区别,不是重复 D-14**。
  - B3d: 两个**已计算好**的概率源 (ctx_lgbm_v1 OOF [179867,5] + whisper OOF [179867,5]) → 2×5d = 10d input → neural calibrator → 5d output。训练目标是校准已有 BCE 输出。
  - N1': **原始 whisper 1280d 帧特征** + ctx 46d → neural head with DB-Loss+SupCon → 5d output。训练目标是**从音频帧直接学到更好的表达**。
  - 文献支撑: DB-Loss (Wu et al ECCV 2020) 在 COCO-MLT/VOC-MLT 多标签长尾上对比 BCE 有显著增益;SupCon (Khosla et al NeurIPS 2020) 对长尾类聚类效果在多个基准验证。**这两者都是在原始特征上训练时生效,不是在校准已训练好的概率时生效**。
  - B3d 失败因为输入已经是两个 BCE-trained model 的 OOF,新 loss 只能重排已有信号不能创造新信号。N1' 输入是 raw frames,新 loss 可以学到 BCE 学不到的 BC 微弱模态。

- **(b) 决策门**: **0.6289 基准合理但有细差**。
  - hubert head 0.6239 用的是 **stride40** (36104 窗),N1' 用 **stride5** (179867 窗 = 5× 数据量)。数据量差异可贡献 0.003-0.005 的 cap1,让 0.6289 不够保守。
  - **更合理的 gate**: stride5 whisper head 的 cap1 baseline 是多少? 从 `whisper-fusion-20260531-0143/probs.npz` 反推 whisper OOF cap1 macro ≈ 0.629 (CONTEXT §3.3 whisper cap1 ~0.629)。**N1' gate 应设为 0.629 + 0.005 = 0.634**,不是 0.6289。
  - 但即使 gate 再严,2h 成本不值得为 gate 调 0.005 而放弃。

- **(c) 机会成本**: **N1' 的机会成本极低**。
  - 2h = 1h GPU + 1h 本机 rsync/setup/orthofuse 重做
  - vs 复赛镜像准备 (可并行,不冲突 — 本机写 Dockerfile 时可同时等云端结果)
  - vs 评审新方向 (评审已快完成,不冲突)
  - vs B2 整通对话 (已否,成本 16 天 vs 2h)
  - **EV 计算**: 假如 30% 概率 +0.003 真分,70% 概率 SKIP → EV = 0.3 × 0.003 = +0.0009 macro。按提交成本 ~1/80 配额 + 2h → EV > 成本。**结论: 即使期望涨幅极小,跑 N1' 的微薄机会也优于不跑**。

---

## Q4 过早判死路线: 1 个值得重启 — D-3 文本路线(LLM 版)

**结论**: 5 条红旗经审查,仅 D-3 的"文本路线"值得用 LLM fine-tuning 方式重启,其余 4 条证据链完整。

- **(a) D-1 VAP**: **不值得重启**。VAP unfreeze 全量 fine-tuning 真分 0.6337 (−0.079 vs SOTA) 已经是使用 BC 标签重训整个模型的结果,不是只用 VAP 原 head 的 256 类标签。`DECISIONS.md:13` 明确写了 "全量 VAP unfreeze 微调真分 0.6337"。这与命题"微调 backbone + 重训 head 用本数据集 BC 标签"是一回事。VAP 的归纳偏置 (turn-taking prediction, not backchannel) 在 backbone 层面就抓不住 BC,不是 head 的问题。

- **(b) D-3 文本路线**: **最值得重启!** D-3 否的是 **sklearn TfidfVectorizer 词袋 + LGBM**,不是 **LLM semantic embedding fine-tuning**:
  1. **两者本质不同**: 词袋 = 词汇共现统计,无句法/语义;Qwen3-0.6B 末层 hidden state = 句法+语义+对话意图。ASR 文本里的 "嗯…好的" / "我明白了" / "那你觉得呢" 这些 turn-yielding / backchannel-inviting 信号,词袋捕捉不到,但 LLM 能。
  2. **关键风险 (需验证)**: D-3 的同款风险 — LLM 特征会不会也只在 cap1 (对话开头) 有效,test 任意切片上蒸发? 根因: 对话开头的 ASR 文本模式 (招呼/寒暄) 可能与中段/末段不同。
  3. **低风险验证方案 (2-3h 本机 MPS)**:
     - 取 30 通随机切片 (不只 cap1,含各处切片),提取 Qwen3-0.6B 末层 1024d
     - 训练 per-class 二分类 head → 5fold OOF F1
     - **关键验证**: 分别看 cap1 切片 vs 非 cap1 切片的 head F1 → 若差距大 = D-3 同款风险;若接近 = 新正交源成立
     - 若通过 → orthofuse 加 text head OOF 做 3 源融合 (ctx+whisper+text),期望 cap1 +0.005~0.010
  4. **算力**: Qwen3-0.6B 63ms/segment (CLAUDE.md) + 5fold head 训练 ~30min → 总 2-3h。**成本极低**。
  5. **为什么 D-3 当时没试**: D-3 之前的认知是"文本特征帮 T/I",用词袋试了发现 cap1 虚高就全否了。但当时正确的做法应该是: 词袋失败 → 升级到 LLM embedding,不是丢弃整个文本路线。

- **(c) D-6 融合上限 + LoRA hubert/w2v2**: **不值得重启**。`DECISIONS.md:82-88` D-7 已证 LoRA whisper 需要 cap5→cap20→full 数据 scaling,cap5 欠拟合 (线上 0.6155)。hubert 比 whisper 小几倍,LoRA 适配需要更少数据,但单源 baseline 0.6239 < whisper 0.629 → 即使 LoRA 成功,增量上限有限。且已经困于 cap1 天花板 0.6540 (D-10),再多源也不破。**计算: 若 LoRA hubert 让 I 从 0.532→0.58 (+0.048),T 从 0.639→0.67 (+0.031),cap1 macro 0.6540→~0.670。但需要 cap20+ 数据 (30-63h 全量,同 D-7 成本不可行)。**

- **(d) B2/Omni zero-shot**: **不值得作为主攻,但值得 1h 快速 probe**。Cycle 11 Omni-3B zero-shot "全答是" 是严酷负面信号,但 ①Omni-3B≠Omni-7B (架构重做,CLAUDE.md 新认知),②cycle 11 是 zero-shot 文本 prompt 没做 prompt engineering。**值得用一个精心设计的 prompt (per-class 5 轮独立二分类,带 ASR 文本+音频上下文) 对 100 段 test 做快速 probe**,测 per-class recall/precision。若任一类 F1>0.3,投入更多;否则 0 成本放弃。1h 云推理 (100 段@~5s/段=8min + prompt 迭代) 值得。

---

## Q5 完整新路线 (0.0091 缺口真路径)

### Top 方向排序

**1. 文本 LLM fine-tuning (Qwen3-0.6B) → 第三独立信号源 for T/I (期望 +0.003~0.007, 成本 2-3h 本机)**
- 理由: ASR 文本含句法/语义对话轮换线索 (句末完整体/问句/邀请发言标记),是完全正交于 context 标签时序 + whisper 音频的第三模态。D-3 的词袋失败不否定 LLM embedding。Q4(b) 详述。
- 风险: D-3 同款 (ASR 文本模式在 cap1 vs test 任意切片可能分布不同) — 但可通过 OOF 5fold validation on diverse slices 验证泛化性。
- 实施: 本机 MPS, Qwen3-0.6B 提取 1024d → 5fold simple MLP head → 验证 cap1 F1 是否≥0.646 (Push 门)。若 pass → orthofuse 3 源融合 (ctx+whisper+text)。

**2. N1' 云上 DB-Loss+SupCon on whisper raw frames (期望 +0.001~0.004, 成本 2h 云)**
- 理由: DB-Loss+SupCon 从原始音频帧训练,与 B3d OOF 校准本质不同。文献支撑充实 (ECCV 2020 + NeurIPS 2020)。即使 cap1 不涨,在线可能因 better calibration of long-tail 类边际改善。
- 风险: D-14 "无新源不涨" 可能成立 (whisper frames 已是 whisper head 训练过的,换 loss 可能效果有限)。成本低不怕失败。
- 实施: 已写好 `cloud/train_head_n1.py`,待 rsync。

**3. SOTA 榨油组合: 阈值窄调 + max() strat + isotonic 校准 (期望 +0.001~0.003, 成本 0.5h 本机)**
- 理由: Q1(c)+(a) — per-class 阈值微调配合 max()/isotonic。成本极低,可与 #1/#2 并行。
- 实施: `cycle_orthofuse.py` 加 max strat + isotonic step + ±0.05 阈值窄扫。

**4. Omni-7B zero-shot probe (期望 0~+0.005, 成本 1h 云)**
- 理由: Cycle 11 失败可能因 3B + 弱 prompt。7B + per-class 独立 prompt 可能不同。但风险高 (Omni 对 turn-taking 没训练信号)。
- 实施: 100 段 test probe,若任一类 F1>0.3 再投入。

### 最优先 1 个: 文本 LLM (Qwen3-0.6B)

**实施步骤**:
1. 本机提取: 随机抽样 train 数据 (100 通,每通取 cap1 + random 5 slices,共 ~600 slices),用 Qwen3-0.6B 读 ASR JSON 提取末 token 1024d → `text_emb.npz`
2. EDA: 训练 per-class simple logistic regression on text 1024d → 5fold OOF F1。分别报告 cap1 切片 vs non-cap1 切片的 per-class F1 → 看泛化 gap
3. 若 gap < 0.03: 全量提取 (去重 unique context 约 8.3%, ~126min 本机)
4. 5fold MLP head → OOF probs → `cycle_orthofuse_nsrc.py` 三源融合 (ctx+whisper+text)
5. Push 门: cap1 ≥ 0.6460

---

## Final 综合建议

**D-15 决策建议**: **PIVOT — 开文本 LLM 路线 + 并行 N1'**

当前任务"等用户决定开云机"是被动的。主动策略应该是:

1. **本机立即启动** 文本 LLM probe (2-3h,不阻塞云决策)
2. **用户开云机后** rsync `train_head_n1.py` 上去跑 N1' (2h)
3. **两路并行出结果后** orthofuse 融合

理由:
- 文本 LLM 是 D-1~D-14 全关闭的唯一**真正新模态** — 不是"加第 N 个音频源"(红旗),不是"context 内同源集成"(红旗),不是"cap1 选 strat"(红旗)
- N1' 成本 2h,即使失败也把 B4 Knowledge Layer 的 N1 方向闭合,不留"万一能涨"的悬疑
- 两路预算: 本机 3h + 云 2h + 提交配额 2-3 次 — 在 16 天内微不足道
- 若两路全失败,0.71529 才是**诚实接受**,不是过早放弃 (D-12→D-13→现在才算"真试过所有合理新方向")

**第一步行动**: 写 `tools/climb/cycle_text_llm_probe.py` — Qwen3-0.6B 文本嵌入提取 + 5fold head EDA,本机跑 2-3h。
