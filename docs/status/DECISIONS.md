# DECISIONS — FinVCup Turn-Taking 架构决策账本

> 带 rationale 的架构/范式决策（"选 X 不选 Y 因为 Z"）。negative cache 细节见 MEMORY reference_negative_cache.md / research-tree falsified 段。

## 2026-05-30

### D-1: VAP/CPC 音频路线整条证伪，停止投入

**决策**: 放弃 VAP（及所有纯音频编码器）攻 BC 的路线。

**Rationale**:
- 全量 VAP unfreeze 微调真分 0.6337（−0.079 vs SOTA 0.7124），BC=0.222 仅打平 LGBM 基座。
- 信号探针（纯前向，800 切片）证 VAP 预训练 head 原生信号（p_now/p_future/256类/vad/熵）对 BC 区分度 **|r|<0.04**——不是"没用对 head"，是 VAP 归纳偏置本身抓不住 BC。VAP 原文 `objective.py:325` backchannel 预测是未完成 TODO，作者自己没解决。
- 与 mel/whisper 三连败同根：纯声学/语音活动表征预测不了 backchannel 时机。

**含义**: 不再试 HuBERT/MMS 换 encoder（同属音频路线，同盲区）。BC 可预测信号在 context 标签时序（r≈0.13），不在音频。

### D-2: BC 攻击战略从"硬攻 BC"转"攻 T/I"

**决策**: 不把算力全压最难的 BC，转攻 T(0.542)/I(0.434)——更确定的 macro 增量。

**Rationale**:
- 全战场诊断：等权 Macro-F1 下每类 +0.05 对 macro 贡献相同（÷5），但难度天差地别。
- BC 单源三方堵死：context 时序已榨干（F1 顶 0.21=信息上限）/ 音频 r<0.04 / 词袋否。提 0.08 是撞信息论上限的硬仗。
- T/I 远未饱和 + negative cache 有证据文本特征帮过 T/I。干净对照（baseline v1 底）确认 T+0.038/I+0.048，C/NA 不动。
- BC 暂搁置为"信息论上限"，非永久放弃——若 T/I 突破后仍需 BC，再回头找语义触发信号。

### D-3: T/I 文本路线证伪（CV 虚高不泛化）

**决策**: 放弃文本词汇/语义特征攻 T/I。

**Rationale**:
- ti-robust（修标点 bug 后）真分 **0.6392**，cap1 CV 0.6358≈SOTA 0.6402，**但 gap 仅 +0.003**（变体F gap +0.072）。同样的 cap1 CV，变体F 线上涨 0.072，文本几乎不涨 → **cap1 增益虚高，不泛化到 test**。
- 文本特征过拟合 cap1（对话开头切片）的局部分布，test 各处切片增益蒸发。
- 印证旧 H-T2 当初"CV+0.004 但 test 预测剧烈偏移可疑→没提交"的判断是对的。
- 教训：被 OOF +0.0217 大数字带偏，没充分尊重阈值铁律"CV 高分≠线上高分"。

### D-4: BC 确认信息论上限 ~0.22，停止单点攻坚

**决策**: 停止 BC 单点攻坚（所有角度已交叉验证到顶）。

**Rationale**（完整证据链，多构型/多数据规模/多信号源交叉验证）:
- context 时序（含导数/突发性/周期）→ 0.21 到顶（baseline 46维已榨干）
- VAP mean-pool 0.222 / attention-pool 0.080（全量单变量对照，pool 非瓶颈）/ 微调 / 冻结 → 全 ≤0.22
- F0/pitch（音频对 BC **最强**分支 |r|0.128）融合 → +0.005（正交但增量太小）
- 文本词汇 → OOF 虚高不泛化（D-3）
- whisper/mel → <0.22
- **所有信号源 r≈0.13，无强信号，叠加（context+F0）仅 +0.005** → BC 在此数据集接近信息论上限，"未来2s会不会backchannel"本身高度难预测。

**含义**: 不再投入 BC 单类。转务实：用已验证最优（变体F 0.7124）+ 零风险稳健化 + 接受 BC≈0.22。

**方法论教训（用户 6 次质疑均驱动了一个验证，全收敛同结论）**: ①音频证伪过快→F0 确是最强分支但弱 ②"同分布=拟合test"→揪出 T/I CV 虚高 ③构型/参数→mean-pool 反优于 attention-pool ④哪个音频分支→F0/pitch ⑤数据增强→AUC 不升反降 ⑥高维映射→核/RFF/MLP 全输线性。补充证据（9+ 角度）：高维可分性 AUC 0.64（样本量 40→150 通不变）、序列/计数框架 0.206<0.212、Omni encoder=WhisperFeatureExtractor（=已证伪 whisper）、Omni LLM zero-shot 无判别力（全答是）。

### D-5: 卡 0.712 的根因 = 单一有效信号源；融合救不了

**决策**: 接受当前条件下融合无正增益；瓶颈是缺第二个强且正交的模型，非融合方法。

**Rationale**（实测）:
- 弱模型融合（vap 0.634/whisper 0.671 + 变体F）→ 估 −0.02~−0.05（NA/I 病态分布污染）
- 现有 context 变体（E/F/G/cycle1）→ 0（不正交，cycle1 vs F 仅 24 处不同）
- 算法正交集成（LGBM/XGB/CatBoost/MLP over context）→ −0.023（三树不正交 + MLP 坏 BC=0 拖累）
- 变体F 本身已是 5seed 集成，10seed 无增益（同范式榨干）
- **融合不是独立杠杆，依赖先有多个强模型。我们只有 context-LGBM 一个强信号源，音频/文本/序列全证伪 → 没有第二个强且正交的可融。**

**榜单框架修正（用户提供分布：前10 0.73-0.75 / 11-20 0.72-0.73 / 我们 0.712）**: BC 不是天花板（榜首也没把 BC 做高），领先来自全类各榨一点 + 融合 + 工程，增益 1-3 分。但这要求有多个强模型——这正是缺的。**未解的真问题：如何造出第二个独立强信号源**（context 之外）。

## 2026-05-31

### D-6: 模型融合冲 0.75 — context 内融合证伪，跨源(whisper)正交是真路径

**决策**: 放弃 context 信号源内部的算法/特征融合；转 context × 跨源音频(whisper/VAP) 的 per-class 正交融合。

**Rationale（实测 + nested 验证）**:
- **context 内融合 nested 证伪**: 4 成员(lgbm_v1/xgb_v1/lgbm_v2/mlp)cap1 全在 0.622-0.624 不正交。per-class grid 加权 in-sample cap1 0.6655 看着 +0.0428，但 **nested-CV 泛化只 0.6198 < base 0.6228**（BC 0.364→nested 0.200 全是过拟合蒸发，同 ti-robust 陷阱）。等权/stacking 也无真增益。grid 在 369 cap1 样本搜 5^4 权重 = 变相调参。
- **发现真正交杠杆**: 逐类核对 cloud-whisper cap1 **T=0.667 > context 0.625 / I=0.555 > context 0.539**——音频在 T/I 有 context 没有的信号。之前 whisper 整体判死(0.671<0.712)是 BC 拖累 + 整体弱，但**逐类 T/I 强**。这是 D-5"全类各榨一点"从未用过的点。
- **方法论**: 跨源融合用**固定权重凸组合候选**(ctx/whisper/eq/w70/w30)，无 cap1 权重搜索(防 grid 过拟合)，per-class 选最优 + 保守 +0.003 门。

**含义**: 融合的价值依赖正交性，正交性来自**不同信号源**(音频 vs context)，不来自同源换算法。冲 0.75 的路径 = 多个独立信号源 × per-class 借强。

### D-7: BC 信息论上限 0.22 需松动 — "冻结路线下 0.22，可学 encoder 下 0.267 但成本不可行"

**决策**: 修正 D-4 的绝对表述。BC 不是数据信息论到顶，是**可学 encoder + 大数据**的工程受限。

**Rationale（reconcile D-4 与诊断报告的矛盾）**:
- D-4 的"BC 信息论上限 ~0.22"全部基于**冻结**编码器(VAP 各 pool / 冻结 whisper / mel)+ context 时序测得。
- 但诊断报告硬证据: **LoRA 让 whisper encoder 可学后，BC cap5-CV 顶到 0.267**(全项目最高 > LGBM 0.222 > 冻结 whisper 0.20)。可学 encoder 才顶得上去 = **音频里确实有更多 BC 信号，冻结提不出**。
- **但兑现成本不可行**: 全量 LoRA 30-63h(193ms/前向, 5fold×全量)，cap5(1845样本)欠拟合 → 线上仅 0.6155。

**含义**: BC 不是永久放弃，是"现有算力下 0.22 是冻结路线极限"。若未来有更高效的可学 encoder 路径(更快前向 / 更小 encoder / 蒸馏)，BC 0.22→0.27 仍是 +0.01 macro 的活路。当前阶段先吃 T/I 跨源正交(更确定)。

## 2026-05-31

### D-8: 跨源融合范式撞 0.715 上限 — 加 hubert 第三源无线上增益，撤 cycle 17 扩容计划

**决策**: 停止"加更多音频源融合"路线。不扩 stride8 hubert，不投 chinese-w2v2 第四源。融合范式天花板 = 0.715 (跨源 ctx×whisper, 上限锁定)。

**Rationale（cycle 16 三源真分实测）**:

| run | cap1 | 线上 | gap | 备注 |
|---|---|---|---|---|
| orthofuse (双源, stride40 弱基座) | 0.6410 | **0.71529** | +0.0743 | 旧 SOTA, gap 异常高 = cap1 过拟合 |
| orthofuse-s5 (双源, stride5 强基座) | 0.6455 | 0.71233 | +0.0668 | 强基座反而线上 -0.003 |
| **orthofuse-3src (ctx+whisper+hubert)** | **0.6540** | **0.71523** | +0.0612 | 三源 cap1 +0.013 vs 双源 → 线上几乎同分 |

**核心模式**: 三个 push cap1 涨 0.0045 → 0.0130，但**线上紧锁 0.712-0.715 窄带**。cap1 收益无法转化为线上。

**根因分析（chain-first）**:
1. **gap 越大代表 cap1 越虚高**（旧 orthofuse gap 0.0743 > 三源 0.0612），多源后 cap1 369 样本估计噪声加剧。
2. **hubert 在 test 切片末分布上贡献蒸发**：cap1 (per-class选 hubert/三源等权) 0.6540 - 双源 0.6410 = +0.013 真增益（无 grid），但 test 切片末分布上消失。
3. **不是过拟合（无 grid 搜索）**，是 **train (stride 全切片) vs test (切片末) 的分布差异**让 hubert/whisper 信号失活——同 ti-robust 文本路线证伪根因（D-3）。
4. **多源 = 更多分布敏感特征叠加 = 更难泛化**，与 cap1 收益方向相反。

**含义**:
- **真 SOTA 仍 = orthofuse-20260531-0319 = 0.71529**（双源 stride40 弱基座，偶然 cap1 估准 + 简单融合泛化好）
- **冲 0.7285 前 10 不能靠加音频源融合**（已饱和）
- **cycle 17 候选取消**: 不扩 autodl-fs 200→400G, 不投 chinese-w2v2-large 第四源（守 validate_before_full_run），下载好的 w2v2 模型留备但不投训练算力。
- **新方向需求（Knowledge Layer 触发）**: 既不是"加更多音频源"（同 D-3/D-8 撞 train/test 分布差），也不是"换 LGBM"（同 D-5 单源问题）。可能方向 = ①后处理（test 切片末专属规则）②半监督（用 test 自身分布对齐）③ test-time adaptation。这些都未试。

**红旗**: 任何下一步若发现自己又在"加第 N 个音频源" → STOP，已 D-8 闭合。要走非"加源"路径。

### D-9: D-8 根因 chain-first 第二诊断 — 不是"分布差"，是 test 抽样噪声 + 真信号 ≤0.003

**决策**: 修正 D-8 的"train/test 分布差"判断。撤销基于此判断的 cycle 17 候选（D 方向 "切片末加权"）。新决策 = **接受 SOTA 0.71529，转复赛镜像准备**（best-effort 模式诚实兑现）。

**第二诊断（仔细看 4 个真分）**:
| run | cap1 | online | gap | rel SOTA |
|---|---|---|---|---|
| variant-F (单模 5seed) | 0.6402 | 0.71242 | +0.0722 | baseline |
| orthofuse (双源弱基座 + whisper) | 0.6410 | 0.71529 | +0.0743 | **+0.003** ← whisper T/I 真信号 |
| orthofuse-s5 (双源强基座 + whisper) | 0.6455 | 0.71233 | +0.0668 | -0.003 ← 强基座+whisper 冲突 |
| orthofuse-3src (三源 + hubert) | 0.6540 | 0.71523 | +0.0612 | +0.0001 (噪声) ← hubert 0 增益 |

**真模式**:
1. **真信号最大 = whisper T/I 跨源 +0.003**（4 push 中唯一可重现的 SOTA 跳点）
2. **cap1 vs 线上 noise floor ≈ 0.003**（hubert cap1 +0.013 → 线上 +0.0001 噪声）
3. **3 个不同 push 线上分散在 [0.71233, 0.71529] 0.003 内 = 接近 test 1000 段抽样不确定性**
4. **不是 train/test 分布差**（context shape 验证完全同 375 chunk），是单次 push test marginal 噪声 ~0.003 淹没了 < 0.003 的真信号

**含义**:
- 前 10 门槛 0.7285 vs SOTA 0.71529 = 差 0.0135
- 已有路径最大单次贡献 = whisper T/I +0.003，要达成 0.7285 需 **4-5 个独立 +0.003 量级真信号**
- 已穷尽方向（D-1~D-8）每路径预期增量都在 noise floor 附近，多次 push 无法稳定累积
- **诚实判断: 前 10 在初赛阶段已不可达**。继续 push 是浪费提交配额 + 模型迭代时间
- **正确动作: 保 SOTA orthofuse-20260531-0319 = 0.71529 作初赛终态, 转**:
  - ① 准备复赛镜像（Docker, 含 ctx-LGBM-stride40 + whisper-cloud-head + orthofuse-3src 完整 pipeline, 需 hubert 也带上验证用）
  - ② 写复赛 README / 配置 / 推理脚本
  - ③ 报备非 Qwen 模型（chinese-hubert 用于 orthofuse-3src，2026-06-10 前 xinyebei@xinye.com）— 即使不破 SOTA 也是合规要求

**红旗**: 不要再启动 cycle 17 "切片末加权" / "TTA" / "半监督" — 这些都是基于 D-8 错诊断的方向，第二诊断已撤。

**multi-AI quorum 反思**: cycle 17 3/3 quorum 投 D 基于我错的 prompt（"train/test 分布差"为前提）。AI quorum 不是真相裁判，只在前提正确时有效。再 chain-first 重读数据时，4 个 push 的线上分散度直接证伪"分布差"假设。教训: quorum 前自己先 chain-first 跑透，避免共识偏置 (group think) 放大错诊断。

### D-10: 5 源融合 cap1 实测上限 = 0.6540（3 源即顶，加 w2v2/e2v 0 贡献）

**决策**: 关闭"加更多音频源" cycle 17. 用户撤 D-9 投降后投入 ~3h 云时间跑 w2v2+e2v 第四第五源, 实测无任何增量, 把 D-8 从 3 源量级推到 5 源量级证实, 不再尝试任何"加第 N 源" 路线。

**实测数据 (4/5 源 orthofuse decision gate)**:

| 组合 | cap1 macro | gain vs ctx | margin vs (N-1) src | 被选源/策略 |
|---|---|---|---|---|
| ctx-only | 0.6228 | — | — | base |
| 2源 ctx+whisper | 0.6410 | +0.0182 | — | T=ctx_w_70, I=whisper |
| 3源 +hubert | 0.6540 | +0.0313 | **+0.0131** ★ | T=ctx_w_h_eq, I=ctx_h_eq |
| **4源 +w2v2** | **0.6540** | +0.0313 | **+0.0000** ❌ | 同 3 源 (w2v2 无类被选) |
| **5源 +e2v** | **0.6540** | +0.0313 | **+0.0000** ❌ | 同 3 源 (e2v 无类被选) |

**单源 cap1 对照**:

| 源 | cap1 | T | I | BC | C | NA | 关键 |
|---|---|---|---|---|---|---|---|
| ctx | 0.6228 | 0.621 | 0.455 | 0.200 | 0.975 | 0.863 | base |
| whisper | ~0.629 | **0.667** | 0.509 | 0.182 | 0.975 | 0.859 | T 最强 |
| hubert | 0.6239 | 0.639 | **0.532** | 0.000 | 0.974 | 0.864 | I 强但 BC 崩 |
| w2v2 | **0.6395** | 0.611 | 0.543 | 0.200 | 0.975 | **0.869** | 单 macro 最高但融合 0 贡献 |
| e2v | 0.621 | 0.622 | 0.491 | 0.154 | 0.968 | **0.870** | 副语言独特但被 whisper/hubert 覆盖 |

**根因分析 (D-8 在 5 源量级证实)**:
1. **ctx 类强度 + gate +0.008 即门槛**: 任何新源在某类 cap1 必须 >= ctx 类 + 0.008 才被选。w2v2 各类全部 <whisper, e2v 各类大多 <hubert，被先占类排除
2. **cap1 369 样本上限**: cap1 macro 0.6540 是 3 源时 T=0.676 + I=0.557 + 其它 ctx 同的算术结果，再加源不会进一步改善这 5 个 per-class 数值（whisper+hubert 已是该类最强源）
3. **同范式 SSL 撞墙**: hubert/w2v2 同 WenetSpeech/TencentGameMate 系列 = 高相关, 不互补; e2v 副语言情感不同范式但任何类未超 whisper/hubert
4. **真分大概率撞 0.71523**: cycle 16 三源真分 0.71523, 4源/5源 cap1 完全同 3 源 → 4源/5源真分**必然同 0.71523** (cap1 同 + per-class strat 同 → test 概率同 → 0/1 csv 同). 不提交浪费配额

**实测投入**: ~3h 云时间 (extract + 2 head train) + 1.2G + 360M 模型下载 + 本机写 2 个 extract 脚本

**实测产出**: D-8 D-9 在 5 源量级证实，无任何提交价值。**诚实总结 = cycle 17 烧资源换 paradigm 闭合**。

**新红旗**: 不要再写 "extract_<新模型>_cuda.py" 脚本. 加任何第 N+1 源 cap1 都不会超过 0.6540. 唯一可能突破路径需要的不是"换音频源", 是**改 ctx 基座本身**或**改融合算法/策略空间设计** (cap1 369 样本上的统计上限)。

**含义 (重新激活 D-9 但保留路径)**:
- 前 10 门槛 0.7285 vs SOTA 0.71529 差 0.0135, 已穷尽方向真信号最大 +0.003 (whisper T/I)
- **重读 D-9**: "已穷尽方向" 现在 = 5 源融合 + 多模型, 实测加源天花板 0.7152 不能破 0.7285
- 守 SOTA orthofuse-20260531-0319 = 0.71529 作初赛最强提交
- **如果用户还想继续**: 真正未试 = ① ctx 基座升级 (XGB/CatBoost/MLP 替 LGBM v1; 但 D-5 已证 4 成员不正交) ② 融合策略改 (per-class isotonic 校准, BC 专用 boost) ③ 接受 0.71529 转复赛镜像

### D-11: cycle 18 BC cap1 +0.108 = cherry-pick (9 样本不可信), 关 BC 单类替换路线

**决策**: 关闭"在 cap1 上选 BC 单类替换策略"的路径. cycle 18 实测 BC 改 mlp+whisper_70 真分 0.69358 = -0.022 vs SOTA 0.71523 (完全归因 BC 一类 F1 跌 ~0.11).

**实测 (chain-first 三层诊断)**:

| 提交 | BC pos | 真分 | Δ vs SOTA |
|---|---|---|---|
| orthofuse-3src (SOTA) | 27 | 0.71523 | base |
| **cycle18 (BC 改 mlp+whisper_70)** | **17** | **0.69358** | **-0.022** |

vs orthofuse-3src 唯一差异 = BC 单类 (27→17, 16 真 BC→neg / 6 非 BC→pos = 净砍 10 + 22 处翻转)。其它 4 类完全相同 pos count.

**cap1 上看好的真实根因**:
- cap1 BC 总正例 9 个 → mlp+whisper_70 strat 上 4 pred / 2 TP / 7 FN → F1=0.308
- lgbm ctx strat 上 1 pred / 1 TP / 8 FN → F1=0.200
- **+0.108 = 多 1 个 TP 在 9 样本上的 F1 跳跃**, 而非"mlp 真学到 BC 信号"
- test 1000 段 BC 真正例量级估计 ~25-50, mlp 选 17 pred 高假阴率 = 真 BC 漏 16 个

**为什么 lgbm strat 在 cap1 看着弱 (BC F1=0.200) 但 test 上对 (BC pos=27, 真分高)**:
- lgbm ctx 在 cap1 BC=0.200 的 P=1.0 R=0.111 → **召回低但 precision 高**, test 上扩展性好
- mlp 在 cap1 BC=0.308 是 P=0.5 R=0.222 → **precision 已经很低**, test 上 FP 大爆发

**含义 (累积 D-3/D-9/D-11 同根 lesson)**:
- **cap1 369 上稀有类 (BC 9 正例 / I 60 正例) 的 strat 选择本质是过拟合验证集**, 不论是 grid 搜权重 (D-7) / OOF +0.0217 大数字 (D-3 ti-robust) / 还是 +1 TP 跳 F1 (D-11)
- 唯一稳定 cap1→test 转化的增益类型 = **多源融合在 T (150 正例) / I (60 正例) 中等样本类的真实信号叠加**, BC (9 正例) cap1 增益**永远不可信**
- 不再追 cap1 BC 增益 (任何形式), 守"BC 用 ctx-only strat" 死规则

**红旗**: 任何 cycle 看 cap1 BC > 0.22 时, STOP, 直接拒该 strat. 这是 D-3/D-9/D-11 累积学到的极硬约束.

**ctx 基座升级路线状态**:
- xgb_v1 / lgbm_v2 fused 0.6529 略低 lgbm_v1 0.6540 → 单独换 base 无收益
- mlp_v1 fused 0.6745 的 +0.0205 全是 BC cap1 cherry-pick → 真分反烧 -0.022
- **结论: ctx 基座升级 (cycle 18) 全证伪**, 守 lgbm_v1 base 不动

### D-12: cycle 19 所有 ctx-内方向全证伪 — 初赛已达个人天花板 0.71529

**决策**: 关闭所有 ctx-内攻击路线 (LGBM sweep / 融合策略改 / mlp 子策略). 接受 SOTA **orthofuse-20260531-0319 = 0.71529** 作初赛终态. 转复赛镜像准备.

**Rationale (cycle 19 双路线完整证伪)**:

**19c (T/I 用 mlp 子策略, 守 BC=ctx 死规则)**:
- T (150 正例) SOTA strat=ctx_w_h_eq cap1=0.676, **任何含 mlp 的 strat 最高 0.635 (-0.04)**
- I (60 正例) SOTA strat=ctx_h_eq cap1=0.557, **任何含 mlp 的 strat 最高 0.475 (-0.08)**
- mlp 在 T/I 上系统性弱 ~0.04-0.08, **不只是 BC 噪声, mlp 整体就比 lgbm 弱**
- 不浪费提交配额, 立即弃 19c

**19b (LGBM 超参 sweep, 用 OOF full 选避 D-3 cap1 cherry-pick)**:
- quick 4 组合 stride40 5-fold OOF: baseline (300/0.05/31/1.0) full_macro 0.5909 = 最高
- 增 n_estimators / 增 lr+leaves / 加 feat_frac → 全部 full_macro 略降
- cap1 上 baseline 0.6268, 其它最高 0.6290 (+0.002 < +0.005 gate)
- **gate 未过, baseline 是最优 = LGBM 在当前 46d 特征+stride40 OOF 全量上已饱和**
- 全量 36 组合大概率同结论 (主要轴向已覆盖), 不浪费 ~70min

**累积 cycle 16-19 全路径汇总**:

| Cycle | 方向 | 决策门结果 | 真分代价 |
|---|---|---|---|
| 16 | 三源融合 (ctx+whisper+hubert) | ✅ cap1 0.6540 → 真分 0.71523 ≈ SOTA | 0 (反正同 SOTA) |
| 17 | 加 w2v2/e2v 4/5 源 | ❌ cap1 锁 0.6540 (D-10) | 0 (不重复提交) |
| 18 | ctx 基座 mlp BC | ❌ cap1 +0.108 cherry-pick → 真分 -0.022 (D-11) | -0.022 烧 1 提交 |
| 19c | T/I mlp 子策略 | ❌ mlp 全类系统性弱 | 0 (cap1 已说差) |
| 19b | LGBM 超参 sweep | ❌ baseline 即最优 (cap1 饱和) | 0 |

**真 SOTA = orthofuse-20260531-0319 = 0.71529** (双源 ctx + whisper)
- 前 10 真门槛 0.7285, 差 0.0135
- 已穷尽所有路径, **每路最大可能增益 < 0.005, 凑不到 0.014**

**含义 (诚实总结)**:
- **初赛个人天花板 = 0.71529**, 进前 10 在剩余路径上**确认不可达**
- 接受这个数字, 不再烧云时间/提交配额
- **转复赛镜像准备**:
  1. Docker 镜像 (含 ctx-LGBM stride40 + whisper-cloud-head + orthofuse 跨源融合 pipeline)
  2. 推理脚本 (单段输入→输出 5 列 CSV)
  3. README + 数据约定文档
  4. 报备 chinese-hubert/w2v2/e2v 非 Qwen 模型 (2026-06-10 前 xinyebei@xinye.com)
     - 即使三源未破 SOTA, 用过的模型仍需合规报备
     - 邮箱身份 = 问用户取真邮箱 (非 userEmail)

**红旗 (本路径永久关闭)**:
- 不再启动任何 "加更多模型源" cycle (D-10)
- 不再启动任何 "在 cap1 上选 strat" cycle (D-3/D-9/D-11)
- 不再启动 LGBM 超参 sweep 任何变体 (D-12)
- 若用户后续仍想突破, 真正未试 = 改特征工程 (新 46d 改进版 → 重训整个 ctx base) 或彻底换架构 (整通对话神经预测), 不在现 cycle 套路内

## 2026-06-01

### D-13: 撤 D-12 接受论, 激活前 20 攻坚（目标 0.7243+, 三轨并行）

**触发**: 用户 6/1 上午通知 — 当前 SOTA 0.71529 在排行榜**第 37 名**（前 40 进复赛），距前 20 门槛 **0.724337 差 0.0091**, 距前 40 安全线非常危险。**目标修正: 必须冲到 0.7243+ 保证进前 20**, 不再"接受 0.71529 转复赛镜像准备"。

**Rationale (D-12 表面成立但战略前提变了)**:
- D-12 写的是"已穷尽**规划路径** + 每路 <0.005 凑不到 0.0135 (前 10 门槛)" — 这个**算式仍然成立**, 改的是输入: 0.0091 (前 20) 比 0.0135 (前 10) 容易, **2 个独立 +0.005 真信号即可达成**, 不是 4-5 个。
- D-12 红旗自己写过"真正未试 = ①改特征工程 ②整通对话神经预测", IV.B 补 ③后处理 / ④Knowledge Layer。**4 条都没真跑过, 不该判死。** 之前判死是"为 0.0135 缺口判死", 不是"为 0.0091 判死"。
- 风险评估: 第 37 名距前 40 仅 3 名 buffer, 其他队伍 6/2-6/16 还在 push, 我们不动会被挤出, **不冲是更大风险**。

**新战略 (三轨并行, 0 算力轨先走)**:

1. **B4 Knowledge Layer (今天起, 0 算力, 最高 ROI)**: consult-AI / 2025-2026 turn-taking SOTA 论文 / 类似比赛技术分享 → 找 D-1~D-12 范围**外**的全新正交信号源（domain 知识）。可能省 5 天死路, 也可能否定 B1/B2 投入。**1 天内必出方向判断**。
2. **B1 ctx 特征工程 v3 (本机, 中期 1-2 天)**: 改进现 46d context 特征 (新增导数/突发/跨声道韵律统计/对话动力学), 重训整个 4 ctx base + 重做 orthofuse 跨源融合。**关键避 cap1 陷阱 = 保数据规模, 不在 cap1 选 strat**。期望 +0.003~0.010。
3. **B3 后处理 / TTA / 半监督 (本机, 短期 0.5-1 天单跑)**: 用 test 自身分布做 self-distillation / pseudo-label / TTA。期望 +0.001~0.005 偏小但成本极低, 可叠加任何 base。**最早可单独 push 验证**。
4. **B2 整通对话神经预测**: 视 B4 结果定 — 如 B4 发现全新正交方向且需要架构换, 启动; 否则 16 天来不及做完整 5fold。

**铁律保留 (D-1~D-12 不撤)**:
- ❌ 不再"加第 N 个音频源" (D-1/D-8/D-10) — w2v2/e2v 不动
- ❌ 不再"在 cap1 369 上选 strat" (D-3/D-9/D-11) — 阈值搜索 / per-class grid / BC 单类替换全禁
- ❌ 不再"context 内同源算法集成" (D-5) — 4 成员不正交
- ✅ 唯一允许的 cap1→线上转化路径 = **多源融合在 T (150 正例) / I (60 正例) 中等样本类的真实信号叠加**

**Push 门重新校准 (基于 D-9 noise floor)**:
- cap1 vs 线上 noise floor ≈ 0.003 (4 个 push 实测)
- 要 push 必须 cap1 macro **≥ SOTA cap1 0.6410 + 0.005** 才有 +0.001 线上信号期望
- 要破前 20 要 cap1 macro **≥ 0.66** (相当于 +0.025 vs SOTA cap1) 才有 +0.009 线上期望
- 仍是保守门, 但比"接受 0.71529"积极得多

**配额预算**:
- 剩余 5×16 = 80 次提交, 实际不可能用完。按"每路径 1-2 次决定生死"分配:
- B3 后处理: 2 次 (sliced TTA / pseudo-label) — 1 周内
- B1 ctx v3: 2 次 (v3 base 单跑 / v3+orthofuse) — 第 2 周
- B4 触发的全新方向: 2-3 次 (未知)
- 总用量预算: 6-7 次, 充裕

**Push 触发**: 任何 cycle cap1 ≥ 0.6460 (= SOTA + 0.005) → 直接 push。低于则 SKIP-advance 下一个 hypothesis (climb §5 best-effort 自动决策)。

**预期路径 (mlestone)**:
- 6/1: B4 Knowledge Layer 出方向判断 + B3 后处理草案
- 6/2-6/4: B3 后处理 push 验证 (期望 +0.001~0.005)
- 6/4-6/8: B1 ctx v3 + 视 B4 启动新方向
- 6/8-6/10: 合规报备邮件 (硬截止) + 第二轮 push
- 6/10-6/16: 最后一周 polish + 预备复赛 Docker 草稿

**Reconcile D-12 与 D-13**: D-12 的"路径穷尽"判断没错, 错在用前 10 门槛标尺判定"不可达"。D-13 = 改用前 20 门槛重新评估同样路径 → 2 条独立 +0.005 即可。**D-12 关闭的红旗仍生效**（cap1 选 strat / 加源 / context 内算法集成 都不复活）, **D-12 自己列的"真正未试"4 条**才是 D-13 的攻击面。

**风险记录**:
- B4 可能空手而归 (turn-taking domain 知识可能没有 2024-2026 新突破)
- B1 ctx v3 增益可能 <0.003 (现 46d 已榨干假设可能成立)
- B3 后处理增益可能 <0.001 noise floor
- 三轨全死 → 真分仍 0.71529 → 排名仍 37 → 进前 40 看其他队是否撞墙

**红旗 (D-13 失效条件)**: 三轨全跑完 cap1 都 <0.6460, 或 push 2 次线上无 +0.003 提升 → D-13 失效, 回 D-12 接受 0.71529 + 寄希望其他队不动。

### D-14: B3d DB-Loss+SupCon 校准 OOF 涨 +0.031, cap1 = SOTA → SKIP-advance + chain-first 阈值/cap1 定义 bug 捞出

**触发**: 6/1 上午执行 B4 → B3d cycle (DB-Loss + SupCon 校准头 on [ctx_lgbm_v1, whisper] 10d OOF).

**实测数字**:
- B3d OOF (179867 stride5 GroupKFold) macro F1 = **0.6012** (+0.031 vs ctx_lgbm_v1 OOF baseline 0.5701) ✅ 真训练增益
- B3d cap1 (首窗 order=0) macro F1 = **0.6240** vs SOTA orthofuse 0.6410 (-0.017) ❌
- B3d × SOTA per-class fusion cap1 = 0.6410 = SOTA, **每类 gate +0.008 全部未过** → SKIP
- 决策: SKIP-advance, 不浪费提交配额

**关键 chain-first 救命 bug (要记入教训)**:
- 早期我在 cycle_b3d_calib_dbloss.py / cycle_b3d_orthofuse.py 用了**错的 cap1 定义**（每通 order max 末窗）和**错的阈值** [0.05,0.5,0.5,0.5,0.05]，**得到假高数字** B3d cap1 0.6690 (+0.041 vs 假 SOTA 0.6281)
- 真定义：cap1 = order==0 首窗（与 cycle_orthofuse.py 一致），THR_VARF = {0:0.05,1:0.5,2:**0.75**,3:**0.65**,4:**0.25**}
- 修正后 B3d cap1 = SOTA = 0.6410，**真增益 = 0**
- **不修正直接 push 错 csv，BC pos 会从 SOTA 的 27 飙到 193 → D-11 真分崩**
- chain-first 在融合候选 + 真分预测前对**所有 baseline 数字 + 阈值 + cap1 定义**完整复核救命

**B3d 失败的本质原因**:
- DB-Loss + SupCon 在 OOF stride5 上的 +0.031 是**校准 ctx_lgbm_v1 和 whisper 两个已知信号源的 BCE 输出更精准**
- 但 SOTA orthofuse 已经在 cap1 上 per-class 选了 ctx 和 whisper 最强组合，**B3d 学的是这两个源的 calibration 没新信号源贡献**
- OOF→cap1 转化 = 0 = 校准头不是新源
- **未来 N1/N2/N3 想叠 loss 升级**, 必须先有**新信号源**（hubert/Omni/F0），不能只校准已有信号

**收获 (虽 SKIP 但有价值)**:
1. DB-Loss + SupCon pipeline 验证通过本机 MPS 可跑 (~7min 5fold×3seed)，未来叠新源可直接复用
2. B1 EDA 同步完成，找到 30+ 个 info>0.15 的强 context v3 特征（runlen_*, burst_*, trans_*, diff1/2_*）
3. 算力账：本机 MPS 5fold 校准头 7min / B1 EDA 50通 1min — 周期很快，可以多试

**下一步 (D-13 战略下)**:
- 优先 B1 ctx 特征工程 v3: 现 46d → 76d, 重训 ctx_lgbm_v1 + 重做 orthofuse. 期望 +0.003~0.010
- B3 其它后处理路径 (TTA / pseudo-label) 暂不投入 (B3d 已确认"无新源就不涨 cap1")

**保留**: B4 / B3d cycle.py 代码 + OOF 校准头训练 pipeline 都保留, 复赛阶段叠 hubert 等新源时复用
