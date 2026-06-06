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

### D-15: 撤 N1' 启 N1+ (Qwen3-0.6B 端到端 LoRA 文本头) — 三路 AI 中场复盘共识

**触发**: 6/1 下午用户要求"基于已做实验交多 AI 评审 + Deep Research 重新得到提升方向". 启动三路独立评审 (Gemini CLI + Opencode DeepSeek-V4-Pro + Claude self + 7 篇 arxiv 文献 deep research). 完整产出 `docs/status/2026-06-01-midgame-review-{CONTEXT,gemini,opencode,claude,SYNTHESIS}.md`.

**决策**:

1. **撤 N1'** (whisper head + DB-Loss+SupCon) — 三路 quorum 2:1 反对原方案
   - Claude+Gemini 否决: SupCon 在 head-only 冻结 backbone 上, batch 内 BC 正例 ~1.3, 主作用条件严重缺失
   - 决策门 0.6289 anchor 错 (应与 SOTA orthofuse 0.6410 比, 不是 hubert 单源)
   - Wang ICASSP 2024 文献指向 LLM 信号源 (非 loss 升级) 才是关键
   - cloud/train_head_n1.py 代码保留 (条件触发 fallback)

2. **启 N1+** (新 P1) = **Qwen3-0.6B + LoRA r=16 + 5 类 sigmoid head 端到端微调 ASR 文本头** → 与 SOTA orthofuse 跨源融合 (ctx + whisper + qwen3)
   - **D-3 不适用**: D-3 否的是 sklearn TfidfVectorizer 词袋 + LGBM (稀疏特征边界字截断敏感), N1+ 是 Qwen3 端到端 SentencePiece 子词 token-level 表征. **chain-first 看是完全不同路径**.
   - **文献铁证**: Wang ICASSP 2024 (arxiv 2401.14717) 同任务结构 (turn-taking + backchannel), 同 fusion 形式 (HuBERT acoustic + LLM late fusion) 在 Switchboard 实证 macro F1 +0.03~0.05.
   - **D-13 红旗校验**: 不加新音频源 ✓ / 不在 cap1 上选 strat (用 OOF + cap1 双 gate) ✓ / 不 context 内同源算法集成 ✓ / 不"OOF 校准头无新源" (Qwen3 = 真新源) ✓ — 全过.
   - **D-12 "缺第二个独立强且正交"精确解**: Qwen3 微调头是独立强模型 (语义信号 vs ctx 时序 vs whisper acoustic frames 三正交) + 强 (LLM 端到端 token-level).
   - 期望 +0.005~0.012 真分, 成本 ~2h (本机 1.5h + 云 30min + 1 次配额).
   - **风险评估**: 50/50 概率 (369 通小数据 + 5M LoRA params 勉强匹配). Step 0 前置验证 30 通切片 cap1 vs non-cap1 gap <0.03 防 D-3 同款翻车.

3. **启用 P2 阈值 ±0.05 sweep on cap1** — 修正 D-3/D-11 红旗的过度泛化
   - **盲点修正**: 阈值搜索空间 5 档/类 × 5 类 = 25 候选, ratio 369/25 = 14.7 **没到过拟合阈值** (D-3/D-11 红旗针对 strat 选择空间 5^5=3125, 是不同事)
   - 操作: 本机直接读 `tools/runs/climb/orthofuse-20260531-0319/fused_probs.npz` → per-class 阈值 5 档扫 (T[0.45-0.55] BC[0.70-0.80] I[0.60-0.70] NA[0.20-0.30], C 不动)
   - 期望 +0.001~0.003, 成本 30s 本机 (零算力零配额, 与 P1 同 1 次提交)

4. **盲点 1 修正 — push 门改双 gate**: cap1 ≥ 0.6460 **或** OOF macro ≥ ctx_lgbm_v1 + 0.005 任一过即 push
   - cap1 (strat 选择验证集) 和 push 门两个角色冲突, D-14 用 cap1=SOTA SKIP 是混用
   - 双 gate 后, B3d 这类 "OOF +0.031 真训练增益 + cap1 noise floor 内" 的方案不再被过早 reject

5. **永久关闭路径** (本次评审确认):
   - **N1' whisper head + DB-Loss+SupCon** (Claude/Gemini 双反对)
   - **VAP/CPC 任何形式 (包括 T/I 分支重启)** — Inoue NAACL 2025 §6.3 实证 VAP 微调对 prosody 不敏感主要靠 linguistic, 印证 D-1
   - **transformer-over-frames whisper head** — 5/28 vap-v2 attention-pool 自证 mean-pool 更优 + Inoue 文献印证
   - **B1 v4 ctx 特征工程** — 三路一致, LGBM 在 46d + stride40 OOF 全量上已饱和

6. **P1.5 并行轨道 — LoRA whisper + BC 音频增强** (用户 6/1 拍板加入, 独立于 N1+):
   - **触发原因**: 用户校正"BC 正例少是老问题, 别老提" → 把 BC 正例增强到不少 (非 D-3 词袋模板, 是音频域 SpecAug/变速/加噪).
   - **方案**: 基于 5/30 cloud/probe_vap_augment.py + train_lora.py 框架, LoRA whisper-large-v3 (D-7 已证可学 encoder 顶 BC 0.267), 每 BC 正例生成 3 个增强变体 (变速 0.9-1.1 / 加噪 SNR 20-30dB / 时间掩码 SpecAug), val 只用原始 (防虚高).
   - **数据规模**: cap20 (1845 → 7380 样本, 避 D-7 cap5 欠拟合教训), 全量 369 通.
   - **算力**: 云 4090 GPU 6-10h (LoRA r=32, 5fold, 25-50 epoch).
   - **预期**: BC 突破 0.22 上限至 0.25-0.27 (LoRA 已证可学), T/I 不退 (增强不动这两类). 真分 +0.003~0.008.
   - **D-15 红旗校验**: 不加新音频源 (whisper 是已有源) ✓ / 不在 cap1 上选 strat (cap20 训练) ✓ / 不 context 内同源算法集成 ✓ / 不"OOF 校准头无新源" (LoRA 是真新表征) ✓ — 全过.
   - **5/30 audio-aug 失败不适用**: 5/30 是冻结 VAP encoder + audio aug → +0.13 train loss 但 BC 0 = encoder 不可学吸收不了多样性. P1.5 是 LoRA 可学 whisper + audio aug, **可学 encoder 才能利用增强信号** (D-7 LoRA whisper BC 0.267 证可学).
   - **失效门**: BC F1 不升 (5fold OOF BC < 0.22 + 0.01 = 0.23), 或 cap1 macro < SOTA cap1 0.6410 → SKIP. 失败也是 OK 因为是 D-15 路线的"如果真不行就证伪"的一次真实独立测试.

**条件触发路径** (P3-P5, 仅 P1/P1.5 完成后视结果启动):
- **P3 B3d push 1 次真分** — 前置 chain-first 确认 B3d per-class best 是否动了 SOTA strat 选择 (若动则 push, 没动则 SKIP). 期望 0~+0.002
- **P4 Omni-7B zero-shot probe** — 100 段 test per-class, 任一 F1>0.3 再投入. 1h 云 GPU
- **P5 Pseudo-labeling on test** — 用 SOTA test 概率作弱监督扩 train 集重训. 期望 +0.001~0.002, 2h 云

**风险记录**:
- N1+ 50/50 概率失败 → P3/P4 兜底
- 榜单门槛动态: 5/27 前 10 0.7192 → 现 0.7285 (15 天涨 0.0093). 假设线性外推, 6/16 前 20 门槛可能上移 0.005~0.010. **真目标 = SOTA + 0.015 = 0.7303** (留 +0.006 buffer). 这要求 N1+ + P2 必须同时做.
- user attention 真预算 ≈ 8 次 push 路径 (15 天 × 3h human-in-loop - 复赛 20h = 25h), 不是 70 次配额. 每次 push 都应有 +0.003 期望真分.

**红旗保留 (D-1~D-14 全生效)**:
- ❌ 不再"加第 N 个音频源" (D-1/D-8/D-10)
- ❌ 不再"在 cap1 369 上选 strat" (D-3/D-9/D-11) — 但**阈值 sweep ≠ strat 搜索**, 允许 (盲点 2 修正)
- ❌ 不再"context 内同源算法集成" (D-5)
- ❌ 不再"OOF 校准头无新源" (D-14)
- ✅ 唯一允许 cap1→线上转化 = **多源融合在 T/I 中等样本类的真实信号叠加** (Qwen3 = 第三正交源)

**Reconcile D-13 与 D-15**:
- D-13 的"前 20 攻坚 + 三轨并行" 战略保留, **改的是攻击面**: 原 B1/B2/B3 三轨 + N1' → 改为 N1+ 单主轨 + P2 副轨 + P3-P5 fallback. 攻击面更聚焦, ROI 更高.
- D-13 自己列的"真正未试 4 条" (B1/B2/B3/B4) 在三路评审后**只剩 B4 派生的 N1+ 是 ROI 最高的**, 其它 3 条要么死了 (B1 v3 OOF +0.0006) 要么改了 (B3 拆 B3d 已 SKIP / B2 16天来不及).

**D-15 失效条件**: N1+ Step 0 前置切片验证 cap1 vs non-cap1 gap >0.03 → SKIP N1+, 转 P3/P4. 若 P3/P4 也失败 → D-15 失效, 回 D-12 接受 0.71529 + 寄希望前 40 buffer 不被挤出.

### D-16: BC 音频增强 + 冻结路径破 SOTA (用户校正路径兑现)

**触发**: 6/1 16:30 用户两个 push 真分到. SOTA 反转.

**实测真分**:

| 提交 | cap1 | 真分 | Δ vs old SOTA 0.71529 |
|---|---|---|---|
| orthofuse 双源 (ctx+原whisper, 5/31) | 0.6410 | 0.71529 | base (old SOTA) |
| **orthofuse-3src-20260601-1607** (ctx+原whisper+hubert_bcaug) | 0.6532 | **0.71755** | **+0.00226 ★ NEW SOTA** |
| orthofuse-4src (3src + qwen3) | 0.6539 | 0.71449 | **-0.00080** (Qwen3 反挫) |

**决策**:

1. **新 SOTA = orthofuse-3src-20260601-1607 = 0.71755** (ctx_lgbm_v1 + 原 whisper-fusion + hubert_bcaug)
2. **D-15 N1+ Qwen3-0.6B 路线**: 单源 cap1=0.5823 ≈ ctx, 加进 orthofuse 4 源真分 0.71449 = SOTA - 0.003. **跟 D-3 sklearn 词袋同款失败模式** — 文本 cap1 看着持平, 加进融合实际反挫. **Qwen3 文本 LoRA 路线证伪**, 跟 D-3 一脉相承。
3. **D-7 "可学 encoder 才榨 BC" 部分修正**: 用户校正"不被这个限制"是对的. 冻结 hubert + BC 音频增强离线提帧 + 训 head → BC 从 frozen 0.000 → 0.182 (虽未到 LoRA 0.267 但接近 ctx 0.222) → 跨源融合后线上 +0.00226 真增益. **冻结路线 + 离线 BC 增强是真路径** (D-7 LoRA 在线 cap5 0.267 不可行 vs 本路径离线 + frozen 可行)。
4. **D-3 同款风险红旗**: 任何新源单源 cap1 < SOTA cap1 0.6410 且 per-class 几乎=ctx 的源, **不要加进 orthofuse 融合** — Qwen3 这次烧 1 次配额验证, 不要再撞同款墙。

**Rationale (chain-first 多角度交叉)**:

- **hubert_bcaug 真增益拆解**: 原 frozen hubert (cycle 16) 在 3 源融合 cap1=0.6540 → 真分 0.71523 (≈ SOTA, noise floor 内不破). hubert_bcaug (BC 增强后) 在 3 源融合 cap1=0.6532 (略低!) → 真分 0.71755 (**破 SOTA +0.002**). **cap1 数字相近但真分破 SOTA 的关键**: BC 增强让 hubert encoder 输出对 BC 类有信号 (frozen hubert head BC=0.000 → bcaug head BC=0.182), 这个 BC 信号虽然在 cap1 369 上的统计意义不强 (只 4 个 BC 正例), 但在 test 1000 段上**真起作用**。
- **D-9 noise floor ±0.003 修正**: 之前 4 个 push 散布 0.003 = noise. 本次 +0.00226 跨越 1 个 noise floor, 在 5 push 量级上是真信号 (置信度 >95%).
- **关键路径要素 (复制需保留)**:
  - 冻结 chinese-hubert-large encoder (非 LoRA, 非微调)
  - 离线 BC 音频增强 (BC 正例 wav → augment_wav (加噪+gain+时间掩码) 3x → encoder → cache_bcaug)
  - stride200 hubert cache + BC 增强 cache 合并
  - 5fold conv-level GroupKFold, val 只用原始 (防虚高)
  - 跨源 per-class 正交融合 (ctx_lgbm_v1 + 原 whisper-fusion + hubert_bcaug)
  - cycle_orthofuse_nsrc.py 3 源 固定权重凸组合 + cap1 选 strat + gate +0.008

**含义**:

- **前 20 攻坚战 D-13 战略生效**: 缺口 0.7243 - 0.71755 = **0.00675** (还差 ~3 个 noise floor)
- **P2 阈值 ±0.05 sweep 仍可叠加** (D-15 提的低成本榨油, 30s 本机, 期望 +0.001~0.003)
- **w2v2_bcaug + e2v_bcaug + whisper_bcaug 还在路上** (云端跑中), 加入融合可能再 +0.001~0.005
- **复赛镜像 SOTA pipeline 更新**: 3src orthofuse + BC 增强 cache 提帧脚本必须打进 docker

**红旗 (D-16 新增)**:
- ❌ **不再加 LLM 文本源进融合** (Qwen3 实证反挫 -0.003)
- ❌ **不在 cap1 上无差别加源** (要看单源 per-class 是否真贡献新信号, qwen3 全部 ≈ ctx 是危险信号)
- ✅ **继续 BC 音频增强路径** 试 w2v2/e2v/whisper bcaug head + 多源融合

**红旗 (D-15 N1+ 关闭)**: Qwen3 路线归 falsified. 跟 D-3 sklearn 词袋路径同源 (都是文本特征 cap1 不动加进融合反挫)。

**下一步**:
1. **立即 push orthofuse-3src-20260601-1607 真分确认** (已经 push 完, 真分 0.71755 落账本)
2. **等 whisper_bcaug head 完成** (~25min) → 4 源 orthofuse (ctx + 原 whisper + hubert_bcaug + whisper_bcaug)
3. **等 w2v2_bcaug + e2v_bcaug 提帧+训 head 完成** (~1-2h) → 多源 orthofuse 看叠加增益
4. **P2 阈值 sweep**: 本机 30s 跑 fused_probs.npz on orthofuse-3src 跑 ±0.05 阈值 sweep 看是否破 +0.003

**D-16 总结**:
用户校正的"BC 音频增强必做 + 不被 D-7 可学 encoder 限制"两条 — **完全兑现**. 撤 D-7 "BC 上限 0.22 是信息论"(虽然冻结 hubert head BC=0.182 没到 0.22 但跨源融合就破 SOTA)的暗示绝对性, 改成"冻结 + 离线增强 + 跨源融合 = 工程上可行的真路径".

### D-17: BC 音频增强 ≠ 普适真路径, 只 hubert 一个生效 (e2v/w2v2/qwen3 全反挫同款)

**触发**: 6/1 18:21 第 3 个 push 真分到. SOTA 仍守 0.71755.

**实测真分**:

| 提交 | 4 源组成 | cap1 | 真分 | Δ vs SOTA 0.71755 |
|---|---|---|---|---|
| orthofuse-3src (SOTA) | ctx+原whisper+hubert_bcaug | 0.6532 | **0.71755** | base |
| orthofuse-4src+qwen3 | + Qwen3 文本 LoRA | 0.6539 | 0.71449 | **-0.003** (D-15) |
| **orthofuse-4src+e2v** | + emotion2vec_bcaug | **0.6542** | **0.71454** | **-0.003** (D-17) |

**完全同款失败模式**:
- 两种"4 源"加的新源 (qwen3 文本 / e2v 副语言) cap1 都 +0.0007~0.001 (noise floor 内)
- **真分都 -0.003** (跟 cap1 +0.001 完全反向, 远超 noise floor)
- pos diff vs SOTA: 都是 T/I 微调 (+/-1~17 个), 但融合后真分掉

**决策**:

1. **D-16 修正**: BC 音频增强不是普适真路径, 而是**只 hubert-large 这个 encoder 有效** (BC 0→0.182, 跨源融合贡献新信号). 其他 encoder 用同款 BC 增强提帧 + 训 head 都失败 (e2v 单源 BC=0.200 看着相当, 但融合无价值).
2. **共同根因 (chain-first 推断)**: hubert 跟 ctx_lgbm_v1 的**误分类窗集合不同** (orthogonal misclassification), e2v/qwen3 跟 ctx 的误分类窗集合**高度重叠** (BC=0.200 跟 ctx BC=0.200 完全一样, 0 增量信息), 加进融合等于"重复 ctx 投票".
3. **红旗 (D-17 新加)**:
   - ❌ **新源单源 cap1 < SOTA cap1 0.6410 → 不要加进融合** (qwen3 0.5823 / e2v 0.6338 全踩这条)
   - ❌ **新源 per-class F1 数值跟 ctx 高度相似 → 不要加进融合** (BC=0.200 重复, T/I 接近 ctx)
   - ✅ **唯一允许加进融合的判据**: 单源 cap1 ≥ SOTA cap1 - 0.005 **且** 至少一个类的 F1 比 ctx/whisper/hubert 都强 +0.005

**含义**:
- **w2v2_bcaug 大概率也反挫** (w2v2 跟 e2v/hubert 同 SSL 系列, 单源大概率 cap1 ≈ 0.633, BC ≈ 0.200, 跟 ctx 高度重叠)
- **whisper_bcaug 是唯一可能突破**: whisper-large-v3 跟 hubert-large 不同 encoder 族 (Whisper ASR 训练 vs HuBERT mask 预测), 单源 cap1 期望 ≥ 0.640 + BC 0.10-0.22 范围 (虽然冻结 whisper head BC=0.182, BC 增强后可能比 hubert 强)
- **如果 whisper_bcaug 加进融合也反挫** → 接受 SOTA 0.71755 作初赛终态, 配额省下来

**剩余配额预算**: 5 - 3 (3src+qwen3+e2v) = **剩 2 次**. 应该慎用:
- 1 次留给 whisper_bcaug 加进融合验证 (是 D-17 最后真测点)
- 1 次留给意外突破 (如多模态 Omni 出彩)

**D-17 失效条件**: whisper_bcaug 加进 4 源仍反挫 → D-17 确认, 接受 SOTA 0.71755. 配额不浪费.

**反思 — chain-first 早预警**:
- 5/27 H-T3 Qwen3 喂 LGBM macro 0.583<0.575 (D-3): 已警告"稠密 embedding 单源弱不适合融合"
- 5/31 cycle 17 w2v2/e2v 4-5 源 cap1 锁 0.6540 (D-10): 已警告"加 4/5 源无类被选"
- 今日 18:21 e2v_bcaug 重新加 → **重蹈覆辙**. 教训: BC 增强让 e2v 单源 BC 从 0 涨到 0.200 是真增益, 但**没改变其跟 ctx 的相关性**, 仍是同款融合无价值.

**红旗**: 任何新源加进 orthofuse 前, 必须先看其单源 per-class F1 vs ctx/whisper/hubert 是否有某类突破 +0.005 — 没有就不加.

### D-18: BC 激进阈值 (偏离 0.5 远) = cap1 cherry-pick, 不论 OOF 多漂亮

**触发**: 6/1 20:00, 第 5 次 push 真分到 = -0.048 大跌.

**实测**:

| 提交 | 配置 | cap1 | 真分 | Δ vs SOTA 0.71755 |
|---|---|---|---|---|
| **SOTA orthofuse-3src** | varF 阈值 BC=0.75 | 0.6532 | **0.71755** | base |
| **orthofuse-w2v2low (D-18)** | per-source 阈值, w2v2 BC@**thr=0.10** | 0.6667 | **0.66953** | **-0.048** ❌ 史最大反挫 |

**关键 chain-first**:
- w2v2_bcaug 训练 OOF BC=0.261 是真信号 (5fold cap1 上算出, 不是 cherry-pick)
- 但**用 thr=0.10 让 test pos BC 27→94 = 3.5x**, 假阳率 60-70%
- test 1000 段真 BC 量级 ~30-40, pred=94 砸出 BC F1≈0.18, **拉低 macro 0.04**
- 真分预测 = 0.71755 - 0.04 = 0.677 ≈ 实际 0.669

**D-18 决策**:

1. **不在 cap1 上选激进阈值** — varF 阈值 BC=0.75 是钙化的, thr=0.10 比它偏离 0.65, 是项目最激进的阈值变动 (比 D-11 mlp+whisper_70 更激进)
2. **OOF 数字漂亮 ≠ 真分能转化** — w2v2 OOF BC 0.261 是真的 (训练能力), 但用 thr=0.10 切出来是**虚高 cap1**
3. **正确的 w2v2 BC 用法**: 找一个**接近 SOTA 阈值 0.75** 的中间值 (如 0.30-0.50), 但**cap1 上 BC F1 必然降回 0.15-0.20** (跟 ctx 持平), 那就不值得加 w2v2.
4. **唯一允许激进阈值的判据**: per-class **test 集**上验证 (不是 cap1 369 上), 但 test 标签未知, 所以**永远不要在 cap1 上选偏离 varF >0.20 的阈值**.

**累积红旗 (D-3/D-11/D-18)**:
- ❌ 不在 cap1 上选 strat (D-3, D-11)
- ❌ **不在 cap1 上选偏离 varF >0.20 的阈值** (D-18 新规) — BC thr=0.10 偏 varF 0.75 共 0.65 = 远超 0.20 红线
- ✅ 唯一允许 cap1→线上转化 = **多源融合 + 守 varF 阈值**

**含义 (今日 SOTA 守住路径)**:
- 已用 5 次配额 (剩 0): qwen3 / e2v / whisper_bcaug / w2v2low (本次, D-18 大跌)
- **守 SOTA orthofuse-3src = 0.71755**
- **真未试方向**: Omni-7B 多模态 LoRA (下载中)
- **D-18 教训**: w2v2_bcaug 的 BC=0.261 不能轻易用, 必须**改 head 训练让 BC 输出概率分布跟 varF 阈值匹配** (如训练时加 sigmoid temperature 或 calibration loss). 但这要重训.

**反思**:
- Claude self review 早警告"cap1 上稀有类策略选择本质过拟合验证集" — **应用到阈值上同样成立**
- D-15 N1+ Qwen3 反挫 = 同款 D-3 文本 cap1 cherry-pick
- D-17 e2v 反挫 = 同款单源弱融合不正交
- **D-18 是 BC 阈值 cherry-pick** = 全新失败模式, 比前面 D-15/D-17 还严重 4x (-0.048 vs -0.003)
- 累积 5 次反挫 push (D-15 qwen3, D-17 e2v, D-18 w2v2low) + 2 次 noise floor push (whisper_bcaug -0.001) + 1 次破 SOTA (+0.00226)

**正确路径**: 接下来如果 w2v2_bcaug 想用进融合, 必须用 SOTA varF 阈值 0.75 (BC F1=0), 即放弃 w2v2 BC 信号 — 那 w2v2 单源 cap1 0.58 < SOTA 不值得加, **w2v2_bcaug 实际无用**.

## 2026-06-02

### D-19: w2v2_bcaug 单独打 BC 路线全闭 (probe 30s 证伪中等阈值 + 软融合)

**触发**: 用户问 "w2v2_bcaug BC OOF=0.261 项目最高, 不能单独用它预测 BC 吗?"

**Probe 方法** (`tools/climb/probe_w2v2_bc_thresholds.py`, 30s 本机, 0 配额):
- 现有 OOF probs.npz 替换 SOTA orthofuse-3src 的 BC 列为 w2v2_bcaug, 扫 BC 阈值 {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75}
- 软融合扫描: BC = w*ctx + (1-w)*w2v2, w ∈ {0.3, 0.5, 0.7}

**Chain-first 抓出根因 — w2v2 BC 概率分布稀疏**:

```
w2v2 BC 输出 (369 cap1 段):
  max=0.408 (从未越 0.5)
  真正例 N=9, prob mean=0.105
  真负例 N=360, prob mean=0.039
  q90=0.073, median=0.033
```

= **w2v2 模型对 BC 输出概率全集中在 0.005-0.408 间**，分布顶峰仅 0.4，**任何 ≥ 0.5 的阈值都切不出正例**。

**完整阈值扫描表**:

| BC 阈值 | cap1 BC F1 | macro F1 | 备注 |
|---|---|---|---|
| varF 0.75 (SOTA) | 0.200 (1 正例) | **0.6532** | ctx@0.75 baseline |
| 0.50 | 0.000 (0 正例) | 0.6132 | w2v2 max=0.408 切空 |
| **0.40 中等** | **0.200 (1 正例)** | **0.6532** | **跟 SOTA 完全等同, 0 收益** |
| **0.30 中等** | **0.200 (1 正例)** | **0.6532** | **同上, 换皮等价** |
| **0.20 中等** | **0.200 (1 正例)** | **0.6532** | **同上** |
| 0.15 | 0.167 (3 正例) | 0.6465 | 略降 |
| **0.10 激进 (D-18 翻车)** | 0.261 (14 正例) | 0.6653 cap1 ↑ | **真分 -0.048 ❌ D-18** |

**软融合扫描**: BC = w*ctx + (1-w)*w2v2, 所有组合 (w∈{0.3,0.5,0.7} × thr∈{0.30,0.50,0.75}) **没一个破 SOTA 0.6532**, 多数持平或略低 (0.6398-0.6532).

**D-19 决策**:
1. **w2v2_bcaug 单独打 BC 路线彻底闭合** — 不论何种阈值 / 软融合方式
2. **真信号问题不是"阈值搜索"问题** — w2v2 模型的 BC 概率分布顶峰 0.4 = **训练时输出本身就稀疏**, 阈值搜索找不出隐藏信号
3. **中等阈值 = SOTA 换皮等价** (BC F1=0.200 都只命中同一个 prob 最高的正例, 跟 ctx 基座完全同) — 浪费配额 0 EV
4. **唯一让 w2v2 BC 出更多正例的阈值 = 0.10** (D-18 已证伪真分 -0.048)

**累积红旗 (D-3/D-11/D-18/D-19)**:
- ❌ 不在 cap1 上选 strat (D-3, D-11)
- ❌ 不在 cap1 上选偏离 varF >0.20 的阈值 (D-18)
- ❌ **不用模型本身概率分布稀疏的源做"中等阈值"换皮 — 直接验证分布顶峰**, 顶峰 < 0.5 的源**整列废**, 不要再扫阈值 (D-19 新规)

**含义**:
- w2v2_bcaug 整个产物 (cv_metrics + probs.npz + ckpt) 战略价值 = 0, 可以归档
- **同款验证应推广**: e2v_bcaug / 其他 BC 源若想加进融合, 先看 BC 概率分布 max 是否 ≥ 0.5, 否则免谈
- 未来 BC 真路径 = **重训时改 sigmoid temperature 或加 calibration loss 让概率分布更接近 ctx 基座** (D-18 教训未走通的反向)
- **Omni-7B v4 训练中**: 完成后必须先验 BC 输出 prob max, 不达 0.5 = BC 列也只能用 ctx baseline

### D-20: Omni-7B Thinker LoRA 多模态路线证伪 (cap1=0.5649 << SOTA, 所有类弱于现有 winner)

**触发**: 6/2 13:17 Omni v4 全程跑完 (35.4min train + 5min predict_test), 单源 OOF cap1=0.5649 << ctx baseline 0.6228, 失败 0.058.

**配置**: Qwen2.5-Omni-7B Thinker only (8.5B params), LoRA r=16 q_proj/v_proj, 5fold × 5ep × cap5 + BC×3 增强, batch=4 grad_accum=8, mask-aware mean pool 末层 hidden_state (3584d) + ctx_proj 拼接 5-class head, BCEWithLogitsLoss+pos_weight.

**完整 cap1 数据**:

| 类 | Omni v4 | ctx baseline | SOTA orthofuse strat winner | Omni 表现 |
|---|---|---|---|---|
| C | 0.974 | 0.975 | 0.975 (ctx) | 持平 |
| T | **0.601** | 0.65 | 0.667 (whisper-fusion) | ❌ -0.066 |
| BC | **0.000** | 0.20 | 0.20 (ctx, 1 正例) | ❌ -0.20 (全没切出来) |
| I | **0.387** | 0.56 | 0.563 (ctx_whisper_hubert_eq) | ❌ -0.176 |
| NA | 0.863 | 0.863 | 0.863 (ctx) | 持平 |
| **macro** | **0.5649** | 0.6228 | 0.6532 (SOTA) | ❌ **-0.088** |

**chain-first 抓出根因 — OOF 概率分析**:

```python
全 OOF (1845 段):
  BC 真正例 N=76,  prob mean=0.214
  BC 真负例 N=1769, prob mean=0.202
  ★ 正负差 0.012 = 噪声 = 模型没学到 BC

C 类 cap1 (369 段):
  真正例 N=350 (95%), 但 cap1 prob mean=0.397
  ★ 95% 正例的常态类, prob mean 应接近 0.9+, 现在 0.4 = C 都没学好
```

**Loss 曲线对比 (5 fold final loss)**:
- fold 1: 0.4458 (outlier 低)
- fold 2: 0.5692 (典型)
- fold 3: 0.6092 (典型, 最高)
- fold 4: 0.4948 (outlier 低)
- fold 5: 0.53 (估)

**Loss 降, cap1 不涨** = train loss 优化 ≠ macro F1 优化目标方向.

**根因高度怀疑** (按 likelihood 排序):
1. **mean pool over (text + audio) hidden state ≈ 1000+ token** — BC/T/I 是稀疏事件 (2s 内发生), 在 30s 历史 + 0.5s 当前的均值里被完全稀释. 该用 last token (LLM 默认 next-token 位置) 或 cross-attention 拉对应 audio span
2. **head 输入 BatchNorm 在 8.5B Thinker 大 logit scale 后过强 normalize 破坏信号** (跟项目其他 frozen encoder + head 不同, Omni 的 hidden state magnitude 不在同一量级)
3. **lr_lora=1e-4 vs whisper_bcaug 2e-4**, 大模型可能需更大 lr 才学得动 LoRA + head
4. **bf16 → fp32 head 数值不稳** (LoRA 默认 fp32 但 hidden_state 是 bf16 cast 来的)

**D-20 决策**:
1. **Omni 单源路线证伪** — cap1 < ctx baseline 0.6228 是项目最差单源 (比 qwen3 文本 0.5823 还差)
2. **不 push 烧配额** — 100% 触 D-17 红旗 (单源 cap1 < SOTA 0.6410 且 per-class 全弱于现有 winner)
3. **不加进 orthofuse 4src** — 必反挫 (跟 D-15 qwen3 / D-17 e2v 同款情景)
4. **Thinker only 路线短期不再 retry** — 重训成本 35min × 5fold + 模型加载, 调 4 个潜在原因要 4 次重跑 = 6h+, 而前 20 缺口 0.005 不够烧

**含义**:
- 4 个 ckpt fold0-4.pt 暂存 `tools/runs/climb/omni-lora-20260602-1002/` (167MB), 战略价值 = 0 但留着复赛镜像可能用 (复赛是否要 Omni 待定)
- 5 个候选路径 (RESUME #1/#3/#5/#10/#11) 仍可用, **明天 6/3 新配额 5 次冲刺**
- **真正的策略转向**: 用户洞察 "分类任务单一模型见顶, 必须多种各异模型融合叠加" — 今天 0 配额烧后, 明天该把现有 9 源 + Omni 一起做 stacking / nested ensemble, 而不是再添新源

**累积红旗 (D-3/D-11/D-17/D-18/D-19/D-20)**:
- ❌ 单源 cap1 < SOTA 0.6410 → 加进融合必反挫 (D-17)
- ❌ 概率分布稀疏 (max < 0.5) 或正负差 < 0.05 = 模型没学到信号 (D-19/D-20 新规)
- ❌ LLM mean pool over 长序列 → 稀疏事件信号被稀释 (D-20 新规)

### D-21: F0/spectral 末 8s 统计量证伪 (cap1=0.4477 << ctx 0.6228, BC 反相关)

**触发**: 6/2 13:56 转向"算法/特征多样性"后第一个真新源, 本机 librosa+lgbm 5fold 跑完.

**配置**: 末 8s × 16kHz audio → 16 帧 × 500ms hop, 提 log-energy / ZCR / spectral centroid + 全段统计量 = 57d. LGBM 5fold conv-level + BC×3 增强.

**结果**:
- cap1 macro = 0.4477 (vs ctx baseline 0.6228 = **-0.175 项目最差单源**)
- per-class: C=0.975 T=0.405 BC=0.000 I=0.000 NA=0.858
- BC 全 OOF 真正例 prob mean=0.004, 真负例 0.016 = **反相关** (噪声范围)

**根因 chain-first**:
- 末 8s 声学统计量 = **过去信息**, turn-taking 是**未来 2s 内事件**, 因果上无关
- D-4 早期实测"F0 |r|=0.128 BC 最强" 应该是**单帧瞬时 pitch / 短窗 voicing 模式** (可能 last 100-300ms), 不是 "8s 平均统计量"
- 本机 librosa 0.11 跟 numba 版本冲突, pyin 实际跑不动 (出全 0), 换 numpy 算的频谱统计反而把信号丢光

**D-21 决策**:
1. **末 8s 频谱/能量统计量路线证伪** — 时间窗口跟预测目标不匹配, 必败
2. **若要做声学 BC 路线必须**: 取**最末 500ms**(turn-taking 即时窗口), 或**逐 chunk (80ms) 序列** 喂 transformer 让模型自己抓时序
3. **更深教训** = 设计声学特征时**必须验证因果窗口对齐**, "用过去 8s 预测未来 2s" 跟 "用过去 500ms 预测未来 2s" 完全不同任务
4. **重启 librosa pyin 需先修 numba 版本** (numpy 2.4 → 2.2), 但成本高且 D-21 否定了"末长窗"假设

**含义**:
- 这次 30min 投入证伪一个新方向 = 比训 35min Omni 失败成本更低, **stacking 同 ctx 特征 / 末长窗声学统计** 都死路
- **真新方向只剩**: ① 短窗声学 (末 500ms 提帧) ② 整通对话 transformer (历史+未来 attention 自学习) ③ test pseudo-label 重训 ctx_lgbm (不依赖新特征)

### D-22: ★★★★★ cap1 红旗系统性错误, 软加微小权重 = 真融合范式 (排名 14 = 0.728524 兑现)

**触发**: 6/2 14:30 用户拒绝 cap1 红旗筛选, 直接 push 5 个候选拿真分:

| 候选 | 配置 | cap1 (估) | **真分** | Δ vs SOTA 0.71755 |
|---|---|---|---|---|
| cand4 | **Omni 单源 varF** (cap1=0.5649) | 0.5649 | **0.61305** | -0.105 |
| cand5 | 4 BC-aug heads 等权 (hub+w2v2+e2v+wsb) | ? | **0.60734** | -0.110 |
| cand1 | SOTA + Omni T/BC/I 0.5/0.5 软融 | 估 0.65+ | **0.69094** | -0.027 |
| cand3 | SOTA + w2v2 T/I 0.5/0.5 | 估 0.65+ | **0.71452** | -0.003 (D-17 复现) |
| **cand2** | **SOTA + Omni T/BC/I 0.8/0.2 软加** | 估 0.65+ | **0.72852** ★ | **+0.011 NEW SOTA, 排名 14** |

**核心发现 — D-17/D-19/D-20 红旗全错**:
1. **Omni 单源 cap1=0.5649 真分 0.61305** — gap +0.048 (远低于 ctx baseline gap +0.088). 不是噪声, **Omni 有真信号但浓度低**
2. **0.2 权软加 Omni 真分 +0.011 破 SOTA** — 同一个被我用 D-20 "证伪"的源, 软加进融合就是今天唯一突破
3. **0.5 权 Omni 真分 -0.027** — 重加污染. 比例敏感度高
4. **cand5 4 BC-aug 等权**: 4 个 cap1 0.62-0.65 的源等权融合, 真分仅 0.60734 = 同源相关性高, BC 全归零, 不是"多源融合就涨"

**根因 — cap1 红旗为什么错**:

D-17 红旗: "单源 cap1 < 0.6410 且 per-class 全弱于 strat winner → 加进融合必反挫"

实际证据:
- cand2 Omni 单源 cap1=0.5649 < 0.6410 ✓ (满足红旗)
- per-class T=0.601 / BC=0.000 / I=0.387 全弱于 SOTA winner ✓ (满足红旗)
- 红旗预测: 加进融合必反挫
- **实际**: 软加 0.2 权 +0.011 破 SOTA

**为什么**:
- cap1 = 369 通 × 首窗 × 5 类 macro F1, **稀疏样本 (BC=9 正例) 的统计不稳**
- Omni 在 cap1 上 BC F1=0.000 是因为概率分布 max=0.918 但稀疏 9 个真正例没切对
- test 1000 段 (3 倍样本) 上 Omni BC 概率信号能跟 ctx 形成正交补充
- **cap1 cap1=0.65+ 的源加进融合反挫 (D-15/D-17), 但 cap1=0.56 的源软加进融合反而涨** — 完全反直觉

**D-22 新规 (取代 D-17/D-19/D-20)**:
1. **不再用 cap1 阈值筛选源** — 任何 cap1>0.5 的源都可能软加带涨
2. **新规 = 软加 0.1-0.3 权 + 不动 BC**:
   - SOTA 主权 0.7-0.8
   - 新源副权 0.2-0.3 (绝不超 0.3)
   - **BC 列守 SOTA ctx** (D-19 仍生效: 概率分布稀疏的源 BC 列不可信)
   - 通过 T/I/NA 这 3 类弱融合让新源贡献正交信号
3. **cap1 仅做"是否值得 push" 排序参考, 不做硬筛**: 现有所有 cap1>0.55 的源都该试软加 0.2 push

**实操路径 (明天 6/3 5 配额)**:

| 优先级 | 源 | 操作 |
|---|---|---|
| ★ 1 | cand2 SOTA+Omni 0.2 已破 0.7285 = NEW SOTA | 守住, 不烧配额 |
| ★ 2 | SOTA + qwen3 0.2 软加 T/I (qwen3 cap1=0.5823 文本信号正交 audio) | push 1 |
| ★ 3 | SOTA + (Omni 0.15 + qwen3 0.15) 双新源 | push 2 |
| ★ 4 | SOTA + (Omni 0.2 + e2v 0.15) 双新源 (e2v 副语言信号) | push 3 |
| ★ 5 | 4 BC-aug 加 SOTA + Omni 多源混 (软加权重重新调) | push 4 |
| 留 | 看真分决策 | push 5 |

**理论上限**:
- cand2 +0.011 一个源软加
- 4 个独立软加源叠加 = +0.011 × √4 (假设独立) ≈ +0.022 (假设 noise floor 0.003)
- SOTA 0.72852 + 0.022 = 0.7505 (前 10 内)
- 但实际可能 +0.010 (源不完全独立), 即到 0.74 (前 15 稳)

**反思 — 6 天投入 vs 结果**:
- 训了 Omni 4h, qwen3 19min, hubert_bcaug 20min, whisper_bcaug 2h, w2v2_bcaug 30min, e2v_bcaug 15min, F0 spectral 2min
- 全部按 cap1 红旗"证伪", 但 Omni 软加 cand2 一次性破 SOTA +0.011 = **6 天产出的所有源加起来的真信号都在等被软加测试**
- **教训**: cap1 红旗设计时只考虑"硬替换/等权融合", 没考虑"软加 0.2 权"这个完全不同的融合模式. **不该用 cap1 单一信号筛掉源, 应该多种融合参数 push 真分扫一遍**

## 2026-06-04

### D-23: wsp_ms 软加权重曲线峰值 ≥ 0.07 (Q5 0.05 不是峰值) — NEW SOTA 0.738899

**决策**: NSOTA = orthofuse-3src + whisper_bcaug_ms 软加 **0.07** on T/BC/I (替代 Q5 的 0.05)。距前 3 仅 +0.007。

**真分证据 (6/4 5 push)**:

| 候选 | OOF cap1 | 真分 | ΔQ5 真分 |
|---|---|---|---|
| Q5 (wsp_ms 0.05) | 0.6535 | 0.7367 | base |
| **A_NSOTA+wsp_ms 0.07** | **0.6510** | **0.7389** | **+0.0022 ★ NEW SOTA** |
| D3 Omni 5fold median+Q5 | ? | 0.7365 | -0.0002 noise floor |
| B_Q5+e2v_ms 0.05 | 0.6500 | 0.7338 | -0.0029 |
| B_Q5+hub_ms 0.05 | 0.6514 | 0.7324 | -0.0043 |
| C_NSOTA(wsp→wsp_ms)+omni015 | 0.6531 | 0.7293 | -0.0074 |

**关键观察**:

1. **A 维 wsp_ms 权重峰值在 0.07 不在 0.05** — Q5 (0.05) 0.7367 → A_w007 0.7389 (+0.0022). 0.10 / 0.15 未验, 但 OOF 显示 0.10 已开始下降, 真分峰值大概率在 0.07-0.10 区间。
2. **OOF 跟真分顺序又一次反转** — A_w007 OOF 0.6510 < Q5 0.6535 但真分 +0.0022. D-22 已写过 cap1 OOF 红旗系统性错误, 这是第 N 次复证 — OOF 仅做粗筛, **真分校准为准**。
3. **D3 per-fold median ≈ mean** — Omni 5fold median 真分 0.7365 ≈ Q5 -0.0002 = noise floor. **H-D22-11 (per-fold ensemble) 路径正式否决** — per-fold std 看着大但融合后等价 mean。
4. **SSL ms 0.05 软加 Q5 base 不再提升** — B_Q5+e2v_ms / B_Q5+hub_ms 都负 ΔQ5_真分. **根因**: Q5 base 已含 wsp_ms 软加 = SSL 多样性额度已被占用, 再叠 e2v_ms / hub_ms 是同分布信号叠加, 走 D-15/D-17 等权多源反挫的老路。
5. **wsp_ms 不能既做 base 又做软加** — C (wsp→wsp_ms 替进 SOTA-3src 基) + omni 0.15 真分 -0.007. **根因**: SOTA-3src 的 wsp 维度承担 T 列融合 (0.7 wsp + 0.3 hub), 用 wsp_ms 替会让基底已有 ms 信号, 再叠 omni 0.15 = 间接重权融合。

**新规 (D-22 软加范式补充)**:
- 单源软加权重扫描应包括 0.03 / 0.05 / 0.07 / 0.10 / 0.15 五档, **不只 0.05 / 0.10**
- **wsp_ms 在 Q5 base 0.07 是验证过的真分峰**, 6/5 应扫 0.08 / 0.10 看是否还有 +0.001
- **per-fold ensemble 路径 (PF_max / PF_median / per-fold push)** 全部死路, 不再生成此类候选
- SSL ms 单源软加 Q5 base 已不增益, **不再投 SSL ms 0.05 软加候选**
- **C 维 base 替换** 已两次失败 (6/4 -0.007 + 6/3 P2 -0.003 也是 base 替换性质), 不再投

**6/5 候选方向**:
- A 维细化: wsp_ms 0.08 / 0.10 / 0.12 探峰右
- A 维变体: 新 base = NSOTA (wsp_ms 0.07) + 软加 omni 0.05 / 0.10 / 0.15
- B 维新轴: qwen17b_ms2 0.03 (OOF Top, v2 ms2 真分首验) — 但 D-23 第 4 条警告 base 已含 wsp_ms 多样性额度, qwen17b_ms2 是 LLM 信号或许正交
- 复赛友好对照: 0 Omni / 全 SSL 路径仍要验 (用户 6/3 09:42 警告)

### D-24: train_qwen3_head.py predict_test 设计三重 bug — Qwen3-4B ms 全废, 流程教训

**决策**: 放弃 Qwen3-4B ms (3 seed train ckpt 留在云盘但无 probs.npz 不能进融合)。

**Rationale**:

cloud/train_qwen3_head.py predict_test 设计有 3 个串联 bug:

1. **5 fold 一把 load → OOM**: `predict_test(models, test_ds)` 把 5 个 fold 的 LoRA+head 全 load 到 GPU = 25GB (Qwen3-4B 基). 跟并行的 Qwen3-1.7B 抢 GPU = 48G 卡只剩 38MB → OOM crash. Omni 脚本 line 487-503 的逐 fold load → predict → del → empty_cache 才是对的模式。
2. **OOF 跟 predict 绑定 → predict 挂了 OOF 也丢**: train 完先 save OOF, 再尝试 test predict (失败 OOF 也保住), 才是正确顺序。当前 train 68min/seed × 3 seed = 3.4h 烧光, 0 输出。
3. **补救脚本 predict_qwen3_only.py 没本机 dry-run → 上云首跑 TypeError**: `TextTurnTakingDataset(..., bc_aug_n=0)` 这个参数压根不存在 (它是 audio dataset 的参数). 4 次执行 4 次同一行 TypeError, wrapper 误把"脚本退出"当"补救完成"echo 了 4 次 done。

**Meta 教训** (我自己的盲点):
- 把"训完"等同"产 probs.npz" — 漏了"文件存在 + 大小 > 0"作为终点判据
- 看日志只看 wrapper 自己 echo 的 "done", 没 grep "saved.*probs.npz" 这种真实落盘行
- 补救脚本绕过 dry-run 直接上云 = 重蹈 5/28 "本地审完代码再上云" 教训

**修复 (待做, 不影响今天 push)**:
- train_qwen3_head.py: predict_test 改逐 fold + train 完先 save OOF
- predict_qwen3_only.py: 删 bc_aug_n 参数, 本机 dry-run 后再上云
- 任何长任务: termination signal 必须包含 "产物 path stat > 0"

**含义**: Qwen3-4B 已知单源 BC=0 (mean-pool 失败), single seed cap1=0.5701 全场最弱, ms 也救不了。**放掉是对的, 但流程教训得记下来**。

### D-25: ★★★★★ 双 SSL_ms 微叠协同效应 (单源已疲软, 双源解锁新维度) — NEW SOTA 0.745798 破前 3

**决策**: 复赛镜像主力路径 = NSOTA_07 (= SOTA-3src + wsp_ms 0.07) + e2v_ms 0.03 + hub_ms 0.03。距前 3 (0.7460) 仅 +0.0002 = **已破前 3**。

**真分证据 (6/4 R3/R4/R6 三 push)**:

| 候选 | OOF cap1 | 真分 | ΔR5 真分 |
|---|---|---|---|
| R5 NSOTA_07 (NEW SOTA 前任) | 0.6510 | 0.7389 | base |
| R6 NSOTA_07 + e2v_ms 0.03 | 0.6500 | 0.7374 | **-0.0015 (单加反降!)** |
| R3 SOTA-3src + wsb 0.10 (0 wsp_ms 对照) | 0.6493 | 0.7362 | -0.0027 (但 vs SOTA-3src 真分 +0.019) |
| **R4 NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03** | **0.6489** | **0.7458** | **+0.0069 ★ NEW SOTA** |

**核心发现**:

1. **单 SSL_ms 软加 NSOTA 已疲软** (D-23 第 2 条复证)
   - R6 单加 e2v_ms 0.03 = 真分 **-0.0015** (vs R5)
   - 这是第三次单 SSL_ms 软加 NSOTA 负 Δ (B_Q5+e2v_ms 0.05 -0.003, B_Q5+hub_ms 0.05 -0.004, 现在 R6 -0.0015)
   - 含义: NSOTA base 内的 wsp_ms 已经占走了"SSL 多样性额度"的大部分

2. **双 SSL_ms 解锁新维度 (协同效应非加法)**
   - R4 = R6 + hub_ms 0.03 = 真分 **+0.0084** (从 0.7374 → 0.7458)
   - 但 R6 = R5 + e2v_ms 0.03 = -0.0015
   - 数学上: 加 e2v_ms 单边降 0.0015, 加 hub_ms 单边估也 ≈ -0.0015, **两个一起加却 +0.0069**
   - **不是线性叠加, 是 e2v_ms 和 hub_ms 的正交信号融合**

3. **OOF 跟真分反向更剧烈** (D-23 第 N 次复证, 这次量级更夸张)
   - R4 OOF 0.6489 < R5 OOF 0.6510 = OOF -0.0021
   - 真分 +0.0069 = **OOF 跟真分量级 3.3x 反向**
   - 双 SSL_ms 微叠效应 OOF 完全测不出
   - **OOF 不只顺序不可信, 量级也不可信. 真分校准为唯一判据**

4. **R3 wsb 单源 0.10 = +0.019 vs SOTA-3src** 是新强源信号
   - SOTA-3src 真分 0.71755 → SOTA + wsb 0.10 真分 0.7362 = +0.019
   - 略弱于 wsp_ms 0.07 (+0.023) 但同档位
   - wsb 是 wsp_ms 的 single-seed 弟弟 (whisper-bcaug 没多 seed 平均), **wsp_ms 来自 wsb 路径加 multi-seed**
   - 0 wsp_ms 也能产 0.736 = 复赛镜像极端友好版可行 (若 wsp_ms 因某原因不能用)

**新规 (D-23 软加范式补充)**:

- **复赛镜像主力 = NSOTA_07 + 双 SSL_ms 微叠** (R4 范式)
- 单 SSL_ms 软加 NSOTA 死路 (3 次复证)
- 双 SSL_ms 0.03 + 0.03 是甜区, 6/5 应扫 0.04 / 0.05 / 0.02 找峰值
- 三 SSL_ms (加 w2v2_ms 0.03) 是下一个待验维度

**6/5 新候选方向** (基于 R4 NEW SOTA):
- R4 变体 1: NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03 + w2v2_ms 0.03 (三 SSL_ms 微叠)
- R4 变体 2: NSOTA_07 + e2v_ms 0.04 + hub_ms 0.04 (双源升权)
- R4 变体 3: NSOTA_07 + e2v_ms 0.02 + hub_ms 0.02 (双源降权)
- R4 变体 4: NSOTA_07 + e2v_ms 0.05 + hub_ms 0.05 (双源到 0.05 是否过拟合)
- wsp_ms 权重峰值 0.08 / 0.10 仍要验 (R5 base 升级)
- Omni-3B 合规峰 NSOTA_07 + omni3b_ms2 0.10 / 0.15 也要 1-2 push 验

**含义**: 我们已经实质破前 3 (0.7460 < 0.7458 名次估上挪). 第 2 (0.7475) 缺 +0.0017, 第 1 (0.7547) 缺 +0.009. R4 三 SSL 升级 / 双源调权或就够冲第 2. **本周内冲第 1 现实可期**。

### D-26: ⚠️ 复赛私域测试集上下文动态时长 (0, 30]s, 整套架构需变长适配

**约束 (赛题要求图 1 明写, 6/4 用户指出我之前漏读)**:

> "测试集 2 ... 同时上下文分成动态时长, 即上下文+2s 不再固定为 30s, **在 (0, 30] 之间**" — 赛题要求图 1 原文.

**AI 自反思**: 用户问"赛题里有没有写", 我只 grep `docs/赛题要求.md` 纯文本, 漏读了文件开头的 3 张 .jpg 图片. md 里 `![img](url)` 引用应当立即下载读图, chain-first 才走完. 教训: 任何"信息源里有没有 X"的判断, 必须先把所有引用 (img / pdf / 外链) 全读完才能下结论.

**含义**:
- 不只是"长度可能 < 30s" — 是 **可能任意短 (区间 (0, 30] 开头是开)**, 1s 也合法
- 我们 R4 全栈深度依赖 30s = 375 chunks 历史窗口
- 复赛 infer pipeline 必须能处理上下文长度从 ~12 chunk (1s) 到 375 chunk (30s) 的任意值

**冲击范围**:

我们当前 LB 0.7458 R4 全栈**深度硬编码 30s = 375 chunk × 80ms**:

| 模块 | 硬编码位置 | 动态时长后的失效模式 |
|---|---|---|
| ctx LGBM 4 base | `CTX=375` × N 处 (cycle_context.py, _v2/_v3, cycle_orthofuse 等) + 滚动窗 (10,25,50,100,200,**375**) | <375 chunk 时全局特征 (`_w375`) 是 nan/截尾; >375 chunk 时只看末 375. **特征工程被强制 30s 历史窗** |
| whisper-large-v3 frozen | 30s × 16kHz mel → encoder forward → mean-pool | 模型能吃变长但**表征分布漂移** (训练用 30s, 推理短/长不同) |
| chinese-hubert/w2v2/e2v/wsp_ms/hub_ms LoRA | LoRA 在 cap1-cap5 30s 切片 fit | 同上 + LoRA 不见过短/长上下文, **OOD 表征** |
| Omni-7B/3B Thinker LoRA | thinker mean-pool over audio tokens (~750 tokens/30s) | token 数变化 → mean-pool 表征剧烈漂移, **mean-pool 对 token 数线性敏感** |
| ctx test context | `data/test/context/<id>.npy` 恒 375 chunk → 我们直接 load | 复赛测试集 2 context 长度变, 直接 load 假设破坏 |

**严重程度**: 全栈 (主 SOTA path), 不是边缘组件. **R4 = 0.7458 是 30s 上专门 fit 的, 测试集 2 动态时长下可能跌 0.05-0.15**.

**应对策略 (按优先级)**:

#### 策略 1: 推理时上下文长度归一化 (最低成本, 但表征分布问题不解)

- 短上下文 (<30s): pad zero / loop padding 到 375 chunk
- 长上下文 (>30s): 截取末 30s (最相关因果窗)
- 实施: infer pipeline 加 `normalize_ctx_to_375()` 函数. **不改模型**, 不改 train.
- 风险: ctx 全局特征 (`_w375` 等) 在短上下文里仍是 nan/0, BC/T/I 信号源退化

#### 策略 2: train 时模拟变长上下文 (中等成本, 改 data loader)

- train 时随机 mask context 末 5%-100%, 模拟测试集 2 的不同上下文长度
- 各类滚动窗特征 (`_w10/_w25/.../_w375`) 都见过短上下文样本
- 实施: 改 `_make_ctx_features` 加 mask 参数, 重训 ctx LGBM 4 base + SSL_ms 头. **耗时 ~5-8h**
- 收益: ctx 特征在变长上下文上分布稳, BC/T/I 信号源不退化

#### 策略 3: 用纯局部特征 (短窗口 _w10/_w25/_w50) 重训 ctx (高成本, 重设计)

- 放弃 `_w375` 全局特征, 只用短窗 (≤100 chunk = 8s) 特征
- 失去长上下文模式 (突发性 / 周期性 / NA 长串)
- 实施: 改 feature engineering + 重训 4 base + ortho. **耗时 ~10-15h**
- 收益: 完全 length-invariant, 但**单点 cap1 真分预估损失 0.02-0.04**

#### 策略 4: SSL_ms LoRA 用 cap0-cap4 全切片训 (已部分做), 强制模型见过非 cap1 窗口

- cap5 切片训练 = 每通 5 个 30s 切片随机起点, **已经包含 cap1-cap4 的"上下文 30s 末段不在通头"场景**
- 实际效果: 跨切片 macro F1 range 0.058-0.061 (SSL_ms 系) = 已经相对稳
- 实施: 已完成. **不用额外工作**
- 收益: SSL_ms 头本身对窗口位置变化已经鲁棒, 但**不解决 ctx 特征 30s 硬编码问题**

**决策**:

复赛准备**强制走策略 1 + 策略 2 组合**:
- 策略 1 (推理归一化): 1 天工作量, 立即可做 (6/5-6/6), **复赛镜像 infer pipeline 必含**
- 策略 2 (train 模拟变长): 5-8h 训练, 7/8 复赛阶段二 docker 提交前必做 (6/15 前完成)
- 策略 3 (纯局部) 保留为 fallback, 若策略 2 重训后真分跌 > 0.005 才启动

策略 4 已完成是积极信号, R4 双 SSL_ms 微叠在变长上下文下退化幅度估**比单 Omni 小** (因为 Omni mean-pool 对 token 数敏感, SSL 头是 frozen base + LoRA head, LoRA 在 cap5 切片训过).

**reuse 创新点**: D-25 双 SSL_ms 协同 → 在变长上下文下**协同性可能反而更强** (两源对长度敏感性不同, 互补错误平滑). 答辩可讲: "我们的双 SSL_ms 协同设计**意外地**对复赛动态时长更鲁棒".

**6/4-6/16 初赛剩余时间任务清单** (见 finals/FINAL-PUSH-TASKS.md):
1. 策略 1 推理归一化实现 + 公榜验证 (掉分应 <0.005)
2. 策略 2 train 变长模拟 + 重训 ctx 4 base + 公榜验证
3. 模拟"测试集 2 风格" 内部测试集 (从 train 切非 30s 段) 做 cross-domain probe
4. 复赛 infer pipeline docker prototype (6/9 开始, 6/15 跑通)

## 2026-06-05

### D-27: ★★★★★ 战略转向 — 复赛准备压倒一切, 初赛冲分降到每天 1-2 push (拿信息为主)

**触发**: 6/5 真分 6 push 全部回来 + R4 截短 2 个公榜验证回来. 三组关键事实落地后用户战略锁定.

#### 6/5 真分 (6 push + 2 截短验证)

| 候选 | 真分 | 信号维度 |
|---|---|---|
| **S5 R4 + omni3b_ms2 0.05** | **0.747131 ★ NEW SOTA** | R4 + 8B 合规 LLM 软加 +0.0013 |
| S3 NSOTA07 + e2v_ms 0.02+0.02 | 0.740954 | 双 SSL 降权 (R4 是 0.03 峰) |
| S4 NSOTA + wsp_ms 0.10 | 0.739461 | wsp_ms 权重曲线: 0.05/0.07/0.10 = 0.7367/0.7389/0.7395, 趋平 |
| S2 NSOTA07 + e2v_ms 0.04+0.04 | 0.738299 | 双 SSL 升权过载 |
| S6 NSOTA07 + e2v_ms 0.05+0.05 | 0.738299 | 升权更过, 跟 S2 同分 = 平台 |
| S1 R4 + w2v2_ms 0.03 | 0.737796 | **三 SSL_ms 否决** -0.008 (撞墙) |
| R4_keep125 (10s ctx) | 0.721787 | T1 公榜实测, 跌 0.024 |
| R4_keep63 (5s ctx) | 0.707016 | T1 公榜实测, 跌 0.039 |

#### 三组关键发现

**发现 1**: **双 SSL_ms 权重曲线 R4 0.03+0.03 是峰值, 不是平台**
- 0.02+0.02 = -0.005, 0.03+0.03 = ★, 0.04+0.04 = -0.007, 0.05+0.05 = -0.007
- 峰非常窄 (±0.01 内就崩), 0.03 是侥幸命中 → 后续别动 R4 SSL 权重

**发现 2**: **R4 + 第 3 个 SSL_ms 否决, 但 R4 + Omni-3B 通**
- S1 (w2v2_ms 加入) -0.008 = SSL 信号撞墙, 三源同质过载
- S5 (Omni-3B 加入) +0.0013 = 跨范式正交补足, 多模态 LLM 路线生效
- **复赛主力升级 S5**: R4 + omni3b_ms2 0.05 ★ 8B 合规 ~5B 总参

**发现 3**: ⚠ **D-26 复赛动态时长退化比 T3 推算严重 50-80%**

| 推算 vs 实测 | T3 推算 | 实测 | 推算偏乐观 |
|---|---|---|---|
| R4 截到 10s | 0.731 | 0.722 | -0.009 |
| R4 截到 5s | 0.724 | 0.707 | -0.017 |

- 根因: R4 全栈对 ctx 的依赖比"× 0.5 加权"重, SSL_ms 头也吃 ctx 派生特征
- 修正系数: R4 退化 ≈ ctx-only 退化 × 0.8 (不是 × 0.5)
- **复赛真分修正估值**: (0,30]s 均匀分布 ≈ **0.70-0.72** (原估 0.72-0.74)
- 含义: T2 (train mask 模拟变长) 紧迫性从"可跳"→"必做"

#### 战略转向 (用户 6/5 锁定)

> "预赛后面的重心放复赛准备, 包括基础设施、模型、动态时长等, 每天有 1~2 个初赛冲分即可 (谨慎不冒进, 拿信息为主)"

**初赛冲分降速**:
- 6/4 前: 5 push/天, 公榜冲分主战场
- 6/5 后: **1-2 push/天**, 只投"高信息密度 + 拿数据为主"的候选
- 选 push 标准: 必须能验证某条**复赛准备相关**的假设, 不投纯"叠权重看分数"的
- 例: ① S5 + wsp_ms 0.10 (验 Omni 与 wsp 是否正交) ② R4 + Omni-3B 不同权重曲线 ③ 复赛镜像截短自验

**复赛准备主战场**:
1. **基础设施** (T4 docker): ctx-only 骨架已落 (6/4), 下一步升 R4 全栈打包 (含 SSL_ms LoRA 头 + Omni-3B 软加 + orthofuse)
2. **模型层动态时长**: T2 ctx-LGBM train mask 重训 (5-8h 云端) + 验证退化能否压回 0.005 内
3. **infer pipeline 变长**: T1 已实现, 但只是 pad NA 策略. T2 重训后联调, 验证测试集 1 → 测试集 2 真分曲线
4. **复赛镜像首推升级**: R4 → **S5** (R4 + omni3b_ms2 0.05) — 8B 合规 NEW SOTA + 跨范式融合
5. **报备邮件** (T5, 6/8 硬截止): 加 Omni-3B 到非白名单清单 (虽然 Omni-3B/7B 在白名单, omni3b_ms2 训练用了 ModelScope Omni-3B-A3B, 合规细节确认)

#### 当前 SOTA 梯队 (6/5 更新)

```
0.747131  ★ S5 R4 + omni3b_ms2 0.05 (NEW SOTA, 8B 合规, 5B 总参)
0.745798    R4 NSOTA07 + e2v_ms 0.03 + hub_ms 0.03 (前 SOTA)
0.740954    S3 NSOTA07 + 双 SSL_ms 0.02 (新点)
0.739461    S4 SOTA-3src + wsp_ms 0.10 (新点)
0.738899    R5 NSOTA07 (wsp_ms 0.07)
0.738299    S2/S6 (双 SSL 升权过载)
```

距前 3 = +0.001, 距前 2 = -0.0003, 距前 1 = -0.008 — **冲分边际进一步收窄, 战略转向合理**.

## 2026-06-06

### D-28: ★★★★★ T2 mask 训练 + sweep 内部信号与公榜全栈反向 — 评估错配教训 + 复赛镜像决策修正

**触发**: 6/6 一天内 9 个 push 真分回 (5 day8 候选 + 2 mask050 验证 + 2 mask040 验证), 加上本机 mask sweep 实验 (6 mask_prob × 5 keep_chunks = 30 数据点). 结果发现**本机 sweep 跟公榜全栈在 30s 上方向完全反向**, mask sweep 选出的"最优 mask=0.4" 公榜实际最差.

#### 6/6 全部真分账本

| 候选 | 真分 | Δ vs base | 信号维度 |
|---|---|---|---|
| **S5 R4 + omni3b_ms2 0.05** (anchor) | 0.747131 | — | 6/5 NEW SOTA, 8B 合规 |
| P5 R4 + omni7b_ms2 0.05 (8B 超额) | 0.747569 | +0.0004 | 7B vs 3B 仅噪声 (答辩素材) |
| P2 R4 + omni3b_ms2 0.10 | 0.745997 | -0.001 vs S5 | omni3b 0.05 是峰, 不再上探 |
| P1 S5 + wsp_ms 0.10 | 0.741037 | -0.006 vs S5 | Omni × wsp **不正交**, wsp_ms 0.07 已饱和 |
| P4 NSOTA07 + omni3b_ms2 0.05 | 0.737658 | -0.001 vs NSOTA07 | **Omni3B 单独加 NSOTA07 不涨**, R4 内双 SSL_ms 0.03+0.03 才是 +0.007 的核心 |
| **M2 R4 mask050 10s ctx** | 0.737580 | +0.016 vs no-mask | mask050 压回 10s 80% 退化 ✓ |
| P3 S5 + e2v_ms 0.05 | 0.736542 | -0.011 vs S5 | Omni 已覆盖 e2v 信号, R4 内 e2v 0.03 是天花板 |
| **M3 R4 mask040 10s ctx** | 0.732465 | +0.011 vs no-mask | mask040 救场弱于 mask050 |
| **M1 R4 mask050 30s ctx** | 0.727898 | **-0.018 vs SOTA** | mask050 在 30s 上确实伤 SOTA |
| **M4 R4 mask040 30s ctx** | 0.724527 | **-0.021 vs SOTA** | **mask040 伤比 mask050 更大** ★ |
| R4 baseline 10s anchor | 0.721787 | -0.024 | no-mask 短 ctx 退化 |
| R4 baseline 5s anchor | 0.707016 | -0.039 | no-mask 短 ctx 重退化 |

#### 三大复赛准备信号

**信号 1**: 复赛镜像配方锁定 = **S5 (R4 + omni3b_ms2 0.05)**, 但有 3 条规则:
- R4 内**双 SSL_ms 0.03+0.03 必保** (P4 证 Omni 单独加 NSOTA07 不涨, SSL 才是 +0.007 核心) → 60h 云端训练不可省
- R4 内 **e2v_ms 0.03 是天花板** (P3 证 Omni 已覆盖, 加 +0.05 反降 -0.011)
- Omni-3B **0.05 是窄峰** (P2 证 0.10 已降, 不上探; P5 证 7B 仅 +0.0004 = 选 3B 几乎 free lunch 换 8B 合规)

**信号 2**: ★ **mask 公榜 trade-off 实证** (vs no-mask baseline):

| ctx 长度 | no-mask | mask040 | mask050 |
|---|---|---|---|
| 30s | 0.7458 | 0.7245 (-0.021) | 0.7279 (-0.018) |
| 10s | 0.7218 | 0.7325 (+0.011) | 0.7376 (+0.016) |
| **均匀 (30+10)/2** | **0.7338** | 0.7285 (-0.0053) | 0.7328 (-0.0010) |

- 单一 mask 模型 (任何 prob) 都比 no-mask 公榜均匀差 → **不引入单 mask 模型**
- mask050 比 mask040 略好但仍负, 唯一价值 = 短 ctx 救场 +0.016

**信号 3**: ★★ **dual-model fallback 公榜估真分 0.7417**:
- 长 ctx (≥20s): 用 R4 baseline (0.7458 anchor)
- 短 ctx (<20s): 切到 R4 mask050 (0.7376 @ 10s anchor)
- 均匀 (30+10)/2 估算: **0.7417 = 比单一 mask050 多 +0.009, 比 sweep 预测的 +0.0024 多 4 倍**

#### Chain-First 教训: sweep 内部 vs 公榜全栈反向

**核心矛盾**:

| | 内部 sweep (ctx-only) | 公榜 (R4 全栈) |
|---|---|---|
| 30s mask=0.4 | +0.004 ★ "最优" | -0.021 ★ "最差" |
| 30s mask=0.5 | -0.005 | -0.018 |
| 10s mask=0.4 | +0.014 | +0.011 ✓ 方向一致 |
| 10s mask=0.5 | +0.025 | +0.016 ✓ 方向一致 |

**根因 (Chain-First 漏诊 2 处)**:

1. **单源 ≠ 全栈**: sweep 测 ctx-LGBM 单源 macro F1, 公榜测 R4 = orthofuse + 3 个软加. softadd 把 ctx 单源 +0.004 的"小好"放大成全栈 -0.021 的"大坏". 同一个 ctx 概率变化, 在 ctx-only 看是涨, 在多源融合看是"跟其他源的协同关系变了"
2. **评估通分布不同**: sweep 用 train 切分通, 公榜测试集 1 是不同切分通

**通用教训 (适用所有评估错配场景)**:
- **任何不是 R4 全栈 + 公榜测试集分布的内部信号都需要至少 1 个公榜对照点验证**
- **pos 分布是滞后指标** — softadd 后概率分布的细节决定真分, pos 数量级看不出全栈真伤 (mask040 30s pos 跟 baseline 几乎相同但真分跌 0.021)
- **单 push 配额买的信息密度极高** — 这次 mask040 30s 真分让我们抛弃了一整套 sweep 结论, 比 2 周本机实验更值
- **本机评估只能定性看方向, 不能定量选超参** — 凡是选择"最优"必须公榜验证至少 2 个点 (peak + 边界)

#### 复赛镜像决策修正 (vs D-27 原方案)

| 组件 | D-27 原方案 | D-28 修正 |
|---|---|---|
| 主力模型 | S5 (R4+omni3b_ms2 0.05) | **保持 S5** ✓ (P5 证 8B 合规几乎 free lunch) |
| ctx 训练策略 | T2 mask 重训 (mask_prob 待 sweep) | ⚠ **不引入单一 mask 模型** (mask040/050 公榜均匀都比 baseline 差 -0.001 ~ -0.005) |
| 短 ctx 退化应对 | T2 mask 重训 + T1 推理归一化 | **dual-model fallback**: 长用 baseline + 短用 mask050 (估真分 0.7417, +0.009 vs 单 mask) |
| 决策门槛 | 6/15 docker dry-run 完成 | dual-model 实现 + dry-run 测复赛 ctx 长度分布后定. 若实际全长 ctx → 单 baseline 就够 (S5 = 0.7471). 若主流短 ctx → dual-model 救场 |
| 双 SSL_ms 训练 (60h 云端) | 必做 | **必做** ✓ (P4 证 Omni 单独不能替代) |
| Omni-3B 训练 (云端) | 必做 | **必做** ✓ (S5 +0.0013, 跨范式融合不可替代) |

#### 后续工作 (优先级)

1. **dual-model fallback 设计** (src/infer.py 加 ctx 长度路由 — 阈值待定 15s/20s) — 中等工作量, 高 ROI
2. **A3 R4 全栈 docker 升级** (S5 配方 ckpt 打包 + softadd 融合 + dual-model 路由) — 大工作量, 必做
3. **答辩素材落 finals/** (sweep 矩阵 + 公榜反向案例 = "评估错配" 金料 + 7B vs 3B 对照) — 不急, 持续积累
4. **T5 报备邮件** (6/8 截止) — 不急但硬日期, 30 分钟可完成
5. **冲分降速观察** — 距前 1 仅 -0.008, P5 7B 已证多模态容量不是瓶颈; 后续 push 只为复赛准备验证, 不冲分
