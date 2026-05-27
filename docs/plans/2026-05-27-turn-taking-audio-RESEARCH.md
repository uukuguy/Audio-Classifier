# Turn-Taking 音频建模 + Macro-F1 极不均衡优化 — Research

**Researched:** 2026-05-27
**Topic:** ① turn-taking/backchannel 音频建模 SOTA ② Qwen/电话域语音编码器选型(≤8B) ③ Macro-F1 极不均衡多标签优化
**Libraries:** transformers (v4.56.2 / v5)
**Codebase scanned:** yes（baseline）
**Companion:** `2026-05-27-finvcup-turn-taking-CONTEXT.md`（决策契约）

> 背景：EDA 确认纯上下文标签地板分 Macro-F1=0.59，BC 是瓶颈类（F1 仅 0.22，窗口正样本率 3.8%），音频/文本必须把它拉起来。

---

## 0. 最重要的一条发现（直接定调）

**有一篇论文用了几乎相同的设定**：Apple **"Talking Turns" (ICLR 2025, arXiv:2503.01174)** —— 30s 因果窗、电话语音(Switchboard)、预测下一 chunk 事件，标签集就是 **{C, T, BC, I, NA}**，用**冻结 Whisper encoder + 线性 softmax 头**，per-class ROC-AUC 89-95 且能 OOD 泛化。

**结论**：我们的赛题本质 = 已发表的可复现任务。冻结大编码器 + 轻量头在全 30s 窗上跑，对这个标签集**已被证明够用**。不需要从零发明架构。

---

## Findings

### A. Turn-Taking 音频建模 SOTA

#### A1. VAP (Voice Activity Projection) — 最对口模板
- **核心论文**: Ekstedt & Skantze, Interspeech 2022, arXiv:2205.09812 · code https://github.com/ErikEkstedt/VoiceActivityProjection
- **架构**: 帧级音频编码器(冻结 CPC, ~50-100Hz) → 小因果 transformer(256d, 4层4头) → VAP 头。**关键创新**：不独立预测每个未来 bin，而是预测**未来 2s 窗口的联合状态**，离散成 4bin/speaker × 2speaker = 8bit = **256 类**，bin 宽递增(200/400/600/800ms，近未来更可预测)。
- **BC 的处理**：256 态分布上读出 4 个 zero-shot 任务(SHIFT/HOLD、SHIFT-pred、**BC-pred**、SHORT/LONG)。**联合建模对 BC-pred 增益最大**(F1 .723 vs 独立 .685)——因为 BC 是两说话人最复杂的相互依赖。
- **双声道实时版**(Inoue, IWSDS 2024, arXiv:2401.04868, code **https://github.com/inokoj/VAP-Realtime**)：每声道→冻结 CPC→channel-wise self-attn(1层)→**cross-attention transformer(3层4头，一声道 query 另一声道 key/value)**→512d→VAP头。**两声道间的 cross-attention 是结构核心**——建模"对方在干什么"，这是 BC 的主导线索。

#### A2. Backchannel 专项（最对口我们的瓶颈）
- **Inoue et al. NAACL 2025, arXiv:2410.15929**（最对口的 BC 论文）：不均衡真实数据(~10% 正)上 zero-shot VAP 的 BC 仅 **F1 15.1**，微调后最高 **42.85**。**SOTA 在不均衡 BC 上也只 0.35-0.50** → 校准预期，别指望 BC F1 很高。
  可直接搬的招：①**两阶段训练**(先 VAP 自监督预训练，再加 BC 头微调，+5 F1) ②**多任务损失** `L=α·Lvap+β·Lvad+γ·Lbc`，**γ 加权更高(5×)** ③**类不均衡**：正样本 loss 加权 5×，且**把正标签提前 500ms**(预测性标注)。
  反直觉发现：压平 intensity 比压平 pitch 更伤 BC，但**两者都不致命**→ BC 更依赖**词汇/上下文线索**而非纯韵律。
- **Amazon ICASSP 2024**（几乎完全是本赛题）：冻结 HuBERT(音频) + GPT-2/LLM(文本) 晚融合。**决定性发现：文本/LLM 单模态 >> 音频单模态(所有类含 BC)**；融合对 C/T 增益大但**对 BC 仅边际**——"BC 更related to 句法语义信息、是局部线索"。**BC 最大增益来自 LLM 多任务指令微调**。
- **Xiamen MM-F2F, ACL 2025, arXiv:2505.12654, code https://github.com/Linyx1125/MM-F2F**（最鼓舞人的 BC 数据点）：他们数据上**音频单模态 HuBERT 的 BC F1=.805 > 文本 .707**("音高/音调不连续触发这些动作")；文本+音频融合 BC F1=.894，三模态 .906。**编码器 bake-off：文本 GPT-2>BERT；音频 HuBERT>Wav2Vec2(BC .805 vs .779)，端到端解冻微调**。融合用 **Low-rank Multimodal Fusion(LMF)** + **Random Modality Dropout**。

> **注**：Amazon 说"文本 > 音频 for BC"，Xiamen 说"音频 > 文本 for BC"——分歧来自数据/标注。对我们的启示：**BC 必须 text+audio 都上，靠融合 + 对方声道**，不要赌单一模态。

#### A3. 赢家通用架构模式（强收敛）
1. **音频保持时序序列，不要 mean-pool 成单向量**（baseline 的 Whisper/CNN 都 pool 了 = 漏分点）。VAP/Apple 都在帧序列上跑因果 transformer。
2. **冻结大编码器 + 小可训头**是默认；数据大才端到端微调。
3. **双声道 cross-attention** 对 BC 和 I(interruption) 结构性关键(都由两说话人同时行为定义)。
4. **多任务/辅助损失**稳住稀有类，防止头塌缩到多数类。
5. **文本融合高杠杆**(LMF / 晚融合 / prompt 注入)，对 BC/T 尤甚。
6. **不均衡招**：下采样 C、稀有类 loss 加权 ≥5×、预测性标注、按 per-class 指标优化。

### B. 电话域(8kHz)中文语音编码器选型（≤8B 约束）

| 编码器 | 参数 | 域 | 评价 |
|---|---|---|---|
| **TencentGameMate/chinese-hubert-large** | 317M | WenetSpeech 10k 时中文多域 | **非 Qwen 首选**：中文原生、电话域邻近、中文 ASR/副语言特征 bake-off 第一(Aishell CER 3.3) HF: huggingface.co/TencentGameMate/chinese-hubert-large |
| chinese-wav2vec2-large | 317M | 同上 | 备选 |
| **Qwen2-Audio 的 Whisper encoder** | ~640M | 通用 | **Qwen 路线首选**：满足"优先白名单"，冻结当特征提取器，总参 ≤8B。但 Whisper 是 content-oriented + mono + 固定 30s log-Mel，**不 channel-aware**，要每声道分别跑 |
| Qwen2.5-Omni audio | (含在 3B/7B) | 通用 | `get_audio_features()` 可抽特征；但 Omni 整体重，可只取 audio 塔 |
| CPC | 小 | — | VAP 原版用；多语言 VAP 论文发现 CPC>多语言 wav2vec2 even 中文(future-prediction 目标契合 turn-taking) |
| emotion2vec / emotion2vec+ | 90-300M | 中文可用 | 副语言/情感帧特征，做"对方能量/情绪"辅助特征 |

**8kHz 处理（明确答案）**：所有 SSL 编码器要 16kHz 输入。**电话域标准做法 = 上采样 8k→16k**(VAP 处理 Switchboard 就这么干)。**不要直接喂 8kHz**(conv stem 的 stride/感受野按 16k 标定，会帧率/特征错位)。**每声道分别处理、保持双声道**。

**中文专项**(Multilingual VAP, arXiv:2403.06487，在 HKUST 中文电话 8kHz 上训)：①英文 VAP **不迁移**到中文，必须中文微调 ②**中文 turn-taking 比英文更依赖 pitch**(中文是声调语言)→ **显式 F0/pitch 特征对我们的中文任务可能比英文 benchmark 更有价值**。

### C. Macro-F1 极不均衡多标签优化

#### C1. 逐类阈值（已验证 +0.07，理论保证正确）
- **Macro-F1 可分解**：每类 F1 只依赖该类自己的(TP,FP,FN)→**逐类独立调阈值 = 最大化 Macro-F1，无跨类耦合**（Pillai/Fumera/Roli, Pattern Recognition 2020）。
- **最优阈值≈F1*/2**（Lipton et al. ECML 2014, arXiv:1402.1892）→ **弱分类器(BC)的最优阈值远低于 0.5**；EDA 里 BC 调到 0.75 是因为 pos_weight，去掉后预计更低。
- **防过拟合（BC 关键，验证集 BC 正样本极少）**：
  1. **在 pooled OOF 预测上调阈值**(K-fold，每样本被未训练它的模型预测一次，合并所有 OOF)
  2. **跨折最小偏差阈值**（Kaggle Quora 1st place）：不取单折最优，取"各折 F1 与该折最优的偏差均值最小"的固定阈值——为 BC 牺牲一点最优换稳健
  3. 定阈值后在全量数据 refit 最终模型

#### C2. 损失（最大可靠增益来源）
- **ASL 非对称损失**（Ben-Baruch ICCV 2021, arXiv:2009.14119, code Alibaba-MIIL/ASL）—— **多标签不均衡首选**。`γ+=0`(正样本保留完整梯度，保护稀缺 BC 正样本) + `γ-=2~4`(压低易负样本) + `m=0.05`(丢弃极易负样本)。**优于普通 focal**(focal 同 γ 会连 BC 正样本梯度一起压)。
- **pos_weight 别用裸 25×**(=neg/pos for BC)→ 会塌精度、F1 反降。用**适度值(5-10× 或 √倒频)，在 OOF Macro-F1 上调**。baseline 现在就是裸 neg/pos，是个改进点。
- **logit adjustment**(Menon ICLR 2021)：减去 log 类先验，**可证明优化 balanced error**（最接近 Macro-F1 哲学），post-hoc 近乎免费。
- 顺序：**ASL 主损失 → 适度 class weight → post-hoc logit adjustment → 逐类阈值**（损失重加权 + 阈值是叠加的，都做）。

#### C3. soft-F1（baseline 有 `compute_gaussian_soft_f1_sequence`，慎用）
- sigmoidF1(Bénédict TMLR 2022, arXiv:2108.10566) 直接优化 F1 surrogate。
- **致命陷阱**：soft-F1 在 **batch 级**算，需每 batch 每类都有足够正样本填满混淆矩阵。**BC 3.8% 下 batch=64 平均才 2.4 个正、常为 0** → BC 梯度退化。
- 缓解：**大 batch / 平衡采样保证每 batch 有 BC 正 / 梯度累积后再算 soft-F1**。
- **结论**：BCE/ASL + 阈值是低方差可靠路径；soft-F1 只作**后期微调 add-on**，且必须先解决 batch 内 BC 正样本问题。**别指望 soft-F1 单独解决 BC**。

#### C4. 重采样/增强（别误伤多数类）
- **平衡 batch 采样器保证每 batch 有 BC 正**(同时解决 C3 的问题) + **SpecAugment** + **随机 mixup 混合**(Mix² 式，arXiv:2403.09598：单一 mixup 会伤性能，随机混合 Mixup/Manifold/MultiMix 才稳，稀有类 macro-F 35→46 且多数类不降)。
- 警告：**naive 过采样/平衡会降 per-class 平均指标**(arXiv:2307.00079, AudioSet)。在 OOF 上验证 C/NA 不回退。

---

## Reusable Assets in Repo

| Path | 是什么 | 怎么用 |
|---|---|---|
| `baselines/.../src/utils.py:compute_gaussian_soft_f1_sequence(probs[B,C,T],targets[B,T],sigma=2,avg_class_indices=(1,2,3))` | 时序高斯平滑 soft-F1，只平均 T/BC/I 三类 | C3 的 soft-F1 微调 add-on 起点；注意它只覆盖 1,2,3 类 |
| `baselines/.../src/data/dataset.py:_read_wav_slice / _load_wave_segment` | 按需切片 8k wav、**保持双声道**[2,T]、resample 到 16k、pad/trim | 直接复用——双声道已保留(VAP 需要)，无需重写音频 IO |
| `baselines/.../src/models/multimodal_baseline.py:WhisperAudioEncoder` | 冻结 Whisper，**但 mean 成 mono + tail attn pool 成单向量** | 漏分点：要改成保序列 + 每声道分别 + cross-attention |
| `...:ContextLabelEncoder / HandcraftedFeatures` | 上下文标签 tail+full 双分支 conv + 19 维手工特征 | 方案 C 现成；HandcraftedFeatures 已实现 EDA 里的 tail-ratio/dist-to-last 特征 |
| `baselines/.../src/train.py:243` | `BCEWithLogitsLoss(pos_weight=neg/pos)` 裸倒频 + 固定 0.5 阈值 + save by roc_auc | 改进点：换 ASL + 适度权重 + 阈值搜索 |
| `baselines/.../src/data/dataset.py:build_train_samples_multitask` | 滑窗(ctx375/tgt25/stride5) + 未来 2s any-出现多标签 | 复用样本构造；验证集要改 30s 切片形式 |

**注**：baseline 所有编码器都 **pool 成单向量 = event-level**，与 SOTA 的"保帧序列"相悖。这是最大架构改进空间。

**对口可 clone 的参考实现**：VAP-Realtime(架构)、MM-F2F(text+audio LMF 融合)、Inoue NAACL2025(不均衡/多任务 recipe)。

---

## Recommendations

### Option A（推荐 · bake-off 主线）：双声道帧级 VAP 式融合
**Approach:** 冻结音频编码器(Qwen2-Audio Whisper encoder 或 chinese-hubert-large) **每声道分别跑、保留帧序列** → 双声道 cross-attention transformer(借 VAP-Realtime) → 融合文本编码器(ASR，LMF)+ 上下文标签(复用 ContextLabelEncoder)→ 5 类头。损失 ASL + 多任务辅助；BC 预测性标注 + 5× 权重；逐类 OOF 阈值。
**Why:** 最贴 Apple/VAP/Amazon/Xiamen 一致 SOTA；双声道 cross-attn + 文本融合正打 BC/I 瓶颈；冻结编码器单卡可训。
**Cost:** 中（要重写音频塔为序列+双声道，但 VAP-Realtime 可借）

### Option B（低风险锚点）：强化 baseline
**Approach:** 保留 baseline 三模态，但①Whisper 改保序列+双声道（不 mean-mono）②损失换 ASL③加逐类阈值搜索④pos_weight 改适度值。
**Why:** 改动小、快出第一个改进分；验证"光修 baseline 漏分点"能涨多少。
**Cost:** 低

### Option C（集成锚点 + 已验证）：纯上下文标签 LGBM
**Approach:** 已实现(`tests/main/eda_context_baseline.py`)，Macro-F1=0.59。补逐类 OOF 阈值 + 做成可提交 CSV。
**Why:** 分钟级训练、集成强一路、第一个公榜锚点。
**Cost:** 极低（已完成大半）

**推荐路径**：C 先拿公榜锚点 → B 修 baseline 漏分点拿改进分 → A 冲分 → 集成 A+B+C。

---

## Open Questions（for writing-plans / 用户）

1. **音频编码器最终选型**（spike）：Qwen2-Audio Whisper-encoder(合规优先) vs chinese-hubert-large(中文电话域更优但需报备)——EDA bake-off 小实验决定，看 BC/I 增益是否值得报备非 Qwen。
2. **显式 F0/pitch 特征**（spike）：中文 turn-taking 更依赖 pitch，是否加 pyworld/torchaudio 的 F0 作为额外帧特征？低成本值得一试。
3. **验证集 30s 切片构造**的精确实现（plan）：从 train 长对话采独立 30s 段 + 对应 context/text/label 模拟 test，避免滑窗乐观偏差。
4. **K-fold vs 单 split**：逐类阈值防过拟合需要 OOF；初赛先单 split 拿锚点，中期上 K-fold 给阈值+集成喂 OOF（CONTEXT 已 defer）。
5. **预测性标注(提前 500ms)** 是否采用——会改变标签构造，需在 CV 上验证对 Macro-F1 的净效应。

---

## Exit

Ready for: `/superpowers:writing-plans`（方向已清晰，决策契约 + 本 research 足够）
Input: 本文件 + CONTEXT.md + EDA 脚本结果
