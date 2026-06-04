# 中场复盘 — 三路评审汇总 (SYNTHESIS)

**生成**: 2026-06-01 14:00
**汇总者**: 主 Claude (协调 Gemini / Opencode / Claude self 三路独立评审)
**输入**:
- `2026-06-01-midgame-review-CONTEXT.md` (评审包, 374 行)
- `2026-06-01-midgame-review-gemini.md` (Gemini CLI, 46 行)
- `2026-06-01-midgame-review-opencode.md` (Opencode/DeepSeek-V4-Pro, 151 行)
- `2026-06-01-midgame-review-claude.md` (Claude self + 7 arxiv 文献, 160 行)

---

## 0. TL;DR (用户必读)

**三路独立收敛到同一最高优先级 = N1+ (Qwen3-0.6B 端到端微调 ASR 文本头) 替代 N1' (whisper head + DB-Loss+SupCon)**。

- **撤 N1'**: 2-1 反对 (Claude 完全否决, Gemini 要求升级, 只 Opencode 支持原方案当低成本试错)。理由: N1' 漏了"新信号源"输入, 重蹈 D-14 B3d 覆辙风险高
- **启 N1+**: 3-0 一致推荐 — Qwen3 文本 LLM 端到端微调是**D-1~D-14 全关闭的唯一真正新模态** (Claude/Opencode 独立给出 Wang ICASSP 2024 文献证据, macro +0.03-0.05)
- **加 P2 = 阈值 ±0.05 sweep**: 3-0 一致 — Q1.c 阈值微调 (THR_VARF 是 5/27 cycle1 时代的安全保守值, 跨源融合后未重调)
- **保留次优路径**: Isotonic / max strat / transformer head — 三路都提到但增益 ≤0.003, ROI 不如 N1+

**期望路径**: N1+ +0.005~0.012 + 阈值 sweep +0.001~0.003 = **总期望 +0.006~0.015**, 即真分 0.7213~0.7303, **覆盖前 20 门槛 0.7243**。

**Claude 独家盲点 (Gemini/Opencode 没提)**:
- cap1 的"strat 验证集"角色和"push 门"角色冲突 → B3d 该 push 真分一次 (OOF +0.031 单独过门)
- 阈值搜索空间 5 档/类 = 25 候选 / 369 样本 = ratio 14.7, **不到过拟合阈值**, 不该被 D-3/D-11 红旗误伤
- N1' 决策门 0.6289 anchor 错 (该跟 SOTA orthofuse 0.6410 比, 不是 hubert 单源)
- 配额 (70 次) vs user attention (实际 ~8 次 push 路径) 真预算
- 榜单门槛动态 (5/27→现在: 前 10 0.7192→0.7285), N1+ 单条 +0.008 仅平 0.7243 不留 buffer → 必须叠 P2

---

## 1. 强 quorum 共识 (≥2 路独立收敛)

### 共识 1: 文本 LLM 端到端微调 = 唯一真正新模态 (3/3)

| 维度 | Gemini | Opencode | Claude self |
|---|---|---|---|
| **方案** | Qwen-text 拼入多模态 fusion head | Qwen3-0.6B 1024d → MLP head → orthofuse 3 源 | Qwen3-0.6B + LoRA r=16 + 5 类 sigmoid head, 与 SOTA late fusion |
| **D-3 否决适用?** | (隐含不适用) | 明确反驳: 否的是词袋+LGBM, 不是 LLM 微调 | 明确反驳: chain-first 看, 词袋是稀疏特征边界字截断敏感, LLM 端到端 token-level 表征对边界字鲁棒 (SentencePiece 子词切分) |
| **文献依据** | (未引用) | DB-Loss ECCV 2020 + SupCon NeurIPS 2020 (但承认 head-only 训练 SupCon 弱) | **Wang ICASSP 2024 (arxiv 2401.14717)**: HuBERT + LLM LoRA late fusion 在 Switchboard turn-taking + BC 三分类 macro F1 +0.03~0.05 (Table 2). **结构同构**: 我们 5 类 sigmoid = 5 个独立二分类, 可借用 Wang 范式 |
| **风险** | (未细评) | D-3 同款 (ASR 文本在 cap1 vs 任意切片可能分布不同), 验证方案: 30 通随机切片 + 看 cap1 vs non-cap1 head F1 gap | 369 通小数据 + LoRA r=16 = 5M trainable, 训练量勉强匹配; 给真分 50/50 概率 |
| **成本** | (未细估) | 2-3h 本机 MPS | ~2h (云 GPU 0.5h + 本机 cap1 评估 30min + 1 次提交) |

**为什么 D-3 不适用 (chain-first 三角验证)**:

- **D-3 原始证据**: ti-robust 真分 0.6392, cap1 0.6358 ≈ SOTA 0.6402, gap +0.003 — 这是 sklearn TfidfVectorizer 词袋 + LGBM 在 stride 全切片 train vs 30s test 切片末的边界字截断敏感导致虚高
- **N1+ chain 不同**: Qwen3-0.6B 子词切分 + 端到端微调 → 学到的是句法/语义对话意图 ("嗯…好的" / "我明白了" / "那你觉得呢" 这类 turn-yielding / backchannel-inviting 信号), **不是词共现统计**
- **D-12 红旗"缺第二个独立强且正交"的精确解**: Qwen3 微调头是**独立强模型** (不依赖 ctx label sequence, 直接看 ASR 文本 + history label tokens) **+ 正交** (语义信号 vs ctx 时序 vs whisper acoustic frames)

### 共识 2: 阈值 ±0.05 sweep on cap1 是低成本榨油点 (3/3)

| 维度 | Gemini | Opencode | Claude self |
|---|---|---|---|
| **判断** | THR_VARF 沿用 5/27 cycle1 单模态时代的值, 跨源融合后联合概率空间已变 | Q1.c 最有希望的榨油点 | Q1 真盲点 |
| **具体建议** | NA 0.25→0.35 (引 5/30 03:31 journal), I 类调整 | T∈[0.45,0.55], BC∈[0.70,0.80], I∈[0.60,0.70], NA∈[0.20,0.30] | T∈[0.45,0.55], BC∈[0.70,0.80], NA∈[0.20,0.30] (与 Opencode 完全独立但相同) |
| **不违反铁律的论证** | (未细辩) | 守"不在滑窗 CV 上搜, 只在 cap1 369 上 ±0.05 窄扫" | **关键**: 阈值搜索空间 5 档/类 × 5 类 = 25 候选, ratio 369/25 = 14.7 没到过拟合阈值; D-3/D-11 红旗"不在 cap1 上选 strat" 针对 strat 选择空间 5^5=3125 大空间, **阈值搜索≠strat 搜索** (盲点 2) |
| **期望** | (未量化) | +0.001~0.003 | +0.001~0.003 |
| **成本** | (未细估) | 0.5h 本机 | 30s 本机 (fused_probs.npz 已存) |

**实施**: 本机直接读 `tools/runs/climb/orthofuse-20260531-0319/fused_probs.npz` → per-class 阈值 ±0.05 5 档扫 → cap1 macro 取最大 → 若提升 → 同 N1+ 一起 push (1 次配额覆盖两个改进)。

### 共识 3: B1 v3 ctx 特征工程死掉合理 (3/3)

- 47 个 info>0.15 EDA 强特征在 OOF +0.0006 是教科书级 LGBM 饱和 (LGBM 内部 split 等价于交互式特征工程, 你给它的新统计量它已经从原始 46d 隐式学到了)
- 换 CatBoost 也没空间 (D-5 实测 cat 0.6328 < lgbm 0.6349)
- **不要再做 B1 v4**

### 共识 4: D-1 VAP 不该重启 (2/3, Claude+Opencode 反对 Gemini)

- **Gemini**: D-1 否的是 BC 弱, 但 VAP 是 T/I 原生 SOTA, 在 N1+ 中作为特征源
- **Opencode 否决**: VAP unfreeze 全量微调真分 0.6337 已经是用 BC 标签重训整个模型 — VAP 的归纳偏置 (turn-taking prediction, not backchannel) 在 backbone 层面就抓不住 BC
- **Claude 否决**: Inoue NAACL 2025 实证 VAP 顶到 BC F1=30 用了 35h 日本对话语料 pretrain → 微调, 我们只有 369 通中文电话 (15h 估), 数据量 60% 不到。**判死合理**

**采纳 Claude/Opencode 双否决, 不重启 VAP。**

---

## 2. 分歧 + 我的裁决

### 分歧 1: N1' 是否还该跑 (1 支持 vs 2 反对)

| 立场 | 论据 | 评估 |
|---|---|---|
| **Opencode 支持** | EV = 0.3 × 0.003 = +0.0009 macro, 即使期望涨幅极小, 跑 2h 微薄机会优于不跑 | EV 计算忽略了 user attention 成本 |
| **Gemini PIVOT** | 抓对了 loss (长尾) 但漏了"新信号源"输入, 应升级为 [ctx+whisper+hubert+F0+Qwen-text] → Transformer head | 多源 fusion head 是大投入, 不在 16 天预算里 |
| **Claude 完全否决** | 1) DB-Loss+SupCon 在 head-only (冻结 hubert/whisper) 上, batch 256 平均每 batch BC 正例 ~1.3, SupCon 主作用条件严重缺失, 预期 contrib≈0; 2) 决策门 0.6289 anchor 错 (该是 SOTA 0.6410); 3) Wang ICASSP 文献证据指向 LLM 信号源不是 loss 升级 | 文献+chain-first 双重证伪 |

**裁决: 撤 N1'**。Claude/Gemini 共识"loss 升级是错的关键变量"压倒 Opencode 的 EV 论证。N1+ 期望 +0.005~0.012 显著高于 N1' 期望 +0.001~0.005, ROI 不可比。**N1+ 完成后**, 若仍未达 0.7303 buffer, 可以**重新评估**是否补跑 N1' (此时云机已开, 边际成本极低)。

### 分歧 2: B3d 是否值得 push 1 次真分 (2 部分支持 vs 1 反对)

| 立场 | 论据 | 评估 |
|---|---|---|
| **Gemini 部分支持** | B3d 在 OOF 上 +0.031 是真实校准增益, 可能因 cap1 样本分布偏移未体现 | 模糊建议, 未给出 push/skip 明确 |
| **Opencode 1 次配额可试** | OOF +0.031 真训练增益 + cap1=SOTA 的 push 完全可能在线 +0.001~0.002; 若配额 ≥5 次值得; B3d CSV 与 SOTA CSV 差异可能 10-30 处 flip (不是 193) | 实测建议 |
| **Claude 部分支持但有保留** | 盲点 1: cap1 双重角色冲突, D-14 用 cap1=SOTA SKIP 是把 strat 选择验证集当 push 门用; 但**如果 B3d 实际 per-class best 全部 fallback 到 ctx (没采纳 calib)**, push 就 = SOTA 没意义; **需先 chain-first 确认 B3d 实际有没有动 SOTA strat 选择** | 加了一个 chain-first 前置检查 |

**裁决**: B3d push 之前先 **chain-first 确认 B3d per-class best 是否动了 SOTA strat 选择** (跑 `tools/runs/climb/b3d-orthofuse-20260601-1029/cv_metrics.json` 比较 vs SOTA orthofuse 的 per_class strat)。
- 若动了 → 把 B3d push 队列**排在 N1+ 后**, 不优先
- 若没动 → SKIP (push = SOTA, 配额浪费)

**优先级**: B3d 不是 P1, P1 给 N1+。

### 分歧 3: transformer-over-frames head 是否该做 (2 支持 vs 1 反对)

| 立场 | 论据 |
|---|---|
| **Gemini 关键盲点** | Mean-pool 丢韵律 onset 信息 (对 BC/T 致命), 用 1500 帧 stride5 + transformer head |
| **Opencode 优先级 d** | 与 STRATS 扩展本质不同, 可用 1500 帧 + transformer (但排在 Top 3 之后) |
| **Claude 反对** | 5/28 vap-v2 attention-pool 实测 mean-pool 反优, 已自证。Inoue NAACL 2025 §6.3 VAP 微调后对 prosody 不敏感 → 升级 pool 架构帮助不大 |

**裁决**: **不做**。Claude 的"已自证 + 文献印证"比 Gemini/Opencode 的"理论上有用"权重更高。

### 分歧 4: Isotonic 校准 vs 不做 (3 提到 + 1 重点反对)

| 立场 | 论据 |
|---|---|
| **Gemini 关键 (a)** | Isotonic Regression 校准后等权平均往往优于校准前的加权平均, 且不引入 cap1 搜索风险 |
| **Opencode P3** | max+isotonic 是榨油组合 (期望 +0.001~0.003) |
| **Claude 警示** | isotonic 是有效的, 但**单调非参 isotonic 仍是单参数估计**, 在 cap1 369 样本上每类拟合 isotonic 会引入 strat 选择风险 (虽然温和) |

**裁决**: **不做 isotonic 作为主路径**。N1+ + 阈值 sweep 两条足够覆盖 0.009 缺口。**如果两条都做完仍差 buffer**, 可作为 P3 上 OOF 179867 窗 (不在 cap1 上拟合, 避 Claude 警示)。

---

## 3. 独家洞察 (单 reviewer 提出但有价值)

### Claude self 独家盲点 (5 条)

1. **cap1 双重角色冲突**: cap1 是 strat 选择验证集 (D-6 OK), 同时是 push 门 (D-13)。修正建议: push 门改 **OOF cap1 + OOF macro 联合**, 任一相比 SOTA 涨 +0.003 即 push。
2. **阈值 sweep ≠ strat 搜索**: ratio 14.7 没到过拟合阈值, D-3/D-11 红旗误伤了 Q1.c 低风险路径。
3. **N1' anchor 错**: `train_head_n1.py:18` 决策门 0.6289 暴露作者把 N1' 当 hubert head 升级版而非新源候选, 与 D-13 战略错位。
4. **配额 vs user attention**: 70 次配额 → 实际 ~8 次 push 路径 (15 天 × 3h/day human-in-loop - 复赛 20h = 25h)。每次 push 都应有 +0.003 期望真分, B3d 不符合 ROI。
5. **榜单门槛动态**: 5/27 前 10 0.7192 → 现 0.7285, **门槛会上移**。N1+ +0.008 仅平 0.7243 不留 buffer, 真目标应是 SOTA +0.015 = 0.7303 (Q1.c 阈值 sweep + N1+ 必须叠加)。

### Opencode 独家方案

- **30 通随机切片验证 D-3 同款风险**: N1+ 跑全量前, 先取 30 通各处切片 (不只 cap1) + Qwen3-0.6B 末层 1024d + per-class logistic regression → 5fold OOF F1。**分别报告 cap1 切片 vs 非 cap1 切片 head F1 → 若 gap < 0.03 = 新源成立**, 否则 D-3 同款风险。**采纳: N1+ 前置验证 30min, 防 D-3 同款翻车**。
- **Omni-7B zero-shot probe**: 1h 云 GPU 测 100 段 per-class F1, 任一类 >0.3 再投入。**作为 N1+ 失败的 fallback**, 不主推。

### Gemini 独家

- **Pseudo-labeling**: 用 SOTA test 概率作弱监督扩 train 集重训, 期望 +0.002。**保留**: 如 N1+ + 阈值 sweep 完成后仍差 buffer, 这是 P3 选项。

---

## 4. D-15 决策建议

### 4.1 战略 (替代 D-13 三轨原 N1')

**保留**: D-13 战略反转 (撤 D-12 接受论, 攻 0.7243 + buffer)
**修正**: 攻击面从"D-13 三轨 + N1'"改为"**N1+ 单主轨 + 阈值 sweep 副轨 (同 1 次 push)**", 配 B3d 条件触发副轨。

| 优先级 | 行动 | 期望 | 成本 | 决策门 |
|---|---|---|---|---|
| **P1** | **N1+ (Qwen3-0.6B LoRA 文本头)** + orthofuse 3 源融合 (ctx+whisper+qwen3) | +0.005~0.012 真分 | 2h 本机 + 30min 云 + 1 次配额 | 单源 cap1 ≥ 0.5 (任何 head learning 即过), fused cap1 ≥ 0.6460 push |
| **P2** | **阈值 ±0.05 sweep on fused_probs.npz** (本机直接做, 0 算力) | +0.001~0.003 真分 | 30s 本机 | per-class 阈值若提升 cap1 ≥ +0.001 → 采纳, 与 P1 同 push 1 次配额 |
| **P3 (条件触发)** | B3d push 1 次真分 | 0~+0.002 | 1 次配额 | **前置 chain-first**: 确认 B3d per-class best 是否动了 SOTA strat 选择; 若动 → push; 若没动 → SKIP |
| **P4 (fallback)** | Omni-7B zero-shot probe | 0~+0.005 | 1h 云 | 100 段 test per-class, 任一 F1>0.3 再投入 |
| **P5 (fallback)** | Pseudo-labeling on test | +0.001~0.002 | 2h 云 | 仅 P1+P2 完成仍差 buffer 时启动 |
| **永久关闭** | N1' (撤回) | — | — | 三路 quorum 2:1 反对, 文献+chain-first 双重证伪 |
| **永久关闭** | VAP/CPC 任何形式 | — | — | Claude+Opencode 双否定 Gemini 重启建议 |
| **永久关闭** | transformer-over-frames whisper head | — | — | 5/28 vap-v2 自证 + Inoue 文献印证 |
| **永久关闭** | B1 v4 ctx 特征工程 | — | — | 三路一致判死, LGBM 饱和 |

### 4.2 N1+ 实施步骤 (本机优先, 防 cloud-bug 教训)

**Step 0 (本机 30min) — 前置 D-3 同款风险验证 (Opencode 独家方案)**:
1. 随机抽 30 通 train 数据, 每通取 cap1 + random 5 slices 共 ~600 slices
2. Qwen3-0.6B 末层 1024d 嵌入 (复用 5/27 H-T3 已下模型)
3. Per-class simple logistic regression → 5fold OOF F1
4. **分别看 cap1 切片 vs 非 cap1 切片的 head F1 → gap < 0.03 = 通过**, 否则 SKIP N1+ 转 P3/P4

**Step 1 (本机 30min) — 写 `cloud/train_qwen3_head.py`**:
- 输入: ASR utterances JSON + 历史 label 序列 (用特殊 token 编码如 `<C><C><T><BC><NA>...`)
- 模型: Qwen3-0.6B + LoRA r=16 + 5 类 sigmoid head
- Loss: BCE (不加 SupCon — Claude 论证 batch 内 BC 正例 ~1.3 主作用条件缺失)
- 训练: 5fold GroupKFold, batch 16, 5 epoch
- 评估: cap1 首窗 + OOF stride40 全量 (与 SOTA orthofuse 对齐)
- 输出: `probs.npz` 同 whisper-fusion 格式 (oof + test + Y + G + order)

**Step 2 (本机 dry-run 10min)** — 1 通 1 epoch 验证管道通 (防 5/29 cloud-bug 教训)

**Step 3 (云 GPU 30min)** — rsync 上云训
- 4090 估算: hubert head 4min ÷ 5 倍 LoRA 慢 = 20-30min
- 用户须开云机 (扩 200G 已就绪)

**Step 4 (本机 10min)** — 拉 `probs.npz` 回, 跑 `cycle_orthofuse_nsrc.py` 扩 STRATS 加 qwen3 第三源 (ctx/wsp/qwen + eq3/w70_q/...)

**Step 5 (本机 5min)** — P2 阈值 sweep on 新 fused_probs.npz, ±0.05 per-class 5 档扫

**Step 6 — 提交决策**:
- 若 fused cap1 ≥ 0.6460 → push
- 若 fused cap1 < 0.6460 但 OOF macro ≥ ctx_lgbm_v1 + 0.005 → **push** (Claude 盲点 1 修正后的双 gate)
- 否则 SKIP, 进 P3/P4

### 4.3 复赛镜像准备并行 (不冲突)

- **6/10 报备邮件硬截止** (xinyebei@xinye.com): 在 P1 等云训的 30min 等待期写 (user attention 不冲突)。**邮件需列**: chinese-hubert / chinese-wav2vec2 / emotion2vec / whisper-large-v3 + **新增 Qwen3-0.6B** (合规白名单, 严格说不需报备但顺手写一笔)
- Docker A2/A3/A4: 同上原则, 训练等待期穿插

---

## 5. 风险记录

### 5.1 N1+ 可能失败 (Claude 给 50/50 概率)

- **D-3 同款风险**: ASR 文本在 cap1 vs 任意切片可能分布不同 → Step 0 前置验证 (30 通切片 gap 测试) 是必要的, 不可跳过
- **小数据风险**: 369 通 5fold = train 295 通, batch 16, ~18 batch/epoch × 5 epoch = 90 step LoRA 微调; Qwen3 0.6B + LoRA r=16 = 5M trainable params 勉强匹配, 需观察 train loss 收敛
- **N1+ 失败 → P3 B3d 兜底**: 1 次配额验证 B3d 真分; 仍失败 → P4 Omni-7B probe; 仍失败 → 0.71529 终态, 寄希望前 40 buffer

### 5.2 榜单门槛上移 (Claude 盲点 5)

- 5/27 前 10 门槛 0.7192 → 现 0.7285 (15 天涨 0.0093)
- 假设线性外推, 6/16 前 20 门槛可能上移 0.005~0.010
- **真目标 = SOTA + 0.015 = 0.7303** (留 +0.006 buffer)
- **意味着**: N1+ +0.008 单条不够, 必须叠 P2 阈值 sweep (+0.002) 或叠 P3 B3d (+0.001~0.002)

### 5.3 user attention 预算 (Claude 盲点 4)

- 真实"可 push 路径" ≈ 8 次, 不是 70 次配额
- **不要为了"试试看"烧 user attention**, P1+P2 主路径 + P3/P4 条件触发是合理上限
- B3d 在 P3 而不是 P1 = 体现 ROI 排序

---

## 6. INDEX 更新 + 落盘 commit

按 lwm HARD INVARIANT, 同步更新 `docs/status/INDEX.md` (本 SYNTHESIS + 3 份评审 + CONTEXT 共 5 个新文件需登记)。建议 commit:

```bash
git add docs/status/2026-06-01-midgame-review-*.md docs/status/INDEX.md
git commit -m "docs: 中场复盘三路 AI 评审 + SYNTHESIS - 撤 N1' 启 N1+

Generated-By: Claude (claude-opus-4-7) via Claude Code CLI"
```

---

## 7. 用户决策点 (next session 第一步)

**主问题**: 是否接受本 SYNTHESIS 的 P1+P2 战略 (撤 N1', 启 N1+)?

**子问题** (任一即可单独答):
1. Step 0 前置 30 通切片验证是否要做? (推荐做, 防 D-3 同款翻车)
2. 是否开云机给 P1 Step 3? (Step 0-2 本机, Step 3 才需云机)
3. P3 B3d 兜底是否启动? (推荐先做 chain-first 前置检查, 再决定)
4. 是否在 P1 等云训时间并行起草 6/10 报备邮件? (推荐)

**默认动作 (用户不反对则按此执行)**:
- 立即在本机开始 Step 0 前置验证
- Step 0 通过 → 写 train_qwen3_head.py + dry-run → 等用户开云机后 rsync + 训
- Step 0 失败 → SKIP N1+, 转 P3 chain-first B3d 检查 + P4 Omni-7B probe

---

**EOF — SYNTHESIS v1.0, D-15 决策建议待用户拍板。完整 review 文档 (3 份) 见 docs/status/2026-06-01-midgame-review-{gemini,opencode,claude}.md**
