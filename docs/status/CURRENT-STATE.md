# CURRENT-STATE — FinVCup 2026 Turn-Taking

> 结构快照：架构 / 关键文件 / 焦点。**不写** session 级进度/下一步（归 RESUME）。

**Last updated:** 2026-06-06（D-28 mask sweep 教训 + dual-model 战略 + T4 docker 骨架 + S5 NEW SOTA 0.747131 公榜合规第 3 / 总第 2 (P5 8B 超额 alt-id)）

## 任务

初赛: 过去30s（音频+ASR+历史标签）→ 预测未来2s内 5类事件(C/T/BC/I/NA)是否出现。多标签 sigmoid+BCE，**Macro-F1**。

**⚠ 复赛任务变体 (D-26, 赛题要求图 1 明写)**: 测试集 2 **上下文动态时长 (0, 30]s** 任意, 不再固定 30s. 预测窗仍 2s. → 全栈需变长适配 (T1 推理归一化 / T2 train 模拟变长 / T3 cross-context probe).

**当前目标层级**（2026-06-06 用户校准公榜）:
- **公榜实际排位** (6/6 10:18 用户贴): #1 明天会更好 0.754713 / **#2 我们 P5 (SpeechlessAI alt-id) 0.747569 (8B 超额, 不进复赛)** / #3 YanHui 0.747489 / **合规我们 S5 0.747131 (距 YanHui -0.00036)**
- **距 #1 仅 +0.0076** = R4 软加单次提升量级内 (D-22 +0.011 / D-25 +0.007), 冲第 1 重新进入可议区间
- **YanHui (#3) 距合规 S5 仅 -0.00036** = 我们和 #3 极易换位, 一次 push 失败 -0.001 就掉第 4
- 心态转向 (D-27): **复赛准备最优 + 公榜稳 #2/#3** (公榜冲分边际递减但 #1 仍可冲)
- 6/16 初赛结束剩 10 天 × 5 = 50 push 配额; 6/10 前合规报备截止
- T1-T5 任务清单见 `docs/finals/FINAL-PUSH-TASKS.md`

## 当前阶段：D-22 软加范式 + D-25 双 SSL 协同 + D-26 复赛动态时长 + D-27 战略降速 + D-28 mask 教训 + dual-model

**SOTA 梯队 (6/6 更新)**:
- **P5 = R4 + omni7b_ms2 0.05 = 0.747569 公榜总第 2 (SpeechlessAI alt-id)** (9B 超 8B, 仅 +0.0004 vs S5 = 噪声, **不进复赛**, 答辩"7B→3B free lunch"金料)
- **S5 = R4 + omni3b_ms2 0.05 = 0.747131 公榜合规第 3** (8B 合规 ~5B 总参, **复赛镜像主力锚点**, 6/5 NEW SOTA)
- P2 = R4 + omni3b_ms2 0.10 = 0.745997 (-0.001 vs S5, omni3b 0.05 是峰)
- R4 = 0.745798 (8B 合规, D-25 双 SSL_ms 0.03+0.03 协同 +0.007)
- M2 = R4 mask050 10s ctx = 0.737580 (短 ctx fallback 主力, +0.016 vs 10s no-mask)
- R5 = NSOTA_07 = 0.738899 (单 wsp_ms 0.07 软加, 8B 合规)
- 合规 SOTA-3src base = 0.71755 (5/31, ctx + whisper + hubert 正交融合)
- 复赛镜像 5 候选已就位 `submission/finals-20260604/{R1,R3,R4,R5,R6}/` + 6/6 day8 候选 `submission/probe-day8-20260606-0115/{P1-P5}/`

**D-28 mask sweep 教训 + dual-model 战略 (6/6)**:
- 本机 sweep 在 30s 上选 mask=0.4 最优, 公榜实测跌 -0.021 = **完全反向**
- 根因: ① 单源 ctx-only ≠ R4 全栈 softadd 放大 (ctx-only -0.005 → 全栈 -0.018, 4x) ② sweep 评估通分布 ≠ 公榜测试分布
- 单一 mask 任何 prob 公榜均匀都比 baseline 差 -0.001~-0.005 → 路线废
- **dual-model fallback**: 长 ctx 用 baseline + 短 ctx 用 mask050, 估真分 0.7417 = +0.009 vs 单一 mask
- 教训: **本机评估只能定性, 选超参必须公榜验证**

**D-22 核心范式转向**:
- cap1 红旗（D-17/D-19/D-20）系统性错误 — Omni 单源 cap1=0.5649 被红旗判死，但软加 0.2 +0.011 破 SOTA
- **软加 0.2 微小权重 = 真融合范式**：D-1~D-21 大量"证伪"源（Omni/qwen3/e2v/w2v2/F0…）其实信号都在等被低权软加测试
- 重权融合（0.5）普遍过载（-0.027），等权融合普遍内噪（-0.003），**0.05~0.20 软加是正解**

**仍生效的真红旗**（D-22 后修正）:
- ❌ 激进阈值 cherry-pick（D-18，BC 阈值偏离 varF >0.20 实测 -0.048）
- ❌ 重权融合（≥0.5 普遍过载）
- ❌ 等权融合（4 src 等权多次实测 -0.003）
- ❌ 同时扫多维超参（必须单变量轴 — 源 / 权重 / base 一次只换一个）

**新发现（6/2）**:
- **multi-seed 训练自身就涨**（hubert 3 seeds avg cap1=0.6287 vs single-seed 0.6221 = +0.007），跟软加正交
- **per-fold ckpt 各自融合是新维度**（Omni 5 fold test probs fold 间 std BC=0.077 I=0.063 = 真多样性）

## 关键事实校准（vs 早期判断）

- BC **不是**瓶颈类（D-4 早期假设），但 D-22 后软加 0.2 路线下，**BC 仍有可挖空间**（Omni 软加 0.2 时 BC 信号被低权融入有效）
- 真增益来自 **whisper/文本帮 T/I**（T 0.667 / I 0.555 > context）+ **Omni 多模态软加 BC/T/I**
- 神经端到端编码器路线**不是全否**（D-22 撤回 D-20）：Omni 单源 cap1=0.5649 < SOTA 但软加 0.2 = +0.011 ★

## Key Files

### 提交件 & SOTA 工件

| 路径 | 作用 |
|---|---|
| `tools/runs/climb/orthofuse-3src-20260601-1607/{pred_test1.csv,fused_probs.npz}` | **合规 SOTA 0.71755 工件**（8B 内） |
| `tools/runs/climb/probe-5push-20260602-1412/cand2_sota_omni_02/pred_test1.csv` | **NEW SOTA 0.72852 排名 14**（9.4B 超额） |
| `tools/runs/climb/omni-lora-20260602-1002/{fold0-4.pt,probs.npz,probs_perfold.npz}` | Omni-7B 5 fold ckpt + per-fold（多样性源） |
| `tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/` | hubert multi-seed 15 ckpt（cap1=0.6287 +0.007 真涨） |
| `submission/code-20260601.zip` | 初赛代码评审包 42KB（6/17 提交用，待用 NEW SOTA 工件重打） |

### 主路径代码

| 路径 | 作用 |
|---|---|
| `tools/climb/cycle_orthofuse.py` | ★ SOTA 主程（per-class 跨源正交融合） |
| `tools/climb/cycle_stack_fusion.py` | 多 ctx base OOF 缓存 |
| `tools/climb/gen_variants.py` | 变体 F 5seed 集成（context 基座） |
| `tools/climb/sliced_cv.py` | cap1 切片化 CV 协议 |
| `tools/climb/build_softadd_candidates.py` | ★ 软加候选生成（D-22 后核心工具） |
| `tools/climb/sweep_softadd_oof.py` | OOF 软加扫描排序（粗筛，真分校准为准） |
| `tools/climb/build_5submissions.py` | 5 push 配额提交件批量生成 |
| `cloud/extract_whisper_cuda.py` | whisper-large-v3 帧特征提取（云端） |
| `cloud/train_head_cuda.py` / `train_head_hubert.py` / `train_head_bcaug.py` | 神经小头训练（含 multi-seed） |
| `cloud/train_lora_whisper_bcaug.py` / `train_lora_hubert_bcaug.py` | LoRA 微调头 |
| `cloud/train_omni_head.py` | Omni-7B Thinker LoRA 训练 |
| `cloud/predict_omni_per_fold.py` | Omni per-fold predict（per-fold ckpt 多样性源） |

### climb 状态机

| 路径 | 作用 |
|---|---|
| `docs/status/climb/research-tree.md` | 战略可视化（generated，resume 只读这个；**当前滞后 NEW SOTA 待 regen**） |
| `docs/status/climb/session-state.json` | 动态状态 |
| `docs/status/climb/{hypotheses.yaml,runs.csv,calibration.json}` | 📦 storage-layer，按需 grep |
| `tools/climb/{push,apply-lb-score,eval-local,regen-tree}.sh/.py` | climb adapter（manual-csv 模式） |

### 决策账本

| 路径 | 作用 |
|---|---|
| `docs/status/DECISIONS.md` | D-1~D-22 完整决策链（D-22 范式反转） |
| `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` | discuss 阶段决策契约 |
| `docs/plans/2026-06-02-omni-lora-RESEARCH.md` | Omni-7B Thinker LoRA research（Context7+WebFetch 双源） |

### 云端资产（AutoDL 4090，今晚活跃）

| 路径 | 作用 |
|---|---|
| `cloud/{extract_whisper,extract_hubert,extract_w2v2,extract_emotion2vec,extract_bcaug,extract_f0}_cuda.py` | 6 个云端特征提取脚本 |
| `cloud/{train_head_cuda,train_head_hubert,train_head_bcaug,train_omni_head,train_qwen3_head,train_vap,train_lora*}.py` | 8 个云端训练脚本 |
| 云端 `/root/.cache/manual_models/` | Qwen2.5-Omni-7B/3B、Qwen3-0.6B/1.7B/4B、chinese-hubert/w2v2、emotion2vec、whisper-large-v3 |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64G whisper stride5 帧特征 |
| 云端 `/root/runtime/active/<name>.{log,pid}` | 活跃任务日志规范路径（D-21 用户规范化） |

## 工程栈

Python 3.12 + conda env `deep-research`（torch 2.7.1 + torchaudio 2.7.1 + lightgbm/xgboost + transformers 5.5 + peft）。本机 MPS（**必须限线程**）。云端 AutoDL 4090（活跃，每日开机推 head/Omni）。复赛镜像必须在 CUDA 上最终验证。

## climb paradigm 状态（D-22 后）

| paradigm | 状态 | 真分 / 备注 |
|---|---|---|
| **orthofuse + Omni 软加 0.2** | ✅ **NEW SOTA** | **0.72852 排名 14**（9.4B 超 8B） |
| **context-whisper-hubert-orthofuse 3src** | ✅ 合规 SOTA | **0.71755** |
| context-whisper-orthofuse 2src | ✅ confirmed | 0.71523 |
| variant-F (ctx 5seed) | ✅ confirmed | 0.71242 |
| context-only LGBM (cycle1) | ✅ confirmed | 0.71079 |
| **Omni-7B Thinker LoRA 单源** | 🟡 D-22 撤证伪 | cap1=0.5649 单源弱但软加 0.2 +0.011 ★ |
| **multi-seed retrain** | 🟢 新发现 | hubert 3-seed cap1 +0.007 真涨，与软加正交 |
| **per-fold ckpt 多样性** | 🟢 新发现 | Omni 5 fold std BC=0.077 I=0.063 |
| 全部其它 | 🔴 falsified（待 D-22 后重审） | context-v2 / 廉价声学 / 文本词汇 / Qwen3-pooled / VAP-mel / VAP-CPC / 冻结 whisper-large-v3 / 集成 grid/stacking / 等权多源融合 / mlp 基座 / LGBM sweep — 见 DECISIONS D-1~D-21 |
