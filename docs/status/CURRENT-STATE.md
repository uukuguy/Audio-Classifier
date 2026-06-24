# CURRENT-STATE — FinVCup 2026 Turn-Taking

> 结构快照：架构 / 关键文件 / 焦点。**不写** session 级进度/下一步（归 RESUME）。

**Last updated:** 2026-06-24（阶段切换: 初赛已结束 → 复赛端到端镜像阶段）

## ⭐ 当前阶段: 复赛端到端 Docker 镜像 (6/22-7/9 评测)

> **作战入口: `docs/status/2026-06-24-复赛端到端作战图.md`** (评测口径/组件可行性/缺口/路线全在此)

- 复赛评测口径 = **镜像端到端推理**(挂载私有测试集2 5-30s不定长 → run.sh → submit.csv), 跟初赛"离线拼 probs.npz 出 csv"**根本不同** → 初赛 `probs.npz` 工件对复赛**无用**, 必须用模型权重现场重算。
- 约束: **≤60min 推理 / ≤8B 参数 / ≤32G 镜像 / 每天 2 次提交**。
- 镜像管道**已跑通**(v2=0.488 线上有效, opencode/deepseek 阶段成果: GPU优先+CPU fallback), 但配方是"现配简配", 跟初赛 SOTA 无关。
- 当前镜像 v3 = **R4 简配**(单seed×4源, 无Omni, 无multi-seed), 不是 S5。
- **下方"初赛 cross-context"章节为初赛历史**, 复赛沿用其 SOTA 配方知识但评测方式已变。

---

## (以下为初赛阶段历史快照, 6/7)

**初赛 Last updated:** 2026-06-07 17:30（Cycles 25-30 完成: cross-context S5/R4 全曲线 → 现有素材池穷尽 → 新训练阶段）

## 任务

初赛: 过去30s（音频+ASR+历史标签）→ 预测未来2s内 5类事件(C/T/BC/I/NA)是否出现。多标签 sigmoid+BCE，**Macro-F1**。

**⚠ 复赛任务变体 (D-26, 赛题要求图 1 明写)**: 测试集 2 **上下文动态时长 (0, 30]s** 任意, 不再固定 30s. 预测窗仍 2s. → 全栈需变长适配.

**当前目标层级**（6/7 真分校准）:
- **公榜实际排位**: #1 明天会更好 0.754713 / **#2 我们 P5 0.747569 (8B 超额)** / #3 YanHui 0.747489 / **合规 S5 0.747131 (距 YanHui -0.00036)**
- **距 #1 +0.0076** = 新训练可触及量级
- 6/16 初赛结束剩 9 天 × 5 = 45 push 配额; 6/10 前合规报备截止
- **T5 报备邮件 6/8 21:00 截止**

## 当前阶段：Cross-context 数据收集完成 → 新训练阶段

**Cross-context 核心发现 (D-31, 6/7)**:
- S5 和 R4 **退化斜率完全相同** (10s 都跌 ~3.2%, 5s 都跌 ~5.3%)
- **T (turn-taking) 在所有 ctx 长度完全不变 (528/1000)** — 答辩金料
- 退化 **100% 来自 ctx LGBM** (滚动窗口特征对短 ctx 漂移), SSL/Omni 音频信号天然不依赖 ctx 长度
- 20s 谷底 (0.7176 < 10s 0.7225) 是 LGBM 375-chunk 窗口在 keep=250 截断的共性 artifact, R4 同有
- **现有素材池 pos 级穷尽**: per-seed/per-class/SSL pair 全扫描均无增量

**新训练路线 (优先级)**:
- P0: Omni-3B per-fold 推理 (脚本就绪 `cloud/predict_omni_per_fold.py`, 30min 云端, 解锁 fold 多样性)
- P1: SSL multi-seed 扩到 5 (hubert/e2v/whisper, 云端各 1-2h)
- P2: 新 SSL encoder (chinese-hubert-xlarge / wavlm / 其他)

**所有云实例已关机, 需先启动.**

**SOTA 梯队 (6/7 更新)**:
- **P5 = R4 + omni7b_ms2 0.05 = 0.747569** (9B 超 8B, 仅 +0.0004 vs S5, 答辩"7B→3B free lunch")
- **S5 = R4 + omni3b_ms2 0.05 = 0.747131 ★ 合规 SOTA 复赛主力** (8B 合规 ~5B 总参)
- R4 = NSOTA07 + e2v_ms 0.03 + hub_ms 0.03 = 0.745798
- P2 = R4 + omni3b_ms2 0.10 = 0.745997 (-0.001 vs S5, 0.05 是峰)
- R5 = NSOTA_07 = 0.738899 (单 wsp_ms 0.07)
- M2 = R4 mask050 10s = 0.737580

**Cross-context 退化 (公榜真分)**:

| ctx | S5 | R4 | Δ |
|---|---|---|---|
| 30s | 0.7471 | 0.7458 | +0.0013 |
| 20s | 0.7176 | 0.7182 | -0.0006 |
| 10s | 0.7225 | 0.7218 | +0.0007 |
| 5s | 0.7078 | 0.7070 | +0.0007 |

**复赛镜像配方 (D-30 锁定)**:
```
S5 = R4 + omni3b_ms2 0.05 T/BC/I
R4 = NSOTA07 + e2v_ms 0.03 + hub_ms 0.03
NSOTA07 = orthofuse-3src + wsp_ms 0.07
orthofuse-3src = context (variant-F 5 seed) × whisper × hubert per-class

ctx: 单 baseline ckpt + normalize_ctx_to_375 左 pad NA
不带: mask050 / dual-route
```

## 关键事实校准

- S5 比 R4 好 +0.001 (Omni-3B 常数偏移), 但**不提供跨 ctx 鲁棒性** — 退化斜率相同
- T 在所有 ctx 长度不变 (528), C 是唯一退化源
- cross-context 真分退化 = ctx-only 退化 × ~0.5 (跟 T3 推算吻合)
- 现有素材池组合已全部穷尽 — 再叠 src / 换 SSL pair / per-class 调权 均无 pos 级增量
- 要突破必须**新训练**

## Key Files

### 提交件 & SOTA 工件

| 路径 | 作用 |
|---|---|
| `tools/runs/climb/orthofuse-3src-20260601-1607/fused_probs.npz` | 3src 基座 probs (ctx/wsp/hub) |
| `tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/probs.npz` | wsp_ms 3-seed mean |
| `tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/probs.npz` | hub_ms 3-seed mean |
| `tools/runs/climb/e2v-bcaug-multiseed-20260602-1632/probs.npz` | e2v_ms 3-seed mean |
| `tools/runs/climb/omni-3b-ms2-mean-3seed/probs.npz` | Omni-3B 3-seed mean (S5 用) |
| `tools/runs/climb/omni-lora-20260602-1002/probs_perfold.npz` | Omni-7B 5 fold 独立 probs |
| `tools/runs/climb/omni3b-lora-ms2-seed{1,42,7}-*/probs.npz` | Omni-3B 单 seed probs |
| `tools/runs/climb/w2v2-bcaug-multiseed-20260602-1549/probs.npz` | w2v2_ms 3-seed mean (未用, 三 SSL 撞墙) |

### Cross-context 提交件

| 路径 | 真分 |
|---|---|
| `submission/crossctx-cycle25-20260607/S5_keep125_10s/` | 0.722521 |
| `submission/crossctx-cycle25-20260607/S5_keep63_5s/` | 0.707751 |
| `submission/crossctx-cycle27-20260607/S5_keep250_20s/` | 0.717598 |
| `submission/crossctx-cycle30-20260607/R4_keep250_20s/` | 0.718213 |
| `submission/truncated-validation-20260604/R4_keep125/` | 0.721787 |
| `submission/truncated-validation-20260604/R4_keep63/` | 0.707016 |

### 主路径代码

| 路径 | 作用 |
|---|---|
| `tools/climb/cycle_orthofuse.py` | ★ SOTA 主程 (per-class 跨源正交融合) |
| `tools/climb/gen_variants.py` | ctx 5seed 基座 |
| `tools/climb/build_softadd_candidates_v2.py` | 软加候选生成 |
| `tools/climb/sweep_softadd_oof.py` | OOF 软加扫描 |
| `tools/climb/dynamic_ctx_utils.py` | normalize_ctx_to_375 / simulate_truncated_context |
| `tools/climb/build_truncated_r4.py` | R4 截短 csv 生成 |
| `tools/climb/perfold_softadd_scan.py` | per-fold 软加扫描 |
| `cloud/train_omni_head.py` | Omni Thinker LoRA 训练 |
| `cloud/predict_omni_per_fold.py` | ★ Omni per-fold 推理 (P0 用) |
| `cloud/train_head_bcaug.py` | SSL 头 multi-seed 训练 |
| `cloud/train_head_hubert.py` / `train_head_cuda.py` | SSL 头单 seed 训练 |
| `src/infer.py` | 复赛推理单入口 (含 normalize_ctx_to_375) |

### climb 状态机

| 路径 | 作用 |
|---|---|
| `docs/status/climb/research-tree.md` | 战略可视化 (runs 46 条, cycle 43) |
| `docs/status/climb/session-state.json` | session 动态状态 |
| `docs/status/climb/runs.csv` | 46 push 完整记录 |
| `docs/status/climb/hypotheses.yaml` | 假设池 (部分滞后) |
| `tools/climb/regen-tree.py` | 确定性 regen |

### 决策账本

| 路径 | 作用 |
|---|---|
| `docs/status/DECISIONS.md` | D-1~D-31 完整决策链 |
| `docs/status/JOURNAL.md` | 事件日志 |
| `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` | discuss 阶段决策契约 |

### Finals 素材

| 路径 | 作用 |
|---|---|
| `docs/finals/FINAL-PUSH-TASKS.md` | T1-T5 任务清单 |
| `docs/finals/T5-disclosure-email-draft.md` | 报备邮件草稿 (6/8 21:00 截止) |
| `docs/finals/charts/cross-context-s5-degradation-20260607.md` | ☆ Cross-context 退化曲线答辩图表 |
| `docs/finals/dual-model-fallback-design.md` | D-28→D-30 dual-model 决策链 (答辩素材) |

### 云端资产 (AutoDL 4090, 当前关机)

| 路径 | 作用 |
|---|---|
| 云端 `/root/audio-classifier/cloud/` | 训练/提取/推理脚本 |
| 云端 `/root/.cache/manual_models/` | Qwen2.5-Omni-7B/3B、Qwen3-0.6B/1.7B/4B、chinese-hubert/w2v2、emotion2vec、whisper-large-v3 |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64G whisper stride5 帧特征 |
| 云端 Omni-3B 5-fold ckpt | P0 per-fold 推理需要 |

## 工程栈

Python 3.12 + conda env `deep-research` (torch 2.7.1 + torchaudio 2.7.1 + lightgbm/xgboost + transformers 5.5 + peft)。本机 MPS (**必须限线程**)。云端 AutoDL 4090 (当前关机)。复赛镜像必须在 CUDA 上最终验证。

## climb paradigm 状态

| paradigm | 状态 | 真分 / 备注 |
|---|---|---|
| **S5 = R4 + omni3b 0.05** | ✅ **合规 SOTA** | **0.747131** 复赛主力 |
| **R4 = NSOTA07 + e2v+hub** | ✅ confirmed | 0.745798 |
| context-whisper-hubert-orthofuse 3src | ✅ confirmed | 0.71755 |
| variant-F (ctx 5seed) | ✅ confirmed | 0.71242 |
| context-only LGBM | ✅ confirmed | 0.71079 |
| **Omni-7B per-fold 多样性** | 🟢 待 Omni-3B 验证 | fold 间 std BC=0.077 I=0.063, pos 级 dampened |
| **cross-context 退化曲线** | 🟢 信息收集完成 | S5/R4 斜率相同, T 不变 |
| dual-model fallback | 🔴 D-30 证伪 | R4 全栈 -0.005 |
| mask 训练 | 🔴 D-28 证伪 | 公榜全反向 |
| 单源叠加 (S5 + 新 src) | 🔴 D-29 证伪 | 全 -0.001~-0.011 |
| 全部早期路线 | 🔴 falsified | 见 DECISIONS D-1~D-21 |
