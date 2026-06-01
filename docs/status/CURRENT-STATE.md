# CURRENT-STATE — FinVCup 2026 Turn-Taking

> 结构快照：架构 / 关键文件 / 焦点。**不写** session 级进度/下一步（归 RESUME）。

**Last updated:** 2026-06-01

## 任务

过去30s（音频+ASR+历史标签）→ 预测未来2s内 5类事件(C/T/BC/I/NA)是否出现。多标签 sigmoid+BCE，**Macro-F1**。目标前3(≥0.7357)/保底前10(**≥0.7285** 真门槛，非首日榜的 0.7192)。

## 当前阶段：初赛收口 + 复赛镜像准备

**初赛 SOTA = `orthofuse-20260531-0319` 真分 0.71529**（双源 ctx+whisper per-class 正交融合，T=w70 / I=whisper）。距前 10 门槛 0.7285 仅差 0.0135，**D-1~D-12 全路径证伪闭合**：①加 N 源（VAP/HuBERT/w2v2/e2v 4-5 源） ②ctx 基座升级（xgb/v2/mlp） ③T/I mlp 子策略 ④LGBM sweep ⑤BC 单类替换 — 每路最大增益 <0.005 凑不到缺口。诚实接受。

**关键事实校准（vs 早期判断）**：
- BC **不是**瓶颈类（早期假设已被证伪）。冻结路线下 BC ≈ 0.22 是极限，所有信号源 r≈0.13 无强信号，"未来 2s 会不会 BC"本身高度难预测（D-4）
- 真增益来自 **whisper/文本帮 T/I**（T 0.667 / I 0.555 > context），而非 BC
- 神经端到端编码器路线全否（VAP/CPC/Omni/HuBERT LoRA 全 falsified，D-1）

**当前阶段动作**：
- 初赛代码评审包 `submission/code-20260601.zip`（42KB）已就绪，待 6/17 TOP 40 公布后上传
- 复赛镜像（Docker + 推理 pipeline）按 6/20-7/7 阶段准备，TOP 30 公布后正式做
- 合规报备（chinese-hubert/w2v2/e2v 等非 Qwen 模型）需 **2026-06-10 前**发邮件给 `xinyebei@xinye.com`

## Key Files

### 提交件

| 路径 | 作用 |
|---|---|
| `submission/code-20260601.zip` | **初赛代码评审包**（42KB，6/17 提交用） |
| `submission/code/{README,MANIFEST}.md` | 中文方案说明 + 文件清单 + 复现流程 |
| `tools/runs/climb/orthofuse-20260531-0319/{pred_test1.csv,fused_probs.npz,cv_metrics.json}` | **真 SOTA 0.71529 工件** |

### SOTA 复现链

| 路径 | 作用 |
|---|---|
| `tools/climb/cycle_orthofuse.py` | ★ SOTA 主程（per-class 跨源正交融合） |
| `tools/climb/cycle_stack_fusion.py` | 4 ctx base OOF 缓存（SOTA 实用 lgbm_v1） |
| `tools/climb/cycle_context.py` / `cycle_context_v2.py` | context-only LGBM（cycle1 / v2 手工特征） |
| `tools/climb/gen_variants.py` | 变体 F 5seed 集成（前 SOTA 0.71242） |
| `tools/climb/sliced_cv.py` | cap1 切片化 CV 协议 |
| `cloud/extract_whisper_cuda.py` | whisper-large-v3 帧特征提取 |
| `cloud/train_head_cuda.py` / `train_head_hubert.py` | 神经小头训练（1280d / 动态 FDIM） |

### climb 状态机

| 路径 | 作用 |
|---|---|
| `docs/status/climb/research-tree.md` | 战略可视化（generated，resume 只读这个） |
| `docs/status/climb/session-state.json` | 动态状态（in-flight / phase / last_cycle） |
| `docs/status/climb/{hypotheses.yaml,runs.csv,calibration.json}` | 📦 storage-layer，按需 grep |
| `tools/climb/{push,apply-lb-score,eval-local,regen-tree}.sh/.py` | climb adapter（manual-csv 模式） |

### 决策账本

| 路径 | 作用 |
|---|---|
| `docs/status/DECISIONS.md` | D-1~D-12 完整证伪链 |
| `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` | discuss 阶段决策契约（10 决策） |
| `docs/plans/2026-05-27-turn-taking-audio-RESEARCH.md` | VAP/BC SOTA + 编码器选型 |
| `baselines/2026_finvcup_baseline/` | 官方 baseline（自带 .git，gitignored） |

### 云端资产（关机中，复赛镜像验证时再开）

| 路径 | 作用 |
|---|---|
| `cloud/{extract_whisper,extract_hubert,extract_w2v2,extract_emotion2vec}_cuda.py` | 4 个云端特征提取脚本 |
| `cloud/{train_head_cuda,train_head_hubert,train_vap,train_lora}.py` | 4 个云端训练脚本（VAP/LoRA 已证伪保留代码） |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64G whisper stride5 帧特征（备份） |

## 工程栈

Python 3.12 + conda env `deep-research`（torch 2.7.1 + torchaudio 2.7.1 + lightgbm/xgboost + transformers 5.5）。本机 MPS（**必须限线程**）。云端 AutoDL 4090D（已关机）。复赛镜像必须在 CUDA 上最终验证。

## climb paradigm 状态（最终）

| paradigm | 状态 | 真分 / 备注 |
|---|---|---|
| **context-whisper-orthofuse** | ✅ confirmed | **★ 真 SOTA 0.71529**（双源 per-class 正交融合） |
| context-only 5seed (变体 F) | ✅ confirmed | 前 SOTA 0.71242（仍是 orthofuse 的 context 基座来源） |
| context-only LGBM (cycle1) | ✅ confirmed | 首个 SOTA 0.71079 |
| 全部其它 paradigm | 🔴 falsified | context-v2 / 廉价声学 / 文本词汇 / Qwen3-pooled / VAP-mel / VAP-CPC / 冻结 whisper-large-v3 / 集成 grid/stacking / context×3-5 源融合 / mlp 基座 / LGBM sweep — 见 DECISIONS D-1~D-12 + research-tree falsified 段 |
