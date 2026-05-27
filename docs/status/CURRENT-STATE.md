# CURRENT-STATE — FinVCup 2026 Turn-Taking

> 结构快照：架构 / 关键文件 / 焦点。**不写** session 级进度/下一步（归 RESUME）。

**Last updated:** 2026-05-28

## 任务

过去30s（音频+ASR+历史标签）→ 预测未来2s内 5类事件(C/T/BC/I/NA)是否出现。多标签 sigmoid+BCE，**Macro-F1**。目标前3(≥0.7357)/保底前10(≥0.7192)。

## 当前架构焦点

**climb 自主迭代框架驱动**（manual-csv 模式）。9 cycle 后结论：手工特征/冻结pooled特征喂浅模型撞 0.71 墙，**BC 是瓶颈类需端到端神经编码器**。本机所有路线实测穷尽，突破需云 GPU。

SOTA = cycle1 纯上下文 LGBM 0.7108。研究树 1 confirmed / 9 falsified。

## Key Files

| 路径 | 作用 |
|---|---|
| `tools/climb/cycle_context.py` / `cycle_context_v2.py` | 纯上下文 LGBM（cycle1=SOTA / v2） |
| `tools/climb/cycle_audio_fusion.py` / `cycle_text_fusion*.py` | 廉价声学/文本词汇融合（均负结果） |
| `tools/climb/cycle_vap_mel.py` / `cycle_vap_fusion.py` / `cycle_vap_whisper.py` | VAP 双声道 cross-attn（mel/whisper，均否） |
| `tools/climb/extract_text_feats.py` | Qwen3 文本特征去重缓存 |
| `tools/climb/{push,apply-lb-score,eval-local,regen-tree}.sh/.py` | climb adapter（manual-csv） |
| `.claude/climb/` | climb 状态机（hypotheses/runs.csv/calibration/session-state） |
| `docs/status/research-tree.md` | 战略可视化（generated，每 LB 注入 auto-regen） |
| `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` | 决策契约（10 决策） |
| `docs/plans/2026-05-27-turn-taking-audio-RESEARCH.md` | VAP/BC SOTA + 编码器选型 research |
| `baselines/2026_finvcup_baseline/` | 官方 baseline（音频IO/ContextLabelEncoder 可复用，gitignored） |
| `~/.cache/manual_models/` | Qwen3-0.6B / whisper-small / whisper-large-v3（curl 直下） |

## climb paradigm 状态

| paradigm | 状态 | 备注 |
|---|---|---|
| context-only | ✅ confirmed | cycle1 SOTA 0.7108 |
| context-v2 / 声学 / 文本词汇 / Qwen3-pooled / vap-stereo | 🔴 falsified | 见 MEMORY negative_cache |
| 云端 whisper-large-v3 端到端/VAP | 未试 | 需云 GPU |

## 工程栈

Python 3.12 + conda env `deep-research`（torch 2.7.1 + torchaudio 2.7.1 + lightgbm/xgboost + transformers 5.5）。本机 MPS（**必须限线程**）。测试/cycle 脚本 `tools/climb/`。
