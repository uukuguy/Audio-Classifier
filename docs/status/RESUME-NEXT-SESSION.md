# RESUME — Next Session

**Last updated:** 2026-05-27 12:12

## TL;DR

FinVCup 2026 对话轮次预测赛题。Discuss 已完成、CLAUDE.md 已重写、docs/status 已初始化。决策契约在 `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md`。下一步进 writing-plans，把 EDA 必答清单排成第一批任务。

## 已完成（本 session）

- /discuss → `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md`（9 决策契约）
- 重写 `CLAUDE.md`（清理 Fusion-Control 残留），删 climb.md 残留 symlink
- 初始化 `docs/status/`（INDEX/CURRENT-STATE/JOURNAL/RESUME）
- 尚未 commit

## Next steps（具体动作）

1. **commit** 本次状态整理（CLAUDE.md + docs/status/ + docs/plans/CONTEXT）
2. `/superpowers:writing-plans` 读 CONTEXT.md → 出 PLAN.md
3. **第一批任务 = EDA**（CONTEXT Decision 9 清单）：
   - test context 标签分布 vs train 末段 → 是否泄漏未来事件先验
   - 各模态边际贡献（context-only / +text / +audio 的 CV Macro-F1）
   - 稀有类窗口级正样本率（决定重采样比例 + 阈值搜索范围）
   - 8kHz 原生 vs 16k 重采样对语音编码器影响
   - ASR 文本质量/覆盖（test ~8 句 vs train 355 句）
4. 复现并跑通 baseline 拿第一个公榜锚点分

## Open questions

- 云 GPU 选型 + 数据上传方式（实际传 369 通原始 wav + label/text，按需切片）
- 语音编码器选型（中文电话域：WavLM / Chinese-wav2vec2 / Whisper-encoder 解冻）—— bake-off 内定
- 比赛报备/联系邮箱（6/10 前报备）→ 需问用户，勿用系统 userEmail

## Ruled out

- Qwen2.5-Omni 端到端（推理时延=复赛镜像风险，本机难训）→ 延后，触发：集成到 ~0.72 还差临门一脚
- 全程本机 MPS（大模型训练慢 + 复赛 CUDA 镜像无法本机验证）

## Ready commands

```bash
# 跑通 baseline（需先改 config 路径到 data/train）
cd baselines/2026_finvcup_baseline
# 本机冒烟（MPS，小样本）：python -m src.train --epochs 1 --max_train_samples 200
```
