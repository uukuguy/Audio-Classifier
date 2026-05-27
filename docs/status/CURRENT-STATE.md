# CURRENT-STATE — FinVCup 2026 Turn-Taking

> 结构快照：架构 / 关键文件 / 当前焦点。**不写** session 级进度或下一步（那归 RESUME）。
> 更新时机：架构层级变化 / 关键文件移动 / 焦点切换。

**Last updated:** 2026-05-27（项目初始化）

## 任务

过去 30s（音频+ASR+历史标签）→ 预测未来 2s 内 5 类事件（C/T/BC/I/NA）是否出现。多标签 sigmoid+BCE，**Macro-F1** 评分。目标前 3（≥0.7357）/ 保底前 10（≥0.7192）。

## 当前阶段

**Discuss 已完成** → 决策契约见 `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md`。
下一阶段：重写 CLAUDE.md（✅ 本次完成）→ writing-plans → EDA → bake-off。

## 架构焦点（计划，尚未实现）

- EDA 优先 + 3 方案 bake-off（可微调语音编码器 / 强化 baseline 融合 / 纯上下文标签基线）
- 稀有类专项：focal loss + 逐类阈值搜索（最高杠杆）
- 验证：按会话划分 + 验证集构造成与 test 一致的独立 30s 切片
- 算力：本机 MBP MPS 做开发/EDA，云 GPU 做正式训练

## Key Files

| 路径 | 作用 |
|---|---|
| `baselines/2026_finvcup_baseline/src/data/dataset.py` | 样本构造 / wav 切片 / text 拼接 / collate（复用） |
| `baselines/2026_finvcup_baseline/src/models/multimodal_baseline.py` | Whisper+Qwen+ContextLabel+Handcrafted 融合（方案 B/C 起点） |
| `baselines/2026_finvcup_baseline/src/utils.py` | `compute_multilabel_metrics`（Macro-F1，写死 0.5 阈值） |
| `baselines/2026_finvcup_baseline/src/infer_test.py` | test 推理 → pred.csv（支持 --threshold） |
| `data/train/`, `data/test/` | 数据（gitignored，已解压） |
| `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` | 决策契约 |

## 工程栈

Python 3.12 + uv + PyTorch 2.7.1（本机 MPS / 云 CUDA）。新代码走 `src/`，测试 `tests/{branch}/`，docs 中文。
