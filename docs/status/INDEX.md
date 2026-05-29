# docs/status/ INDEX

> 打开任何 docs/status 文件前**先读这个索引**，避免基于被取代的结论行动。
> 新建/移动任一 docs/status 文件时同步更新本表（HARD INVARIANT）。

## Active

| 文件 | 是什么 | 状态 |
|---|---|---|
| `CURRENT-STATE.md` | 结构快照：架构、关键文件、当前焦点 | 🟢 active |
| `RESUME-NEXT-SESSION.md` | session 交接棒（恢复用 `/project-state resume`） | 🟢 active |
| `JOURNAL.md` | append-only 事件日志 | 🟢 active |
| `research-tree.md` | climb 战略可视化（generated，1 confirmed/9 falsified） | 🟢 active |
| `2026-05-28-sliced-cv-audit.md` | 切片化验证集审计：cap1 可信 CV(gap+0.118→+0.055)，BC 真瓶颈非假象 | 🟢 active |
| `2026-05-29-lora-finetune-plan.md` | LoRA 微调 whisper-large-v3 攻 BC 方案（冻结路线 falsified 后的下一步） | 🟢 active |

## 跨会话记忆（MEMORY，恢复入口）

| 路径 | 是什么 |
|---|---|
| `~/.claude/projects/-Users-...-Audio-Classifier/memory/MEMORY.md` | 记忆索引（project/reference/feedback 分类） |
| └ `project_status_*` | 当前判断（SOTA/决策门，带日期会过时） |
| └ `reference_negative_cache` | 9 负结果 don't re-explore（验证确定事实） |
| └ `reference_mps_hardware_limits` / `reference_threshold_law` | 实测硬约束 + 阈值铁律 |
| └ `feedback_*` | 协作反馈（先验证再全量 / 收口 / 私有文件） |

## Decisions / 契约

| 文件 | 是什么 |
|---|---|
| `../plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` | discuss 阶段决策契约（10 决策 / 开放问题 / 延后项） |
| `../plans/2026-05-27-turn-taking-audio-RESEARCH.md` | research：VAP/BC SOTA + 编码器选型 + Macro-F1 优化（含可复用资产清单） |
| `../../tests/main/eda_context_baseline.py` | EDA：纯上下文标签基线（Macro-F1=0.59，可运行） |

## climb（LLM 驱动迭代框架）

| 路径 | 是什么 |
|---|---|
| `.claude/rules/climb.md`（symlink→全局） | climb 执行手册（项目无关） |
| `.claude/climb/climb.config.yaml` | 本项目 adapter：manual-csv、5 类子分、SOTA 锚点（gitignored） |
| `.claude/climb/hypotheses.yaml` | 假设池（paradigm C/B/A/ensemble，8 假设）（gitignored） |
| `.claude/climb/{calibration,runs.csv,pending-lb,session-*}` | climb 状态机（gitignored，HARD INVARIANT：state on disk） |
| `../../tools/climb/*.sh` | adapter 脚本（push/apply-lb-score/eval-local/train/consult-ais，tracked） |
| `../../tools/climb/cycle_*.py` | 9 cycle 实现（context/context_v2/audio_fusion/text_fusion×2/vap_mel/vap_fusion/vap_whisper + extract_text_feats + regen-tree） |
| `research-tree.md` | **climb 战略可视化（generated，每 LB 注入 auto-regen）**：paradigm ladder + 假设池 + 诊断链 |

## External anchors

| 路径 | 是什么 |
|---|---|
| `../../baselines/2026_finvcup_baseline/` | 官方 baseline（自带 .git，gitignored） |
| `../../data/train/`, `../../data/test/` | 赛题数据（gitignored） |
| `../赛题要求.md` | 赛题原文 + 初赛排行榜 |
