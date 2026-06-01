# docs/status/ INDEX

> 打开任何 docs/status 文件前**先读这个索引**，避免基于被取代的结论行动。
> 新建/移动任一 docs/status 文件时同步更新本表（HARD INVARIANT）。

## Active

| 文件 | 是什么 | 状态 |
|---|---|---|
| `CURRENT-STATE.md` | 结构快照：架构、关键文件、当前焦点 | 🟢 active |
| `RESUME-NEXT-SESSION.md` | session 交接棒（恢复用 `/project-state resume`） | 🟢 active |
| `JOURNAL.md` | append-only 事件日志 | 🟢 active |
| `DECISIONS.md` | 架构/范式决策账本（D-1 VAP路线证伪 / D-2 转攻T/I） | 🟢 active |
| `2026-05-28-sliced-cv-audit.md` | 切片化验证集审计：cap1 可信 CV(gap+0.118→+0.055)，BC 真瓶颈非假象 | 🟢 active |
| `2026-05-29-lora-finetune-plan.md` | LoRA 微调 whisper-large-v3 攻 BC 方案（冻结路线 falsified 后的下一步） | 🟡 decision-history |
| `2026-05-29-diagnosis-zero-lift.md` | **零提升根因诊断（9-agent workflow）**：真根因=数据规模杠杆非频率错配；下一步3动作（动作1文本按类隔离最高ROI） | 🟡 decision-history |
| `2026-05-30-vap-paradigm-pivot.md` | 范式转向 research（whisper→VAP/CPC）。**VAP 路线已证伪(见 DECISIONS D-1)，文档为历史** | 🔴 superseded |
| `2026-05-30-candidate-models-fusion.md` | **候选模型+融合方案(后面做)**：VAP-HuBERT/MMS仓库自带零成本/双encoder集成/LGBM动作3鲁棒化/LGBM基座只换BC列 | 🔴 superseded(D-1/D-8 否) |
| `2026-06-01-experiment-inventory.md` | **★ 6/1 实验全盘点**：15 个 push 真分账本 + HOT 产物路径 + D-1~D-12 决策摘要 + 遗留任务 IV.A-D | 🟡 decision-history |

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

> **2026-05-30**: climb 状态从 `.claude/climb/`（gitignored，fresh clone 丢）迁到 `docs/status/climb/`（**git-tracked**）。加载分层（见 climb.md §9/§10.4）：resume **只读 research-tree**（含 in-flight 段），其余 storage-layer 按需 grep。

| 路径 | 是什么 | 加载层 |
|---|---|---|
| `.claude/rules/climb.md`（symlink→全局） | climb 执行手册（项目无关） | rules |
| `climb/research-tree.md` | **climb 战略可视化（generated）**：in-flight 段 + paradigm ladder + 假设池 + 诊断链。resume 单文件即够。 | 🟢 resume-load |
| `climb/session-state.json` | session 进度（phase/last_cycle/in-flight/best_online）。research-tree in-flight 段数据源。 | 🟢 resume-load（源） |
| `climb/climb.config.yaml` | adapter：manual-csv、5 类子分、state_dir、SOTA 锚点 | 📦 storage |
| `climb/hypotheses.yaml` | 假设池（paradigm C/B/A/ensemble） | 📦 storage |
| `climb/{calibration,runs.csv,pending-lb,session-target,adjudicator-log}` | climb 状态机（tracked，cycle.sh 写 + regen 读） | 📦 storage |
| `../../tools/climb/*.sh` + `hooks/post-commit` | adapter 脚本 + 确定性同步兜底 hook（fresh clone 重装，见 CLAUDE.md） | tracked |
| `../../tools/climb/cycle_*.py` | 9 cycle 实现 + regen-tree（2026-05-30 加 in-flight 段 + 去 datetime 确定性） | tracked |

## External anchors

| 路径 | 是什么 |
|---|---|
| `../../baselines/2026_finvcup_baseline/` | 官方 baseline（自带 .git，gitignored） |
| `../../data/train/`, `../../data/test/` | 赛题数据（gitignored） |
| `../赛题要求.md` | 赛题原文 + 初赛排行榜 |
