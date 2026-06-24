# docs/status/ INDEX

> 打开任何 docs/status 文件前**先读这个索引**，避免基于被取代的结论行动。
> 新建/移动任一 docs/status 文件时同步更新本表（HARD INVARIANT）。

## Active

| 文件 | 是什么 | 状态 |
|---|---|---|
| `2026-06-24-复赛端到端作战图.md` | **★★ 复赛阶段唯一作战入口** — 评测口径(端到端 vs 离线) / S5 组件×60min可行性 / 缺口 / 镜像工程问题 / Phase A-D 路线（含⚡重大修正段） | 🟢 active |
| `2026-06-24-镜像架构设计-H-F1F2.md` | **★ H-F1/F2 镜像架构设计+实现** — 三源干净架构 src/{common,infer_e2e,sources/}, Dockerfile唯一源, tag规范。已实现+构建team26:r3-base-20260624(20.4G), 融合逻辑验证identical初赛0.71755 | 🟢 active |
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
| `2026-06-01-top20-attack-plan.md` | **🚨 前 20 攻坚作战图**：D-13 激活，三轨并行（B4 Knowledge Layer / B3 后处理 / B1 ctx v3）+ push 门 cap1≥0.6460 + milestone | 🟢 active |
| `2026-06-04-submission-strategy.md` | **★ 25 push 真分账本 + 跨切片稳定性 + 6/5-6/16 节奏 + 复赛镜像 R1-R5 合规组合** | 🟢 active |
| `2026-06-01-midgame-review-CONTEXT.md` | **🔍 中场复盘评审输入包**：3 路 AI (Gemini/Opencode/Claude self) 统一输入, 含 SOTA 主路径 + 8 push 摘要 + D-1~D-14 决策 + 4 题套餐 | 🟢 active |
| `2026-06-01-midgame-review-gemini.md` | Gemini CLI 评审 (46 行): Isotonic+Transformer head+多源 fusion+Pseudo-labeling | 🟡 decision-history |
| `2026-06-01-midgame-review-opencode.md` | Opencode (DeepSeek-V4-Pro) 评审 (151 行): Qwen3-0.6B 文本头 + N1' 并行 + Omni-7B probe | 🟡 decision-history |
| `2026-06-01-midgame-review-claude.md` | Claude self 评审 (160 行, 7 arxiv 文献): N1+ LoRA Qwen3 + Wang ICASSP 范式 + 5 项目盲点 | 🟡 decision-history |
| `2026-06-01-midgame-review-SYNTHESIS.md` | **★ 三路汇总 + D-15 建议**: 撤 N1' 启 N1+ + 阈值 sweep, 永久关 VAP/B1 v4/N1' | 🟡 decision-history（D-22 后部分撤） |
| `../plans/2026-06-02-omni-lora-RESEARCH.md` | **Omni-7B Thinker LoRA research**: Context7+WebFetch 双源, Thinker only/hidden_size=3584/q_proj v_proj/mask-aware mean pool | 🟡 decision-history（D-20 单源证伪, D-22 软加翻转） |

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

## 决赛答辩材料 (finals, 持续积累)

> 6/4 建桶, 不预先做完整 PPT. 日常开发顺手塞素材, 7/16 决赛阶段一开始时综合.

| 路径 | 是什么 |
|---|---|
| `../finals/README.md` | 6 个素材桶说明 + 时间节点 |
| `../finals/FINAL-PUSH-TASKS.md` | **★ 初赛剩余时间 6/4-6/16 任务清单 (T1-T5 含动态时长适配 + 复赛 docker + 报备邮件)** |
| `../finals/INNOVATION-CANDIDATES.md` | 创新点候选 (C1-C5 已盘点) |
| `../finals/DECISIONS-HIGHLIGHTS.md` | 决策素材高亮 (D-1~D-25 摘可讲版) |
| `../finals/EXPERIMENT-EVIDENCE.md` | 真分账本 + 跨切片数据 + 可视化候选 |
| `../finals/quotes/` | 金句反思 (session 摘要 + 用户洞察) |
| `../finals/charts/` | 可视化素材 (mermaid + matplotlib + 架构图) |
| `../finals/deep-dives/` | 深度技术 (DD-1~DD-7 题目, 评委追问 backup) |
| `../finals/T5-disclosure-email-draft.md` | **T5 报备邮件草稿 (6/8 21:00 截止, 4 公开模型: whisper/hubert/e2v + Omni3B 白名单)** |
| `../finals/dual-model-fallback-design.md` | **D-28 dual-model fallback 设计 (θ=20s 路由, 3 步实现, 1-2 push 验证, 估真分 +0.009)** |

## External anchors

| 路径 | 是什么 |
|---|---|
| `../../baselines/2026_finvcup_baseline/` | 官方 baseline（自带 .git，gitignored） |
| `../../data/train/`, `../../data/test/` | 赛题数据（gitignored） |
| `../赛题要求.md` | 赛题原文 + 初赛排行榜 |
