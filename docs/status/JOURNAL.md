# JOURNAL — FinVCup 2026 Turn-Taking

> Append-only。一行一个事实（commit/验证/弃路/决策）。永不改历史，修正写新行。
> 格式：`## YYYY-MM-DD` 下 `- HH:MM <事实> [commit]`

## 2026-05-27

- 11:57 /discuss 完成，写 CONTEXT.md（9 决策）：EDA 优先+3 方案 bake-off，本机 MPS 开发+云 GPU 训练，按会话划分+30s 切片验证
- 11:57 实测确认任务=turn-taking 多标签预测（非泛化音频分类），数据 8kHz 双声道电话，类极不均衡 BC0.5%/T1.2%/I4%
- 11:57 排行榜目标确认：#1=0.7475 #3=0.7357 #10=0.7192，#1→#10 仅 0.028（榜首密集，稀有类+阈值是杠杆）
- 12:12 删除 .claude/rules/climb.md 残留 symlink（RL 跑分用，与本任务无关）
- 12:12 重写项目 CLAUDE.md（去 Fusion-Control 残留，填本赛题事实+long-task patterns），初始化 docs/status/
- 12:14 commit 状态整理（CLAUDE.md+CONTEXT+docs/status），赛题要求.md 用户在改不入此 commit [f9fb23e]
- 12:20 决策更新：模型总参 ≤8B 硬上限，优先白名单 Qwen，非 Qwen 需明显增益（CONTEXT Decision 10）
- 12:25 EDA Q1：test context 分布=train 同分布（C62.6/NA31.7/I4.1/T1.1/BC0.5），确认 test 是同分布切片
- 12:25 EDA Q3 关键：窗口级正样本率 ≠ chunk 级！C94.2/NA65.6/T26.0/I14.1/BC3.7%。T 窗口级不稀有，**BC(3.7%)才是真正最难类**
- 12:25 EDA Q4 关键：纯 context 末段对未来同类共现 lift——BC 4.58x / I 2.83x（最难的两类靠历史标签信号最强）→ 方案C(纯上下文)价值高，音频边际待验证
- 12:35 EDA Q5 决定性：纯上下文标签+LGBM+阈值调优 = Macro-F1 0.59（conv-split valid，3min 训完）。逐类 F1：C0.973/NA0.772/T0.527/I0.463/BC0.219
- 12:35 三结论：①阈值调优免费+0.07（必做）②BC(F1只0.22)是真瓶颈→音频信号必须在此发力 ③方案C不够冲榜但是集成强锚点+分钟级成本
- 13:00 research 关键命中：Apple ICLR2025 "Talking Turns" 用几乎相同标签集{C/T/BC/I/NA}+30s 因果窗+电话语音，冻结 Whisper encoder+线性头，per-class AUC 89-95
- 13:00 research：VAP(Ekstedt/Skantze) 是最对口模板——双声道 cross-attention transformer 预测未来2s 联合状态；BC 是文献公认最难类，SOTA F1 也只 0.35-0.50
- 13:00 research：BC 最大杠杆是文本/ASR(嗯/对/uh-huh 词汇线索)>音频，但音频带 onset/timing；双声道 cross-attn 对 BC/I 结构性关键
- 13:00 research：8kHz 必须上采样到 16k 喂 SSL 编码器；中文电话域首选 TencentGameMate/chinese-hubert-large(317M)；Qwen 路线用 Qwen2-Audio 的 Whisper encoder(640M) 冻结
- 13:00 research：损失用 ASL(γ+=0,γ-=2~4,m=0.05)优于 focal；pos_weight 别用裸 25x；阈值在 pooled OOF 上调，BC 用"跨折最小偏差"阈值防过拟合；soft-F1 受 batch 内 BC 正样本稀少所限慎用
- 13:00 research：baseline 所有编码器都 pool 成单向量(event-level)，与 SOTA"保帧序列"相悖——最大架构改进空间；baseline 音频 IO 已保双声道可直接复用
- 13:10 写 RESEARCH.md（VAP/BC SOTA+编码器选型+Macro-F1 优化+可复用资产）。推荐路径 C(纯上下文锚点)→B(修 baseline 漏分点)→A(双声道帧级VAP式)→集成
- 16:10 决策：climb 改造成通用(adapter 层)+本项目沿用，push_mode=manual-csv(我出CSV/你提交贴分)
- 16:25 通用化 ~/.claude/shared-rules/climb.md：新增 §0 adapter 层+§5.1 push 模式(auto-docker|manual-csv)，去 Fusion-Control 硬编码(标为示例)
- 16:25 本项目落地 .claude/climb/(config+8假设池 paradigm C/B/A/ensemble)+tools/climb/ 5 脚本；manual-csv 闭环烟测通过(push→贴分→calibration gap+0.125)
- 16:25 验证不破坏 Fusion-Control：FC runs.csv 子分列名/push.sh auto-docker 仍合法，仅给 FC 补 climb.config.yaml(显式化隐含 adapter)，未改 FC 现有文件
