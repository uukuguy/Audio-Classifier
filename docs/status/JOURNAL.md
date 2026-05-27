# JOURNAL — FinVCup 2026 Turn-Taking

> Append-only。一行一个事实（commit/验证/弃路/决策）。永不改历史，修正写新行。
> 格式：`## YYYY-MM-DD` 下 `- HH:MM <事实> [commit]`

## 2026-05-27

- 11:57 /discuss 完成，写 CONTEXT.md（9 决策）：EDA 优先+3 方案 bake-off，本机 MPS 开发+云 GPU 训练，按会话划分+30s 切片验证
- 11:57 实测确认任务=turn-taking 多标签预测（非泛化音频分类），数据 8kHz 双声道电话，类极不均衡 BC0.5%/T1.2%/I4%
- 11:57 排行榜目标确认：#1=0.7475 #3=0.7357 #10=0.7192，#1→#10 仅 0.028（榜首密集，稀有类+阈值是杠杆）
- 12:12 删除 .claude/rules/climb.md 残留 symlink（RL 跑分用，与本任务无关）
- 12:12 重写项目 CLAUDE.md（去 Fusion-Control 残留，填本赛题事实+long-task patterns），初始化 docs/status/
