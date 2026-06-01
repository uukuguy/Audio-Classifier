# Multi-AI Quorum 决策历史

> 每次 decision gate 触发多 AI consult 时 append 一条。格式见 climb.md §8。
> 3/3 或 2/3 推 → PUSH；1/3 或 0/3 → SKIP（不 pause）。

（暂无 — 待首个 cycle 触发）

## 2026-05-31 cycle15 — 云端GPU方向 (whisper-ASL vs chinese-hubert第三源)
- claude: B (whisper已榨T/I是融合借的来源, ASL增量不确定; 饱和需全新正交源)
- gemini: B (HuBERT对韵律敏感, 比ASR-whisper更易正交破饱和)
- opencode: B (融合饱和后新源更对路, 中文电话域HuBERT有望贡献正交T/I)
- vote: 3/3 → B (chinese-hubert-large 第三独立源)
- caveat: 非Qwen模型, 需2026-06-10前向 xinyebei@xinye.com 报备 (对外身份→问用户)
- 后续兑现: cycle 16 三源真分 0.71523 ≈ 旧SOTA 0.71529, hubert 无线上增益. D-8 关闭"加更多音频源"路线

## 2026-05-31 cycle17 — 新方向 (融合范式撞 0.715 上限后)
- claude: D (Train 切片末采样, 直接消除 D-8 根因 train/test 分布差; 担心 8x 砍数据欠拟合但本机零成本可验)
- gemini: D (消除分布偏移是根本; 150k 对 LGBM 足够; 极端担心可两阶段 pretrain+continue 或 10:1 weight)
- opencode: D (分布 mismatch 是根因, D 最直接同分布; quality over quantity; 可 sample_weight 给 slice-end 高权重而非硬砍)
- vote: 3/3 → D (Train 切片末采样 / sample_weight slice-end)
- 共识 fallback: 不硬砍数据, 用 sample_weight 给末窗高权重 (10:1) 保留全量信号
