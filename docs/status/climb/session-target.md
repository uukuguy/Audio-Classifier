# climb session target — Audio-Classifier / FinVCup turn-taking

> Updated: 2026-05-27 16:10（climb 启用，尚未跑首个 cycle）

## Current target

**Mode**: best-effort + 积极进取（比赛快速冲分，非产品开发）

**Goal**: 公榜 Macro-F1 前3 ≥0.7357 / 保底前10 ≥0.7192。当前自己无提交，地板分 = 纯上下文 CV 0.59。
榜首极密集（#1→#10 仅 0.028）→ 稀有类（BC F1 仅 0.22）+ 阈值调优是杠杆。

## 运营约束

1. **文件 > conversation memory**：每 cycle 写 runs.csv + hypotheses.yaml results，不在对话累积长报告。
2. **积极进取**：§5 best-effort（first-of-paradigm always push + cheap calibration check < 2h push）。
3. **代码定位 codegraph 优先**。
4. **push = manual-csv**：climb 产 pred_test1.csv + 提示，用户手动提交（每天 2 次）+ 贴回真分。
5. **模型 ≤8B，优先白名单 Qwen**，非 Qwen 需明显增益（合规 6/10 前报备）。

## Budget

| 资源 | 限制 |
|---|---|
| wall_clock | unlimited |
| 公榜提交 | 2/day（手动）|
| context_fill | 85% hard pause |
| 算力 | 本机 MPS（轻）+ 云 GPU（重训练）|

## Active focus（两腿并行，自主推进）

> 用户 5/27 指令：①用 climb 就自主推进，不逐步问（silent mode）②不一定非复现 baseline ③两腿现在并行

- ✅ **H-001 context-only** 公榜 0.7108（排22），gap +0.12 已校准
- **腿 B（本机，立即）**：榨 context-only 破前10 —— K-fold OOF + 更多序列特征（transition n-gram/conv 模式）+ XGB/CatBoost 集成 + 跨折稳健阈值。零依赖分钟级。
- **腿 A（本机，准备中）**：**不复现 baseline**，直接 research Option A 轻量版 —— 冻结编码器**预提取特征缓存**（避每step重跑大编码器）+ MPS 训双声道小头。先装 torchaudio + 下 Qwen2-Audio/Whisper encoder。专攻 BC（音频救 BC 是唯一天花板突破口）。

## 自主决策记录

- 跳过 baseline 复现：架构有漏分点（pool 单向量/单声道），research 指出直接 VAP 式更对口。baseline 仅作代码资产参考（dataset.py 音频IO/ContextLabelEncoder 可复用）。
- MPS 可行性：冻结编码器预提取特征后 MPS 训小头可行，不必一定云 GPU。云作为放大手段（更大编码器/更快迭代），非必需前提。
- 5/27 战略门后用户定：**本机 MPS 训（冻结编码器+特征缓存）**。便宜路线(LGBM+手工特征)穷尽，最优 0.7108。
- 神经编码器路线优先级：**先 H-T3（Qwen3-0.6B 文本编码器，攻 T/I）** 后音频——文本对 T/I 已证有真实增益、模型小下载快 MPS 提特征快；音频(攻BC)留待文本验证后。
- 网络：HF/hf-mirror 可达(HTTP 200)，ModelScope 被墙(000)。从 HF 下模型。
- H-T3 在下 Qwen3-0.6B + 实测 MPS 特征提取速度（绿灯前先走数据流，速度决定可行性）。

## Falsified path（don't ladder）

（暂无）

## Session metadata

- created_at: 2026-05-27T16:10:00
- session_id: 2026-05-27-climb-init（will rotate per /clear）
- phase: 启用完成，待跑首个 cycle

<!-- TARGET-BEGIN (机器可读, check-target.py 读; LB 落/cycle 末自动判定 §4.1) -->
target_metric: online
target_value: 0.724337
<!-- TARGET-END -->

## 2026-06-01 D-13 target 更新

- 旧 target 0.75 (前 3 / 前 10 冲击) 失效, 当前真分 0.71529 = 排行榜第 37 名 (前 40 进复赛, buffer 3 名)
- 新 target = **0.724337** (前 20 真门槛), 缺 +0.009
- 失效条件: 三轨全跑 cap1 <0.6460 或 push 2 次线上无 +0.003 → 回 D-12
- 作战图: `docs/status/2026-06-01-top20-attack-plan.md`
