# 前 20 攻坚作战图 — 2026-06-01 启动

> **状态**: 🟢 active（D-13 激活，0.71529 → 0.7243 +0.009 攻坚战）
> **截止**: 2026-06-16 23:59（初赛阶段一结束）
> **决策依据**: 见 DECISIONS.md D-13。本图是执行级展开。

## 形势速读

| 指标 | 值 |
|---|---|
| 当前真分 | **0.71529** |
| 排行榜排名 | **第 37 名**（前 40 进复赛，3 名 buffer 极危险） |
| 前 20 真门槛 | **0.724337** |
| 缺口 | **+0.009**（≈ 2 个独立 +0.005 真信号即可达） |
| 剩余时间 | **16 天**（到 6/16） |
| 提交配额 | 5/天 × 16 = 80 次，预算用 6-7 次 |
| 算力 | 本机 MPS + 云机（关机中，需时再开） |

## 关键数字（D-13 校准的 Push 门）

- cap1 vs 线上 noise floor ≈ **0.003**（D-9 实测）
- 要 push 必须 cap1 macro **≥ SOTA cap1 0.6410 + 0.005 = 0.6460**
- 要破前 20 cap1 macro **≥ ~0.66**（+0.025 vs SOTA cap1）

## 三轨并行（按 ROI 排）

### 🟢 轨道 1: B4 Knowledge Layer（**今天起，0 算力**）

**目标**: 1 天内出方向判断 — 找 D-1~D-12 范围**外**的全新正交信号源。

**动作**:
1. consult-AI（gemini + opencode + Context7）三方咨询 turn-taking SOTA 2025-2026 趋势
2. WebSearch: "turn-taking prediction 2025 SOTA"、"backchannel prediction SOTA"、"VAP 改进版"、"dialogue act prediction"
3. 类似比赛回顾：找类似多标签事件预测 + 语音对话比赛的获奖技术（中科院/讯飞/CMU 杯之类）
4. 读 SSRN / arXiv 近 6 个月 turn-taking 论文
5. 看是否有 D-1~D-12 没覆盖的方向（如: 语义+韵律混合特征 / 多任务学习 / 显式对话状态机 / RLHF / instruction tuning）

**产出**:
- `docs/status/2026-06-01-knowledge-layer-findings.md`（research 报告）
- 决策: 是否有新方向值得启动 B2？B1 是否仍是最高 ROI？

**门槛**: 1 天内出报告，超时则继续 B1/B3 不等。

### 🟡 轨道 2: B3 后处理（**B4 之后/并行，本机 0.5-1 天**）

**目标**: SOTA orthofuse 上叠后处理 / TTA / pseudo-label。期望 +0.001~0.005 偏小但成本极低。

**候选方案**:
- B3a: **Test-time augmentation** — 对 test segment 做轻量 perturbation（噪声 / pitch shift）后 ensemble probs（whisper head 上做）
- B3b: **Pseudo-label self-distillation** — 用 orthofuse fused_probs 做 test 集软标签，加权进 train 重训 ctx base
- B3c: **Sliced TTA on whisper** — 对 30s test segment 做不同 stride 提取多组 frame 特征，averagepool probs

**已有产物**:
- `tools/runs/climb/orthofuse-20260531-0319/fused_probs.npz` — SOTA 5×1000 融合后概率
- `tools/runs/climb/whisper-fusion-20260531-0143/probs.npz` — whisper OOF+test
- `tools/runs/climb/_stack_cache_s40.npz` — 4 ctx base OOF+test

**Push 门**: cap1 ≥0.6460

### 🟠 轨道 3: B1 ctx 特征工程 v3（**中期 1-2 天**）

**目标**: 46d → ~70-80d 改进版 context 特征，重训 4 ctx base + 重做 orthofuse 跨源融合。

**避坑（已 D-12 红旗确认）**:
- ❌ 不在 cap1 369 上选 strat — 保数据规模
- ❌ 不调 LGBM 超参 — D-12 已证 baseline 即最优
- ❌ 不动 BC 单类 strat — D-3/D-11 cap1 cherry-pick 陷阱

**新特征候选**（待 EDA 验证）:
1. **导数特征**: chunk-level 标签概率的 1阶/2阶差分（已有 D-3 测过基础版，v3 应做高阶版）
2. **突发性 burstness**: 短窗口标签密度 stats（C/T 簇集时序）
3. **跨声道韵律统计**: 谁说话 / 沉默间隙长度 / 说话重叠率（双声道）
4. **对话动力学**: 句长分布 / pause-to-speech 比 / 轮转频率
5. **位置/时间**: 30s 窗口在整通对话的相对位置（开头 vs 中段 vs 末段）

**步骤**:
1. EDA 验证每个新特征 vs labels 的 mutual info / 单特征 LGBM AUC（先验证再投入）
2. 增量加 5 个特征进 ctx 特征 → 重训 lgbm_v1
3. cap1 验证 macro ≥0.6460 才进 stack_cache + orthofuse 重做
4. push

**Push 门**: cap1 ≥0.6460

### 🔴 轨道 4: B2 整通对话神经预测（**条件触发，仅 B4 有方向才启动**）

视 B4 Knowledge Layer 结果定 — 如发现需架构换的全新方向 → 启动云上长时序 transformer over 整通对话。

**风险**: 16 天可能来不及做完整 5fold。

## 风险记录

| 风险 | 概率 | 应对 |
|---|---|---|
| B4 空手而归（无新方向）| 中 | 仍跑 B1+B3，期望 0.71529 + 0.001~0.008 |
| B1 ctx v3 增益 <0.003 | 高 | 接受，回 B3 + B4 |
| B3 后处理增益 <0.001 | 中 | 接受，全力 B1 |
| 三轨全死 → 仍 0.71529 → 排名仍 37 | 低（但可能）| 寄希望其他队也撞墙 |
| 其他队 6/2-6/16 push 把我们挤出前 40 | 中 | 持续监控排行榜 |

## D-13 失效条件（红旗）

三轨全跑完 cap1 都 <0.6460，或 push 2 次线上无 +0.003 提升 → D-13 失效，回 D-12 接受 0.71529 + 寄希望其他队不动。

## Milestone

| 日期 | 节点 |
|---|---|
| **6/1** | B4 Knowledge Layer 启动 + B3 草案 |
| **6/2-6/4** | B3 后处理 push 验证（期望 +0.001~0.005） |
| **6/4-6/8** | B1 ctx v3 + 视 B4 启动新方向 |
| **6/8-6/10** | 🔴 合规报备邮件硬截止 + 第二轮 push |
| **6/10-6/16** | 最后一周 polish + 预备复赛 Docker 草稿 |
| **6/17** | TOP 40 公布，提交代码评审包 |
