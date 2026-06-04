# Next-Session Handoff

**Updated:** 2026-06-04 15:00 (R4 NEW SOTA 0.7458 第 4 + D-26 复赛动态时长 + T1/T3 完成)
**恢复命令:** `/project-state resume`

## TL;DR (3 句)

1. **🏆 R4 = 0.745798 排第 4** (NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03 双 SSL 协同, 8B 合规). 距第 3 (0.74603) 仅 +0.0002, 距第 2 (0.747489) +0.0017, 距第 1 (0.75471) +0.009.
2. **⚠ D-26 复赛动态时长**: 赛题要求图 1 明写"测试集 2 上下文 (0, 30]s 任意". 全栈硬编码 30s=375 chunk 需变长适配. T1/T3 已完成实测: **截到 10s ctx 仅跌 0.029 (5%)**, R4 全栈估退化 0.01-0.02 = 真分估 0.72-0.74, 远好于事前估的 0.60-0.70.
3. **战略转向**: 公榜冲分边际递减 (对手在动), **重心转复赛准备** + 公榜稳第 4-5. T1-T5 任务进度见 `docs/finals/FINAL-PUSH-TASKS.md`.

## R4 NEW SOTA 详情 (6/4 9:30 push 真分回)

| 候选 | 真分 | 备注 |
|---|---|---|
| **R4 NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03** | **0.745798** | 🏆 NEW SOTA, 8B 合规 ~1.7B 总参 |
| R5 NSOTA_07 (wsp_ms 0.07) | 0.738899 | 6/4 早 SOTA, 单 SSL 软加 |
| D3 Omni 5fold median + Q5 | 0.736531 | per-fold median ≈ mean (H-D22-11 否决) |
| B_Q5+e2v_ms_w005 | 0.733773 | 单 SSL ms 0.05 软加 Q5 (反降 -0.003) |
| B_Q5+hub_ms_w005 | 0.732356 | 同上 |
| C_NSOTA(wsp→wsp_ms)+omni015 | 0.729264 | base 替换失败 (-0.007) |

**D-25 核心发现**: 单 e2v_ms 0.03 软加 NSOTA_07 真分 -0.0015 (反降), 单 hub_ms 0.03 同样估也 -0.0015, **但两者一起加 +0.0084** = R4 0.7458. **非加法的协同效应**, OOF 完全测不出 (R4 OOF -0.0021 真分 +0.0069 = 3.3x 反向).

## ⚠ D-26 复赛动态时长 (我之前漏读图 1, 用户纠正)

**约束**: 赛题要求图 1 原文: "测试集 2 ... 同时上下文分成动态时长, 即上下文+2s 不再固定为 30s, **在 (0, 30] 之间**"

**应对进度 (T1-T5)**:

| 任务 | 状态 | 完成 % |
|---|---|---|
| T1 推理归一化 (`normalize_ctx_to_375`) | ✅ 实现 + 单元测试通过 | 70% |
| T1 公榜验证 R4 截短 csv | ⏳ csv 就绪 (R4_keep125/63), 等 push | 30% |
| T2 train 变长模拟重训 | ⏳ 未启动 (T3 显示退化小, 可能不需要做) | 0% |
| **T3 cross-context 内部对照** | ✅ **实测完成** | 100% |
| T4 复赛 docker prototype | ⏳ 未启动 | 0% |
| T5 报备邮件 (6/8 前发) | ⏳ 草稿就绪未发 | 0% |

**T3 实测核心数据** (`docs/finals/charts/cross-context-degradation-20260604.md`):

| 上下文 | ctx-only macro F1 | Δ vs 30s | R4 推算真分 |
|---|---|---|---|
| 30s | 0.5797 | base | 0.7458 |
| 20s | 0.5617 | -0.018 | ~0.737 |
| 10s | 0.5505 | -0.029 | ~0.731 |
| 5s | 0.5355 | -0.044 | ~0.724 |
| 2s | 0.5047 | -0.075 | ~0.708 |
| 1s | 0.4945 | -0.085 | ~0.703 |

**含义**: 测试集 2 (0, 30]s 均匀分布估真分 = **0.72-0.74**, 远好于事前估的 0.60-0.70. 主要因 ctx 滚动窗特征 (10/25/50/100/200 chunk) 在短上下文仍有信号 + SSL_ms LoRA 不直接吃 context.

## 6/5 5 push 候选 (已就位)

主推: 留 1 个验 R4 截短 (T1 公榜验证), 其余 4 个继续冲分

| Push | 候选 | 路径 | 期望 |
|---|---|---|---|
| **P1 (复赛验证)** | R4_keep125_ctx10s | `submission/truncated-validation-20260604/R4_keep125_ctx10s/` | 0.728-0.735 (验 T3 推算) |
| P2 | S1 R4+w2v2_ms 0.03 (三 SSL_ms 微叠) | `submission/probe-day7-20260604-1005/S1_R4+w2v2_ms_003/` | 0.747-0.750 冲第 2 |
| P3 | S5 R4+omni3b_ms2 0.05 (R4+LLM 合规) | 同上 S5/ | 0.745-0.752 |
| P4 | S2 NSOTA07+e2v_ms 0.04+hub_ms 0.04 双源升权 | 同上 S2/ | 0.745-0.748 |
| P5 | S4 NSOTA+wsp_ms 0.10 (wsp_ms 右探) | 同上 S4/ | 0.737-0.740 |

或者全部 5 个都用 truncated-validation 探索 R4 在 5s/10s/20s 上下文真分曲线 (答辩金料更厚) — 看 6/5 决定。

## 复赛镜像 5 候选 (`submission/finals-20260604/`, 真分全到齐)

```
0.7458  R4_NSOTA+e2v_ms_003+hub_ms_003     🏆 ★★ NEW SOTA / 复赛镜像首推
0.7389  R5_NEW_SOTA_NSOTA+wsp_ms_007       🟢 复赛主力 备份#2
0.7374  R6_NSOTA+e2v_ms_003_ultralow       🟡 R4 同源对照
0.7362  R3_SOTA+wsb_010_no_wsp_ms          🟡 0 wsp_ms 极端友好版
0.7338  R1_NSOTA+e2v_ms_005_stable         🟢 跨切片最稳 安全网
```

5 个**全部 8B 合规** (~1.5-1.7B), 跨切片 range 0.058-0.061 (R1 最稳). **不要再 push 它们** (真分已锁).

## 排行榜实时 (2026-06-04)

| 排 | 真分 | 队 | 距 R4 |
|---|---|---|---|
| 1 | 0.75471 | — | -0.009 |
| 2 | 0.747489 | — | -0.0017 |
| 3 | 0.74603 | — | -0.0002 |
| **4** | **0.7458** | **我们 R4** | base |
| 5+ | < 0.74 | ... | + |

## 决赛答辩素材桶 (`docs/finals/`)

6/4 建桶, 7/16 决赛阶段一前持续积累. 不预先做完整 PPT.

- `README.md` — 6 桶说明
- `INNOVATION-CANDIDATES.md` — C1-C5 候选 (软加范式 / 双 SSL 协同 / orthofuse / climb 工具链 / cap1 红旗自省)
- `DECISIONS-HIGHLIGHTS.md` — D-1~D-26 摘可讲版
- `EXPERIMENT-EVIDENCE.md` — 25 push 账本 + T3 cross-context 表 + R4 截短验证 csv
- `quotes/` — 用户金句 + 自反思
- `charts/` — T3 cross-context 表 (已落)
- `deep-dives/` — DD-1~DD-7 题目
- **`FINAL-PUSH-TASKS.md`** — 初赛剩余 13 天任务清单 T1-T5

## 下次 session 第一步

```bash
# 1. resume
/project-state resume

# 2. 拿 6/5 push 真分 (用户提供, 必含 R4_keep125 if push 了)
# 3. 跑 calibration_push_results.py 校准
# 4. T1 公榜验证若回 → 写答辩 slide 草稿 (charts/r4-truncated-real-score.md)
# 5. T4 docker prototype 起手 (6/5-6/6 一天)
# 6. 6/8 前发 T5 报备邮件
```

## Open Questions (待用户确认)

1. **6/5 5 push 是否留 1 个给 R4_keep125 截短验证?**
   - 利: 真分回来 = 答辩金 slide + T1/T3 实证
   - 弊: 少 1 个公榜冲分 push (但边际递减, 不太亏)
2. **T2 train 变长重训是否需要做?**
   - 看 R4_keep125 真分: 若跌 < 0.02 → T2 可跳过
   - 若跌 > 0.03 → T2 必须做 (6-8h 训练)
3. **6/10 报备清单是否包含 Qwen3-0.6B/1.7B?**
   - 严格按白名单只有 Qwen3-0.8B, 我们用 Qwen3-0.6B/1.7B 是边缘
   - 安全做法: 主动报备 (草稿在 submission-strategy.md)

## Ready-to-paste commands

```bash
# 看 T3 完整数据
cat docs/finals/charts/cross-context-degradation-20260604.md

# 重生成 R4 截短 csv (其他挡, 如 250 chunk = 20s)
OMP_NUM_THREADS=4 python3 tools/climb/build_truncated_r4.py --keep 250

# 跑 T3 cross-context 实测 (扩大样本)
OMP_NUM_THREADS=4 python3 tools/climb/eval_dynamic_ctx.py

# 看 R4 vs 截短 R4 的 pos 对比
for d in submission/truncated-validation-20260604/*/; do
  echo "=== $d ==="
  python3 -c "import pandas as pd; df=pd.read_csv('$d/pred_test1.csv'); print({c:int(df[c].sum()) for c in ['c','na','i','bc','t']})"
done
```
