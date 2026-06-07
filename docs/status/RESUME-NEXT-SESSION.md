# Next-Session Handoff

**Updated:** 2026-06-07 13:15 (V1/V2 真分回 → R4 全栈 dual-route ctx 3 候选就绪 D1/D2/D3)
**恢复命令:** `/project-state resume`

## TL;DR (4 句)

1. **✅ V1/V2 公榜真分回完, dual-model 路线公榜实证**: V1 (全 30s) = 0.710789 = 5/27 cycle1 一字不差 → 路由实现 100% 正确; V2 (一半 10s 一半 30s) = 0.720935 = **+0.010 vs V1** → dual-model 在动态长度上首次公榜实证.
2. **🎯 R4 全栈 dual-route 3 候选已就位 (commit 29c0238, 13:08)**: `submission/probe-day9-r4dualv2-20260607-1308/{D1,D2,D3}/`. D2 sanity 已精确复现 R4 baseline pos 975/947/80/15/528. D1 真候选 押"V2 ctx +0.010 经 softadd 进 R4 估 +0.005-0.015 = 0.750-0.760". D3 全 mask050 不路由对照看路由价值.
3. **📤 下次 session 用户立即投 D1 + D3** (D2 sanity 可省): D1 真分回完判 dual-route 在 R4 全栈下真增益; D1 > D3 + 0.005 → 复赛镜像走 dual-ckpt, D1 ≈ D3 → mask050 直接替 baseline 即可不需路由.
4. **⏰ T5 报备邮件 6/8 21:00 截止 (剩 ~32h)**: 草稿就绪 `docs/finals/T5-disclosure-email-draft.md`, 队名 SpeechlessAI, 用户用 531045572@qq.com 发到 xinyebei@xinye.com.

## 关键工件链 (V1/V2 → D1/D2/D3 严格同纲)

| 层 | 工件 | 期望真分 | 实际真分 |
|---|---|---|---|
| baseline ctx-only single | `models/ctx_only/` (cycle1) | — | V1 = 0.710789 ✓ |
| dual-route ctx-only | V1 sanity / V2 截短 | V1=baseline / V2=+0.010 | V1=0.710789 / V2=0.720935 ✓ |
| **R4 全栈 baseline** | S5 = 0.747131 (anchor) | — | 0.747131 (6/5) |
| **R4 全栈 + dual ctx 真候选** | D1 | **0.750-0.760 (押 +0.005-0.015)** | **⏳ 待投** |
| R4 全栈 + 全 mask050 ctx (不路由) | D3 | 0.737-0.755 (押 ≈ S5) | **⏳ 待投** |
| R4 全栈 sanity (全 baseline ctx) | D2 | = 0.7458 | **可省** (pos 已精确复现) |

## 6/6 全部 9 push 真分账本

| 候选 | 真分 | Δ vs SOTA | 关键信号 |
|---|---|---|---|
| **🏆 P5 R4 + omni7b_ms2 0.05** ⚠ 8B超 | **0.747569** | +0.0004 | 7B vs 3B 仅噪声 (答辩 free lunch) |
| **★ S5 R4 + omni3b_ms2 0.05** (anchor) | **0.747131** | — | 6/5 NEW SOTA, 8B 合规, **复赛主力** |
| P2 R4 + omni3b_ms2 0.10 | 0.745997 | -0.001 vs S5 | omni3b 0.05 是峰 |
| R4 baseline 30s anchor | 0.745798 | -0.001 | — |
| P1 S5 + wsp_ms 0.10 | 0.741037 | -0.006 vs S5 | Omni × wsp **不正交**, wsp_ms 饱和 |
| P4 NSOTA07 + omni3b_ms2 0.05 | 0.737658 | -0.001 vs NSOTA07 | **双 SSL_ms 才是 R4 核心 +0.007** |
| **M2 R4 mask050 10s ctx** | 0.737580 | +0.016 vs 10s no-mask | mask050 压回 80% 退化 |
| P3 S5 + e2v_ms 0.05 | 0.736542 | -0.011 vs S5 | Omni 覆盖 e2v, R4 内 e2v 0.03 天花板 |
| M3 R4 mask040 10s ctx | 0.732465 | +0.011 vs 10s no-mask | mask040 救场弱于 mask050 |
| **M1 R4 mask050 30s ctx** | 0.727898 | **-0.018 vs SOTA** | mask 长 ctx 伤 |
| **M4 R4 mask040 30s ctx** | 0.724527 | **-0.021 vs SOTA** | mask040 伤更大 (sweep 反向案例) |
| R4 baseline 10s anchor | 0.721787 | -0.024 | — |
| R4 baseline 5s anchor | 0.707016 | -0.039 | — |

## 6/6 T4 docker 骨架完成 (ctx-only) — 全 committed (75799ad)

**已落盘 commit**:
- `Dockerfile` + `.dockerignore` + `requirements.docker.txt` (linux/amd64, python:3.12-slim, 390MB)
- `src/__init__.py` + `src/infer.py` (双 ckpt 路由 + LF 行尾, sprint 1 升级 84b56bf)
- `models/ctx_only/` (5 LGBM ckpt + thresholds.json + feature_spec.json, ~5.3MB)
- `tools/__init__.py` + `tools/climb/__init__.py`
- `tools/climb/cycle_context.py` (改造: build_train 加 mask_prob 参数 + ckpt dump)
- `tools/climb/{build_day8_candidates, build_r4_mask_truncated, eval_mask_sweep, build_dual_model_validation, train_mask050_ckpt.sh}.py`

**验证四重通过**:
- 1000 段 docker run 出 csv = src.infer 本机 csv = cycle_context.py 原 csv 二进制相同
- 变长入口测: 截短 125 chunk → normalize_ctx_to_375 自动 pad → pos 按预期变化
- docker --platform linux/amd64 build 16s, 单次 run ~5s
- **dual-ckpt 路由测**: 6 单测 + 1000 段端到端 (long/short 同 ckpt mode = single mode binary identical)

## D-28 复赛镜像决策修正 (核心交付)

| 组件 | D-27 原方案 | D-28 修正 |
|---|---|---|
| 主力模型 | S5 | **S5 保持** ✓ |
| ctx 训练 | T2 mask 重训 | ⚠ **不引入单一 mask** (均匀公榜都比 baseline 差) |
| 短 ctx 退化 | mask + T1 归一化 | **dual-model fallback** (长用 baseline, 短用 mask050, 估真分 0.7417) |
| 双 SSL_ms 训 60h | 必做 | **必做** ✓ |
| Omni-3B 训 | 必做 | **必做** ✓ |

## 下次 session 第一步 (按 deadline 排序)

```bash
# 1. resume
/project-state resume

# 2. 用户立即提交 (R4 dual 3 候选已就位):
#    ① T5 报备邮件 (6/8 21:00 截止 = 剩 ~32h, 用户操作 30 min)
#        草稿: docs/finals/T5-disclosure-email-draft.md (队名 SpeechlessAI 已填)
#        用 531045572@qq.com 发, 收件 xinyebei@xinye.com
#    ② 🎯 D1 公榜投 (R4 全栈 dual-route 真信号, 押 +0.005-0.015 vs S5)
#        submission/probe-day9-r4dualv2-20260607-1308/D1_R4_dual_half_truncated/pred_test1.csv
#    ③ D3 公榜投 (R4 全栈 + 全 mask050 不路由, 对照看路由价值)
#        submission/probe-day9-r4dualv2-20260607-1308/D3_R4_all_mask050_30s/pred_test1.csv
#    ④ D2 sanity 可省 (pos 已精确复现 R4 baseline 975/947/80/15/528)
#
# 3. D1/D3 真分回来后判:
#    - D1 > D3 + 0.005 → 路由真有价值, 复赛镜像走 dual-ckpt ctx
#    - D1 ≈ D3 (±0.003) → 路由价值 ≈ 0, mask050 直接换 baseline 更省工程
#    - D1 < 0.740 (远低于 S5 0.7458) → dual-route 在 R4 全栈下反向 (D-28 类), 弃
#
# 4. 平行可做 (不阻塞 D1/D3 公榜回分):
#    ⑤ A3 R4 全栈 docker 升级骨架 (S5 配方 + softadd + dual-ckpt 路由, 按 D1/D3 结论定)
#    ⑥ 答辩素材落 finals/ (V1/V2 dual-model 公榜实证 + D-29 路线终结 + 7B/3B 对照)
#    ⑦ 复赛镜像下一突破方向: 改 R4 内部某个源 (新 hub_ms / e2v_ms / ctx base 重训) — 见 D-29
```

## 路由阈值已锁 θ=20s (250 chunk, D-28 策略 A 保守)

理由: D-28 教训"评估错配"刚被狠狠教训, 不再相信本机线性插值. 先发简单策略 A 拿公榜真分, V2 push 回来后看是否调激进到 θ=15s (策略 B). 文档: `docs/finals/dual-model-fallback-design.md`.

## Open Questions (待用户决策)

1. **D1/D3 用哪个账号投?** D1/D3 是 R4 全栈, 估真分 0.737-0.760 区间, 押 D1 可能涨过 S5. 主账号 (D1 涨 → 涨名次, D1 跌 → 跌名次) vs SpeechlessAI alt-id (不影响合规位). **推荐主账号投** — D1 是真候选, 涨了就锁第 2.
2. **D2 sanity 要不要投?** pos 已精确复现 R4 baseline pos=975/947/80/15/528, sanity 在本机已通过. 投 D2 = 浪费 1 push 配额验已确定的事. 推荐**不投 D2**.

## Ready-to-paste commands

```bash
# 看 D-28 完整决策
sed -n '/^### D-28/,/^## /p' docs/status/DECISIONS.md | head -100

# 复现 mask sweep 矩阵
cat tools/runs/climb/mask-sweep-20260606-0203/matrix.txt

# T4 docker 骨架运行
docker run --rm --platform linux/amd64 \
  -v $PWD/data/test:/data/test:ro \
  -v $PWD/tools/runs/climb/_docker_test:/output \
  finvcup-infer:ctx-only

# 复现 R4 全栈 mask050 / mask040 csv
OMP_NUM_THREADS=4 python3 tools/climb/build_r4_mask_truncated.py --keep 375 --mask-prob 0.5
OMP_NUM_THREADS=4 python3 tools/climb/build_r4_mask_truncated.py --keep 125 --mask-prob 0.5

# 看今天 5 个 day8 候选 (P1-P5 全部真分回完, 别再投同款)
ls submission/probe-day8-20260606-0115/
cat submission/probe-day8-20260606-0115/MANIFEST.json | python3 -m json.tool | head -30

# 复赛镜像核心配方 (S5):
#   R4 = orthofuse-3src + wsp_ms 0.07 + e2v_ms 0.03 + hub_ms 0.03 (软加 T/BC/I)
#   S5 = R4 + omni3b_ms2 0.05 (软加 T/BC/I)
# 真分锚: 30s = 0.747131, 10s (no mask) = 0.7218 → 复赛短 ctx 需 dual-model

# 当前 SOTA 梯队 (6/6):
#   0.747569  P5 R4 + omni7b 0.05 (8B 超额, 答辩素材)
#   0.747131  S5 R4 + omni3b 0.05 ★ 合规 SOTA 复赛主力
#   0.745997  P2 R4 + omni3b 0.10
#   0.745798  R4 baseline NSOTA07+e2v0.03+hub0.03
#   0.741037  P1 S5 + wsp 0.10 (wsp 饱和)
#   0.738899  R5 NSOTA07
#   0.737580  M2 R4 mask050 10s ★★ 短 ctx fallback 主力
```

## 关键不要重走 (D-29 落定)

**D-29 新增 (6/7)**: 单源软加叠加路线终结. 现有素材池上"在 S5/R4 上再加一个 src 软加"全否:
- ❌ S5 + omni7b 0.03 (A 6/7 -0.001, 噪声内但不正交)
- ❌ S5 + qwen17b 0.03 (B 6/7 -0.0056, 跨 LLM 范式不正交)

**D-28 已落定 (6/6)**:
- ❌ 单一 mask 模型路线 (任何 prob, 公榜均匀都比 baseline 差 -0.001~-0.005)
- ❌ mask=0.4 (公榜 30s 伤 -0.021, sweep 反向案例)
- ❌ 信任本机内部 sweep 选超参 (本机定性 ✓, 定量选 ✗)
- ❌ 用 pos 数量级估全栈真分 (softadd 放大效应, pos 几乎不变可能真分跌 0.021)
- ❌ Omni × wsp_ms 叠加 (P1 -0.006, 不正交)
- ❌ S5 + e2v_ms 升权 (P3 -0.011, Omni 已覆盖)
- ❌ Omni3B 权重上探 0.10+ (P2 是峰, 不再扩)
- ❌ R4 + 第 3 个 SSL_ms (S1 +w2v2 -0.008, 三 SSL 撞墙)
- ❌ NSOTA07 单加 Omni3B 跳过双 SSL_ms (P4 -0.001, 双 SSL 是核心 +0.007)
- ❌ 7B vs 3B 多模态升级 (仅 +0.0004 = 噪声)

## 当前 git 工作树状态 (6/7 13:15)

工作树**干净** (除 RESUME 这一改本身), 6/7 增量:
- 8e85a9f: day9 push-1 真分回完 + mask050 训完 + V1/V2 出件 (15 文件)
- 1d93510: D-29 写入 + handoff 反映 V1/V2 就绪投 (3 文件)
- ed1409b: journal 补 1d93510 hash (1 文件)
- 29c0238: 🎯 **R4 全栈 dual-route 3 候选 D1/D2/D3 + mask050-fast + r4_dual_v2 工具** (8 文件)

6/6 历史:
- 75799ad / 84b56bf / 2d9e27d / 499d602 / 24625c3 / 6a13c16 (D-28 sprint 1 完成链)

下次 session resume 后**无需先 commit**, 直接按"下次 session 第一步" → 用户发 T5 邮件 + 提 D1 + D3 csv 即可.
