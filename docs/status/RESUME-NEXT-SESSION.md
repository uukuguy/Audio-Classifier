# Next-Session Handoff

**Updated:** 2026-06-07 01:45 (day9 押风险博 #1 失败 → D-29 单源叠加路线终结 + mask050 训完 3min + V1/V2 出件就绪投)
**恢复命令:** `/project-state resume`

## TL;DR (4 句)

1. **🔴 D-29 路线终结**: 在 S5/R4 上线性叠加任何新 src 软加都跌 (wsp/e2v/omni7b 0.03 or 0.05/qwen17b 全 -0.001 ~ -0.011, 6/7 A=0.7462 / B=0.7415). **现有素材池冲 #1 走线性叠加路 已关闭**. 真要冲 #1 必须改 R4 内部某个源 (换更强 hub_ms/e2v_ms/ctx base), 不再单纯软加. 接受 #2/#3 公榜位, 复赛准备压倒.
2. **✅ mask050 ckpt 训完 (commit 8e85a9f, 6/7 01:33)**: 本机 OMP=4 仅 3 分钟 (RESUME 估 5-8h 严重高估). OOF=0.5901, 7 文件落 `models/ctx_only_mask050/`. V1/V2 验证 csv 同步落盘到 `submission/dual-model-validation-20260607-0135/`. V1 sanity LF-binary 等同 single-ckpt baseline csv → 路由不破 ✓.
3. **📤 下次 session 第一动作 = 用户提 V1 + V2 公榜验路由**:
   - V1 全 30s sanity → 估真分 0.7458 (验路由实现不破 baseline)
   - V2 一半 10s 一半 30s → 估真分 0.730-0.745 (验 dual-model 在动态长度上真改善)
   - V1 真分 ≠ 0.7458 → 路由实现有 bug 修
   - V2 < V1 → mask050 路线本身证伪
   - 两者通过 → 进 A3 (R4 全栈 ctx 源升级 dual-ckpt 路由)
4. **⏰ T5 报备邮件 6/8 21:00 截止 (剩 ~43h)**: 草稿就绪 `docs/finals/T5-disclosure-email-draft.md`, 队名 SpeechlessAI, 用户用 531045572@qq.com 发到 xinyebei@xinye.com.

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

# 2. 用户立即提交 (V1/V2 已就位):
#    ① T5 报备邮件 (6/8 21:00 截止 = 剩 ~43h, 用户操作 30 min)
#        草稿: docs/finals/T5-disclosure-email-draft.md (队名 SpeechlessAI 已填)
#        用 531045572@qq.com 发, 收件 xinyebei@xinye.com
#    ② V1 公榜投 (验路由实现不破 baseline, 估真分 0.7458 ± 0.001)
#        submission/dual-model-validation-20260607-0135/V1_full_30s_sanity/pred_test1.csv
#    ③ V2 公榜投 (验 dual-model 在动态长度上真实改善, 估真分 0.730-0.745)
#        submission/dual-model-validation-20260607-0135/V2_half_truncated_to_10s/pred_test1.csv
#
# 3. V1/V2 真分回来后:
#    - V1 ≈ 0.7458 ✓ + V2 > V1 → dual-model 工作, 进 A3 R4 全栈升级
#    - V1 ≠ 0.7458 → 路由实现有 bug, 修 src/infer.py
#    - V2 < V1 → mask050 在 10s 上比 baseline 还差, dual-model 路线证伪
#
# 4. 平行可做 (不阻塞 V1/V2 公榜回分):
#    ④ A3 R4 全栈 docker 升级骨架 (S5 配方 + softadd + dual-ckpt 路由)
#    ⑤ 答辩素材落 finals/ (sweep 矩阵 + 公榜反向 + 7B/3B 对照 + D-29 路线终结)
#    ⑥ 复赛镜像下一突破方向: 改 R4 内部某个源 (换更强 hub_ms / e2v_ms / ctx base 重训) — 看素材池有什么没榨干
```

## 路由阈值已锁 θ=20s (250 chunk, D-28 策略 A 保守)

理由: D-28 教训"评估错配"刚被狠狠教训, 不再相信本机线性插值. 先发简单策略 A 拿公榜真分, V2 push 回来后看是否调激进到 θ=15s (策略 B). 文档: `docs/finals/dual-model-fallback-design.md`.

## Open Questions (待用户决策)

1. **V1/V2 用哪个账号投?** V1/V2 是 ctx-only 路由验证 (不是 R4 全栈), 估真分 0.73-0.745 量级, 远低于 S5 0.747. 主账号 (公榜会暂时掉 #3 → ?) vs SpeechlessAI alt-id (不影响合规排位). 推荐用 alt-id 投, 不动主账号合规位.
2. **6/7 主账号剩 3 push 怎么用?** D-29 后"再加 src 软加"路线关; 现有素材池无新方向. 选项: (a) 不投, 配额留到下个素材变化点; (b) 投 V1/V2 主账号验路由 (但跌名次).

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

## 当前 git 工作树状态 (6/7 01:45)

工作树**干净** (所有改动已 commit), 6/7 增量:
- 8e85a9f: 6/7 day9 push-1 真分回完 + mask050 训完 + V1/V2 出件 (15 文件)

6/6 历史:
- 75799ad / 84b56bf / 2d9e27d / 499d602 / 24625c3 / 6a13c16 (D-28 sprint 1 完成链)

下次 session resume 后**无需先 commit**, 直接按"下次 session 第一步" → 用户发 T5 邮件 + 提 V1/V2 csv 即可.
