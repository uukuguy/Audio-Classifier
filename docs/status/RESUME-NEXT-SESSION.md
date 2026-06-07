# Next-Session Handoff

**Updated:** 2026-06-07 16:05 (D1/D2/D3 真分回 → D-30 dual-model 在 R4 全栈下证伪 → 复赛镜像锁 S5 单 baseline ctx)
**恢复命令:** `/project-state resume`

## TL;DR (4 句)

1. **🔴 D-30 dual-model fallback 在 R4 全栈下证伪**: D1 (R4 + dual ctx) = 0.742064 (-0.005 vs S5), D3 (R4 + 全 mask050) = 0.733222 (-0.014). ctx-only V2 +0.010 经 R4 softadd 反向变 -0.005, 跟 D-28 mask 4x 反向放大同形. **复赛镜像锁 S5 单 baseline ctx ckpt, 不上 mask050, 不上路由**.
2. **✅ D2 sanity 六位精度命中 S5 = 0.747131**: 用户投了 D2 (本来推荐省, 但投了反而**消除了"是不是代码 bug"的疑虑** — D1 -0.005 是真信号). 工件链 100% 正确, R4 baseline 完全可复现.
3. **📤 下次 session 公榜剩余配额**: D-29 + D-30 后, 现有素材池"再加 src 软加" + "改 R4 内部 ctx 源" 两条路全关. 真要冲 #1 必须**新训练** (新 hub_ms / 新 e2v_ms / 新 ctx base / 新多模态 LLM 头). 公榜降速观察, 复赛准备压倒 (D-27 重申).
4. **⏰ T5 报备邮件 6/8 21:00 截止 (剩 ~29h)**: 草稿就绪 `docs/finals/T5-disclosure-email-draft.md`, 队名 SpeechlessAI, 用 531045572@qq.com 发到 xinyebei@xinye.com.

## 公榜真分账本 (V1 → D3 完整学习链)

| 候选 | 真分 | Δ vs S5 (0.747131) | 信号 |
|---|---|---|---|
| V1 ctx-only 全 30s single | 0.710789 | — (ctx-only 锚) | 路由实现 100% 正确 (= 5/27 cycle1 一字不差) |
| V2 ctx-only 一半 10s dual | 0.720935 | +0.010 vs V1 | dual-model 在 ctx-only 动态长度上**首次公榜实证** |
| **D2 R4 全栈 sanity** | **0.747131** | **= 0 六位精度** | **R4 baseline 完全可复现, 工件链 100% 正确** |
| **D1 R4 全栈 + dual-route ctx** | **0.742064** | **-0.005** | ⚠️ **dual-route 在 R4 全栈下反向** (ctx-only 涨, R4 全栈跌) |
| D3 R4 全栈 + 全 mask050 ctx | 0.733222 | -0.014 | mask050 直接替 baseline 更伤 |

## 复赛镜像配方 (D-30 锁定)

```
S5 = R4 + omni3b_ms2 0.05 = 0.747131 (8B 合规 ~5B 总参)
R4 = NSOTA07 + e2v_ms 0.03 + hub_ms 0.03 = 0.745798
NSOTA07 = orthofuse-3src + wsp_ms 0.07
orthofuse-3src = context (variant-F 5 seed) × whisper × hubert per-class

ctx 源: variant-F 5 seed te_lgbm_v1 (= R4 baseline 用的, 不换)
短 ctx 应对: normalize_ctx_to_375 左 pad NA (单 ckpt 直接喂, 不路由)
不带: mask050 ckpt / dual-ckpt 路由
保留代码: src/infer.py dual-route 代码留 (--ckpt_dir_short 为空走单 ckpt, 默认关闭)
```

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

## D-28 → D-30 复赛镜像决策最终态

| 组件 | D-27 原方案 | D-28 修正 | **D-30 终态 (6/7)** |
|---|---|---|---|
| 主力模型 | S5 | S5 保持 | **S5 = 0.747131** ✓ |
| ctx 训练 | T2 mask 重训 | 不引入单一 mask | ✓ 不变 |
| 短 ctx 退化 | mask + T1 归一化 | dual-model fallback (估 0.7417) | ❌ **dual-model 在 R4 全栈反向 -0.005 → 弃**; **normalize_ctx_to_375 左 pad NA** 单 ckpt 直接喂 |
| ctx ckpt | baseline + mask050 双 ckpt | dual 路由 θ=20s | **单 baseline ckpt** (variant-F 5 seed te_lgbm_v1) |
| 双 SSL_ms 训 60h | 必做 | 必做 | **必做** ✓ |
| Omni-3B 训 | 必做 | 必做 | **必做** ✓ |
| src/infer.py 路由代码 | (未开发) | sprint 1 写好 | **保留代码默认关闭** (--ckpt_dir_short 空 = 单 ckpt) |
| models/ctx_only_mask050/ | (未训) | 训完 | **保留, 不进复赛镜像** (答辩素材) |

## 下次 session 第一步 (按 deadline 排序)

```bash
# 1. resume
/project-state resume

# 2. 必做 (deadline 顺序):
#    ① T5 报备邮件 (6/8 21:00 截止 = 剩 ~29h, 用户操作 30 min)
#        草稿: docs/finals/T5-disclosure-email-draft.md (队名 SpeechlessAI 已填)
#        用 531045572@qq.com 发, 收件 xinyebei@xinye.com
#
# 3. 复赛镜像方案 (D-30 已锁):
#    ② A3 R4 全栈 docker 升级 — 用 D-30 终态:
#        - ctx 源: variant-F 5 seed te_lgbm_v1 (= R4 baseline 用的)
#        - 不带 mask050 ckpt
#        - src/infer.py --ckpt_dir_short 默认空 = 单 ckpt 路径
#        - softadd 栈: orthofuse-3src + wsp_ms 0.07 + e2v_ms 0.03 + hub_ms 0.03 + omni3b_ms2 0.05
#        - 短 ctx 应对: normalize_ctx_to_375 左 pad NA
#
# 4. 平行可做:
#    ③ 答辩素材落 finals/:
#        - V1/V2 ctx-only +0.010 → R4 全栈 D1 -0.005 反向放大 = "评估错配"金料
#        - D2 = S5 六位精度 = 工件链可信
#        - D-28 → D-30 完整决策链
#        - 7B/3B 对照 / D-29 单源叠加路线终结
#    ④ 公榜配额 (5/天, 剩 ~9 天 × 5 = 45 push):
#        D-29 + D-30 后无新方向, 等"新训练"或"新素材"才有进展.
#        选项 (a) 不投, 配额留; (b) 投 sanity / 跨 ctx 长度对照拿信息.
#        推荐 (a), 不冒进.
#
# 5. 真要冲 #1 (距 +0.0076), 必须新训练:
#    - 新 hub_ms / 新 e2v_ms (换更强 SSL encoder 或 multi-seed 扩到 5+ seed)
#    - 新 ctx base (跟 variant-F 不同算法的 ctx, 如 transformer 替 LGBM)
#    - 新多模态 LLM 头 (Omni-3B 已是局部峰, 换 SeaMoss 等)
#    - 这些都是云端工作, 几小时到几天
#    - D1 < 0.740 (远低于 S5 0.7458) → dual-route 在 R4 全栈下反向 (D-28 类), 弃
#
# 4. 平行可做 (不阻塞 D1/D3 公榜回分):
#    ⑤ A3 R4 全栈 docker 升级骨架 (S5 配方 + softadd + dual-ckpt 路由, 按 D1/D3 结论定)
#    ⑥ 答辩素材落 finals/ (V1/V2 dual-model 公榜实证 + D-29 路线终结 + 7B/3B 对照)
#    ⑦ 复赛镜像下一突破方向: 改 R4 内部某个源 (新 hub_ms / e2v_ms / ctx base 重训) — 见 D-29
```

## 路由阈值 θ=20s — D-30 后**作废**

D-30 已证 dual-model 在 R4 全栈下反向, 复赛镜像不上路由, θ 无意义. 文档 `docs/finals/dual-model-fallback-design.md` 留作答辩素材 (展示"我们试过 dual-model fallback, 公榜证伪, 最后选 normalize 单 ckpt").

## Open Questions (待用户决策)

1. **复赛镜像方案是否一次直接走 D-30 终态?** (S5 单 baseline ctx + normalize_ctx_to_375 左 pad NA). 推荐**直接走** — D-30 已闭环证据, 不再迭代.
2. **公榜剩余 push 配额怎么用?** D-29 + D-30 后无新方向, 选项: (a) 不投, 配额留到新训练完; (b) 投 sanity / 跨切片对照. 推荐 (a) 不冒进.
3. **冲 #1 是否启动新训练?** 距 +0.0076. 候选: 新 SSL encoder / 新 ctx base / 新多模态. 云端工作, 用户决定优先级.

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

# 复赛镜像核心配方 (S5, D-30 锁定):
#   R4 = orthofuse-3src + wsp_ms 0.07 + e2v_ms 0.03 + hub_ms 0.03 (软加 T/BC/I)
#   S5 = R4 + omni3b_ms2 0.05 (软加 T/BC/I)
# ctx 源: variant-F 5 seed te_lgbm_v1 (单 baseline, 不上 mask050, 不上路由)
# 短 ctx 应对: normalize_ctx_to_375 左 pad NA (单 ckpt 直接喂)
# 真分锚: 30s = 0.747131 ✓ D-30 D2 sanity 六位精度命中

# 当前 SOTA 梯队 (6/7):
#   0.747569  P5 R4 + omni7b 0.05 (8B 超额, 答辩素材)
#   0.747131  S5 R4 + omni3b 0.05 ★ 合规 SOTA 复赛主力 (D2 D-30 复现)
#   0.745997  P2 R4 + omni3b 0.10
#   0.745798  R4 baseline NSOTA07+e2v0.03+hub0.03
#   0.742064  D1 R4 dual-route ctx (D-30 证伪 -0.005)
#   0.741037  P1 S5 + wsp 0.10
#   0.738899  R5 NSOTA07
#   0.737580  M2 R4 mask050 10s
#   0.733222  D3 R4 全 mask050 ctx (D-30 -0.014)
#   0.720935  V2 ctx-only dual half (ctx-only 锚)
#   0.710789  V1 ctx-only baseline (= cycle1, 路由 sanity ✓)
```

## 关键不要重走 (D-30 落定)

**D-30 新增 (6/7 16:00)**: dual-model fallback 在 R4 全栈下证伪. ctx-only 涨 ≠ R4 全栈涨:
- ❌ R4 全栈 + dual-route ctx (D1 -0.005)
- ❌ R4 全栈 + 全 mask050 ctx 不路由 (D3 -0.014)
- ❌ "ctx-only V2 +0.010 经 softadd 进 R4 估 +0.005-0.015" 假设 (D-30 实测 -0.005, 4x 反向放大)
- ❌ mask050 ckpt 进复赛镜像 (保留 ckpt 文件作答辩素材, 不带进镜像)

**D-29 (6/7 凌晨)**: 单源软加叠加路线终结. 现有素材池上"在 S5/R4 上再加一个 src 软加"全否:
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

## 当前 git 工作树状态 (6/7 16:05)

工作树**干净** (除 handoff commit 本身), 6/7 增量:
- 8e85a9f: day9 push-1 真分 + mask050 训完 + V1/V2 出件 (15 文件)
- 1d93510: D-29 写入 + handoff V1/V2 就绪投 (3 文件)
- ed1409b: journal 补 1d93510 hash (1 文件)
- 29c0238: R4 全栈 dual-route 3 候选 D1/D2/D3 + mask050-fast + r4_dual_v2 工具 (8 文件)
- 3ef30b6: handoff RESUME 反映 V1/V2 真分 + D1/D3 待投 (2 文件)
- 6bd0ea6: journal 补 3ef30b6 hash (1 文件)
- **本次**: D-30 + RESUME 反映 D1/D2/D3 真分 + 复赛镜像 D-30 终态

6/6 历史:
- 75799ad / 84b56bf / 2d9e27d / 499d602 / 24625c3 / 6a13c16 (D-28 sprint 1 完成链)

下次 session resume 后**无需先 commit**, 直接按"下次 session 第一步" → 用户发 T5 邮件 + 提 D1 + D3 csv 即可.
