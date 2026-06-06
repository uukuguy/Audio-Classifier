# Next-Session Handoff

**Updated:** 2026-06-06 11:50 (sprint 1 完成 + DECISIONS 同步修正 + 流程教训记一笔)
**恢复命令:** `/project-state resume`

## TL;DR (4 句)

1. **🏆 6/6 公榜实位 (用户 10:18 校准)**: #1 明天会更好 0.754713 / **#2 我们 P5 SpeechlessAI alt-id 0.747569 (8B 超额不进复赛)** / #3 YanHui 0.747489 / **合规 S5 0.747131 第 3 距 #1 +0.0076 (R4 软加单次量级内)**. 9 个 push 真分回完, 复赛镜像配方锁定 = S5; R4 内**双 SSL_ms 0.03+0.03 必保** (+0.007 核心), **Omni3B 0.05 是峰**, Omni-7B vs 3B 仅 +0.0004 (选 3B 几乎 free lunch).
2. **⚠ T2 mask 训实验大教训**: 本机 sweep 选出 mask=0.4 "最优", 公榜实际跌 -0.021 = sweep 与公榜全栈在 30s 上**完全反向**. 根因: ① 单源 ctx-only ≠ R4 全栈 softadd 放大; ② sweep 评估通 ≠ 公榜测试通分布. **本机评估只能定性, 选超参必须公榜验证**.
3. **D-28 sprint 1 完成 (6/6 11:28 commit 84b56bf, DECISIONS 同步 24625c3)**: dual-model fallback 工程链全就绪. C (src/infer.py 双 ckpt 路由, 6 单测 + 端到端回归 binary identical) + E (build_dual_model_validation.py V1/V2 工具, fallback 验证通过) + B (train_mask050_ckpt.sh 训练脚本就绪) + T5 报备邮件草稿就绪 (队名 SpeechlessAI). **路由阈值锁定 θ=20s (策略 A 保守, D-28 注解)**. **下次 session 唯一阻塞 = 用户启 mask050 训练 (5-8h 本机 / 1-2h 云端)**, 训完一键出真 V1/V2 push.
4. **流程教训 (24625c3 修正)**: 11:30 commit 499d602 的 Edit 因没先 Read DECISIONS 而失败, 但 commit msg 已宣称改了 = 短暂双重状态 (commit msg ≠ diff). 靠 post-commit hook 二次警告救回, 24625c3 真改 DECISIONS 闭环. **教训**: Edit 报 tool_use_error 立即修, 不带 bug 走下一 commit; post-commit hook 警告是兜底信号不能忽略.

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

# 2. 立即决策 (deadline 顺序, 半天闭环):
#    ① T5 报备邮件 (6/8 21:00 截止 = 剩 ~57h, 用户操作 30 min)
#        草稿在 docs/finals/T5-disclosure-email-draft.md, 队名 SpeechlessAI 已填
#        用 531045572@qq.com 发, 收件 xinyebei@xinye.com
#    ② mask050 ckpt 训练启动 (5-8h 本机 / 1-2h 云端, 用户选)
#        ./tools/climb/train_mask050_ckpt.sh
#        训完产物自动到 models/ctx_only_mask050/ (7 文件 ~5.3MB)
#    ③ V1/V2 验证 csv 生成 (1 min, ckpt 训完后)
#        python3 tools/climb/build_dual_model_validation.py \
#            --ckpt_dir models/ctx_only \
#            --ckpt_dir_short models/ctx_only_mask050 \
#            --out_dir submission/dual-model-validation-$(date +%Y%m%d-%H%M)/
#    ④ 用户 push V1 sanity (验路由不破 baseline, 期望 0.7458 ± 0.001)
#    ⑤ 用户 push V2 real (验 dual-model 实际改善, 期望 0.73-0.745)

# 3. 平行可做 (不阻塞 mask050 训练):
#    ⑥ A3 R4 全栈 docker 升级 (S5 配方 ckpt 打包, ~6G 镜像)
#    ⑦ 答辩素材落 finals/ (sweep 矩阵 + 公榜反向 + 7B/3B 对照)
#    ⑧ 6/7 1 push 冲 #1 (S5 + 新软加组合, perfold 多样性)
```

## 路由阈值已锁 θ=20s (250 chunk, D-28 策略 A 保守)

理由: D-28 教训"评估错配"刚被狠狠教训, 不再相信本机线性插值. 先发简单策略 A 拿公榜真分, V2 push 回来后看是否调激进到 θ=15s (策略 B). 文档: `docs/finals/dual-model-fallback-design.md`.

## Open Questions (待用户决策)

1. **mask050 训练在哪跑?** 本机 (5-8h, OMP=4 不卡) vs 云端 4090 (1-2h, 需先 rsync cycle_context.py 跟最新版同步)
2. **6/7 push 量?** 公榜距 #1 仅 +0.0076 = R4 软加单次量级内; YanHui (#3) 距合规 S5 仅 -0.00036, 易换位. 投 V1/V2 还是单独冲 #1 候选?

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

## 关键不要重走 (D-28 落定)

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

## 当前 git 工作树状态 (6/6 11:50)

工作树**干净** (所有改动已 commit):
- 75799ad: 6/6 D-28 mask 教训 + T4 docker 骨架 + 公榜排位校准 (30 文件)
- 84b56bf: 6/6 D-28 sprint 1 dual-model 工程链就绪 + T5 队名 (8 文件)
- 2d9e27d: 6/6 handoff sprint 1 完成态 (2 文件, RESUME + JOURNAL)
- 499d602: D-28 注解 dual-model θ=20s 锁定 ⚠ Edit 失败漏改 DECISIONS = 短暂双重状态
- 24625c3: **修正** 499d602 漏改 DECISIONS + T5 SpeechlessAI 队名同步 (2 文件)
- 6a13c16: journal 11:32 补 24625c3 hash (1 文件)

下次 session resume 后**无需先 commit**, 直接按"下次 session 第一步"启动 mask050 训练.
