# Next-Session Handoff

**Updated:** 2026-06-06 10:25 (D-28 mask 教训 + dual-model + T4 docker 骨架 + 公榜排位校准: 总 #2 (P5 alt-id) / 合规 #3)
**恢复命令:** `/project-state resume`

## TL;DR (3 句)

1. **🏆 6/6 公榜实位 (用户 10:18 校准)**: #1 明天会更好 0.754713 / **#2 我们 P5 SpeechlessAI alt-id 0.747569 (8B 超额不进复赛)** / #3 YanHui 0.747489 / **合规 S5 0.747131 第 3 距 #1 +0.0076 (R4 软加单次量级内)**. 9 个 push 真分回完, 复赛镜像配方锁定 = S5; R4 内**双 SSL_ms 0.03+0.03 必保** (+0.007 核心), **Omni3B 0.05 是峰**, Omni-7B vs 3B 仅 +0.0004 (选 3B 几乎 free lunch).
2. **⚠ T2 mask 训实验大教训**: 本机 sweep 选出 mask=0.4 "最优", 公榜实际跌 -0.021 = sweep 与公榜全栈在 30s 上**完全反向**. 根因: ① 单源 ctx-only ≠ R4 全栈 softadd 放大; ② sweep 评估通 ≠ 公榜测试通分布. **本机评估只能定性, 选超参必须公榜验证**.
3. **战略转向 dual-model fallback**: 单一 mask 模型 (任何 prob) 公榜均匀都比 no-mask 差 -0.001~-0.005. dual-model (长 ctx baseline + 短 ctx mask050) 估真分 **0.7417 = +0.009**. T4 docker 骨架已完成 (ctx-only 390MB), 下一步: 实现 dual-model 路由 + S5 全栈打包.

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

## 6/6 T4 docker 骨架完成 (ctx-only)

**已落盘** (`git status` 未 commit):
- `Dockerfile` + `.dockerignore` + `requirements.docker.txt` (linux/amd64, python:3.12-slim, 390MB)
- `src/__init__.py` + `src/infer.py` (单入口 --ckpt_dir --test_root --output_csv --ctx_mode)
- `models/ctx_only/` (5 LGBM ckpt + thresholds.json + feature_spec.json, ~5.3MB)
- `tools/__init__.py` + `tools/climb/__init__.py`
- `tools/climb/cycle_context.py` (改造: build_train 加 mask_prob 参数 + ckpt dump)
- `tools/climb/build_day8_candidates.py` + `build_r4_mask_truncated.py` + `eval_mask_sweep.py`

**验证三重通过**:
- 1000 段 docker run 出 csv = src.infer 本机 csv = cycle_context.py 原 csv 二进制相同
- 变长入口测: 截短 125 chunk → normalize_ctx_to_375 自动 pad → pos 按预期变化
- docker --platform linux/amd64 build 16s, 单次 run ~5s

## D-28 复赛镜像决策修正 (核心交付)

| 组件 | D-27 原方案 | D-28 修正 |
|---|---|---|
| 主力模型 | S5 | **S5 保持** ✓ |
| ctx 训练 | T2 mask 重训 | ⚠ **不引入单一 mask** (均匀公榜都比 baseline 差) |
| 短 ctx 退化 | mask + T1 归一化 | **dual-model fallback** (长用 baseline, 短用 mask050, 估真分 0.7417) |
| 双 SSL_ms 训 60h | 必做 | **必做** ✓ |
| Omni-3B 训 | 必做 | **必做** ✓ |

## 下次 session 第一步 (优先级排序)

```bash
# 1. resume
/project-state resume

# 2. 决定先做哪个 (用户点头):
#    ① T5 报备邮件 (6/8 截止, 30 分钟)
#    ② dual-model fallback 设计实现 (改 src/infer.py 加 ctx 长度路由, 阈值 15s/20s 待定)
#    ③ A3 R4 全栈 docker 升级 (S5 配方 ckpt 打包 + softadd 融合 + dual 路由)
#    ④ 答辩素材落 finals/ (sweep 矩阵 + 公榜反向 = "评估错配"金料 + 7B vs 3B 对照)
#    ⑤ 今天 6/7 1-2 push (按 D-27 节奏, 不冲分只拿信息)

# 3. commit 当前 git 工作树 (大量未 commit: src/ + Dockerfile + models/ + 4 个新脚本 + 4 个 truncated csv + DECISIONS D-28 + JOURNAL + RESUME)
git add -A
git commit -m "6/6: D-28 mask sweep 教训 + T4 docker 骨架 + 9 push 真分 + dual-model 战略"
```

## Open Questions (待用户确认)

1. **6/7 今天投多少 push?** 按 D-27 = 1-2 push/天拿信息. 公榜校准后: 距 #1 +0.0076 在 R4 软加单次提升量级内 (D-22 +0.011, D-25 +0.007), 冲 #1 不再"必输的赌"; 但 YanHui (#3) 距合规 S5 仅 -0.00036, 一次失败 -0.001 就掉 #4
   - 选项 A: 0 push 今天, 全转 docker / dual-model 实现
   - 选项 B: 1 push, 投 dual-model 模拟 csv (验证 dual 策略真分 ≈ 0.74+)
   - 选项 C: 2 push, 加投复赛镜像答辩素材 csv
   - 选项 D: 1 push 冲 #1 (S5 + perfold 多样性 / 新软加组合), 同时备 dual-model docker

2. **dual-model 路由阈值定多少?** mask050 sweep 显示 15s/20s 都是边界点
   - 选项 A: ≥ 20s 用 baseline, < 20s 用 mask050 (保守)
   - 选项 B: ≥ 15s 用 baseline, < 15s 用 mask050 (激进, 更多场景用 baseline 保 SOTA)
   - 推荐: B (先粗后细, mask050 实测真分曲线还不全)

3. **是否补 mask030 + mask020 公榜验证?** mask040/050 已知, 但**未知 mask 系列的真实公榜峰值**
   - 选项 A: 不补 — 单一 mask 都比 baseline 差, ROI 低
   - 选项 B: 补 mask030 30s + 10s 1 push (找窄峰)
   - 推荐: A (按 D-28 教训, 单一 mask 路线已废)

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

## 当前 git 工作树状态

```
M docs/status/DECISIONS.md          # D-28 已写
M docs/status/JOURNAL.md            # 6/5-6/6 全 entries
M docs/status/RESUME-NEXT-SESSION.md  # 本文件
M tools/climb/cycle_context.py      # build_train 加 mask_prob + ckpt dump
?? .dockerignore + Dockerfile + requirements.docker.txt
?? models/                          # ctx-only ckpt (5 LGBM + thresholds + spec)
?? src/                             # __init__.py + infer.py
?? tools/__init__.py + tools/climb/__init__.py
?? tools/climb/build_day8_candidates.py  + build_r4_mask_truncated.py + eval_mask_sweep.py
?? submission/truncated-validation-20260604/R4_mask040_keep125_ctx10s/
?? submission/truncated-validation-20260604/R4_mask040_keep375_ctx30s/
?? submission/truncated-validation-20260604/R4_mask050_keep125_ctx10s/
?? submission/truncated-validation-20260604/R4_mask050_keep375_ctx30s/
```

下次 session **必须先 commit** 这批改动 (≥ 10 文件), 否则 fresh clone 就丢全部 docker 骨架 + D-28 决策.
