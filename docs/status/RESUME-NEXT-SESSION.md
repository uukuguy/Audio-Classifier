# Next-Session Handoff

**Updated:** 2026-06-01 11:21（D-13 第一日 91min 全收口 + 三轨本机证伪 → N1' 云上待启动）
**恢复命令：`/project-state resume`**

## TL;DR

1. **形势**: SOTA 0.71529 = 排行榜**第 37 名**(前 40 进复赛, buffer 3 名极危险), 目标 0.7243(+0.009), 剩 **15 天** 到 6/16.
2. **D-13 第一日完整收口** (6/1 09:30-11:21, 0 提交配额消耗): **三轨本机攻击面全证伪**:
   - ✅ B4 Knowledge Layer (1h): gemini consult + 9 路 WebSearch + arxiv 2 篇. 锁 N1 优先, B2 取消
   - ❌ B3d (校准头 ctx+whisper OOF): OOF +0.031 但 cap1=SOTA → SKIP. **D-14 闭合: 校准头无新源不涨 cap1**
   - ❌ B1 v3 (46d→93d 加 EDA 强特征): OOF +0.0006, cap1 -0.004 → SKIP. **D-12 "46d 榨干"实证**
   - 🟡 N1 原方案 (本机 whisper head 重训) chain-first 否决 (frames 在云端 64GB, 本机不可行)
3. **唯一活路 = N1' 云上**: `cloud/train_head_n1.py` 已写, 派生 train_head_hubert.py, 替换 BCE 成 DB-Loss + α·SupCon. 用 whisper frames + ctx 训, **whisper 单源 BC cap1 0.20 (vs hubert 0.0) = SupCon 有 BC 正样本可学**. ETA 20-30min GPU + rsync.
4. **用户已扩容云机系统盘 200G** ✅ (准备好做 N1' / 复赛镜像)
5. **handoff 时未推进的开放问题**: 是否开云机做 N1', 还是直接收手转复赛镜像准备.

## 攻击面状态 (D-13 攻坚战)

| 轨道 | 状态 | 真实结果 | 下一步 |
|---|---|---|---|
| B4 Knowledge Layer | ✅ | 找 N1/N2/N3 三方向, B2 取消 | 报告完整 |
| B3d (本机 OOF 校准) | ❌ SKIP | OOF Δ+0.031, cap1=SOTA, BC corr 0.69 但 cap1 持平 | D-14 闭合 |
| B1 v3 (ctx 特征工程) | ❌ SKIP | OOF Δ+0.0006, cap1 Δ-0.004 | D-12 验证 |
| **N1' (云上 whisper head DB-Loss+SupCon)** | 🟡 待启动 | — | **等用户决策开云机** |
| N2 (Omni-7B LLM judge) | ⚪ 备 | cycle 11 Omni-3B zero-shot 已踩坑(全答是), Omni-7B 新架构未试 | 仅 N1' 失败再考虑 |
| N3 (韵律 token + Qwen2.5-7B LoRA) | ⚪ 备 | 6-10h 重投入 | 仅 N1'+N2 失败再考虑 |

## 当前阶段时间线

| 日期 | 节点 | 状态 |
|---|---|---|
| **6/1** | D-13 第一日 91min 三轨证伪 + N1' 准备 | ✅ |
| 6/2 (下一 session) | 启动 N1' 云上 + 若 SKIP 走 N2/N3 决策 | 🔄 |
| 6/3-6/9 | 视 N1' 结果, polish + 第二轮 push | ⚪ |
| **6/10 前** | 🔴 合规报备邮件硬截止 → `xinyebei@xinye.com` | ⚪ |
| 6/16 | 初赛阶段一结束 | |
| 6/17 | TOP 40 公布 + 代码评审包提交(已就绪) | 🎯 |

## In-flight（下次 session 第一步）

**🟡 N1' 云上启动** (cloud/train_head_n1.py 已写好):

```bash
# 1. 开云机 (用户操作 AutoDL 后台)
# 2. rsync N1 脚本上云
rsync -avz -e "ssh -p 46379" cloud/train_head_n1.py root@connect.westd.seetacloud.com:/root/audio-classifier/cloud/

# 3. 云端启动 (用 hubert cache 测先, 或重新提 stride40 whisper)
ssh -p 46379 root@connect.westd.seetacloud.com "cd /root/audio-classifier && \
  WCACHE=/root/autodl-fs/hubert_cache OMP_NUM_THREADS=4 \
  setsid nohup python cloud/train_head_n1.py \
    --epochs 15 --alpha 0.3 \
    --run-dir tools/runs/climb/n1-hubert-dbloss-\$(date +%Y%m%d-%H%M) \
    </dev/null >/root/n1.log 2>&1 &"

# 4. heartbeat + 等结果 (ETA 20-30min)
```

**N1' 决策门 (D-13 校准)**:
- 单源 cap1 macro ≥ hubert head baseline 0.6239 + 0.005 = **0.6289** → 进 orthofuse 重做评估
- orthofuse 替换 hubert/whisper 后 cap1 macro ≥ SOTA 0.6410 + 0.005 = **0.6460** → push
- 任一不达 → SKIP, 转 N3 或接受 0.71529

**如果用户决定不开云机**:
- 转复赛镜像准备 (task #6) — 复赛 Docker 草稿 + 推理 pipeline + 6/10 报备邮件
- D-13 失效, 回 D-12 接受 0.71529 + 寄希望其他队不动 (排名 37 → 实际进前 40 概率 80%+)

## Push 门 (D-13 校准)

- cap1 vs 线上 noise floor ≈ 0.003 (D-9 实测)
- **要 push 必须 cap1 macro ≥ 0.6460** (= SOTA cap1 0.6410 + 0.005)
- 要破前 20 cap1 ≈ ≥ 0.66 (+0.025 vs SOTA cap1)
- 低于则 SKIP-advance, 不浪费配额

## 铁律保留 (D-1~D-14 红旗全生效)

- ❌ 不再"加第 N 源" (D-1/D-8/D-10)
- ❌ 不再"在 cap1 369 上选 strat" (D-3/D-9/D-11)
- ❌ 不再"context 内同源算法集成" (D-5)
- ❌ 不再"OOF 校准头无新源" (D-14 — 本次新增)
- ✅ 唯一允许 cap1→线上转化 = **多源融合在 T (150 正例) / I (60 正例) 中等样本类的真实信号叠加**

## 真分账本（5 个 push 全表，跟上次同）

| run_id | strat | cap1 | 真分 | Δ vs SOTA | 备注 |
|---|---|---|---|---|---|
| variant-F-20260528-0559 | 5seed LGBM stride5 | 0.6402 | 0.71242 | base | 旧 SOTA |
| **orthofuse-20260531-0319** | **双源 ctx+whisper (T=w70, I=whisper)** | **0.6410** | **0.71529** | **+0.003** | **★真 SOTA, 排行榜 37** |
| orthofuse-s5-20260531-0627 | 双源 stride5 强基座 | 0.6455 | 0.71233 | -0.003 | 强基座反不如 |
| orthofuse-3src-20260531-0813 | 三源 ctx+whisper+hubert | 0.6540 | 0.71523 | +0.0 | noise floor 同 SOTA |
| cycle18-mlpbc-20260531-1244 | BC 改 mlp+whisper_70 | 0.6756 (cap1) | 0.69358 | -0.022 ❌ | D-11 BC cap1 cherry-pick 实证 |

完整 15 push 账本 + HOT 产物路径 → `docs/status/2026-06-01-experiment-inventory.md`

## 关键资产（磁盘验证）

| 路径 | 内容 | 用途 |
|---|---|---|
| `submission/code-20260601.zip` | **初赛代码评审包**（42KB） | 6/17 提交用 |
| `tools/runs/climb/orthofuse-20260531-0319/{pred_test1.csv,fused_probs.npz}` | **真 SOTA 0.71529** | base |
| `tools/runs/climb/whisper-fusion-20260531-0143/probs.npz` | whisper OOF 179867+test (stride40) | N1' 比对 base |
| `tools/runs/climb/_stack_cache_s40.npz` | 4 ctx base OOF/test 缓存 (36M) | orthofuse 重做用 |
| **`tools/runs/climb/b3d-calib-20260601-1008/probs.npz`** | **B3d OOF +0.031 真训练增益但 cap1=SOTA** | 教训证据 |
| **`tools/runs/climb/ctx-v3-20260601-1055/{oof.npz,cv_metrics.json}`** | **B1 v3 实测 OOF/cap1 ≈ v1** | D-12 验证证据 |
| **`cloud/train_head_n1.py`** | **N1' 云上 DB-Loss+SupCon 实现** (派生 train_head_hubert.py) | 待 rsync |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64G whisper stride5 帧 | N1' base |
| 云端 `/root/autodl-fs/hubert_cache/` | 11G hubert stride40 帧 | N1' 替代 base |

## 关键文档 (resume 后必读)

1. **`DECISIONS.md` D-13 + D-14** (战略激活 + B3d 教训)
2. **`2026-06-01-top20-attack-plan.md`** (作战图)
3. **`2026-06-01-knowledge-layer-findings.md`** (B4 报告, N1/N2/N3 候选)
4. **`2026-06-01-b1-eda-v3-features.json`** (B1 EDA, 47 候选特征)
5. **`2026-06-01-experiment-inventory.md`** (完整盘点)

## Open Questions（下次 session 优先确认）

1. **开云机做 N1' 吗?** — 已写 `cloud/train_head_n1.py` 待 rsync. ETA 20-30min GPU + 30min rsync/setup. 期望 cap1 0.65-0.66 (whisper head 0.6521 + DB-Loss+SupCon 改造). 失败代价 ~1h GPU 钱 + 0 提交配额. **强烈建议开**.
2. **6/10 报备邮件什么时候起草?** — 硬截止 9 天, 用户对外邮箱 `531045572@qq.com`. 可以并行 N1' 等待时写.
3. **N1' 失败后下一步?** — 选项 A: 转 N3 (韵律 token + Qwen2.5-7B LoRA, 6-10h 云上重投入) B: 接受 0.71529, 转复赛镜像准备.

## Ready-to-paste commands

```bash
# 1. 开云机后 rsync N1
rsync -avz -e "ssh -p 46379" cloud/train_head_n1.py \
  root@connect.westd.seetacloud.com:/root/audio-classifier/cloud/

# 2. 云端启动 N1' (hubert cache 测先)
ssh -p 46379 root@connect.westd.seetacloud.com "cd /root/audio-classifier && \
  HCACHE=/root/autodl-fs/hubert_cache OMP_NUM_THREADS=4 \
  setsid nohup python cloud/train_head_n1.py \
    --epochs 15 --alpha 0.3 \
    --run-dir tools/runs/climb/n1-hubert-dbloss-\$(date +%Y%m%d-%H%M) \
    </dev/null >/root/n1.log 2>&1 &"

# 3. N1 heartbeat (5min 间隔)
ssh -p 46379 root@connect.westd.seetacloud.com "tail -5 /root/n1.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# 4. N1 完成 rsync probs.npz 回本机
rsync -avz -e "ssh -p 46379" root@connect.westd.seetacloud.com:/root/audio-classifier/tools/runs/climb/n1-hubert-dbloss-*/ \
  tools/runs/climb/

# 5. 本机 orthofuse 重做 (N1' probs 加进去看 per-class)
python tools/climb/cycle_orthofuse.py --whisper-npz tools/runs/climb/n1-hubert-dbloss-*/probs.npz --submit

# 6. climb 状态查看
cat docs/status/climb/research-tree.md
cat docs/status/climb/session-state.json | python3 -m json.tool
```

## 提交配额状态

- 5/31 已用 3 (orthofuse-s5 / 3src / cycle18-mlpbc)
- **6/1 D-13 第一日 = 0 提交配额损耗** (本机攻坚 cap1 自动 SKIP)
- 配额 5/天, 初赛阶段一持续到 6/16, **剩余配额预算 4-5 次**
- N1' 若 push 后, 还剩 3-4 次缓冲

## Pending commits

需要 commit:
- `cloud/train_head_n1.py` (N1' 实现, 待 rsync)
- `tools/climb/cycle_context_v3.py` (B1 v3 SKIP 留存)
- `docs/status/{JOURNAL,RESUME-NEXT-SESSION}.md` (本次 handoff 更新)

`docs/赛题要求.md` modified 是用户私有 — 铁律不动。

## D-13 失效条件

三轨本机已全 SKIP。**若 N1' 也 SKIP** → D-13 失效, 回 D-12 接受 0.71529 + 寄希望其他队不动。但 N3 (韵律 token Qwen2.5) 还是个候选, 用户决策是否值得 6-10h 投入。
