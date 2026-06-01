# 实验盘点 — 2026-06-01 初赛收口

> **目的**: 系统盘点 5/27~6/1 共 6 天实验获得的可用研究信息 + 物理产物 + 未完成探索任务。供 6/2-7/7 复赛镜像准备 + 复赛阶段二决策参考。
> **状态**: 🟡 decision-history — 初赛已收口，这是终态快照；复赛阶段开始后产物清单会增加，决策清单不会变。
> **生成时机**: 6/1 上午用户主动 dump 知识资产，trigger `/project-state` 落盘。

## I. 真分账本（15 个 push/未提交 完整记录）

| # | run_id | paradigm | cap1 | 真分 | Δ SOTA | 状态 |
|---|---|---|---|---|---|---|
| 1 | `cycle1-context-only` (5-27) | context-only LGBM | 0.5908 | **0.71079** | base | 🥇 首 SOTA |
| 2 | `cycle1b-context-v2` (5-27) | context-only K-fold+XGB | 0.5921 | 0.68327 | -0.027 | 🔴 激进阈值 |
| 3 | `variant-F` (5-28) | 5seed 概率平均 + cycle1 阈值 | 0.6402 | **0.71242** | +0.002 | 🥇 SOTA #2 |
| 4 | `variant-B` (5-28) | 切片 cap1 阈值搜索 | 0.6480 | 0.69301 | -0.018 | 🔴 砸 NA |
| 5 | `variant-C` (5-28) | 5seed rank 平均 | 0.6341 | 0.64128 | -0.070 | 🔴 BC 崩 |
| 6 | `cloud-whisper-smoke` (5-29) | 冻结 whisper 40通 | 0.6413 | 0.63378 | -0.077 | 🔴 |
| 7 | `cloud-whisper-full-cycle1` (5-29) | 冻结 whisper 全量+cycle1 阈值 | 0.6521 | 0.67091 | -0.041 | 🔴 冻结上限 |
| 8 | `cloud-whisper-full-balanced` (5-29) | 冻结 whisper+balanced 阈值 | 0.6521 | 0.64370 | -0.068 | 🔴 |
| 9 | `vap-full` (5-30) | VAP-CPC stereo unfreeze | 0.6403 | 0.63368 | -0.078 | 🔴 |
| 10 | `ti-robust` (5-30) | context+文本词汇 | 0.6358 | 0.63916 | -0.073 | 🔴 文本虚高 |
| 11 | `stack-fusion-grid` (5-31) | 4 ctx base 算法集成 | 0.6198 (nested) | 0.67914 | -0.033 | 🔴 不正交 |
| 12 | **`orthofuse-20260531-0319`** (5-31) | **ctx×whisper per-class 正交** | **0.6410** | **0.71529** | **+0.003** | 🏆 **真 SOTA** |
| 13 | `orthofuse-s5` (5-31) | 双源 stride5 强基座 | 0.6455 | 0.71233 | -0.003 | 🔴 强基座+whisper 冲突 |
| 14 | `orthofuse-3src` (5-31) | ctx+whisper+hubert | 0.6540 | 0.71523 | -0.00006 | ⚪ 同 SOTA |
| 15 | `cycle18-mlpbc` (5-31) | BC 改 mlp+whisper_70 | 0.6756 | 0.69358 | -0.022 | 🔴 cap1 cherry-pick |

**总投入**：~30h 云时间 + 15 次提交配额 + 5 个非 Qwen 模型下载。

**结论**：真 SOTA = **0.71529**（orthofuse-20260531-0319），距前 10 真门槛 **0.7285 差 0.0135**。D-12 后已穷尽规划路径，剩余 16 天初赛阶段不可达前 10。

## II. 可复赛复用的核心产物（HOT，复赛必带）

### II.1 SOTA pipeline 物理位置（本机）

| 路径 | 大小 | 作用 |
|---|---|---|
| `tools/runs/climb/orthofuse-20260531-0319/{pred_test1.csv, fused_probs.npz, cv_metrics.json}` | 76K | **真 SOTA 提交件 + 5×1000 融合后概率** |
| `tools/runs/climb/whisper-fusion-20260531-0143/probs.npz` | 3.2M | **whisper OOF 179867×5 + test 1000×5**（T/I 信号源） |
| `tools/runs/climb/_stack_cache_s40.npz` | 36M | **4 ctx base OOF+test 缓存**（lgbm_v1/xgb_v1/lgbm_v2/mlp + Y + G） |
| `tools/runs/climb/variant-F-20260528-0559/` | 20K | 前 SOTA 5seed 集成 |
| `tools/runs/climb/hubert-fusion-20260531-0750/probs.npz` | 764K | hubert OOF + test（1024d，3 源融合用） |
| `tools/runs/climb/w2v2-fusion-20260531-1120/probs.npz` | 744K | w2v2 OOF + test（无融合价值但已用→报备） |
| `tools/runs/climb/e2v-fusion-20260531-1120/probs.npz` | 752K | e2v OOF + test（同上） |

### II.2 关键代码资产

| 路径 | 作用 | 进 SOTA pipeline？ |
|---|---|---|
| `tools/climb/cycle_orthofuse.py` | per-class 跨源正交融合主程 | ✅ |
| `tools/climb/cycle_orthofuse_3src.py` / `cycle_orthofuse_nsrc.py` | 3/N 源通用版（备份） | 🟡 备 |
| `tools/climb/cycle_stack_fusion.py` | 4 ctx base OOF 生成 + cache | ✅ |
| `tools/climb/cycle_context.py` / `cycle_context_v2.py` | LGBM 基线（v1=SOTA / v2 已证伪） | ✅ v1 |
| `tools/climb/gen_variants.py` | 变体 F 5seed 集成 | ✅ |
| `tools/climb/sliced_cv.py` | cap1 切片化 CV 协议 | ✅ |
| `cloud/extract_whisper_cuda.py` | whisper 帧特征提取 | ✅ |
| `cloud/train_head_cuda.py` / `train_head_hubert.py` | 神经小头训练 | ✅ |
| `cloud/extract_hubert_cuda.py` / `extract_w2v2_cuda.py` / `extract_emotion2vec_cuda.py` | 3 个云端特征提取脚本 | 🟡 备/合规 |
| `cloud/Dockerfile` | CUDA base（已有草稿） | 🟡 复赛 |

### II.3 云端备份（关机中，复赛验证时再开）

| 路径 | 大小 | 作用 |
|---|---|---|
| `/root/autodl-fs/backups/whisper_cache_full/{train,test}/` | 64GB | whisper stride5 帧特征（**最贵的资产**） |
| `/root/autodl-fs/hubert_cache/{train,test}/` | 11GB | hubert stride40 帧特征 |
| `/root/autodl-fs/w2v2_cache/` + `emotion2vec_cache/` | ~20GB 估 | 已提取，融合无价值但报备需要 |
| 云端 `/root/.cache/manual_models/` | — | chinese-hubert-large / w2v2-large / e2v_base / whisper-large-v3 / Qwen2.5-Omni-3B |

### II.4 提交件

| 路径 | 状态 |
|---|---|
| `submission/code-20260601.zip` (42KB) | ✅ **初赛代码评审包就绪**，等 6/17 |
| 复赛 Docker 镜像 | ⚪ 未做 |
| 复赛技术报告 PDF (≤3 页) | ⚪ 未做 |

## III. 研究信息资产（COLD，知识沉淀）

### III.1 已锁定的决策（D-1~D-12，详 docs/status/DECISIONS.md）

| ID | 决策 | 类型 |
|---|---|---|
| **D-1** | VAP/CPC 音频路线整条证伪 | 范式否决 |
| **D-2** | BC 攻击战略从"硬攻 BC"转"攻 T/I" | 战略转向 |
| **D-3** | T/I 文本路线证伪（CV 虚高不泛化） | 范式否决 |
| **D-4** | BC 冻结路线信息论上限 ~0.22 | 范式天花板 |
| **D-5** | 0.712 卡点根因 = 单源问题，融合救不了 | 诊断 |
| **D-6** | context 内融合证伪，跨源 whisper 正交是真路径 | 真路径 |
| **D-7** | BC 可学 encoder 上限 0.267（成本不可行） | reconcile D-4 |
| **D-8** | 跨源融合范式锁 0.715，加 hubert 第三源无线上增益 | 范式天花板 |
| **D-9** | 5 源融合 cap1 锁 0.6540（3 源即顶） | 实测上限 |
| **D-10** | 实测加源全量 5 源天花板 0.7152 | 兜底 |
| **D-11** | cycle 18 BC cap1 cherry-pick（9 样本不可信） | cap1 陷阱 |
| **D-12** | cycle 19 所有 ctx-内方向全证伪 — 初赛个人天花板 0.71529 | 收官诊断 |

### III.2 工程铁律（已沉淀进 CLAUDE.md / memory）

1. **阈值铁律** — 滑窗 CV 调激进阈值线上更差 0.027（cycle 1b 真分实证）
2. **稠密 embedding 不喂 LGBM** — Qwen3-0.6B 1024d pooled→LGBM macro -0.008
3. **cap1 369 样本 BC 增益永远不可信** — D-3/D-11 累积同根（9 正例 +1 TP 跳 F1）
4. **本机训练必须限线程**（OMP/MKL/VECLIB/OPENBLAS=4）+ MPS 高水位别乱设
5. **whisper 类大编码器本机 MPS 不可行**（45h），必须云 GPU
6. **下载源**：云端 modelscope >> hf-mirror（国内 IDC）/ 本机相反
7. **实验值永不写 default**（W_DIAG 旧 bug 同款风险）

### III.3 关键探针/诊断结论（不属决策但是宝贵 negative cache）

- VAP 预训练 head 原生信号对 BC **|r|<0.04** — VAP 本身归纳偏置抓不住 BC
- BC 所有信号源 r≈0.13，叠加(context+F0) 仅 +0.005 — 信息论接近上限
- whisper 逐类 cap1：**T=0.667 / I=0.555 强于 context** — 真正交杠杆
- mlp ctx base 在 T/I 系统性弱 0.04-0.08（不只是 BC 噪声）
- w2v2/hubert/e2v 单源虽不弱（cap1 0.62-0.64）但融合 0 贡献（同范式 SSL 撞墙）

## IV. 遗留未完成探索任务

按**短期紧迫度 × 突破期望**排序。

### IV.A 🔴 复赛截止硬约束（必做，不做有合规风险）

| # | 任务 | 截止 | 备注 |
|---|---|---|---|
| A1 | **报备邮件** → `xinyebei@xinye.com` | **2026-06-10** | 列非 Qwen 模型：chinese-hubert / chinese-wav2vec2 / emotion2vec / whisper-large-v3。from `531045572@qq.com` |
| A2 | **复赛 Docker 镜像**（CUDA, ≤20GB, 模型 ≤8GB） | 6/20-7/7 阶段 | 含完整推理 pipeline。已有 `cloud/Dockerfile` 草稿待完善 |
| A3 | **复赛推理脚本**：单段 30s 音频 + ASR JSON + context.npy → 5 列 0/1 CSV | 同上 | 接口对齐手册 §复赛输入输出 |
| A4 | **复赛技术报告 PDF**（≤3 页） | 同上 | 方法 + 实验 + 诚实声明（D-1~D-12 摘要） |

### IV.B 🟡 初赛剩余配额可探索（D-12 后概率低，但配额免费）

| # | 任务 | 期望 | 风险 |
|---|---|---|---|
| B1 | **真正未试**：改特征工程 — 把现 46d context 特征做"v3 改进版"重训整个 ctx base（保数据规模 → 避 cap1 陷阱） | +0.003~0.01 | D-12 红旗"不在现 cycle 套路内"，是少数可能突破方向 |
| B2 | **真正未试**：整通对话神经预测（彻底换架构，长时序 transformer over 整通对话） | +0.005~? | 算力投入大，初赛仅剩 16 天可能来不及 |
| B3 | **真正未试**：后处理（test 切片末专属规则 / TTA / 半监督）— D-9 之前认为是错诊断方向，但 D-10 实测后仍未试。基于 cap1=test 标签的 self-distillation | +0.001~0.005 | D-9 撤了"分布差"前提但路径本身未验证。低风险可试 |
| B4 | **Knowledge Layer 触发**：consult-AI / WebSearch 2026 turn-taking SOTA 新论文 / 比赛技术分享 — 找全新正交信号源（domain 知识） | 未知 | 0 算力，纯研究投入 |

### IV.C ⚫ 假设池 stale 项（已部分被 D 证伪，但 yaml 未标）

| ID | 描述 | 真实状态 |
|---|---|---|
| H-002 | baseline 逐类阈值搜索 | 🟡 cycle1 阈值实际已隐含解决，单独跑无价值 |
| H-003 | ASL 损失替 BCE | 🟡 D-4 后 BC 战略转移，ASL 价值降 |
| H-004 | baseline 保帧序列 + 双声道 | 🔴 D-1 后整条音频路线否，等同已证伪未标 |
| H-005 | 双声道 cross-attn VAP 冻结 Qwen2-Audio | 🔴 D-1 否 |
| H-006 | chinese-hubert spike | ✅ 已做（D-8 三源融合用过，无增益但已合规报备） |
| H-007 | F0/pitch 帧特征 | 🟡 D-4 探针已测（最强分支 +0.005），未独立 push |
| H-008 | A+B+C rank 平均集成 | 🔴 D-5 ensemble-grid 一系否 |
| H-L1/L2/L3 | LoRA whisper（r=32 / ASL / cap1） | 🔴 D-7 全量不可行 / cap5 欠拟合 |

**清理建议**：所有 🔴 的应在 `docs/status/climb/hypotheses.yaml` 标 `falsified`，否则 climb resume 仍当 active 排序污染决策。本盘点暂不动 hypotheses.yaml 文件本身（避免风险），仅记录待办。

### IV.D 🧹 工程/状态债务

| # | 任务 | 影响 |
|---|---|---|
| D1 | climb `session-state.json` 的 `next_action`/`phase` 仍是 5-31 fusion-075 阶段（"改 gen_variants.py 存 cap1 OOF+test 连续概率"）— D-12 闭合后未刷新 | 下次 climb resume 会读错战略，应清空或改为"复赛镜像准备" |
| D2 | `hypotheses.yaml` 多个 stale active 项（见 IV.C） | 同上，污染决策 |
| D3 | `research-tree.md` SOTA 段写 "0.72"（应 "0.71529"），由 regen-tree 数据源四舍五入 | 视觉误导 |
| D4 | `runs.csv` 第 15-18 行 stack-fusion + orthofuse 重复 | 历史重复 append，不影响读但脏 |
| D5 | 7 个 `cloud/probe_*.py` 探针脚本未入 git（确认正交性时一次性使用） | 评估是否归档 |
| D6 | 本机 `~/.cache/manual_models/`（whisper-small / large-v3 / Qwen3-0.6B）共 ~6GB | whisper-small 已证本机不可行可删；其余复赛镜像可能用 |

## V. 单点总结（10s 速读）

- **真 SOTA** = `orthofuse-20260531-0319` 真分 **0.71529** | 距前 10 门槛 0.7285 差 0.0135
- **初赛收口** D-12：剩余路径每路 <0.005 增益凑不到缺口，进前 10 不可达
- **当前阶段** 6/2-6/16：等待 + 复赛镜像准备
- **最高优先级** = 6/10 报备邮件（合规硬截止）→ Docker / 推理 pipeline 草稿 → 6/17 触发提交
- **未死路径**（用户若想冲）= B1 ctx 特征工程 v3 / B3 后处理 TTA / B4 Knowledge Layer 找全新信号源
