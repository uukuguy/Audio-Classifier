# Next-Session Handoff

**Updated:** 2026-06-01 09:30（D-13 激活 — 前 20 攻坚战启动）
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **形势反转**：当前 SOTA 0.71529 = 排行榜**第 37 名**（前 40 进复赛，buffer 3 名极危险）。**D-12 接受论已撤** → D-13 激活前 20 攻坚。
2. **新目标 = 0.7243（前 20 真门槛），缺口 +0.009**。比前 10 缺口 0.0135 容易，**2 个独立 +0.005 真信号即可达**。
3. **三轨并行启动**：🟢 B4 Knowledge Layer（今天，0 算力）→ 🟡 B3 后处理（0.5-1 天）→ 🟠 B1 ctx v3 特征工程（1-2 天）。B2 整通对话视 B4 触发。
4. **首要参考**: `docs/status/2026-06-01-top20-attack-plan.md` 作战图 + `DECISIONS.md` D-13 + `2026-06-01-experiment-inventory.md` 盘点。
5. **D-13 失效条件**：三轨全跑 cap1 <0.6460，或 push 2 次线上无 +0.003 → 回 D-12 接受。

## 当前阶段时间线

| 日期 | 节点 | 状态 |
|---|---|---|
| **6/1（今天）** | B4 Knowledge Layer 启动 + B3 草案 | 🔄 进行中 |
| 6/2-6/4 | B3 后处理 push 验证 | ⚪ |
| 6/4-6/8 | B1 ctx v3 + 视 B4 启动 B2 | ⚪ |
| **6/10 前** | 🔴 合规报备邮件硬截止 → `xinyebei@xinye.com` | ⚪ |
| 6/10-6/16 | polish + 复赛 Docker 草稿 | ⚪ |
| 6/16 | 初赛阶段一结束 | |
| 6/17 | TOP 40 公布 + 代码评审包提交（已就绪 42KB） | 🎯 触发点 |

## In-flight（恢复后第一步）

**B4 Knowledge Layer（今天必做）**:
1. consult-AI 三方 (gemini/opencode/Context7) 咨询 turn-taking SOTA 2025-2026
2. WebSearch: "turn-taking prediction 2025 SOTA"、"backchannel SOTA"、"VAP 改进版"
3. 找 D-1~D-12 范围**外**的全新方向
4. 1 天内出 `docs/status/2026-06-01-knowledge-layer-findings.md` 报告

## Push 门（D-13 校准）

- cap1 vs 线上 noise floor ≈ 0.003
- **要 push 必须 cap1 macro ≥ 0.6460**（= SOTA cap1 0.6410 + 0.005）
- 要破前 20 cap1 macro ≈ **≥ 0.66**（+0.025 vs SOTA cap1）
- 低于则 SKIP-advance（climb §5 best-effort 自动决策）

## 铁律保留（D-1~D-12 红旗仍生效）

- ❌ 不再"加第 N 个音频源"（w2v2/e2v 不动）
- ❌ 不再"在 cap1 369 上选 strat"（阈值搜索 / per-class grid / BC 单类替换全禁）
- ❌ 不再"context 内同源算法集成"（4 成员不正交）
- ✅ 唯一允许 cap1→线上转化 = **多源融合在 T (150 正例) / I (60 正例) 中等样本类的真实信号叠加**

## 真分账本（5 个 push 全表）

| run_id | strat | cap1 | 真分 | Δ vs SOTA | 备注 |
|---|---|---|---|---|---|
| variant-F-20260528-0559 | 5seed LGBM stride5 | 0.6402 | 0.71242 | base | 旧 SOTA |
| **orthofuse-20260531-0319** | **双源 ctx+whisper (T=w70, I=whisper)** | **0.6410** | **0.71529** | **+0.003** | **★真 SOTA** |
| orthofuse-s5-20260531-0627 | 双源 stride5 强基座 | 0.6455 | 0.71233 | -0.003 | 强基座反不如 |
| orthofuse-3src-20260531-0813 | 三源 ctx+whisper+hubert | 0.6540 | 0.71523 | +0.0 | noise floor 同 SOTA |
| cycle18-mlpbc-20260531-1244 | BC 改 mlp+whisper_70 | 0.6756 (cap1) | 0.69358 | -0.022 ❌ | D-11 BC cap1 cherry-pick 实证 |

完整 15 push 账本见 `docs/status/2026-06-01-experiment-inventory.md` §I

## 关键资产（磁盘验证）

| 路径 | 内容 | 用途 |
|---|---|---|
| `submission/code-20260601.zip` | **初赛代码评审包**（42KB） | 6/17 提交用 |
| `tools/runs/climb/orthofuse-20260531-0319/{pred_test1.csv,fused_probs.npz,cv_metrics.json}` | **真 SOTA 0.71529 + 5×1000 融合概率** | B3 后处理基础 |
| `tools/runs/climb/whisper-fusion-20260531-0143/probs.npz` | whisper OOF 179867×5 + test 1000×5 | T/I 信号源 |
| `tools/runs/climb/_stack_cache_s40.npz` | 4 ctx base OOF/test 缓存 (36M) | B1 ctx v3 重训基础 |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64G whisper stride5 帧 | 复赛镜像备份 |

## Ready-to-paste commands

```bash
# B4: consult-AI 三方 quorum
bash tools/climb/consult-ais.sh "2025-2026 turn-taking prediction SOTA \
  论文/技术方向。我们卡在 0.71529 (Macro-F1 5 类 C/T/BC/I/NA), \
  已证伪: VAP/CPC 音频(BC |r|<0.04), LoRA whisper(不可行), 文本词汇(不泛化), \
  context 内融合(不正交), 跨源 ctx×whisper-T/I 正交融合(成立, +0.003). \
  缺 +0.009 进前 20, 16 天. 求未试方向"

# WebSearch
# (用 Jina/Tavily/WebSearch 搜 2025-2026 turn-taking SOTA)

# B3 后处理: 用现有 SOTA fused_probs 起点
python -c "import numpy as np; p=np.load('tools/runs/climb/orthofuse-20260531-0319/fused_probs.npz'); print(list(p.keys()))"

# B1 ctx v3 EDA 起点: 现 46d 特征位置
ls tools/climb/cycle_context*.py tools/climb/cycle_deriv_feats.py
```

## Open questions

1. **B4 1 天后若无新方向**：B1 直接全力还是 B1+B3 并行？（推荐 B1+B3 并行，B3 短，不冲突）
2. **B2 触发门**：什么样的 B4 发现算"值得启动 B2"？（暂定: 论文实证 +0.01 以上的全新架构）

## Pending commits

D-13 落盘后 working tree 仅 `docs/赛题要求.md`（用户私有，铁律不动）。
