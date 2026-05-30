# Next-Session Handoff

**Updated:** 2026-05-31 03:25
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **冲 0.75 = 模型融合。本轮关键突破：跨源融合成立**——context×whisper per-class 融合 cap1 0.6228→0.6410(+0.0182)真增益，T/I 借 whisper（whisper 真实强 T0.656/I0.509 vs ctx0.621/0.455）。
2. **context 内融合证伪**：4 成员不正交，grid 过拟合（nested 揭穿 BC 0.364→0.200 蒸发）。正交性来自**不同信号源**，不来自同源换算法。
3. **两个提交件待真分**：grid（预测≈或<SOTA）+ orthofuse（预测+0.0006，但 T/I 是真增益）。
4. **冲 0.75 下一步明确**：把 whisper T/I 增益**叠到变体F 强基座**（当前 orthofuse 基座是 lgbm_v1 单模 0.6228，非变体F 5seed 0.6402）+ 加第二音频源（chinese-hubert 中文域最有潜力）。

## Where things stand

- **SOTA 仍 = 变体F 0.7124**（前10 门槛 0.7285，差 0.016）。
- **whisper 连续概率已存本机**：`tools/runs/climb/whisper-fusion-20260531-0143/probs.npz`（oof 179867×5 + test 1000×5）。跨源融合的基础资产。
- **context 4 成员概率缓存**：`tools/runs/climb/_stack_cache_s40.npz`（--cached 秒复用）。
- **VAP-CPC test 概率**：`tools/runs/climb/vap-full/test_probs.npz`（缺 cap1 OOF，三源融合要补）。
- 工作树未 commit：cloud/ 3 脚本 + train_head_cuda(加 probs.npz 存盘) + 新建 cycle_stack_fusion/cycle_orthofuse + DECISIONS D-6/D-7 + climb 状态。

## What this session delivered

- `tools/climb/cycle_stack_fusion.py` — context 内 4 成员融合 + nested-CV 过拟合验证（证伪 grid）
- `tools/climb/cycle_orthofuse.py` — context×跨源 per-class 正交融合（whisper 成立）
- `cloud/train_head_cuda.py` — 加 probs.npz 连续概率存盘（产物铁律 + 融合必需）
- `tools/climb/cycle_algo_ensemble.py` — 修 MLP（_balance_idx 正类过采样，BC 0→0.154）
- DECISIONS D-6（融合路径）+ D-7（BC 上限 0.22 松动为"冻结路线下"）

## Next steps (immediate, action-level)

1. **【最高 ROI】whisper T/I 叠变体F 强基座**：当前 orthofuse 用 lgbm_v1 单模(0.6228)，改用变体F 5seed 概率(cap1 0.6402)做 context 基座，T 借 w70-whisper / I 借 whisper。预期把 +0.0182 叠到更高起点 → 可能真破 SOTA。需先重出变体F 的 cap1 OOF + test 连续概率（gen_variants 加存盘）。
2. **【待真分】** 用户上传 orthofuse + grid 两提交件，贴回真分校准 whisper-orthofuse paradigm gap（whisper test gap 未知，决定外推准不准）。
3. **【第二音频源】** 真分确认跨源融合线上也涨 → 上 chinese-hubert-large（中文电话域最对口，317M，需 6/10 前报备）当独立特征源。emotion2vec（副语言正交维度）次选。
4. **【提升成员质量】** whisper head 改 ASL 损失替 BCE（文献证优于 BCE，T/I/BC 成员更强 → 融合天花板更高）。

## Don't go down these paths again (ruled out)

- **context 内融合**（算法正交/特征子集/grid/stacking）— nested 证伪，成员不正交。融合需不同信号源。
- **grid 权重搜索** — 369 cap1 样本搜 5^4 权重 = 过拟合（BC 0.364→nested 0.200）。融合用固定权重凸组合。
- **wav2vec2/MMS 当 VAP encoder** — 文献证 CPC > wav2vec2/MMS for VAP（已试 CPC）。但 chinese-hubert 当独立特征源不同。
- 全套旧 negative cache（VAP/whisper冻结/LoRA/文本/Omni/BC 9角度）见 DECISIONS D-1~D-5。

## 关键认知（本轮新增）

- **融合正交性来自信号源不同，非算法不同**（D-6）。whisper(音频) vs context(标签时序)才正交。
- **per-class 借强 + 固定权重凸组合 + 标签对齐** = 防 grid 过拟合的正确融合姿势。
- **BC 信息论上限 0.22 不严谨**（D-7）：冻结路线测的，LoRA 可学 encoder 顶到 0.267，但全量 30-63h 不可行。
- **whisper T/I 强是真的**：cap1 T 0.667/I 0.555（whisper-only），强于 context。这是"全类各榨一点"真杠杆。

## 各音频源逐类 cap1 对照（融合参考）

| 类 | context(变体F) | whisper | VAP-CPC | 谁强 |
|---|---|---|---|---|
| C | 0.974 | 0.975 | 0.972 | 平 |
| T | 0.625 | **0.667** | 0.630 | whisper |
| BC | 0.200 | 0.200 | 0.222 | VAP 略 |
| I | 0.539 | **0.555** | 0.513 | whisper |
| NA | 0.863 | 0.864 | 0.864 | 平 |

## 云主机

- AutoDL 4090D 48GB，`ssh -p 46379 root@connect.westd.seetacloud.com`，python=`/root/miniconda3/bin/python`(torch2.7+cu128)
- whisper 64G 帧特征：`/root/autodl-fs/backups/whisper_cache_full`（软链 `data/whisper_cache`，跳提取直跑 head）
- **限线程铁律**：`OMP_NUM_THREADS=8`（爆 20 核卡死踩过 2 次）
- whisper 训练 PID 73208 已结束（DEAD + probs.npz 双信号验证），0 残留

## Ready-to-paste（下一步1：whisper T/I 叠变体F）

```bash
cd /Users/sujiangwen/sandbox/competitions-2026/Audio-Classifier
# 1. 改 gen_variants.py 让变体F 存 cap1 OOF + test 连续概率 npz (当前只存0/1 CSV)
# 2. cycle_orthofuse.py 的 ctx 基座从 lgbm_v1 换成变体F 5seed 概率
OMP_NUM_THREADS=4 python3 tools/climb/cycle_orthofuse.py \
  --whisper-npz tools/runs/climb/whisper-fusion-20260531-0143/probs.npz --folds 5 --submit
```
