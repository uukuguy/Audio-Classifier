# Next-Session Handoff

**Updated:** 2026-05-31 05:20（环境修复 session：CC→156 + hook error 修 + climb 收口落盘完成）
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## ✅ 环境已修复（上个 session 的 4 项待办全部完成 — 不要再重做）

上个 session 跑在 **CC 2.1.158**（工具回执通道有 bug），本 session 已全部清理：

1. ✅ **CC 版本**：`~/.local/bin/claude` symlink → **2.1.156**；`~/.claude/settings.json` 顶层加 `autoUpdatesChannel:stable` + `minimumVersion:2.1.156`（防自动跳回 158）。**下次启动跑 156**。
2. ✅ **PostToolUse:Bash hook error 根因已修**：那条 error = 8 个 GSD 门禁 `[ -d .planning ] && exec` 在**无 .planning 项目里短路返回 exit 1**，被 CC 记成 hook error（**非真 bug，不阻断**）。修法：8 条全加 `|| exit 0`，无 .planning 时干净退出 0。**本 session 仍会报（hook 配置不热重载），重启即净。**
3. ✅ **climb 状态全落盘**：session-state.json（best_online→0.71529 / confirmed 加 orthofuse / pending 清空 / next 改 whisper叠变体F）+ regen research-tree（SOTA 0.71529）。
4. ✅ **commit 收口**（git add **不带** `research-tree.json` — 该目录只有 .md；**不碰** `docs/赛题要求.md`（用户私有，M 状态非本 session 改）+ 操作手册.pdf（untracked））。

## TL;DR（本轮真正的成果）

1. **★★★ orthofuse 真分 0.71529 破 SOTA**（+0.002866 vs 变体F 0.712424）——context×whisper 跨源 per-class 融合**成立**（真增益，真分铁证）。T 借 w70-whisper / I 借 whisper（whisper 在 T/I 真实强）。
2. **grid 真分 0.679138**（-0.0333）——context 内融合**证伪**，nested 过拟合应验（BC 0.364→0.200 蒸发）。正交性来自信号源不同（D-6），非同源换算法。
3. 新 SOTA = **orthofuse 0.71529**。前10 门槛 0.7285，差 0.013。

## 冲 0.75 下一步（真分校准后的明确方向）

1. **【最高 ROI】whisper T/I 叠变体F 强基座**：当前 orthofuse 的 context 基座是 lgbm_v1 **单模**(cap1 0.6228)，换成**变体F 5seed**(cap1 0.6402) 做基座，T/I 仍借 whisper → 把 +0.0182 叠到更高起点。
   **前置**：改 `tools/climb/gen_variants.py` 让变体F 存 cap1 OOF + test 连续概率 npz（现只存 0/1 CSV）。
   ```bash
   OMP_NUM_THREADS=4 python3 tools/climb/cycle_orthofuse.py \
     --whisper-npz tools/runs/climb/whisper-fusion-20260531-0143/probs.npz --folds 5 --submit
   ```
2. **【第二音频源】** chinese-hubert-large（中文电话域最对口，317M，需 6/10 前报备）当独立特征源；emotion2vec 次选。
3. **【成员质量】** whisper head 改 ASL 损失替 BCE，T/I/BC 成员更强 → 融合天花板更高。

## 关键资产（磁盘已确认在位）

- whisper 连续概率：`tools/runs/climb/whisper-fusion-20260531-0143/probs.npz`（oof 179867×5 + test 1000×5）
- context 4 成员概率缓存：`tools/runs/climb/_stack_cache_s40.npz`（含 oof_lgbm_v1/xgb_v1/lgbm_v2/mlp_v1 + te_* + Y + G，--cached 秒复用）
- VAP-CPC test 概率：`tools/runs/climb/vap-full/test_probs.npz`（缺 cap1 OOF，三源融合要补）
- orthofuse 提交件 + 融合概率：`tools/runs/climb/orthofuse-20260531-0319/{pred_test1.csv,fused_probs.npz,cv_metrics.json}`

## Don't go down these paths again（ruled out）

- **context 内融合**（grid/stacking/算法正交/特征子集）— 真分 0.679 << SOTA，4 成员不正交，nested 过拟合应验。融合需不同信号源。
- **grid 权重搜索** — 369 cap1 样本搜 5^4 权重 = 过拟合。融合用固定权重凸组合 + per-class 借强 + 标签对齐。
- 全套旧 negative cache（VAP/whisper冻结/LoRA/文本/Omni/BC 9角度）见 DECISIONS D-1~D-5，新增 D-6/D-7。

## 各音频源逐类 cap1 对照（融合参考）

| 类 | context(变体F) | whisper | VAP-CPC | 谁强 |
|---|---|---|---|---|
| C | 0.974 | 0.975 | 0.972 | 平 |
| T | 0.625 | **0.667** | 0.630 | whisper |
| BC | 0.200 | 0.200 | 0.222 | VAP 略 |
| I | 0.539 | **0.555** | 0.513 | whisper |
| NA | 0.863 | 0.864 | 0.864 | 平 |

## 工具/环境元信息

- **Claude Code 已回滚 2.1.158 → 2.1.156**（`~/.local/bin/claude` symlink）。158 工具回执有 bug，重启进 156 应恢复。
- 本机 conda env `deep-research`（torch 2.7.1 + torchaudio 2.7.1）。本机训练必须限线程。
- 云主机 AutoDL 4090D：`ssh -p 46379 root@connect.westd.seetacloud.com`，python `/root/miniconda3/bin/python`(torch2.7+cu128)。whisper 64G 帧特征 `/root/autodl-fs/backups/whisper_cache_full`（软链 `data/whisper_cache`）。限线程 `OMP_NUM_THREADS=8`。
- 提交配额 2/day（手动 csv）。orthofuse + grid + vap-full 真分均已落。
