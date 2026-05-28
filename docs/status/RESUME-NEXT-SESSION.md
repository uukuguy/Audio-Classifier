# RESUME — Next Session

**Last updated:** 2026-05-28 06:50（会话收口）
**恢复命令：`/project-state resume`**（本项目 lightweight-memory，**不是** gsd-resume-work）

## TL;DR

1. **新 SOTA = 0.712424**（变体 F：5seed 概率平均集成 + cycle1 固定阈值），破前 SOTA 0.710789（+0.0016）。
2. **② 切片 CV 方法论修正完成**：cap1 切片验证集是可信 CV 协议（gap +0.118→+0.055）。BC 确认是真瓶颈（非 CV 假象）。
3. **① 上云 whisper 产物全部就绪**（`cloud/`），本机冒烟通过，**待用户 AutoDL 开机**。

## 当前最优（线上真分校准确定）

| 变体 | 线上 | cap1 CV | 结论 |
|---|---|---|---|
| **F 概率平均+cycle1阈值** | **0.712424** | 0.640 | ✅ 新 SOTA，集成降方差有用 |
| cycle1 (前SOTA) | 0.710789 | — | 锚 |
| B 切片阈值 | 0.693007 | 0.648 | ❌ 砸 NA（阈值铁律3验） |
| C 5seed rank | 0.641277 | 0.634 | ❌ rank 伤稀有类 BC 崩 |

- F CSV：`tools/runs/climb/variant-F-20260528-0559/pred_test1.csv`（gitignored 在磁盘）
- 脚本：`tools/climb/gen_variants.py`（变体 B/C/E/F，旋钮走 CLI）

## 三大铁律本会话实测确认

1. **集成用概率平均（非 rank）** — F 概率平均 +0.0016；C rank 平均 −0.070（rank 把稀有类 BC 0.5% 的正例被海量负例稀释，假正例爆）。
2. **任何 CV 调阈值偏离 cycle1 近全正都砸 NA** — B 在 cap1 CV 上（0.648）比 F（0.640）高，线上却低 0.019。**CV 高分 ≠ 线上高分。** 阈值铁律第 3 次确认。
3. **BC 是真瓶颈** — 切片三档采样 BC 全钉 0.22，LGBM 路线打不破。需上云 whisper 端到端。

## Next steps

### 主线（待用户）：上云 whisper 攻 BC
1. 用户 AutoDL 开机（`cloud/AUTODL-CHECKLIST.md`：4090 + PyTorch2.7.1 镜像 + 6/10 前报备 `xinyebei@xinye.com`）
2. 开机后 `git clone` → `bash cloud/run_cloud.sh smoke`（前 40 通）→ 贴日志回来看 BC 有无抬头
3. 有信号 → `full` 全量 → 出 CSV → 提交 → 贴真分 → 注入 climb
4. **本机摸不到云实例**：用户贴 SSH 或贴脚本到云终端跑、回贴日志（我盯日志判进度）

### 可选零投入
- F 已是新 SOTA，可提交固化。若想再榨：cycle1 多 seed 数（7/9 seed）概率平均，边际可能 +0.001 级
- 提交配额：初赛每日有限（复赛每天 2 次），优先交预测最稳/最缺校准的

## Ready commands

```bash
/project-state resume
cat .claude/climb/session-state.json          # climb 完整状态(F新SOTA)
cat docs/status/2026-05-28-sliced-cv-audit.md # 切片CV审计
cat cloud/AUTODL-CHECKLIST.md                 # 上云操作单
# 重跑可信CV: OMP_NUM_THREADS=4 ... python3 tools/climb/sliced_cv.py --folds 5
# 生成变体: python3 tools/climb/gen_variants.py --variants F --seeds 5
```

## Ruled out（don't re-explore，详见 MEMORY negative_cache）

context加码 / 廉价声学 / 文本词袋 / Qwen3 pooled / VAP-mel / 本机whisper / **切片阈值(砸NA)** / **rank平均集成(BC崩)**。本机 LGBM 路线确认到顶（集成只 +0.0016）。真突破唯一路径=上云 whisper 端到端攻 BC。
