# 可视化素材 (桶 5)

> mermaid 源 / matplotlib 图 / 架构图 / 流程图. 出过的都存这, 7/16 PPT 时挑.

## 待生成 (重要可视化清单, 后续做)

| 图名 | 类型 | 答辩用途 | 数据源 |
|---|---|---|---|
| `lb-progression-timeline.png` | matplotlib 时间线 | 真分推进图 (0.6197 baseline → 0.7458 R4 NEW SOTA) | docs/status/2026-06-04-submission-strategy.md "真分梯队" |
| `cap0-4-stability-lines.png` | 折线图 | 跨切片 macro F1 稳定性 (12 源, x=cap_i) | 同上 "跨切片稳定性" 段 |
| `oof-vs-real-scatter.png` | 散点图 | OOF cap1 vs 真分 (25 push), 红圈标 D-22 反范式 push | 25 push 账本 |
| `orthofuse-arch.mermaid` | mermaid 流程图 | per-class 路由架构 (C/T/BC/I/NA 5 类各从哪儿来) | tools/climb/cycle_orthofuse.py |
| `decision-tree.mermaid` | mermaid 树 | D-1~D-25 决策链 (撤回/确认/转向) | docs/status/DECISIONS.md |
| `dual-ssl-synergy.png` | 桑基/瀑布图 | R5→R6 (-0.0015) vs R5→R4 (+0.0069) 协同 | DECISIONS D-25 |
| `softadd-weight-curves.png` | 折线图 | wsp_ms/Omni/e2v_ms 软加权重曲线 (0.05/0.10/0.15/0.20) | 25 push |
| `climb-loop-arch.mermaid` | mermaid 状态机 | hypothesis pool → train → push → calibrate 循环 | ~/.claude/shared-rules/climb.md |

## 命名约定

- `<topic>-<date>.png` — 静态图片
- `<topic>-<date>.mermaid` — mermaid 源
- `<topic>-<date>.py` — matplotlib/seaborn 生成脚本
- `<topic>-<date>.drawio` — 复杂架构图 (drawio)

## 工具偏好

- **mermaid** 优先 (流程图 / 决策树 / 状态机) — 文本可 diff, 答辩时一旦改数据/路径可快速重生
- **matplotlib** for 数据图 (折线 / 柱状 / 散点) — 同上, 脚本可 diff
- **drawio** 仅当 mermaid 表达不了 (跨多 swimlane / 复杂层级) — 留作 fallback
- 避免 PPT 内画图 (无版本控制, 改起来痛)

## 风格统一

- 字号: 14pt 标题 / 12pt 标签 / 10pt 注解
- 配色: 高对比 (PPT 投影损失色彩) — 主色 #2E86AB 蓝 + 强调 #E63946 红 + 中性 #6C757D 灰
- 不用三维 / 阴影 (PPT 减分项)
- 答辩听众 5 秒看完原则 — 一图一个点, 不堆信息

## 已生成

(空, 7/16 答辩准备时补)
