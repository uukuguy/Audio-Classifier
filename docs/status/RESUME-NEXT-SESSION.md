# RESUME — Next Session

**Last updated:** 2026-05-27 20:26

## TL;DR

FinVCup turn-taking。climb 跑了 6 cycle，**便宜路线(LGBM/冻结pooled特征)全部撞 0.7108 墙**（6 负结果）。当前 SOTA = cycle1 纯上下文 LGBM **公榜 0.710789（排22）**。用户已定下一步：**做 research 主推的 VAP 双声道音频编码器攻 BC**。

## 当前最优

- **cycle1 context-only：公榜 0.710789**（CV 0.5908，gap +0.12）
- 提交工件：`tools/runs/climb/20260527-1636-h001-context-only/pred_test1.csv`

## 已排除（6 负结果，研究树 negative cache）

便宜路线全到顶——**根因：冻结编码器抽 pooled 特征(喂树/喂神经头)对 turn-taking 无效**（mean-pool 丢时序 + 冻结通用语义无 turn-taking 信号）。BC 铁卡 0.20-0.22。

## Next steps（VAP 路线，吸取教训：先小验证再全量）

1. **先验证架构，再上大编码器**：双声道帧序列 + 轻量 cross-attention 小头，编码器先用便宜的（mel/CNN），看 BC 能否动
2. BC 动了 → 换 Qwen2-Audio Whisper encoder（白名单 640M，冻结，curl 直下绕 hf client）放大
3. **关键铁律**：保帧序列不 pool（前 6 cycle 都栽在 pool）；双声道 cross-attention 是 BC 主导线索
4. 复用 baseline 的 8k 双声道音频 IO（`baselines/.../dataset.py:_read_wav_slice`）

## 关键约束/教训（详见 CLAUDE.md）

- 阈值：滑窗 CV 调激进阈值线上更差，保 ~0.5 / per-class-aware（C 低阈值安全）
- 稠密 embedding 不喂树；冻结 pooled 特征对本任务无效
- **投入前先小规模验证架构匹配**（栽过：Qwen3 提取跑 29% 才发现无效）
- 模型 curl 直下（hf client HEAD 失败）；MPS 可行但端到端训练耗时需重估
- 提交是用户的事（5次/天），climb 只产候选 + 报 CV

## Ready commands

```bash
# 已缓存资产
ls data/cache/qwen_text/   # 110通 Qwen3文本特征(此路已否，但缓存框架可借鉴)
# baseline 音频 IO 参考
sed -n '119,154p' baselines/2026_finvcup_baseline/src/data/dataset.py
# climb 状态
cat .claude/climb/session-state.json; cat docs/status/research-tree.md
```

## Ruled out (don't re-explore)

context-only 加码 / 廉价声学 / 文本词汇 / Qwen3 mean-pool(树or神经头) —— 全是冻结/pooled 特征喂浅模型，撞 0.71 墙。
