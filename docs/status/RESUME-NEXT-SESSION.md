# RESUME — Next Session

**Last updated:** 2026-05-28 (会话收口)
**恢复命令：`/project-state resume`**（本项目 lightweight-memory，**不是** gsd-resume-work）

## TL;DR

FinVCup turn-taking。9 个 climb cycle 后，**本机所有路线实测穷尽**，SOTA 仍是 cycle1 纯上下文 LGBM **公榜 0.710789（排22）**。撞到决策门：真突破需云 GPU，本机零投入只剩 2 个角度没试。

## 当前最优（验证确定）

- **cycle1 context-only：公榜 0.710789**（CV 0.5908，gap +0.12）
- CSV：`tools/runs/climb/20260527-1636-h001-context-only/pred_test1.csv`（gitignored 在磁盘）

## 决策门（待用户定，本会话结束时未定）

本机 LGBM/冻结特征/mel音频/whisper 全否（9 负结果，详见 MEMORY [[reference_negative_cache]]）。两条路：

1. **上云 GPU** — 唯一能跑 whisper-large-v3 端到端/VAP，research 主推攻 BC。模型已下、代码就绪，需用户定云方案。
2. **守 0.7108 换零投入角度**（建议先做，不卡机）：
   - **30s 切片化验证集**（CONTEXT Decision 4）——全程用滑窗 CV 不可信(gap+0.12)，这是欠的方法论修正，对将来上云实验也有用
   - cycle1 多 seed/阈值集成

我的建议：先做 ②（零算力+修 CV 漏洞），同时用户考虑 ①。② 不阻碍 ①。

## Next steps（若选 ②，本机可立即做）

1. 写 30s 切片化验证集构造（从 train 长对话采独立 30s 段 + 对应 context/text/label，模拟 test 分布）→ 让 OOF CV 可信
2. 重测 cycle1/context 在切片验证集上的真实 CV，对齐线上 gap
3. cycle1 多 seed 集成（rank 平均）

## 本机训练铁律（别再犯，详见 MEMORY [[reference_mps_hardware_limits]]）

- **必须限线程**：`OMP/MKL/VECLIB/OPENBLAS_NUM_THREADS=4 + torch.set_num_threads(4)`，否则 load 飙到 39 卡死全机
- 后台 `nohup ... & disown`（macOS 无 setsid），防其它 CC 误杀
- 模型 curl 直下绕 hf client；fp32 加载；whisper 系本机跑不动（需云）

## Ready commands

```bash
/project-state resume          # 恢复完整上下文
cat docs/status/research-tree.md   # 战略可视化(1 confirmed/9 falsified)
cat .claude/climb/hypotheses.yaml  # 假设池
cat .claude/climb/session-state.json
# climb 状态机 + 真分注入: bash tools/climb/apply-lb-score.sh "<run_id> <score>"
```

## Ruled out（don't re-explore，详见 MEMORY negative_cache）

context加码 / 廉价声学 / 文本词袋 / Qwen3 pooled(树or神经头) / VAP-mel / 本机whisper —— 冻结pooled特征喂浅模型撞0.71墙；本机whisper资源不可行。
