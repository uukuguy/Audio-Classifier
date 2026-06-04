# 深度技术 (桶 6)

> 某个技术点能展开 2-3 页讲的素材. 决赛阶段评委会追问技术细节, 答辩时不能"也就那样过去了".
> 现在不写完整 deep-dive, 只**列要展开的题目** + 记**关键细节备忘**.

## 已识别的 deep-dive 题目

### DD-1. orthofuse per-class 正交融合 (数学 + 工程)

**展开点**:
- 为什么 per-class 路由 > 全局 ensemble (类间独立性 + 不同源对不同类强弱)
- 数学: 每类的融合规则 (C/T/BC/I/NA 分别走什么权重组合)
- 与 stacking / blending 的区别 (stacking 是 meta-learn 加权, 我们是先验路由)
- 失败的对照: stack-fusion-grid 实验 (D-N) 在 OOF 上看着好, 上线反挫
- 实际产物: `tools/climb/cycle_orthofuse.py` 200 行代码

**素材出处**: tools/climb/cycle_orthofuse.py, DECISIONS D-7 段

### DD-2. 软加微权融合 (D-22 范式) 的理论解释

**展开点**:
- 软加 0.05-0.20 = 微扰主源, 不破坏 BC 列稀疏信号的"硬"决策
- 为什么 0.5 重权过载 (Omni 0.5 真分 -0.027) 而 0.2 (cand2 +0.011)
- 权重曲线为什么有峰 (wsp_ms 峰在 0.07, Omni 在 0.15)
- 可能机制: 软加是 ensemble 的 soft Bayesian average, 单源 prob 强弱在低权下被适度信任
- 与 boosting / mixture of experts 的关系

**素材出处**: DECISIONS D-22, D-23, JOURNAL 6/2-6/4

### DD-3. 双 SSL_ms 协同效应 (D-25) 的可能机制

**展开点**:
- 实测: 单 e2v_ms 0.03 = -0.0015, 单 hub_ms 0.03 估也 -0.0015, 双加 = +0.0069 非加法
- 假说 1: 两源在 BC/T 类决策边界的"互补错误"被低权融合平滑
- 假说 2: SSL 表征空间正交 (e2v 副语言情感 vs hub 中文音素), 微权融合 = 多视角投票
- 假说 3: 单源加 0.03 把主源 wsp_ms 0.07 的信号"挤掉"了 (一个零和效应), 双源同时加保持 wsp_ms 主导
- 待验: 6/5 三 SSL_ms 微叠 (R4 + w2v2_ms 0.03) 真分能否 +0.005 验证机制

**素材出处**: DECISIONS D-25, 6/5 5 push 候选 (probe-day7)

### DD-4. cap1 OOF 红旗系统性误诊 (方法论自省)

**展开点**:
- cap1 = 每通对话首窗 = 369 通 = 369 样本
- 公榜测试集 = 1000 段独立切片
- BC 类正例只有 9 个 (cap1 369 段里), 统计噪声极高
- 数学: 9 个正例的 F1 方差 (假设独立 Bernoulli) ≈ ...
- 实际后果: 5 次反范式 push (D-17/19/20/22/23) 推翻同一个红旗
- 教训: 在样本稀疏 + 分布漂移场景, OOF 不能选源

**素材出处**: DECISIONS D-17, D-22, D-23, D-25

### DD-5. climb autonomous-loop 机制 (工具链层)

**展开点**:
- hypothesis pool (paradigm × cost × ranking) state machine
- LLM-driven iteration: 每 cycle Claude 自决 (PUSH / SKIP / PIVOT)
- calibration matrix per-paradigm (mean gap, std, last_3)
- push.sh / apply-lb-score.sh / regen-tree.py 协议
- session-state.json + research-tree.md 双层 (active + storage)
- 25 cycles, 25 push 全自动状态机管理
- 跟 Karpathy autonomous research 的关系

**素材出处**: ~/.claude/shared-rules/climb.md (全文), docs/status/climb/

### DD-6. 跨切片 cap0-cap4 稳定性分析 (复赛鲁棒性论证)

**展开点**:
- 每通对话切 5 个 30s 段 (cap0-cap4), 模拟 test 独立 30s 切片
- 每段算独立 macro F1, range = max-min = 跨切片不稳定性
- 实测 12 源 range 0.058-0.097
- 最稳: e2v_ms 0.058 / hub_bcaug_ms 0.059 / w2v2_ms 0.059
- 最不稳: omni3b 0.097 / omni3b_ms2 0.094 / hub_single 0.086
- 含义: SSL ms 系跨切片最友好 → 复赛镜像主力是 R4 双 SSL 微叠 + R1 fallback

**素材出处**: docs/status/2026-06-04-submission-strategy.md, 6/4 实测脚本

### DD-7. negative cache: VAP/CPC 整路证伪 (跟同行讨论的论据)

**展开点**:
- VAP (Ekstedt&Skantze, ACL 2022/2024) 是 turn-taking 学术 SOTA, 我们 D-1 验证它不灵
- VAP 微调真分 0.6337 (-0.079 vs SOTA 0.7124)
- 探针实验: VAP head 原生信号 p_now/p_future/256类/vad/熵 对 BC |r|<0.04
- VAP 原文 backchannel objective 是 unfinished TODO (代码 line 325 注释)
- 含义: 单纯音频路线 (mel/whisper/VAP) 抓不住 BC 时机, BC 信号在文本/语义

**素材出处**: DECISIONS D-1, JOURNAL 5/30, baselines/VAP/

## Deep-dive 不写完整的原因

每个 deep-dive 展开是 2-3 页 PDF, 7 个 = 14-21 页. 复赛技术报告**只 3 页**, 决赛 PPT **每 slide < 1 分钟讲解**. 现阶段只列"题目 + 展开点", 7/16 选 3-4 个真正展开.

## 触发新 deep-dive 题目的信号

- 实验做完发现机制能讲 2 段以上
- 评委可能问"为什么是 X 不是 Y" 的技术点
- 学术上能开新研究方向的发现 (副产物)
