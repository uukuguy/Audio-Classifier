# 创新点候选 (桶 1)

> 任何"我们做法跟主流不同"或"反直觉发现"的瞬间 → 这里. 不要预先排序 / 评强弱, 7/16 决赛阶段一前再筛.

## 已知候选 (2026-06-04 盘点)

### C1. D-22 软加 0.05-0.20 微小权重融合范式 (颠覆 cap1 OOF 红旗范式)

**反直觉一句话**: 我们用 cap1 OOF "证伪"的源 (Omni-7B cap1=0.5649 << SOTA 0.6410), 软加 0.20 权重后真分 +0.011 破 SOTA.

**学术性**: cap1 OOF 选模型是常规做法 (类似交叉验证选超参), 我们用真分校准证明: **在测试集分布漂移 + 稀疏类样本不足的场景下, OOF 跟真分顺序/量级都不可信**. 软加微权 (0.05-0.20) 比硬替换/等权融合更鲁棒.

**素材出处**: DECISIONS D-22, JOURNAL 6/2 14:30, 6/4 09:16/10:00

### C2. D-25 双 SSL_ms 协同效应 (非加法的协同增益)

**反直觉一句话**: 单 e2v_ms 0.03 软加 NSOTA 真分 -0.0015 (反降), 单 hub_ms 0.03 同样估也 -0.0015, **但两者一起加 +0.0069** (R4 NEW SOTA 0.7458).

**学术性**: 多模态 / 多源融合普遍假设增益线性. 我们的实测显示: 双 SSL_ms (whisper-bcaug-ms vs hubert-bcaug-ms) 间的微权融合存在**协同效应**, OOF 完全测不出 (R4 OOF -0.0021 真分 +0.0069 = 3.3x 反向). 可能机制: 两个 SSL 模型在 BC/T 类决策边界的"互补错误"被低权融合平滑掉.

**素材出处**: DECISIONS D-25, JOURNAL 6/4 10:00, finals-20260604/MANIFEST.json

### C3. orthofuse 跨源 per-class 正交融合 (非通用 ensemble)

**做法**: 不是用同一组权重平均所有源, 而是**按类路由**:
- C 类 (94% 恒正): 只用 ctx (LGBM)
- T 类: 0.7 wsp + 0.3 hub (whisper 主, hubert 辅)
- BC 类: 只用 ctx (硬路由保 BC 信号源稀疏不被稀释)
- I 类: ctx + wsp + hub 三投票平均
- NA 类: 只用 ctx

**学术性**: per-class 跨源路由打破"一个 ensemble 通吃 5 类"传统. 类间相关性低 + 各源对各类的强弱不同, 路由化提升单类 F1 互不影响.

**素材出处**: tools/climb/cycle_orthofuse.py, DECISIONS D-7

### C4. climb autonomous-loop 工具链 (LLM 驱动的 hypothesis-push-calibrate 自驱)

**做法**: 整套迭代框架自动化 — hypothesis pool → train → eval cap1 → push.sh 生 csv → 用户贴真分 → calibration matrix 更新 → 新 hypothesis ranked → 循环. 25 cycles, 25 push 全部状态机管理.

**学术性**: 是 Karpathy "autonomous research" 思路在比赛场景的具体实现. 工程价值高, 但答辩听众可能觉得"这是工具不是科研".

**素材出处**: ~/.claude/shared-rules/climb.md, docs/status/climb/

### C5. cap1 红旗系统性误诊 (negative methodological insight)

**反直觉一句话**: 在公榜 1000 段 vs 我们 cap1 369 段的样本量差里, "cap1 选源" 是过滤偏差源.

**学术性**: 比赛文献里很少有人写"我们试图选源的方法是错的". 我们用 4 个 D-17/D-19/D-20/D-22 跨越 1 周的链式分析最终推翻自己, 这是**方法论自省**, 学术诚信高分项.

**素材出处**: DECISIONS D-17, D-19, D-20, D-22, D-23, D-25

## 候选记录格式 (新增时用)

```
### C-N. <一句反直觉/与主流不同的话>

**做法/发现**: <2-3 句>
**学术性**: <为什么值得讲, 跟哪个主流方法对比>
**素材出处**: <文件/JOURNAL 行号/DECISIONS D-N>
```

## 决赛前需要回答的问题 (不填答案, 备忘)

- 哪 1-2 个创新点能撑 1 个 slide + 30 秒讲 (主推)?
- 哪些是 "技术细节 backup" (评委追问时回答)?
- 哪些是 "学术诚信 backup" (评委质疑测试集 2 泛化时回答)?
