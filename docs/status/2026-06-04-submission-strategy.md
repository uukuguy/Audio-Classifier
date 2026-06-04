# 提交策略 — 6/4 真分账本 + 跨切片稳定性 + 6/5-6/16 节奏

> 25 push 全量真分账本分析 + 跨切片 cap0-cap4 macro F1 实测 → 6/5-6/16 提交节奏 + 复赛镜像准备
> 锚点: NEW SOTA = A_NSOTA+wsp_ms_w0070 = **0.738899** (排名 ≤ 7 估)

## 真分梯队 (25 push)

| 排 | 真分 | 候选 | base | 软加 | 备注 |
|---|---|---|---|---|---|
| 1 | **0.7389** | A_NSOTA+wsp_ms_w0070 | NSOTA | wsp_ms 0.07 TBCI | **NEW SOTA** 6/4 |
| 2 | 0.7367 | Q5 | NSOTA | wsp_ms 0.05 TBCI | 6/4 早 SOTA |
| 3 | 0.7365 | D3 Omni 5fold median+Q5 | Q5 | per-fold ≈ mean | 验 median≈mean |
| 4 | 0.7338 | B_Q5+e2v_ms_w005 | Q5 | e2v_ms 0.05 TBCI | |
| 5 | 0.7327 | P3 A_omni_w015 | SOTA-3src | omni 0.15 TBCI | 6/3 SOTA |
| 6 | 0.7324 | B_Q5+hub_ms_w005 | Q5 | hub_ms 0.05 TBCI | |
| 7 | 0.7301 | Q6 跨日复现 | NSOTA | e2v_ms 0.10 repro | 投递可信锚点 |
| 8 | 0.7301 | P1 NSOTA+e2v_ms_w010 | NSOTA | e2v_ms 0.10 | |
| 9 | 0.7296 | Q4 3 LLM 微 | NSOTA | 3 LLM 微融合 | |
| 10 | 0.7293 | Q7 wsp_base→wsp_ms | modified | base 替换 | |
| 11 | 0.7293 | C_NSOTA(wsp→wsp_ms)+omni015 | modified | base 替换+omni | |
| 12 | 0.7285 | cand2_SOTA+Omni_0.20 | SOTA-3src | omni 0.20 TBCI | 6/2 D-22 SOTA |
| 13 | 0.7282 | Q1 NSOTA+qwen4b_005 | NSOTA | qwen4b 0.05 | |
| 14 | 0.7274 | P3 A3B_omni3b_w020 | SOTA-3src | omni3b 0.20 | 8B 合规峰值 |
| 15 | 0.7267 | P5 A3B_omni3b_w015 | SOTA-3src | omni3b 0.15 | 8B 合规 |
| 16 | 0.7264 | P2 A_omni_w0125 | SOTA-3src | omni 0.125 | |
| 17 | 0.7259 | P1 A_omni_w010 | SOTA-3src | omni 0.10 | |
| 18 | 0.7244 | P4 NSOTA+omni3b_w005 | NSOTA | omni3b 0.05 | |
| 19 | 0.7215 | P2 C_omni020+e2v_ms_w010 | SOTA+Omni0.2 | e2v_ms 0.10 | 多源叠加红旗 |
| 20 | 0.7167 | P4 PF_omni_max | SOTA+per-fold | omni max 0.2 | per-fold 否决 |
| 21 | 0.7166 | P5 NSOTA_TI_only | NSOTA | cols=TI_only | cols 限制伤 |
| 22 | 0.7145 | cand3 SOTA+w2v2_TI_0.5 | SOTA-3src | w2v2 0.5 TI | 重权红旗 |
| 23 | 0.6909 | cand1 SOTA+Omni_0.50 | SOTA-3src | omni 0.5 TBCI | 重权红旗 |
| 24 | 0.6131 | cand4 Omni_single | Omni only | single source | 单源死路 |
| 25 | 0.6073 | cand5 4_BC-aug_eq | 4src eq | equal weight | BC=1 归零 |

## 跨切片稳定性 (复赛友好度, range 升序)

```
range=0.0580  e2v_ms        ★ 全场最稳 (e2v_ms 软加 = 复赛镜像安全网)
range=0.0582  wsp           ★ SOTA-3src 主源
range=0.0586  w2v2_ms       ★
range=0.0591  hub_bcaug_ms  ★
range=0.0606  wsp_bcaug_ms  ★ NEW SOTA 主源 wsp_ms (软加, 不是 base)
range=0.0607  omni          中位
range=0.0690  omni_ms2      偏不稳
range=0.0822  qwen3-0.6B    不稳
range=0.0859  hub           不稳
range=0.0867  qwen17b_ms2   不稳
range=0.0940  omni3b_ms2    最不稳 (8B 合规但跨切片差)
range=0.0969  omni3b        最不稳 (合规版风险溢价)
```

**含义**: 4 个 SSL multi-seed 源 (e2v/w2v2/hub/wsp 的 ms) range 都 ≤ 0.061, 是复赛镜像最安全的家族; Qwen / Omni 系全部 range ≥ 0.0607, 复赛风险溢价存在; **Omni-3B 跨切片是 Omni-7B 的两倍不稳** (尽管 cap0 cap1 高)。

## 关键发现

### 1. NSOTA base > SOTA-3src base 是确定性的
- NSOTA (= SOTA + wsp_ms 0.05/0.07): N=10, mean 0.7321
- SOTA-3src: N=8, mean 0.7216
- **base 升级 +0.011 > 任何单源软加 +0.011** (cand2 的)
- 含义: NEW SOTA 锁定 NSOTA+wsp_ms 0.07, 6/5+ 一律以它为基底

### 2. wsp_ms 权重曲线峰值未定 (高 ROI)
- 0.05 → 0.7367
- 0.07 → 0.7389 (+0.0022 NEW SOTA)
- 0.08 / 0.10 未验, 趋势在涨, **必投 6/5 P1/P2**

### 3. SSL ms 单源软加 Q5/NSOTA 已疲软但稳
- e2v_ms 0.05 = 0.7338 (-0.003 vs Q5)
- hub_ms 0.05 = 0.7324 (-0.004 vs Q5)
- 不会涨上限, **但跨切片最稳 = 复赛镜像安全网** (P5 / 决赛阶段必用)

### 4. Omni-3B 是 8B 合规救命稻草
- N=3 真分 0.7244/0.7267/0.7274 都跨过 0.724
- range 0.094 跨切片不稳, ms2 mean 可压
- **6/10 报备前必须锁: NSOTA+omni3b_ms2 不同权重 push 验合规峰**

### 5. Omni-7B 权重曲线已走完, 峰 = 0.15
- 0.05/0.10/0.125/0.15/0.20/0.50: 0.7244/0.7259/0.7264/0.7327/0.7285/0.6909
- 不再做 Omni 细粒度 (0.13/0.17), 真分锚已够

### 6. base 替换不可行
- wsp → wsp_ms 替进 SOTA-3src 基: 6/4 C 真分 0.7293 / Q7 真分 0.7293 都 < Q5
- 根因: wsp_ms 不能既做 base 又做软加 (信号重复)

### 7. 多源叠加 (OOF Top 引诱) 必败
- P2 dawn (omni 0.2 + e2v_ms 0.10): OOF Top1 真分倒数第 2 (0.7215)
- D-23 复证 D-22: OOF cap1 跟真分顺序高度不一致, **OOF 仅做粗筛**

### 8. per-fold ensemble (max/median) 等价 mean
- D3 Omni 5fold median+Q5 真分 0.7365 ≈ Q5 - 0.0002 (noise floor 内)
- H-D22-11 否决, **不再投 per-fold 候选**

## 不要再走的路 (8 红旗 + 真分证据)

| 红旗 | 历史真分 | 否决日 |
|---|---|---|
| 重权 ≥ 0.5 | cand1 -0.027, cand3 -0.003 | D-22 |
| 等权 4src+ | cand5 -0.110 BC=0 | D-15 |
| 单源 single push | cand4 -0.105 | D-20 |
| per-fold max/median | P4 -0.014, D3 ≈ noise | D-23 |
| cols 限制 (TI_only/BC_only) | P5 -0.015 | D-23 |
| base 替换 (wsp→wsp_ms) | C -0.007, Q7 -0.007 | D-23 |
| 多源叠加 | P2 dawn -0.015 OOF Top1 真分倒2 | D-23 |
| Omni 细粒度权重 (0.13/0.17) | 峰值 0.15 已锁 | D-23 |

## 6/5 - 6/16 提交节奏 (修订 — R4 0.7458 第 3 名后, 复赛准备前移)

### 阶段 1: 6/5 — 5 push 冲第 2 (高 ROI, R4 范式深挖)

| Push | 候选 | 路径 | 期望 |
|---|---|---|---|
| P1 | `S1_R4+w2v2_ms_003` (三 SSL_ms 微叠) | `submission/probe-day7-20260604-1005/S1_R4+w2v2_ms_003/` | 0.747-0.750 ★ 冲第 2 |
| P2 | `S2_NSOTA07+e2v_ms_004+hub_ms_004` (双源升 0.04) | 同上 | 0.745-0.748 |
| P3 | `S5_R4+omni3b_ms2_005` (R4 + LLM 8B 合规) | 同上 | 0.745-0.752 |
| P4 | `S4_NSOTA+wsp_ms_010` (wsp_ms 0.07→0.10 验峰) | 同上 | 0.737-0.740 |
| P5 | `S3_NSOTA07+e2v_ms_002+hub_ms_002` (双源降 0.02) | 同上 | 0.741-0.745 |

### 阶段 2: 6/6 - 6/8 — 3 天 15 push 公榜冲第 1 + 同步开始复赛镜像

**Push (5/天 × 3 = 15)**:
- **3 push**: R4 三 SSL 升级变体 (e2v+hub+w2v2 0.02/0.04, e2v+hub+w2v2 0.03+0.03+0.05 等)
- **3 push**: wsp_ms 0.08 / 0.09 探峰精细化 + base 替换 NSOTA_08 + e2v+hub 双微叠
- **3 push**: v2 ms2 三新源 (Omni-7B-ms2 / Omni-3B-ms2 / Qwen3-1.7B-ms2) 加进 R4 base
- **3 push**: 跨切片稳定性优先组合 (双 SSL_ms low-w + minimal Omni) — 复赛镜像 prototype 试水
- **3 push**: 备用 calibration / 紧急 fallback

**复赛镜像同步开工 (背景任务)**:
- 6/6: Dockerfile + entrypoint 框架, base 镜像选 (主办方手册待查具体规范)
- 6/7: 全部模型权重打包脚本 (~50G hf 模型 + LoRA ckpt), data/ 增强数据生成代码入镜像
- 6/8: train pipeline 全流程脚本 (用户提供主办方机器 GPU 估算)

### 阶段 3: 6/9 - 6/10 — 报备 + 复赛镜像 dry-run

**6/8 21:00 前**: 报备邮件发 `xinyebei@xinye.com` (cc 我们邮箱), 列 6 个非白名单模型 (见上节). 2 天缓冲避主办方不回。

**push (5/天 × 2 = 10)**:
- 公榜冲分进入"挑战第 1"模式, 5/10 push 用于 R4 系列细微调
- 5/10 push 用于复赛 friendly 组合的公榜对照 (验跨切片稳源在公榜上分差多少)

**复赛镜像**:
- 6/9: train 全流程本机 / 云端 dry-run, 计时
- 6/10: infer 流程 dry-run + entrypoint 切换开关 (env CHIRP_MODEL=R4|R5|R1|R6) 验证

### 阶段 4: 6/11 - 6/15 — 复赛镜像主战场 + push 最后冲刺 (25 push)

**Push (5/天 × 5 = 25)**:
- 6/11-12: 10 push 全部用于"复赛镜像准备"对照 — 测各复赛 fallback 路径在公榜真分
- 6/13-14: 10 push 用于公榜决赛冲分 (重点 R4 变体峰值)
- 6/15: 5 push 战略保留 (应急 fallback / 跨日复现锚 / 最稳组合提交一次防万一)

**复赛镜像必完成**:
- 6/11: 全部模型权重打包验证 (≤ 200G 量级)
- 6/12: train+infer 全流程在干净环境 (无 cache) 跑通
- 6/13: 数据增强代码 + LoRA train 全流程入镜像验证
- 6/14: 镜像内多 entrypoint 切换测试 (R4 / R5 / R1 / R6 多 model 切换)
- 6/15: docker push 到主办方 register (待手册查)

### 阶段 5: 6/16 — 公榜最后 5 push

- 5 push 全部为**最高真分组合**确认+复现 (避免末日翻车)
- 6/17: 提交评审包 (镜像 docker tar + pred_test1.csv)

## 复赛镜像合规组合 (最终交付候选)

8B 总参 + 跨切片 range ≤ 0.07 加权的组合, 按风险/收益排序:

| 组合 | 模型清单 | 总参 | 估真分 | 跨切片风险 |
|---|---|---|---|---|
| **R1 最稳** NSOTA + e2v_ms 0.05 | ctx + wsp + hub + e2v_ms | ~1.5B | 0.7338 (已验) | ★★★★★ 极稳 |
| **R2 中位** NSOTA + omni3b_ms2 0.15 | ctx + wsp + hub + Omni-3B | ~5B | 0.726-0.732 | ★★★ 中 |
| **R3 复赛对照** SOTA + wsb 0.10 | ctx + wsp + hub + whisper_bcaug | ~1.5B | 0.730-0.735 估 | ★★★★ 高稳 |
| **R4 全 SSL 微叠** NSOTA + e2v_ms 0.03 + hub_ms 0.03 | ctx + wsp + hub + e2v_ms + hub_ms | ~1.7B | 0.732-0.737 估 | ★★★★ 高稳 |
| **R5 NEW SOTA** NSOTA + wsp_ms 0.07 | ctx + wsp + hub + whisper_bcaug_ms | ~1.5B | 0.7389 (已验) | ★★★ 中 (wsp_ms 跨切片 0.061) |

R1/R3/R4/R5 全部 8B 合规 (≤ 2B), 是复赛镜像主力。R2 (Omni-3B) 作为 LLM 路径备份, 跨切片不稳但能产 0.726+。

> **R5 NEW SOTA 本身 8B 合规** — wsp_ms 是 whisper-large-v3 的 multi-seed 微调头, base 1.5B 不到。所以**目前 NEW SOTA 已经是合规的**, 不像 6/2 cand2 (Omni-7B = 9.4B 超额)。

## 决赛冲分推演 (2026-06-04 R4 NEW SOTA 0.7458 后)

### 真实排行榜定位 (2026-06-04 用户实时更新)

| 排 | 真分 | 备注 |
|---|---|---|
| 1 | **0.75471** | 新榜首 (5/27 快照里没出现的新提交) |
| 2 | 0.747489 | (5/27 ListenBeyond 上挪一位) |
| 3 | 0.74603 | (5/27 CapyBara 上挪一位) |
| **4** | **0.7458** | **我们 R4** |
| 5+ | ... | 5/27 快照里第 3 limzero 0.73568 已被挤到 5+ |

⚠ **5/27 快照已严重过时, 榜首密度增加**. 6/4 当天就有新队 0.7547 上榜首, 跟我们 R4 同步推进. **榜单是动态的, 我们的真分推进会被对手抵消**。

### R4 (NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03) = 0.7458 距:
- 第 3 (0.74603) — **缺 +0.0002** ← 6/5 1-2 个 push 即可达
- 第 2 (0.747489) — **缺 +0.0017** ← 6/5-6/6 可达
- 第 1 (0.75471) — **缺 +0.009** ← 决赛冲分目标, 但榜首仍可能继续上涨

**初赛榜首密度估** (实时): #1 - #4 间距 0.009, #1 - #2 间距 0.007 = 头部密集到 0.5%, **任何 +0.002 都可能改名次**, 但同时**对手也在 push**, 我们提 +0.005 不一定能进前 3 (取决于 1-3 名也提 +0.003 平均)。

### 最可能冲第 1 路径
1. wsp_ms 权重峰值右探 (0.08 / 0.10) — base 升级 +0.001-0.002 已知方向
2. R4 三 SSL_ms 微叠 (+w2v2_ms 0.03) — 协同效应可能再 +0.003-0.005
3. R4 双源升权 (0.04+0.04) 或降权 (0.02+0.02) 找峰值 +0.001-0.003
4. v2 ms2 三个新源 (Omni-7B-ms2 / Omni-3B-ms2 / Qwen3-1.7B-ms2) 真分首测, 估 +0-0.003 不确定增量

## ⚠ 复赛风险预警 (FAQ#2 官方警告)

> **"公开测试集打分结果与私有测试集打分趋势仍有可能不同。现在在初赛打分比较好的选手, 复赛成绩并不一定是最好的"**

主办方明确警告 — 初赛公榜 over-fit 是真风险。我们的 R4 0.7458 是基于 cap1 真分校准, 测试集 1 上调到的权重 (wsp_ms 0.07, e2v_ms+hub_ms 0.03+0.03) **在测试集 2 (私有, 含内部私有数据) 可能不是最优**。

**对策 — 复赛镜像不应只交 R4 单点, 而是"R4 + 复赛友好家族"组合可切换**:

| 复赛镜像主力候选 | 真分 (公榜) | 跨切片 range | 复赛友好度 | 角色 |
|---|---|---|---|---|
| **R4** NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03 | **0.7458** | ≈0.060 | 中 | 🏆 公榜峰值, 决赛主力 |
| **R5** NSOTA_07 = SOTA + wsp_ms 0.07 | 0.7389 | 0.061 | 中 | 🟢 base 极简版 |
| **R1** Q5 + e2v_ms 0.05 | 0.7338 | **0.058 最稳** | ★★★★★ | 🟢 跨切片最稳, 复赛 fallback |
| **R3** SOTA + wsb 0.10 | 0.7362 | 0.061 | ★★★★ | 🟡 0 wsp_ms / 0 Omni 极端友好 |
| **R6** NSOTA_07 + e2v_ms 0.03 (单 SSL_ms) | 0.7374 | 0.058 | ★★★★★ | 🟡 R4 单源版, 备用 |

**复赛镜像设计原则**:
1. **Train+infer 全流程入镜像** (6/20-7/7 窗口期, 每天 2 次, 取最后)
2. **生成数据 + 生成代码全包** (FAQ#4): 我们的 BC-aug 增强数据 + 训练脚本必须入
3. **不能在镜像里访问公榜测试集** (FAQ#3 严禁), 提交件必须靠模型在私有测试集 2 上 infer 产出
4. **保留 fallback path**: 镜像 entrypoint 默认输出 R4, 但保留切换到 R1 / R3 / R6 的开关 (env var) 给主办方/我们 debug
5. **耗时控制**: 复赛在云上跑, train 全量耗时 (Omni-3B-ms 87min × 3 seed + 6 个 SSL_ms × 3 seed × 15min ≈ 7-8h) 必须能在主办方机器跑完

## 公开模型白名单确认 (操作手册 6/4 读)

主办方公示的**白名单 5 个**:
- Qwen2.5-Omni-3B / Omni-7B (Omni 系)
- Qwen3.5-4B / Qwen3-4B / **Qwen3-0.8B** (Qwen3 系)

**我们当前用模型 vs 白名单 (6/10 前必须报备非白名单的)**:

| 模型 | 是否白名单 | 6/10 前报备 |
|---|---|---|
| Qwen2.5-Omni-3B | ✅ | 不需 |
| Qwen2.5-Omni-7B (cand2 NEW SOTA 9.4B 超额) | ✅ 但 8B 软约束 | 不需 |
| Qwen3-4B | ✅ | 不需 |
| **Qwen3-0.6B** (我们用了, 不在白名单) | ❌ | ★ 必须报备 |
| **Qwen3-1.7B** (我们用了, 不在白名单) | ❌ | ★ 必须报备 |
| chinese-hubert-large | ❌ 非 Qwen | ★ 必须报备 |
| chinese-wav2vec2-large | ❌ 非 Qwen | ★ 必须报备 |
| emotion2vec_base | ❌ 非 Qwen | ★ 必须报备 |
| whisper-large-v3 (含 wsp_ms = LoRA 微调) | ❌ 非 Qwen | ★ 必须报备 |

**报备邮件** (6/10 前发 `xinyebei@xinye.com` cc `531045572@qq.com`):

```
主题: 公开模型报备 — 队名 [我方队名]

主办方好,

按 6/10 前报备要求, 列出我队使用的公开模型 (除官方白名单外):

1. chinese-hubert-large (TencentGameMate)
   下载: https://huggingface.co/TencentGameMate/chinese-hubert-large
2. chinese-wav2vec2-large (TencentGameMate)
   下载: https://huggingface.co/TencentGameMate/chinese-wav2vec2-large
3. emotion2vec_base (Modelscope/iic)
   下载: https://modelscope.cn/models/iic/emotion2vec_base
4. whisper-large-v3 (OpenAI)
   下载: https://huggingface.co/openai/whisper-large-v3
5. Qwen3-0.6B (官方白名单外 Qwen 子模型)
   下载: https://www.modelscope.cn/models/Qwen/Qwen3-0.6B
6. Qwen3-1.7B (官方白名单外 Qwen 子模型)
   下载: https://www.modelscope.cn/models/Qwen/Qwen3-1.7B

均为可不受限制共享/使用/再传播的公开模型, 符合公示要求.
[队名] 队 [日期]
```

⚠ **Qwen3-0.6B / 1.7B 是否算 Qwen3 系白名单"派生"还需跟主办方确认** — 严格按公示只允许 Qwen3.5-4B / Qwen3-4B / Qwen3-0.8B 三个, 我们的 0.6B 和 1.7B 是同系列不同大小. 安全做法是**主动报备 + 备用 Qwen3-0.8B 替代** (Qwen3-0.6B → 0.8B 是白名单内的 minimal pivot).

## 关键决策点 (6/4)

1. **Qwen3-0.6B / 1.7B 是否能用?** — 我们 Q5 base 没用 Qwen3, 但 Q1/Q4 实验用过. **建议**: 立即把 Qwen3 排除出**复赛主力** R4, 只在公榜冲分时用. 决赛镜像不依赖 Qwen3-0.6B / 1.7B.
2. **报备邮件 6/8 前发** (留 2 天缓冲, 避免 6/10 当天主办方不回)
3. **6/5-6/8 公榜冲分 4 天** vs **6/9-6/15 复赛镜像准备 7 天** 时间分配
4. **复赛镜像需要 train+infer 全流程** — 这是大工程, 不能拖到 6/16 才开始
