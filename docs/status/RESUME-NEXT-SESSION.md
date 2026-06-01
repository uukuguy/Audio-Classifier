# Next-Session Handoff

**Updated:** 2026-06-01 08:42（初赛收口 — 三 Open Question 全清空，进入等待期）
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **初赛已收口**：真 SOTA = `orthofuse-20260531-0319` **0.71529**（双源 ctx+whisper，T=w70/I=whisper）。距前 10 门槛 0.7285 差 0.0135，D-1~D-12 全路径证伪闭合。
2. **初赛代码评审包就绪**：`submission/code-20260601.zip`（42KB），等 **6/17 TOP 40 公布** 后上传。
3. **6/1 收口三事完成**：①云机已关 ②参赛邮箱 `531045572@qq.com` 写入 CLAUDE.md 合规段 ③`57ea52c` 落盘（CURRENT-STATE 5-28→6-1 校准 + 主办方手册 PDF 入库）。
4. **进入等待期 6/2-6/16**：无新 cycle 待跑，全路径已证伪；可做的事只有**复赛镜像前置准备**（Dockerfile 草稿、推理 pipeline 接口设计、合规报备邮件起草）。

## 当前阶段时间线

| 时间 | 动作 | 状态 |
|---|---|---|
| 6/2-6/9 | 复赛镜像前置：Dockerfile 草稿 + 推理 pipeline + 合规报备邮件起草 | 🟡 可选启动 |
| **6/10 前** | **报备邮件 → `xinyebei@xinye.com`**（非 Qwen 模型：chinese-hubert / chinese-wav2vec2 / emotion2vec / whisper-large-v3）。**对外邮箱用 `531045572@qq.com`，不是 CC 账户** | 🔴 硬截止 |
| 6/16 | 初赛阶段一结束（最后机会 push 新想法，配额 5/天） | |
| **6/17 00:00** | **主办方公布 TOP 40**，平台开放代码评审通道（48h） | 🎯 触发点 |
| 6/17-6/18 | 上传 `submission/code-20260601.zip`（≤100MB / 次） | |
| 6/19 | TOP 30 复赛名单公布 | 🎯 触发点 |
| 6/20-7/7 | 复赛阶段一：提交 Docker 镜像（≤20GB，模型 ≤8GB）+ 3 页技术报告 PDF | 🟡 云机届时再开 |

## Next steps（按优先级）

1. **🔴 起草合规报备邮件** 给 `xinyebei@xinye.com`（6/10 前发，不能拖到最后）— 列非 Qwen 模型清单 + 自陈用途。对外 from 用 `531045572@qq.com`。
2. **🟡 复赛 Dockerfile 草稿**：CUDA base + transformers + whisper-large-v3 离线模型 + lgbm / xgb / mlp 基座 + orthofuse 跨源融合脚本。先单写不构建（构建需云机或本地 docker）。
3. **🟡 复赛推理 pipeline 接口设计**：单段 30s 音频 + ASR JSON + context npy → 5 列 0/1 CSV，按手册 §复赛 输入输出规范。
4. **⚫ 可选**：初赛配额仍剩，若**想到全新正交方向**（非 D-1~D-12 范围）可再 push；否则空跑等 6/17。

## 初赛代码评审包（已交付）

### 物理位置

| 路径 | 内容 |
|---|---|
| `submission/code/` | 工作目录（可读，gitignored） |
| `submission/code-20260601.zip` | **提交件（42KB）**，已 zip 准备好 |

### 内容清单

| 包内文件 | 作用 |
|---|---|
| `README.md` | 任务理解 / 方案总览 / 复现 4 步骤 / 已用模型表 / 诚实声明（D-10/D-11/D-12 证伪表） |
| `MANIFEST.md` | 文件清单 + ASCII 数据流图 + 排除清单 + 评估口径 |
| `tools/climb/cycle_orthofuse.py` | ★ SOTA 主程（per-class 跨源融合） |
| `tools/climb/cycle_stack_fusion.py` | 4 base OOF + cache（SOTA 实际只用 lgbm_v1） |
| `tools/climb/cycle_context{,_v2}.py` | LGBM 基线 + v1/v2 手工特征 |
| `tools/climb/sliced_cv.py` | cap1 切片化 CV 评估协议 |
| `tools/climb/gen_variants.py` | 变体 F 5seed 集成（前 SOTA 0.71242，爬坡轨迹） |
| `cloud/extract_whisper_cuda.py` | whisper-large-v3 帧特征提取 |
| `cloud/train_head_cuda.py` | 神经小头训练 |
| `cloud/requirements.txt` | 依赖锚定 |

### 已对齐手册要求

- ✓ 无模型权重（README 给 whisper 公开模型下载源）
- ✓ 无数据（赛方提供）
- ✓ 100% 中文 docstring/README/MANIFEST
- ✓ 42KB ≪ 100MB
- ✓ 清掉内部 docstring 中的 climb 术语（H-XXX/D-XX/"变体F"等），保留代码字段名（自描述）
- ✓ Python 语法 check 全通过（8/8 py 文件 ast.parse OK）

### 何时提交

**不是现在**。提交流程：

| 时间 | 动作 |
|---|---|
| 现在 ~ 6/16 | 初赛阶段一持续进行，可继续 push（5 次/天，配额充裕） |
| 6/17 00:00 | 主办方公布 TOP 40 名单 → 如果进 TOP 40，平台开放代码评审通道（48h） |
| 6/17-6/18 | 上传 `submission/code-20260601.zip` 到平台（≤100MB / 次，每天 5 次） |
| 6/19 | TOP 30 复赛名单公布 |
| 6/20-7/7 | 复赛阶段一：提交 docker 镜像（≤20GB，模型 ≤8GB） |

## 真分账本（5 个 push 全表）

| run_id | strat | cap1 | 真分 | Δ vs SOTA | 备注 |
|---|---|---|---|---|---|
| variant-F-20260528-0559 | 5seed LGBM stride5 | 0.6402 | 0.71242 | base | 旧 SOTA |
| **orthofuse-20260531-0319** | **双源 ctx+whisper (T=w70, I=whisper)** | **0.6410** | **0.71529** | **+0.003** | **★真 SOTA** |
| orthofuse-s5-20260531-0627 | 双源 stride5 强基座 | 0.6455 | 0.71233 | -0.003 | 强基座反不如 |
| orthofuse-3src-20260531-0813 | 三源 ctx+whisper+hubert | 0.6540 | 0.71523 | +0.0 | noise floor 同 SOTA |
| cycle18-mlpbc-20260531-1244 | BC 改 mlp+whisper_70 | 0.6756 (cap1) | 0.69358 | -0.022 ❌ | D-11 BC cap1 cherry-pick 实证 |

## 复赛镜像准备（下一阶段，TOP 30 公布后做）

按手册：复赛阶段一（6/20-7/7）需提交 docker 镜像 + 可执行的训练/推理代码，技术报告 ≤3 页 PDF。

### 复赛镜像必交付物

1. **Docker 镜像**（CUDA base，≤20GB，模型权重 ≤8GB）— 含完整推理 pipeline:
   - context-LGBM stride40 lgbm_v1 base
   - whisper-large-v3 frozen encoder + 神经 head（出 T/I 增强源）
   - orthofuse 跨源融合（per-class strat: T=w70, I=whisper, 其余 ctx）
   - 输入接口：单段 30s 音频 + ASR JSON + context npy → 输出 5 列 0/1 CSV
2. **技术报告**（≤3 页 PDF）：方法 + 实验 + 诚实声明
3. **训练代码** + 推理接口
4. **报备邮件给 xinyebei@xinye.com**（2026-06-10 前）:
   - 用过的非 Qwen 模型：chinese-hubert-large / chinese-wav2vec2-large / emotion2vec_base / whisper-large-v3
   - **用户对外邮箱身份要问用户取真邮箱**（不要用 `girigiri@fastmail.com` 那是 CC 账户）

## Don't go down these paths again（永久 ruled out, D-1~D-12）

- ❌ 任何"加第 N 源"（D-1 VAP / D-10 4-5 源音频）— cap1 锁 0.6540
- ❌ 任何"在 cap1 369 上选 strat / grid / per-class"（D-3/D-7/D-9/D-11）— 全 cherry-pick
- ❌ 任何"ctx 基座算法替换"（D-11 mlp / D-12 LGBM sweep）— 已饱和
- ❌ 任何"BC 单类替换 ctx-only strat"（D-11 红旗）— 9 正例 +1 TP = 假信号
- ❌ 任何"文本词汇/F0/导数手工特征"（D-3/D-4）— cap1 虚高不泛化
- ❌ 任何"VAP / whisper LoRA 重训"（D-1）— 不可行

完整 negative cache 见 `docs/status/DECISIONS.md` D-1~D-12 + MEMORY `reference_negative_cache.md`

## 工具/环境元信息

- **Claude Code 2.1.156**（symlink 锁版本防跳 158）
- 本机 conda env `deep-research`（torch 2.7.1 + torchaudio 2.7.1），训练限线程
- 云主机 AutoDL 4090D：`ssh -p 46379 root@connect.westd.seetacloud.com`
  - **建议先关机省费**（GPU 0% 利用，但实例在线一直收费）
- 完整云端账本：`memory/reference_cloud_instance.md`
- 模型下载源铁律：云端 modelscope >> hf-mirror（国内 IDC）/ 本机相反

## 提交配额状态

- 5/31 已用 3 个：orthofuse-s5 / orthofuse-3src / cycle18-mlpbc
- 配额 5/天，初赛阶段一持续到 6/16，剩余配额充裕
- D-12 后无必要再 push（已穷尽路径，每路 < +0.005 预期增量），但**手册写历史最好成绩排名**，没风险也可再试新想法

## 关键资产（磁盘验证）

| 路径 | 内容 | 用途 |
|---|---|---|
| `submission/code-20260601.zip` | **初赛代码评审包**（42KB） | 6/17 提交用 |
| `tools/runs/climb/orthofuse-20260531-0319/{pred_test1.csv,fused_probs.npz,cv_metrics.json}` | **真 SOTA 0.71529** | 复赛 pipeline 输出参考 |
| `tools/runs/climb/whisper-fusion-20260531-0143/probs.npz` | whisper OOF 179867×5 + test 1000×5 | 复赛 whisper head 复现 |
| `tools/runs/climb/_stack_cache_s40.npz` | 4 ctx base OOF/test (lgbm/xgb/v2/mlp) + Y + G | LGBM base 复现 |
| `tools/runs/climb/hubert-fusion-20260531-0750/probs.npz` | hubert OOF + test (1024d) | 用过需报备但不进 SOTA pipeline |
| `tools/runs/climb/w2v2-fusion-20260531-1120/probs.npz` | w2v2 OOF + test (1024d) | 同上 |
| `tools/runs/climb/e2v-fusion-20260531-1120/probs.npz` | e2v OOF + test (768d) | 同上 |
| 云端 `/root/audio-classifier/cloud/*.py` | extract + train_head_hubert + train_lora + train_vap | 复赛镜像复用 |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64G whisper 帧 stride5 全量 | 复赛镜像离线推理不需要 |

## Open questions（下次 session 优先确认）

1. **复赛镜像前置准备是否现在启动**：还有 16 天到 6/17，可以①闲置等 TOP 40 ②提前起 Dockerfile + 推理 pipeline 草稿（TOP 30 后只剩 17 天紧）。倾向 ②，6/10 报备邮件起草是硬截止必须做。
2. **是否还想再 push 配额**：D-1~D-12 已穷尽规划过的范式；用户若想到**全新正交方向**可再试，否则配额闲置无害（手册按历史最好成绩排名）。

## Ready-to-paste commands

```bash
# 1. 验证云端状态 + 关机决策
ssh -p 46379 root@connect.westd.seetacloud.com "ps aux | grep -E 'python.*train|extract' | grep -v grep; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"

# 2. 查看初赛代码评审包内容
unzip -l submission/code-20260601.zip

# 3. 查看真 SOTA 提交件
head -3 tools/runs/climb/orthofuse-20260531-0319/pred_test1.csv

# 4. climb 状态
cat docs/status/climb/research-tree.md  # 战略可视化（resume 只读这个）
cat docs/status/climb/session-state.json  # 动态状态

# 5. 6/17 真正提交时（不是现在）
#    去赛方平台 → 我的团队 → 提交结果 → 上传 submission/code-20260601.zip
```

## Pending commits

**全清空** — 6/1 08:42 commit `57ea52c` 已落盘最后一批。working tree 仅剩 `docs/赛题要求.md`（用户私有，铁律不动）。
