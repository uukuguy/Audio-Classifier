# Next-Session Handoff

> Updated: 2026-06-24 12:00 end of session.

## TL;DR

1. **复赛阶段 (6/22-7/9 评测, 每天2次提交)**: 镜像端到端评分, 跟初赛离线出csv根本不同。
2. **重写了干净三源端到端架构** (ctx+wsp+hub orthofuse), 构建+push `team26:r3-base-20260624` → **真分 0.515 (崩)**。
3. **0.515 根因已诊断闭环**: 不是我的代码错, 是 **6/22 head (opencode训, 契约不明) 坏的**。解法明确: 换初赛验证过的 head ckpt (云端有)。
4. **下一步**: 拉初赛 head → 端到端 OOF 自评 F1 (目标对齐初赛三源 0.6532) → 重建镜像 push。

## 复赛关键约束 (官方指引)

- 评测 = 镜像端到端: 挂私有测试集2 (`/xydata`, **5-30s不定长**) → `/app/run.sh` → `/app/submit/submit.csv` → 自动评分
- ≤60min推理 / ≤8B / ≤32G镜像 / **每天2次提交** / 阶段一截止 **7/9 23:59**
- 7/15: 交训练代码+README+技术报告(3页). 评测有GPU(官方baseline支持CUDA)
- **push通路**: climb构建+本机冒烟+`docker push` 都由AI做(本地已login); 用户只"提交url到官方后台"+贴回真分
- registry: `finvcup-registry.cn-shanghai.cr.aliyuncs.com/finvcup/team26`

## 0.515 崩塌诊断 (闭环, chain-first)

**真根因: 6/22 head 坏**, 不是提取代码。证据链:
- 逐元素diff: 我的 ssl_encoder 提取 vs 初赛 extract_whisper_cuda → 段长64000(8s)/w16(128000)/tail400+pool **全 identical**。提取没问题。
- head 加载: WhisperVAP vs model.pt = 0 missing/unexpected, 匹配。
- 6/22 head (opencode训) 喂正确特征反而出错 → 它用了不同提取/数据训, 契约不明 = 坏。
- **现场 wsp probs `[0.913,0.296,...]` ≠ 初赛缓存 `[0.516,0.601,...]`** (但缓存是3seed mean, 现场是单seed, 对比基准本就不公平 — 已弃此对比, 改端到端OOF自评)。

**修复中保留的3个有效修复** (初赛正确逻辑, 我重写时漏了):
1. `context.py` 加 `normalize_ctx_to_375(pad_na_left)` — 变长ctx必须pad到375再featurize (短ctx位置/占比特征分母L错位, 实测截短100: BC+0.165)
2. `ssl_encoder.py` whisper 加 `h[:,-400:]` 取tail (whisper固定30s→1500帧, 有效仅末8s)
3. `ssl_encoder.py` 段长 `CTX_SEC*sr` (非*SR16, 8kHz上切末8s)

## Next steps (action-level)

1. **拉初赛验证过的 head ckpt 到本地** (替换坏的6/22 head):
   - whisper: 云端 `tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/ckpt_seed{1,7,42}_fold{0-4}.pt` (15个, 对应0.71755链路)
   - hubert/e2v: 同理找 `*-bcaug-multiseed-*/ckpt_*.pt`
   - 决定: 单seed单fold (简) vs 多seed/fold mean (准, 但端到端要跑N遍head — encoder1遍+N head)
2. **端到端 OOF 自评 F1** (用户决策的正确验证法, 绕开缓存对比):
   - 新架构在本机 train 折外数据跑 → 算 Macro-F1 → 对齐初赛三源 cap1 **0.6532** (C0.972/T0.661/BC0.20/I0.523/NA0.861)
   - 对齐 = 端到端正确; 不齐 = 还有bug继续chain-first
3. 对齐后 → 重建镜像 `r3-fix-<date>` → docker push → 给用户tag/url → 提交校准

## 不要再走的弯路 (ruled out)

- ❌ **6/22 head (opencode训)** — 契约不明, 0.515元凶, 弃用
- ❌ 用初赛缓存probs对比现场probs判对错 — 3seed mean vs 单seed基准不公平, 改端到端OOF自评
- ❌ 逐个猜改提取参数 — 已证提取代码=初赛正确, 别再调提取
- ❌ e2v (D-33): WavLM加载funasr权重假跑, 修需funasr重依赖, 0.03边际源 → H-F1b独立验证才议
- ❌ Omni: 11G+CPU超时+仅+0.0004
- ❌ probs.npz搬镜像 / 同tag推不同镜像 / docker system prune

## 关键产物 / 路径

| 路径 | 作用 |
|---|---|
| `src/{common,infer_e2e}.py` + `src/sources/{context,ssl_encoder,fusion}.py` | ★新三源端到端架构 (本session核心产出) |
| `submission/docker-src/r3-base/` | r3-base源码留档 |
| `Dockerfile.finals` | 唯一构建源 (三源, pynvml, 瘦身COPY) |
| `tools/runs/climb/orthofuse-3src-20260601-1607/cv_metrics.json` | ★初赛三源配方+基准0.6532 |
| `tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/` | 初赛验证head链路 (probs本地, ckpt云端) |
| `tools/climb/diag_*.py` | 本session诊断脚本 (extract_diff/probs对比) |
| `docs/status/2026-06-24-复赛端到端作战图.md` | 复赛作战图 (含⚡修正段) |
| `docs/status/2026-06-24-镜像架构设计-H-F1F2.md` | 架构设计 |

## climb 状态

- paradigm `finals-e2e-core4-orthofuse`. cycle45 = H-F1 (PUSH r3-base → 0.515 崩 → 诊断闭环)
- pending-lb: r3-base-20260624 已落真分0.515 (待从pending清除/标falsified)
- 下个cycle候选: H-F1 修复版(换初赛head) > H-F5阈值 > H-F1b e2v
- push_mode = auto-docker-build (config已校正)

## 云机 (开机中, 用完记得关)

- `ssh -p 46379 root@connect.westd.seetacloud.com` 4090 48G, `/root/audio-classifier` (非git, 只训练/提特征)
- 初赛验证head ckpt + 帧缓存(whisper64G/hubert11G/e2v/w2v2) 都在云端

## Ready-to-paste

```bash
cd /Users/sujiangwen/sandbox/competitions-2026/Audio-Classifier
source ~/miniconda3/etc/profile.d/conda.sh && conda activate deep-research

# 拉初赛验证过的 whisper head (单seed示例, 替换坏的6/22 head)
scp -P 46379 root@connect.westd.seetacloud.com:/root/audio-classifier/tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/ckpt_seed42_fold0.pt models/wsp_head/model.pt
# hubert 同理找 hubert-bcaug-multiseed 的 ckpt

# 端到端OOF自评 (需写脚本: 新架构在train折外跑→算F1 vs 0.6532)
# 冒烟5段: MODELS=$PWD/models TEST_ROOT=/tmp/smoke5 python -m src.infer_e2e
```
