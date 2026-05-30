# Next-Session Handoff

**Updated:** 2026-05-30 21:15
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **SOTA 仍 = 变体F 0.7124**（一整轮无新提交破 SOTA）。前10 门槛 0.7285，差 0.018（约 1.8 分）。
2. **本轮把所有突破路径探到了底**——BC（9+角度）、T/I 文本、序列框架、Omni 双用法、融合，全部证伪或负增益。见 DECISIONS D-1~D-5。
3. **根因诊断清楚（D-5）**：卡 0.712 = **只有 context-LGBM 一个强信号源**，音频/文本/序列全太弱。融合救不了（缺第二个强且正交的模型）。
4. **榜单框架修正（关键）**：BC 不是天花板（榜首 0.73-0.75 也没做高 BC），领先来自**全类各榨一点 + 多模型融合 + 工程**，增益 1-3 分。
5. **真问题（下一步核心）**：如何造出 **context 之外的第二个独立强信号源**（≥0.70 且正交）——这是所有路径的瓶颈。

## 下一步候选（按用户最后未决方向）

用户说"开新会话继续"，未锁定具体方向。候选（待新会话定）：
1. **修好 MLP 再试融合**：当前 MLP 坏了（无 class_weight，BC=0 拖累集成）。修 class_weight + 调架构，看修好的神经模型能否提供树之外的正交性。**最快、零成本（本机）。**
2. **接受 0.712 转复赛镜像收尾**：CUDA Dockerfile + 增强代码合规打包 + 最终提交确认 + 文档。务实路径。
3. **找全新第二信号源**：D-5 指出的真问题。但音频/文本/序列已穷尽，需要真正新角度（用户可能有想法）。

## 本轮完整证据链（不要重走 — DECISIONS D-1~D-5）

| 路径 | 结果 |
|---|---|
| VAP/CPC 全 pool/构型/微调 | BC ≤0.222（D-1）|
| Omni-3B（已下载云端）encoder | =WhisperFeatureExtractor（=已证伪 whisper）|
| Omni-3B LLM zero-shot 推理 | 无判别力（全答"是"）|
| F0/pitch（音频最强分支）| 融合 +0.005 |
| context 导数/突发/周期 | 零增益（0.21 已榨干）|
| 高维核映射（RBF/RFF/MLP）| 全输线性 AUC 0.64（样本量 40→150通不变）|
| 音频增强（3x BC）| AUC 不升反降 |
| 序列/计数框架 | 0.206<0.212 |
| T/I 文本词汇 | CV 虚高不泛化，真分 0.6392（D-3）|
| 算法正交集成 4 算法 | −0.023（三树不正交+MLP坏）（D-5）|
| 10seed 稳健化 | 无增益（5seed 已收敛）|

## 关键资产

- **Omni-3B 已下载云端** `/root/audio-classifier/models/Qwen2.5-Omni-3B`（~12G，3分片完整，modelscope 下的）
- 算法集成脚本 `tools/climb/cycle_algo_ensemble.py`（--stride 控样本量，**用 stride≥40**，stride=5 会 140 万样本卡死 CatBoost）
- 各 BC 探针 `cloud/probe_vap_kernel.py`(高维) / `probe_vap_augment.py`(增强) / `probe_omni_kernel.py` / `probe_omni_reason.py`
- 变体F SOTA `tools/runs/climb/variant-F-20260528-0559/`（cap1 0.6402 / 线上 0.7124）

## 云主机（24h 全开）

- AutoDL 4090D 48GB，`ssh -p 46379 root@connect.westd.seetacloud.com`，PATH=/root/miniconda3/bin
- **下载源铁律**：云主机用 **ModelScope**（`modelscope download --model X --local_dir Y` 断点续传）比 hf-mirror 快 5-6x。本机相反。
- **云端脚本限线程铁律**：`OMP_NUM_THREADS=8` + `torch.set_num_threads(8)`（不限会爆 20 核卡死，踩过 2 次）
- 主盘剩 100G+（已清 whisper 遗留 131G，备份在 14T `/autodl-fs`）
- 当前 0 残留进程，0 显存
- **本地 vs 云**：树模型（LGBM/XGB/CatBoost/MLP over context）本地跑（CPU，不用 GPU）；音频/Omni 提特征+训练上云（CUDA，本机 MPS 有 nan bug）

## 合规

用 Omni 须 6-10 前报备 xinyebei@xinye.com，**用户在准备报备材料**。报备邮箱是对外身份（注册邮箱 531045572@qq.com 体系，非系统 userEmail）。

## 各类天花板锚点（cap1 可信 CV，gap +0.055 → 线上）

| 类 | cap1 F1 | 线上正例 | 空间 |
|---|---|---|---|
| C | 0.974 | 974 | 饱和 |
| NA | 0.863 | 949 | 较饱和，温和阈值可能有零头 |
| T | 0.622 | 504 | 未饱和但文本路径证伪 |
| I | 0.512 | 65 | 未饱和但文本路径证伪 |
| BC | 0.200 | 30 | 信息论上限 |

## Ready-to-paste（候选1：修 MLP 重融合）

```bash
cd /Users/sujiangwen/sandbox/competitions-2026/Audio-Classifier
# 改 cycle_algo_ensemble.py make_clf 的 mlp 分支: 加 sample_weight(类平衡) 修 BC=0
# 然后剔除/加权融合
OMP_NUM_THREADS=4 python3 tools/climb/cycle_algo_ensemble.py --folds 5 --stride 40 --algos lgbm,xgb,mlp
```
