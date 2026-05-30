# Next-Session Handoff

**Updated:** 2026-05-30 18:00
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **SOTA 仍是变体F = 0.7124**（未变）。前10 真门槛 **0.7285**，差 0.016。
2. **BC 确认信息论上限 ~0.22**——7+ 独立角度交叉验证（含用户 3 个质疑驱动的排查），全部证伪"音频/特征/框架能救 BC"。这是有充分证据的结论，不是放弃。
3. **当前在下 Qwen2.5-Omni-3B**（唯一未试大范式，云上 ~10G/12G，分片2 续传中）。下一步 = 冻结提 Omni 表征验 BC 信号（同 kernel 探针逻辑），强才微调。
4. **合规**：用 Omni 须 6-10 前报备 xinyebei@xinye.com，用户在准备报备材料。
5. 云盘已大清理（93G→20G，删 whisper 路线遗留 128G+），主盘剩 100G+。

## 下一步（Omni 下完后，第一动作）

**先验信号，再决定微调**（同 VAP kernel 探针逻辑）：
1. Omni 下完 → 写 Omni 表征提取脚本：用 `Qwen2_5OmniThinkerForConditionalGeneration` 的 audio encoder + text embedding 提 BC 特征
2. 跑 kernel 探针（线性 vs RBF/RFF/MLP）测 BC 可分性 AUC
3. **判据**：若 Omni audio AUC >> VAP 的 0.64 → 值得端到端微调；若 ≈0.64 → Omni audio 侧同命，转测"Omni LLM 联合语义推理"（把它当理解模型 prompt 判断 BC 时机，不是特征提取器）
4. Omni 真正可能赢的不是 audio encoder（SSL 同 VAP），是**音频+文本+context 联合语义**——这要把它当生成/推理模型用

**模型加载**（transformers 5.5 已确认）：`Qwen2_5OmniForConditionalGeneration` / `Qwen2_5OmniThinkerForConditionalGeneration` / `Qwen2_5OmniProcessor` 都有。只需 thinker 不需 talker（语音生成组件用不上）。

## BC 完整证据链（不要重走，全已证伪）

用户连续质疑逼出的彻底排查，全否：
| 角度 | 结果 |
|---|---|
| 全量 VAP 微调 | BC 0.222 |
| v2 attention-pool 单变量 | BC 0.08 < mean-pool（pool 非瓶颈）|
| F0/pitch（音频最强分支）融合 | +0.005（正交但弱）|
| context 导数/突发/周期特征 | 零增益（baseline 已榨干 0.21）|
| **高维可分性**（RBF/RFF/MLP）| 全输线性，AUC 0.64 |
| **样本量**（40→150通）| 结论不变 |
| **音频增强**（3x BC 变速/加噪）| AUC 不升反略降 |
| 序列/计数框架 | 0.206 < 二分类 0.212 |
| T/I 文本（D-3）| OOF 虚高不泛化，真分 0.6392 |

**结论**：BC≈0.22 是数据信息论上限，"未来2s会不会backchannel"高度难预测。

## Don't go down these（全证伪，见 DECISIONS D-1~D-4）

VAP/CPC 全 pool/构型 / whisper / mel / F0 单独 / context 任何新特征组合 / 文本词汇(CV虚高) / 序列框架 / 高维核映射 / 音频增强 / 10seed 稳健化(无增益)

## 云主机

- AutoDL 4090D 48GB **24h 全开**。`ssh -p 46379 root@connect.westd.seetacloud.com`，PATH=/root/miniconda3/bin
- **下载源铁律**：云主机(国内)用 **ModelScope** 比 hf-mirror 快 5-6x（`modelscope download --model X --local_dir Y` 支持断点续传）。本机相反(ModelScope 墙)。
- VAP 仓库 `cloud/VAP`(VAP_ROOT)，数据 `/root/audio-classifier/data/`，Omni 下到 `/root/audio-classifier/models/Qwen2.5-Omni-3B`
- 云端脚本限线程铁律：`OMP_NUM_THREADS=8`（torch 不限会爆 20 核卡死，踩过 2 次）
- 主盘剩 100G+，whisper 备份在独立 14T 盘 `/autodl-fs`

## Ready-to-paste（Omni 下完后）

```bash
HOST=connect.westd.seetacloud.com; PORT=46379
# 1. 确认下载完整
ssh -p $PORT root@$HOST 'du -sh /root/audio-classifier/models/Qwen2.5-Omni-3B; ls .../*.safetensors'
# 2. 写 cloud/probe_omni_kernel.py (仿 probe_vap_kernel, 换 Omni audio encoder)
# 3. 跑 kernel 探针测 BC AUC vs VAP 0.64
```

## 关键数字锚点

| 类 | OOF F1 | test正例 | 备注 |
|---|---|---|---|
| C | 0.971 | 974 | 饱和 |
| NA | 0.797 | 949 | 较饱和 |
| T | 0.542 | 504 | 文本 CV 虚高不泛化 |
| I | 0.434 | 65 | 同 T |
| BC | 0.212 | 30 | 信息论上限 |
