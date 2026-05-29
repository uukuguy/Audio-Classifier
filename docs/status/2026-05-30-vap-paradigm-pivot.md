# 范式转向：从 whisper 微调 → VAP/CPC（turn-taking 学术 SOTA 模板）

> 2026-05-30 凌晨。用户两次关键质疑触发：①"一直绕在 whisper 一个模型上干什么" ②"climb 是探索"。
> Research 结论 + 判断，备查。

## 触发：whisper-large-v3 是双重错误

今天烧了大量云主机时间在 whisper-large-v3 LoRA 上，两次估时严重错误（全量估5h实际30h；cap40探针估3h实际63h）。根因 = whisper-large-v3 **193ms/前向**（32层transformer），任何全量×多epoch×多fold = 几十小时。

**更深的错误**：whisper 是 **ASR（转文字）模型**，而 turn-taking 任务需要的是"谁在说/会不会插话"= 说话人区分 + 韵律 + 重叠语音信号。**用错了模型族**。

## Research 铁证：VAP/CPC 才是对口范式

学术界这个任务（预测未来语音活动/turn-taking事件）的 SOTA 模板是 **VAP (Voice Activity Projection, Ekstedt & Skantze, Interspeech 2022)**，不是 whisper。

### 为什么 VAP 完美契合本赛题
1. **VAP objective = 预测未来 2s 对话状态**（256 states）→ 赛题正是预测未来 2s 内 5 类事件，**任务同构**
2. **VapStereo = 双声道 stereo 模型，只需 stereo 波形**→ 赛题正是 8kHz 双声道电话
3. **CPC encoder 是 causal/incremental（因果）**→ 天然满足赛题因果约束（只用过去信息）
4. **CPC encoder 极小极快**：5层 strided CNN（strides[5,4,2,2,2], 512 hidden, 每10ms一特征），16kHz PCM 直接跑。**比 whisper-large-v3 640M 小几个数量级，前向毫秒级 → 全量训练可行**
5. **跨声道 cross-attention** 捕捉 turn-yielding/interruption/backchannel 交互
6. **微调专门提升 BC**：论文"Yeah, Un, Oh"(2410.15929) 微调 VAP 做实时 BC 预测，F1 显著超 baseline；多语言 VAP 微调 BC 提升 **0.3+**
7. **支持 Mandarin（HKUST 电话语料）**——中文电话域已验证
8. **文献明确：CPC 比 wav2vec2/MMS 更对口 VAP 任务**

### 现成资产
- 官方仓库 `github.com/ErikEkstedt/VAP`（stereo 版，不需 VAD 输入只需双声道波形）
- 自带预训练权重 `examples/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`
- `scripts/finetune.bash` 微调脚本 + hydra 配置
- 加载：`VAPModule.load_model(ckpt)` 或 `VAP(EncoderCPC(), TransformerStereo()).load_state_dict()`
- 依赖：python3.10 + torch≥2.0.1 + CUDA

### VapStereo 架构
```
双声道 stereo 波形(8k→16k)
  → CPC encoder(causal, 每声道) → 帧特征序列
  → self-attention(1层, 共享权重, 每声道)
  → cross-attention(3层, 共享权重, 跨声道)
  → linear → VAP vocabulary(256 states, 未来2s)
  + VAD objective(每声道当前帧语音活动, 处理 bleed-over)
```

## 判断：下一步方案（VAP 微调）

### 为什么有希望破 0.7124
- BC 是唯一瓶颈类（F1 0.20）。VAP 是文献证明能提升 BC 的方法（微调 +0.3）
- CPC causal encoder 天然因果、专为"预测未来语音活动"自监督训练 = 与赛题目标对齐
- 速度可行 → 能用全数据训练（解决之前 800x 数据劣势）

### 风险/未知
- VAP 预训练在英语/瑞典语对话，中文电话域需微调适配
- VAP objective(256 states 未来2s) → 赛题 5 类事件，需设计映射头（VAP 输出转 C/T/BC/I/NA）
- 赛题标签是 chunk 级(80ms)，VAP 是 50Hz(20ms)帧，需对齐
- CPC 16kHz，8kHz 需重采样（但 CNN 比 whisper mel 对重采样不敏感）
- 合规：CPC(facebookresearch)是公开模型，需 6-10 前报备

### 工程计划（本地准备充分再上云，不再烧云主机调试）
1. 本地 clone VAP 仓库，读架构代码，搞清 VAP→5类映射 + chunk/帧对齐
2. 本地写适配脚本（VAP encoder + 赛题头 + 赛题 dataloader），静态审查无误
3. 用赛题数据小样本本地 dry-run 验证逻辑（CPU 也能跑通流程，不求速度）
4. 云主机一把跑通：先小验证(可行性) → 全量

## 已彻底排除（don't re-explore）
- whisper-large-v3 LoRA/冻结：193ms前向×全量=30-63h不可行，且ASR模型族不对口
- LGBM + 任何特征：撞 0.71 墙
- 文本词汇喂 LGBM：线上假正例净负
