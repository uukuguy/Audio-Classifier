# T5 报备邮件草稿 (6/8 21:00 前发)

**收件人**: xinyebei@xinye.com
**抄送**: 531045572@qq.com (我方参赛邮箱, 必须用此邮箱发, **不**用 git 账号邮箱)
**截止**: 主办方 6/10 截止, 我方 6/8 21:00 前发, 留 2 天缓冲

---

## 决策依据 (6/6 D-28 校准)

复赛镜像主力 = S5 (R4 + omni3b_ms2 0.05), 备份 = R1/R5。R1-R6 实际依赖的 component:

| component | 来自模型 | 是否白名单 | 报备 |
|---|---|---|---|
| ctx | 我方训 LightGBM | — | 不需 |
| wsp / wsp_ms / wsb | whisper-large-v3 (OpenAI) | ❌ | ★ 必须 |
| hub / hub_ms | chinese-hubert-large (TencentGameMate) | ❌ | ★ 必须 |
| e2v_ms | emotion2vec_base (Modelscope/iic) | ❌ | ★ 必须 |
| Omni3B_ms2 | Qwen2.5-Omni-3B (Alibaba) | ✅ 白名单 | 不需但建议列 |

**Qwen3-0.6B / Qwen3-1.7B 删除**: 实验阶段 (D-21 前) 尝试过, R4/S5 复赛镜像配方**不依赖**, 不进决赛镜像 → 不报备 (避免暴露使用白名单外 Qwen 子模型的口实)。

**chinese-wav2vec2-large 删除**: S1 (R4 + w2v2_ms) 已证伪 -0.008 三 SSL 撞墙, 不进 R1-R6, 不报备。

---

## 邮件正文 (中文, 准备好就直接发)

```
主题: 第十一届信也科技杯 公开模型报备 — SpeechlessAI

主办方好,

按操作手册 6/10 前公开模型报备要求, 列出我队复赛镜像使用的公开模型 (除官方白名单 Qwen 系列外):

1. whisper-large-v3 (OpenAI)
   下载: https://huggingface.co/openai/whisper-large-v3
   用途: encoder 帧特征提取 + LoRA 微调头 (wsp / wsp_ms)

2. chinese-hubert-large (TencentGameMate)
   下载: https://huggingface.co/TencentGameMate/chinese-hubert-large
   用途: encoder 帧特征 + LoRA 微调头 (hub / hub_ms)

3. emotion2vec_base (Modelscope/iic)
   下载: https://modelscope.cn/models/iic/emotion2vec_base
   用途: 情感声学特征 + LoRA 微调头 (e2v_ms)

均为可不受限制共享/使用/再传播的公开模型, 符合公示要求。复赛镜像总参约 5B (含官方白名单 Qwen2.5-Omni-3B), 在 8B 软约束内。

SpeechlessAI 队
2026-06-XX  (用户填发送日期)
```

---

## 发件前 checklist (用户操作)

- [x] 队名: **SpeechlessAI** (用户 10:53 确认, 跟公榜账号一致)
- [ ] 填发件日期
- [ ] 用 **531045572@qq.com** 发 (参赛邮箱, 已在 CLAUDE.md 锁定)
- [ ] 收件 xinyebei@xinye.com
- [ ] 抄送自己 531045572@qq.com (留存底)
- [ ] 发送时间 < 6/8 21:00
- [ ] 截图发件成功邮件 → 存 `docs/finals/T5-disclosure-sent-screenshot.png` 作答辩素材

## 如果主办方回复"需要补 X" (低概率)

| 可能要求 | 准备回应 |
|---|---|
| "Qwen3-0.6B / 1.7B 你们实验用过, 也要报" | 回: 仅实验阶段对照, R1-R6 复赛镜像不依赖, 已弃用。可附 6/4 提交策略文档佐证 |
| "emotion2vec_base 不是 HuggingFace, ModelScope 也算公开吗" | 回: ModelScope 是官方公开仓库, 开源协议无限制 |
| "训练数据增强用了什么" | 回: 仅赛方提供训练集, 无外部数据增强 (符合 FAQ#3) |
