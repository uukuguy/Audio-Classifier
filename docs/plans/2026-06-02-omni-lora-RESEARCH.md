# Qwen2.5-Omni-7B LoRA 多模态分类头 — Research

**Researched:** 2026-06-02
**Topic:** Omni-7B LoRA 微调用于 audio+text → 5 类 multi-label 分类
**Libraries:** transformers 5.5.0 (cloud), peft 0.19.1 (cloud)
**Codebase scanned:** yes
**触发**: RESUME #4 主路径 — 突破冻结 head 0.71755 SOTA 天花板, 距前 20 门槛 0.7243 缺口 0.00675

---

## Findings

### Library: transformers 5.5.0 — Qwen2.5-Omni 类层级

Omni 是**三段式套娃**, 不要整体加载 (浪费 50% 显存 + Talker/Token2Wav 跟分类任务无关):

```
Qwen2_5OmniForConditionalGeneration     ← 完整 14B 推理用 (不要)
├── Qwen2_5OmniThinkerForConditionalGeneration  ← ★ 这个! audio+text+image+video → text token
│   ├── audio_encoder (Whisper-style, ~600M)
│   ├── vision_encoder (~600M, 我们用不到)
│   └── text_model (Qwen2-style 7B LLM, hidden_size=3584)
├── Qwen2_5OmniTalkerForConditionalGeneration   ← 语音生成不用
└── Qwen2_5OmniToken2WavModel                   ← DiT + BigVGAN, 不用
```

**只加载 Thinker** (Context7 + WebFetch 两路独立确认):
```python
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "/root/.cache/manual_models/Qwen2.5-Omni-7B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = Qwen2_5OmniProcessor.from_pretrained("/root/.cache/manual_models/Qwen2.5-Omni-7B")
```

**省一半显存**: Thinker ~8.5B params × bf16 = ~17GB, 加 LoRA r=16 ≈ +20MB, batch 8 估 25-30GB / 48GB 富余 (vs 整体 Omni ≈ 28GB+talker/token2wav).

### Library: transformers 5.5.0 — Thinker forward 接口

Forward 签名 (省 vision 输入):

```python
out = thinker(
    input_ids,              # (B, L_text) — text token (含 audio placeholder <audio>)
    attention_mask,         # (B, L_text)
    input_features,         # (B, L_audio, 128) — Whisper-style log-mel
    feature_attention_mask, # (B, L_audio)
    output_hidden_states=True,
    return_dict=True,
)
# out.last_hidden_state: (B, L_text+L_audio_embedded, 3584)   ← ★ 接 head 的位置
# out.hidden_states: tuple(L_layers+1) — 想用中间层时取
```

**关键**: audio 通过 `<audio>` token 注入 text 序列, **输出 hidden state 是融合后的 LLM 表征** (不需要分别 pool encoder/text)。

**Processor 多模态输入**:
```python
conversations = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "..."},   # 实际可直 numpy
        {"type": "text", "text": "ASR: ...\nHistory labels: ..."},
    ]},
]
inputs = processor.apply_chat_template(conversations, tokenize=True, return_dict=True, return_tensors="pt")
# 出: input_ids / attention_mask / input_features / feature_attention_mask / audio_feature_lengths
```

也可不走 chat_template, 直接 `processor(text=..., audio=...)` 拿到一样的 keys (跟现有 `train_lora_whisper_bcaug.py` 一致简单些)。

### Library: transformers 5.5.0 — 音频输入约束

- Audio feature extractor 用 **Whisper 风格 log-mel**, **sampling_rate 16000** (跟 whisper_bcaug 兼容, 我们的 wav 是 8kHz 双声道需 resample)
- 长度: `seconds_per_chunk=2`, `position_id_per_seconds=25` → 30s 音频估占 ~750 audio token (vs text 200-400 token, 序列总长 ~1000-1200 安全)
- bf16 dtype 上 4090 上 forward 8s 单段估 ~80-120ms (比 whisper-large-v3 cap5 frozen 慢 2-3x 因 LLM 7B 多)

### Library: peft 0.19 — target_modules 选择

**Qwen2 系列标配**:
```python
LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],   # ← 跟 whisper_bcaug / qwen3 完全相同
    lora_dropout=0.05,
    bias="none",
    # task_type 留空 (不用 PeftModelForSequenceClassification, 我们自己接 head)
)
peft_thinker = get_peft_model(thinker, lora_cfg)
```

**进阶 (期望 +0.001~0.003 但增训练参数 2x)**: 加 `k_proj, o_proj` → 全 attention 4 投影都覆盖。
**MoE 不适用**: Omni 是 dense, 不走 `target_parameters`。

**只 LoRA 文本侧 vs 全 Thinker**:
- 默认 `get_peft_model` 会自动匹配所有 `q_proj/v_proj`, 包括 **audio_encoder + text_model + vision_encoder** (我们不用 vision 但 module 还在)
- 期望 **只 LoRA text_model**: `target_modules=["text_model.*\\.q_proj", "text_model.*\\.v_proj"]` 正则模式 (peft 0.19 支持) — **但实测 audio_encoder 的 LoRA 也常带涨**, 这是音频任务首选保留两个 encoder
- **首版决策**: 默认 `["q_proj", "v_proj"]` 全覆盖, 不正则限制, 跟 whisper_bcaug 同款

### Library: peft 0.19 — 提 hidden state 接 head (不走生成)

**反例**: 不要用 `task_type="CAUSAL_LM"` + 看 logits — 我们不是预测下一个 token。
**正解** (跟 `train_lora_whisper_bcaug.py` 完全同型, qwen3_head.py 也是这个范式):

```python
class OmniClassifier(nn.Module):
    def __init__(self, peft_thinker, hidden_size=3584, ctx_dim=4):
        super().__init__()
        self.thinker = peft_thinker
        self.ctx_proj = nn.Linear(ctx_dim, 64)  # 4 个 ctx OOF base
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size + 64),
            nn.Linear(hidden_size + 64, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def forward(self, input_ids, attention_mask, input_features, feature_attention_mask, ctx_oof):
        out = self.thinker(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            output_hidden_states=False, return_dict=True,
        )
        h = out.last_hidden_state                          # (B, L, 3584)
        # mask-aware mean pool (跟 qwen3_head.py 同款)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # (B, 3584)
        ctx_feat = self.ctx_proj(ctx_oof)                  # (B, 64)
        return self.head(torch.cat([pooled, ctx_feat], 1)) # (B, 5)
```

**Loss**: `nn.BCEWithLogitsLoss(pos_weight=pw)`, `pw` 按 train 类频率倒数估 (跟 train_head_bcaug.py / train_lora_whisper_bcaug.py 完全同款)。

### Pitfall — 5/29 LoRA whisper 卡死教训不重蹈

`train_lora_whisper_bcaug.py` 5/30 实测 **CPU 100% / GPU 0% 24min 卡死** = mel + resample 实时算瓶颈, 不是 GPU 慢。
**对策**: 训练前**先把 audio 离线 resample 到 16kHz 存 npy** (复用 `cloud/extract_bcaug_cuda.py` 的 wav 加载逻辑, 但只存 16k waveform, 不算 mel — processor 自己算 mel 时 GPU 调度不会卡)。

**或**: dataset `__getitem__` 把 `processor()` 调用本身放在 worker (num_workers=4) 而不是 main process 算, 已是 train_lora_whisper_bcaug.py 后期改法 (但仍卡, 因为 76GB whisper cache 抢 IO)。Omni 没有现成 cache, 必须**离线 resample + processor 在 worker 算**。

### Pitfall — `device_map="auto"` 多 GPU 风险

云端单卡 4090D 48GB, `device_map="auto"` 没意义且容易把部分层放 CPU → 训练奇慢。**强制 `device_map="cuda"` 或 `device_map={"": "cuda:0"}`**。

### Pitfall — bf16 LoRA 数值

- Thinker bf16, LoRA 默认 fp32 (PEFT 默认行为), 不需要手动 cast
- Loss + head 也用 fp32 (PEFT 自动 mixed precision)
- **不要** `model.half()` (fp16 训练 LoRA 数值不稳, 5/27 实测 nan)

### Alternatives considered

| 方案 | 状态 |
|---|---|
| 整 `Qwen2_5OmniForConditionalGeneration` 加载 | ❌ 浪费 50% 显存, talker/token2wav 跟任务无关 |
| `task_type="SEQ_CLS"` 用 PEFT 自动 head | ❌ Thinker 不支持, 且我们要拼 ctx_oof |
| 冻结 Omni 提帧 → 训小 head | ⚠️ 备选 — 若 LoRA 训不动 fallback. 但 Omni encoder vs whisper-large-v3 几乎是同一个 (论文级别), 期望 frozen Omni ≈ frozen whisper, 没新增益 |
| Qwen2-Audio-7B (单独的 audio 模型) | ⚠️ 备选, 模型未下载. Omni 是后继并 strictly 更强, 优先 Omni |

---

## Reusable Assets in Repo

| 路径 | 内容 | 在新脚本如何用 |
|---|---|---|
| `cloud/train_lora_whisper_bcaug.py` | LoRA + ctx_proj + 5-class head + 5fold + BCE + pos_weight + OOF + test 完整骨架 | ★ 复制为新文件改 model 部分 |
| `cloud/train_qwen3_head.py` | LLM LoRA + mask-aware mean pool + 5-class head + chat-style text input | ★ pool & head 部分直接抄 |
| `cloud/train_head_bcaug.py` | 简化版 (frozen + head only), BC 增强 cache 消费逻辑 | 训练循环 / fold 切分 / OOF 写盘 都同款 |
| `cloud/extract_bcaug_cuda.py` | wav 加载 + BC 3x 增强 + 8kHz→16kHz resample 模板 | dataset `__getitem__` 内复用 `load_wav_8k_dual` + `augment_wav` |
| `data/train/audio/<id>.wav` + `data/train/text/<id>.json` + `data/train/labels/<id>.npy` | 原始数据 | 跟所有现有训练脚本同 |
| `tools/runs/climb/_stack_cache_s40.npz` | 4 个 ctx OOF base | LoRA head 拼 ctx_proj 输入 |

**Prior art**:
- `docs/plans/2026-05-27-finvcup-turn-taking-CONTEXT.md` — 10 个决策契约
- `docs/status/DECISIONS.md` D-7 — LoRA whisper cap5 顶到 0.267 BC 但全量 30-63h 不可行
- `docs/status/2026-06-01-midgame-review-SYNTHESIS.md` — 三路 AI 评审 (Claude/Gemini/Opencode) 后 D-15 决策, 但 Omni 在评审时还没下载完, 是评审之后用户加的方向

---

## Recommendations

### Option A (★ recommended): Thinker-only LoRA + mask-aware mean pool + ctx_oof 拼接 5-class head

**架构**:
- 加载 `Qwen2_5OmniThinkerForConditionalGeneration` (8.5B, 不要 Talker/Token2Wav)
- LoRA `target_modules=["q_proj", "v_proj"]` 全覆盖 (text + audio encoder)
- 多模态输入: ASR 文本 (含历史标签序列) + 30s 切片 audio (8kHz→16kHz resample)
- mask-aware mean pool last_hidden_state (3584d) + ctx_proj(4d ctx_oof → 64d) 拼接 → 256→128→5 head
- BCE + pos_weight 5fold conv-level OOF

**Why**:
- Thinker-only 省 50% 显存, 4090 48GB 富余
- `q_proj/v_proj` LoRA = whisper_bcaug / qwen3_head 同款, peft 0.19 完全成熟
- mask-aware pool = qwen3_head 同款 (cap1=0.5823 失败原因是单源不是 pool 错), 直接复用
- ctx_oof 拼接是项目 SOTA 范式 (orthofuse-3src 用的就是 ctx 基座)

**Cost**:
- 离线 resample 8k→16k 369+1000 通: ~30min 云
- LoRA 训 cap5: 1845 通 × 5 切片 = 9225 样本, 5fold, 估 5ep × ~2-3min/ep × 5fold × 2-3min(forward 比 whisper slow) ≈ **4-6h GPU**
- BC 3x 增强 (复用 `extract_bcaug_cuda.py` 同款增强参数): 6332 BC × 3 ≈ +19k 样本, 总 ~28k → ETA 5-7h
- 总 ETA: **5-7h LoRA 训 + 30min 提帧 + 30min OOF/test 推理 + 30min orthofuse 融合 = 半天云**

**Push 门**: cap1 ≥ 0.6410 (SOTA) **且** per-class 至少有一类 ≥ 当前 strat winner + 0.005 (D-17 红旗)。
**期望真分**: +0.005~0.015 (RESUME #4 估), 落在 0.7225~0.7325, 距前 20 门槛 0.7243 **有戏**。

### Option B (fallback): Frozen Omni 提帧 + 训小 head

只在 Option A LoRA 训不动 / OOM / 卡死时降级。复用 `train_head_bcaug.py` 完整骨架, 只换 frozen encoder 为 Omni Thinker。

**Why not 首选**: D-1 已证 frozen whisper-large-v3 头封顶 0.222 BC, Omni encoder 跟 whisper-large-v3 几乎同源, 期望几乎不带涨。但写代码成本低 (改 5 行), 留作 plan B。

### Option C (太险, 不推): 整 Omni 加载 + LoRA Talker 出 token 监督

让 Omni 直接输出 "C/T/BC/I/NA" 文本 token, BCE 改 CE。完全没价值: (1) 浪费 talker/token2wav 显存 (2) 5 类 multi-label 强行往生成上套不自然 (3) eval 麻烦。

---

## Open Questions (for plan)

1. **音频切片长度选 30s 还是末段 8s/16s?**
   - 现有 `train_lora_whisper_bcaug.py` 用 cap5 (5 切片/通) audio 是末 8s。Omni 支持长 audio (24kHz 30s 也行, 但 token 多 forward 慢)。
   - 决策: **首版用 8s 末段** (跟 whisper_bcaug 完全对齐, 公平比较增益), 若 cap1 不破 SOTA 再试 16s/30s。
2. **slice-cap 选几?**
   - 全量 1845 通 × cap5 = 9225 + BC 增强 ~19k ≈ 28k 样本, 5ep 估 5-7h
   - cap3 = 1845×3 + BC 3x ≈ 21k, 估 4-5h
   - 决策: **cap5 全量** — Omni 是主路径不省, D-7 教训是 cap5 LoRA whisper 反挫线上, 这次配 ctx_oof 拼接 head 应能避 D-7
3. **LoRA r 选 16 还是 32?**
   - r=16: 跟 whisper_bcaug 完全同, trainable ~3M, 训快
   - r=32: trainable ~6M, 拟合更强但过拟风险 +
   - 决策: **r=16** 首版, 不破 SOTA 再 r=32
4. **训完是否做 BC 阈值搜?**
   - **绝对不**! D-18 教训 (w2v2low BC=0.10 阈值翻车 -0.048)。守 varF 阈值算法 (BC≈0.75)
5. **要不要冒烟测试 (5 通 dry-run) 才上全量?**
   - **必须冒烟**。用户 6/1 硬要求 "训练前环境全检"。dry-run 验: ① Omni Thinker 加载 OK ② processor 多模态输入 ② LoRA inject OK ③ forward → loss → backward 通 ④ 单 batch step VRAM 不爆 ⑤ ckpt 保存 ⑥ resume 加载
6. **跟 SOTA orthofuse-3src 怎么融合?**
   - Omni OOF probs.npz → 加进 orthofuse 4 源 (ctx + whisper + hubert_bcaug + Omni)
   - **关键**: 必须满足 D-17 红旗 — Omni 单源 cap1 ≥ 0.6410 **且** per-class 至少一类 ≥ 0.005 优势, 否则不加进融合 (D-15 qwen3/D-17 e2v 都因这条反挫)

---

## Exit

下一步: 直接进 plan/写代码 (决策已全有, 不需要 /discuss)。

```bash
# 1. 把 train_lora_whisper_bcaug.py 复制改造为 cloud/train_omni_head.py
cp cloud/train_lora_whisper_bcaug.py cloud/train_omni_head.py
# 改: WhisperModel → Qwen2_5OmniThinkerForConditionalGeneration
# 改: WhisperFeatureExtractor → Qwen2_5OmniProcessor
# 改: dataset 加文本 (utts + history labels) 拼成 processor 输入
# 改: model forward 接受 (input_ids, input_features, feature_attention_mask, ctx_oof)
# 保持: LoRA r=16 / target_modules=["q_proj","v_proj"] / 5fold / BCE+pos_weight / OOF+test

# 2. 本机冒烟 5 通 dry-run (--convs 5 --epochs 1 --folds 2 --slice-cap 2 --batch-size 2)
#    本机 MPS 跑 cpu fallback, 验代码逻辑通 (Omni 加载会失败但 dataset+forward 接口能验)
#    或: 直接小 batch rsync 上云试跑

# 3. 上云全量
rsync -avz cloud/train_omni_head.py -e "ssh -p 46379" \
  root@connect.westd.seetacloud.com:/root/audio-classifier/cloud/

ssh -p 46379 root@connect.westd.seetacloud.com '
cd /root/audio-classifier
TS=$(date +%Y%m%d-%H%M)
mkdir -p /root/runtime/active tools/runs/climb/omni-lora-$TS
OMP_NUM_THREADS=4 setsid nohup python cloud/train_omni_head.py \
  --epochs 5 --slice-cap 5 --folds 5 \
  --run-dir tools/runs/climb/omni-lora-$TS \
  </dev/null >/root/runtime/active/omni_lora.log 2>&1 &
echo $! > /root/runtime/active/omni_lora.pid
'
```
