# LoRA 微调 whisper-large-v3 攻 BC 方案

> 2026-05-29 · 状态：方案待实施
> 前置：冻结 encoder 路线云端实测 falsified（最高 0.671 << SOTA 0.7124）

## 1. 为什么冻结路线失败

| Run | 线上 | cap1 CV | 与 SOTA 差距 |
|---|---|---|---|
| cloud-whisper-smoke (40通) | 0.6338 | 0.6413 | -0.079 |
| cloud-whisper-full-cycle1 (全量) | 0.6709 | 0.6521 | **-0.042** |
| cloud-whisper-full-balanced | 0.6437 | 0.6521 | -0.069 |

**根因**：whisper 通用语音表征（训练于 ASR/翻译）**不包含 turn-taking 预测信号**（韵律 timing / onset / 双声道话轮交互）。冻结 encoder → 小头学不到任务所需的细粒度特征。必须通过微调让 encoder 学到 turn-taking 特异信号。

## 2. 方案：LoRA 微调 whisper-large-v3 encoder

### 2.1 硬件可行性

| 项 | 值 |
|---|---|
| 云 GPU | RTX 4090 48GB VRAM |
| whisper-large-v3 encoder | 640M 参数（32 层，d=1280） |
| LoRA r=32, α=16 | 可训练参数 ~5-15M（占总量 <2%） |
| bf16 显存占用 | 基座 ~1.3GB + LoRA ~30MB + 优化器 ~60MB + 激活（按 batch） ≈ **< 10GB** |
| **结论** | ✅ 48GB 绰绰有余，甚至全参数微调都行（但数据量不建议全参数） |

### 2.2 训练数据

当前 `extract_whisper_cuda.py` 用 `--stride-mult 8` 提取，实际窗口数取决于此参数。全量滑窗数据：
- 369 通对话 × 每通数百窗（stride_mult=8 时约 ~1.44M/8 ≈ 180K 窗口）
- 缓存已提取：`data/whisper_cache/train/*.npz`（`frames [W,2,80,1280] fp16`）

**LoRA 微调不走缓存**——需要**重新前向传播 encoder**，让 LoRA 梯度回传。这意味着：
- 不再使用 `extract_whisper_cuda.py` 的缓存特征
- 直接从原始音频 `.wav` → whisper encoder（带 LoRA）→ 帧序列 → 小头
- 训练时显存 = encoder 前向 + LoRA 反向 + 小头

### 2.3 架构

```
原始音频 (.wav, 8kHz 双声道)
  → torchaudio resample → 16kHz
  → WhisperFeatureExtractor (log-mel)
  → whisper-large-v3 encoder + LoRA (r=32, α=16, target: q_proj/v_proj)
      → 帧序列 [2, 80, 1280]（双声道分别过 encoder）
  → proj → cross-attn(query 聚合) → 音频 2 向量
  + context 手工特征 (80d) → MLP head
  → 5 类 sigmoid + BCE loss
```

**核心改动**：encoder 从 `freeze + eval()` 改为 `LoRA + train()`。小头架构基本不变（`WhisperVAP` class）。

### 2.4 LoRA 配置

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],   # whisper attention 的 Q/V
    lora_dropout=0.1,
    bias="none",
)
# 只对 encoder 应用 LoRA
model = WhisperModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).encoder
model = get_peft_model(model, lora_cfg)
# 可训练参数 ~5-15M（< 2%），显存友好
```

### 2.5 训练超参（初始方案）

| 参数 | 值 | 理由 |
|---|---|---|
| LoRA r / α | 32 / 16 | 标准起步，encoder 不太大 |
| lr | 2e-4 (LoRA) + 1e-3 (head) | LoRA 层慢学，新头快学 |
| scheduler | cosine with warmup (5%) | 标准微调 |
| epochs | 10 | 全量数据 × 10 epoch ≈ 2-4h |
| batch_size | 32-64（按显存调） | 48GB 够用 |
| weight_decay | 0.01 | 防 LoRA 过拟合 |
| grad_clip | 1.0 | 与当前一致 |
| precision | bf16 | 4090 支持，稳定 |
| pos_weight | clip(max=10) | 与当前一致 |

### 2.6 CV 协议

- **主 CV**：5-fold 会话级 split（与当前 `train_head_cuda.py` 一致，防泄漏）
- **可信评估**：cap1 切片（每通 1 片段，模拟 test 独立 30s 切片）
- **阈值**：cycle1 固定阈值（C=0.05, NA=0.05, T=0.50, I=0.50, BC=0.50），**不在 OOF 上调阈值**（阈值铁律 3 验）

### 2.7 训练数据加载

关键区别：**不再从缓存读 npz，而是在线过 encoder**。

```python
def collate_batch(indices):
    # 1. 从原始 wav 读音频片段
    # 2. WhisperFeatureExtractor → log-mel
    # 3. encoder(mel) → 帧序列 (LoRA 参与，梯度回传)
    # 4. 帧序列 + context → 小头 → loss
```

**显存估算（单 batch）**：
- log-mel: batch × 2ch × 1500帧 × 128dim × 2B (bf16) ≈ 32 × 2 × 1500 × 128 × 2 = ~23MB
- encoder 前向激活（32层）：~2-4GB（bf16）
- LoRA 反向：~100MB
- 小头：~10MB
- **总计单 batch ≈ 3-5GB**，batch_size=64 也远不超 48GB

### 2.8 时间估算

| 步骤 | 时间 |
|---|---|
| 单样本 encoder 前向（4090, bf16） | ~10-20ms |
| 单 epoch 全量（~180K 窗 × 2声道） | ~60-120min |
| 10 epoch | ~10-20h |
| 5-fold CV × 10 epoch | ~50-100h ⚠️ 太长 |

**问题**：5-fold × 10 epoch × 全量数据 = 50-100h，不可接受。

**解决方案**：降训练规模——

| 方案 | 训练样本 | 单 fold 时间 | 5-fold 总时间 |
|---|---|---|---|
| A: 全量 + stride_mult=8（~180K窗） | 180K × 10ep | 10-20h | ❌ 50-100h |
| **B: cap1 切片训练（~369 窗）** | 369 × 100ep | ~5min | ✅ 25min |
| **C: 全量 + stride_mult=32（~45K窗）** | 45K × 10ep | 3-6h | ⚠️ 15-30h |
| **D: cap5 切片训练（~1845 窗）** | 1845 × 50ep | ~15min | ✅ 75min |

**推荐方案 D（cap5 切片 + LoRA）**：
- 每通 5 个独立片段 = 1845 训练样本
- 50 epoch × 1845 样本 = 足够 LoRA 收敛（LoRA 可训练参数少，数据需求低）
- 5-fold 总时间 ~1.5h
- 切片化训练 = 切片化评估，分布一致

如果 cap5 过拟合（LoRA 仍然 5-15M 参数 × 1845 样本），退守方案 B（cap1 + 更少 epoch + 更强正则）。

### 2.9 输出

与当前 `train_head_cuda.py` 一致：
- `pred_test1.csv`（默认 = cycle1 阈值）
- `cv_metrics.json`（cap1 macro + per-class F1）
- 可选多份阈值 CSV（cap1 / cycle1 / balanced）

## 3. 实施计划

### 3.1 新增/修改文件

| 文件 | 动作 | 说明 |
|---|---|---|
| `cloud/train_lora.py` | **新建** | LoRA 微调训练脚本（核心） |
| `cloud/requirements.txt` | 修改 | 加 `peft` 依赖 |
| `cloud/run_cloud.sh` | 修改 | 加 `lora` 子命令 |

### 3.2 `train_lora.py` 核心逻辑

```python
# 伪代码骨架
from peft import LoraConfig, get_peft_model
from transformers import WhisperModel

class WhisperVAPLoRA(nn.Module):
    def __init__(self, ctx_dim, lora_r=32, lora_alpha=16):
        # 1. 加载 whisper encoder + LoRA
        base = WhisperModel.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).encoder
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                              target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.1)
        self.encoder = get_peft_model(base, lora_cfg)

        # 2. 小头（与冻结版一致）
        self.proj = nn.Sequential(nn.Linear(1280, 192), nn.LayerNorm(192), nn.GELU())
        self.cross = nn.MultiheadAttention(192, 4, batch_first=True, dropout=0.1)
        self.q = nn.Parameter(torch.randn(1, 1, 192) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(ctx_dim + 2*192),
            nn.Linear(ctx_dim + 2*192, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 5))

    def forward(self, mel_ch0, mel_ch1, ctx):
        # 双声道分别过 LoRA encoder
        h0 = self.encoder(mel_ch0).last_hidden_state  # [B, T, 1280]
        h1 = self.encoder(mel_ch1).last_hidden_state
        # 降采样到 80 帧 + proj + cross-attn（同冻结版）
        ...
```

**关键：双声道要分别过 encoder**（不是共享一次前向），因为 LoRA 梯度需要正确回传到每个声道。

### 3.3 数据加载（在线，非缓存）

```python
class TurnTakingDataset(Dataset):
    """直接从 wav + labels 构造训练样本，在线过 whisper feature extractor。"""
    def __init__(self, conv_ids, split, slice_cap=5):
        # 预计算所有窗口端点（切片化）
        self.samples = []
        for cid in conv_ids:
            labels = np.load(f"data/train/labels/{cid}.npy")
            wav = load_wav(cid, split)
            # 切片化：每通取 slice_cap 个独立片段
            ends = pick_slice_ends(labels, cap=slice_cap)
            for e in ends:
                self.samples.append((cid, e, wav, labels))

    def __getitem__(self, idx):
        cid, end, wav, labels = self.samples[idx]
        # 取音频片段 + resample + log-mel
        ctx_feat = ctxfeat(labels[end-375:end])
        target = multi_hot(labels[end:end+25])
        mel_ch0 = extract_mel(wav[0, ...])  # [1, 128, T]
        mel_ch1 = extract_mel(wav[1, ...])
        return mel_ch0, mel_ch1, ctx_feat, target
```

### 3.4 训练循环

```python
# 双优化器：LoRA 慢学 + 头快学
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 2e-4, "weight_decay": 0.01},
    {"params": head_params, "lr": 1e-3, "weight_decay": 1e-4},
])
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup=5% steps, total_steps=...)

for epoch in range(epochs):
    for batch in loader:
        mel0, mel1, ctx, tgt = batch
        logits = model(mel0, mel1, ctx)
        loss = BCEWithLogitsLoss(pos_weight=pw)(logits, tgt)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

## 4. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| cap5 切片 1845 样本仍过拟合 | 中 | 降 cap1 (369) + 更强正则(dropout 0.5, lora_dropout 0.2, weight_decay 0.1) |
| LoRA encoder 表征仍然不对口 turn-taking | 低-中 | 已知冻结不行；LoRA 让 encoder 微调适应任务，理论上应更强。若仍不行=whisper 架构本身不合适 |
| 训练时间超预期 | 低 | cap5 + 50ep × 5fold ≈ 1.5h，可控 |
| 阈值问题（阈值铁律） | 低 | 默认用 cycle1 固定阈值，不调 |

## 5. 成功标准

| 指标 | 目标 | 当前基线 |
|---|---|---|
| cap1 切片 CV macro F1 | > 0.652（冻结版） | 0.6521（冻结 encoder + 小头） |
| cap1 BC F1 | > 0.25 | 0.200（冻结） / 0.222（纯 context LGBM） |
| 线上预测 | > 0.7124（SOTA） | 0.6709（冻结最高） |

**最低可行目标**：线上 > 0.7124（超越当前 SOTA）。如果 LoRA 微调后线上仍低于纯 context LGBM，说明 whisper encoder 架构本身不 turn-taking 对口，应放弃此路线。

## 6. 依赖

- `peft >= 0.10`（LoRA 实现）
- `transformers >= 5.5`（已有）
- `torch >= 2.7` + `torchaudio >= 2.7`（已有）
- 云端 `pip install peft` 即可

## 7. 执行步骤

```bash
# 1. 云端安装 peft
pip install peft

# 2. 冒烟验证（40 通，确认 LoRA 前向+反向正常）
python cloud/train_lora.py --convs 40 --epochs 3 --slice-cap 5 --run-dir tools/runs/climb/lora-smoke

# 3. 看冒烟日志确认 loss 下降 + 梯度正常 + BC 有区分度

# 4. 全量训练
python cloud/train_lora.py --convs 0 --epochs 50 --slice-cap 5 --run-dir tools/runs/climb/lora-full

# 5. 出 CSV → 提交 → 贴分
```
