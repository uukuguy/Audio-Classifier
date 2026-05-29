# Next-Session Handoff

**Updated:** 2026-05-30 02:35
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **SOTA 仍是变体 F = 0.7124**。
2. **范式转向完成**：whisper 弃（ASR族+193ms/前向, 全量30-63h）→ **VAP/CPC**（turn-taking 学术 SOTA）。
3. **VAP 适配脚本 `cloud/train_vap.py` 就绪，本机彻底排查通过**（commit fa5fbf3）。抓修 4 个上云必崩 bug + 确认 nan 是 MPS 平台问题（云端 CUDA 无）。CPU 端到端验证全类有值。
4. **下一步 = 上云三步走**：速度实测 → 小验证 → 全量。云主机当前关机，开机即可执行。
5. **清云端残留**（换模型后）：whisper 路线遗留 `tools/runs/climb/lora-*`、`/tmp/lora-*.log`、`~/.cache/manual_models/whisper-large-v3`(~3GB,VAP不用)。开机先 `pkill -9 -f train_lora; pkill -9 -f train_vap` 确保无残留进程。

## VAP 方案就绪状态

**已本机验证（上云前）**：
- VAP 加载 missing=0/unexpected=0（`baselines/VAP/example/checkpoints/VAP_state_dict.pt`, 23MB, 含CPC）
- `out["x"]` = [B,T,256] 双声道 cross-attn 融合帧序列 @ 50Hz, causal
- 完整流程 dry-run 通过：load→VAP前向→ctx融合头→loss→反向→CV→test预测→CSV
- CSV 格式正确（6列 segment_id,c,na,i,bc,t / 1000段 / 0-1）
- pos_weight labels-only（无 train_lora 的 1.5h bug）；weights_only=True（security）

**架构**（`cloud/train_vap.py`）：
```
过去win_sec双声道8k→重采样16k → VAP → out[x][B,T,256]末pool帧
  + ctx特征(46维,cycle1,C/NA强信号) → 5类头 → sigmoid
```
- CPC 默认冻结（`--unfreeze` 微调，文献 BC+0.3 靠微调）
- 因果：CPC causal + AliBi causal mask + 取末窗帧

## 上云三步走（开机后执行，脚本已本地彻底验证 cloud-ready）

**第0步 清理 + 推代码**：
```bash
ssh -p <PORT> root@<HOST> 'pkill -9 -f train_lora; pkill -9 -f train_vap; nvidia-smi --query-gpu=memory.used --format=csv,noheader'  # 确认0残留
# (可选清whisper残留省盘) ssh ... 'rm -rf tools/runs/climb/lora-* /tmp/lora-*.log'
rsync -avz -e "ssh -p <PORT>" cloud/train_vap.py baselines/VAP root@<HOST>:/root/audio-classifier/  # VAP仓库23MB含权重,首次必传
ssh -p <PORT> root@<HOST> 'cd /root/audio-classifier && export PATH=/root/miniconda3/bin:$PATH && pip install einops omegaconf lightning 2>&1 | tail -1'
```

**第1步 速度实测**（关键！不再凭估算，folds=1 已修可正常训）：
```bash
ssh ... 'cd /root/audio-classifier && export PATH=/root/miniconda3/bin:$PATH && python cloud/train_vap.py --convs 20 --slice-cap 5 --epochs 3 --folds 1 --win-sec 10 --batch-size 32 --run-dir tools/runs/climb/vap-speed'
```
看 fold 完成时间 → 算 ms/段。CUDA 上 loss 应正常(MPS 才 nan)。据此定全量规模(不会再 63h)

**第2步 小验证**（速度OK后，验证 BC 信号）：
```bash
python cloud/train_vap.py --convs 40 --slice-cap 5 --epochs 10 --folds 5 --win-sec 10 --run-dir tools/runs/climb/vap-smoke
# 看 cap1 BC F1 vs LGBM 0.222 / whisper-frozen 0.20。BC升=VAP对口确认
```

**第3步 全量**（小验证有信号）：
```bash
python cloud/train_vap.py --convs 0 --slice-cap 20 --epochs 10 --folds 5 --win-sec 10 --unfreeze --run-dir tools/runs/climb/vap-full
# 8卡A100可用(用户)。出CSV→拉回→提交→贴分
```

## 关键依赖/资产
- VAP 仓库在 `baselines/VAP/`（gitignored，需 rsync 上云）含预训练权重
- 云端依赖：`pip install einops omegaconf lightning`（torchaudio 已装）
- VAP 加载会自动从 fbaipublicfiles 下 CPC 权重（被 VAP_state_dict 覆盖，无所谓）

## Don't go down these again（已证伪）
- whisper-large-v3 LoRA/冻结 → 30-63h + ASR族不对口（0.671/0.6155）
- LGBM+任何特征 → 撞 0.71 墙
- 文本词汇喂LGBM → 线上假正例净负（只T版 0.7013）
- 激进阈值/rank集成/切片阈值 → 全负；滑窗CV估线上 → 系统性误导

## 云主机
- AutoDL 4090D 48GB，**已关机**。SSH 端口/host 开机查控制台（之前 `-p 46379 root@connect.westd.seetacloud.com`）
- 全量可用 8 卡 A100（用户）
- 数据已在云 `/root/audio-classifier/data/`（369训练+1000test）
- **VAP 仓库需 rsync 上云**（之前没传过）
- 限线程铁律(本机)：OMP/MKL/VECLIB/OPENBLAS_NUM_THREADS=4

## Currently Running

| 任务 | PID | 启动 | 预计 | 查 |
|---|---|---|---|---|
| **全量 VAP unfreeze**(369通/cap20/10ep/5fold/win10/微调) | 云 5193 | 03:12 | ~04:40 | `ssh -p 46379 root@connect.westd.seetacloud.com 'tail -15 /tmp/vap-full.log'` |

完成后：拉 CSV 回本机 → 用户提交 → 贴分。
`scp -P 46379 root@connect.westd.seetacloud.com:/root/audio-classifier/tools/runs/climb/vap-full/pred_test1.csv tools/runs/climb/vap-full/`

## VAP 上云已验证（真实数据）
- CUDA 加载/无nan/速度<6ms段(快whisper 32x)/VRAM 0.2GB
- 冻结 40通 macro 0.5338 BC=0; **微调 40通 macro 0.5777 BC=0.077 T 0.43→0.57**(微调对口)
- nan 确认是 MPS 平台问题，CUDA 干净
- VAP 仓库在云端 `cloud/VAP`，用 `export VAP_ROOT=/root/audio-classifier/cloud/VAP`
