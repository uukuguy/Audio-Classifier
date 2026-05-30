# Next-Session Handoff

**Updated:** 2026-05-30 10:50
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **SOTA 仍是变体 F = 0.7124**（无新提交落地）。前10 真门槛 **0.7285**（非旧记的 0.7192）。
2. **VAP/CPC 音频路线整条证伪**（DECISIONS D-1）：全量真分 0.6337，信号探针证 VAP head 对 BC |r|<0.04。**不再试任何音频 encoder 攻 BC**（含 HuBERT/MMS，同盲区）。
3. **战略转攻 T/I**（DECISIONS D-2）：文本词汇特征干净对照确认 **T+0.038 / I+0.048**，C/NA 不动，OOF macro +0.0217。**真增益**。
4. **卡点 = T/I 增益转化为提交时遇位置偏置**：train 整通采样(T率0.253) vs test 切片末(T率0.325)→test T 暴涨 709(SOTA 504)、i/bc 崩。5seed 集成压不动=系统性非方差。
5. **下一步明确 = 修对齐**：train 改切片末采样匹配 test 分布（sliced_cv 风格），很可能破局让 T/I 可安全提交。

## 下一步（第一动作，已诊断到根因）

**核心修复**：让 train 也只在"30s 切片末"位置采样（模拟 test），消除位置偏置。

- 当前 `tools/climb/cycle_ti_text.py` 的 `build()` 用 `range(CTX, len-TGT, stride=40)` 整通各处采样 → 改成 sliced_cv 风格：每通切互不重叠 30s 切片，预测点取每切片相对末尾。
- 参考 `tools/climb/sliced_cv.py` 的 `build_slices_all()`（已有切片化采样逻辑）。
- 验证：重跑 `--submit`，看 test 分布是否回到 t≈504/i≈65（接近 SOTA 锚点）+ cap1 CV 是否仍 >SOTA 0.6402。
- 分布对齐后即可提交（这是 ti-text-fusion paradigm first calibration，值得用 1 次配额拿真分）。

**已验证的位置偏置证据**（下一步的依据）：
```
T正例率: 整通各处=0.253 vs 切片首=0.325  (test 全在切片位置, T 天然高 28%)
```

## 本 session 交付（commit c745696）

- `cloud/probe_vap_signal.py` — VAP 信号探针（纯前向，证 VAP head 对 BC 无信息）
- `tools/climb/cycle_bc_temporal.py` — 增强 context 时序特征攻 BC（零增益，证 context 已榨干）
- `tools/climb/cycle_ti_text.py` — **T/I 文本特征实验（核心，下一步改这个的采样）**
- `tests/main/diag_bc_context.py` — BC precision 诊断（R0.83/P0.05）
- `tests/main/diag_allclass_ceiling.py` — 全战场天花板诊断（导出转攻 T/I）
- `tests/main/analyze_vap_lgbm_complement.py` — VAP-BC vs LGBM-BC 互补性
- `docs/status/DECISIONS.md` — D-1 VAP证伪 / D-2 转T/I
- climb 状态全同步（runs.csv 加 vap-full + 真分校准 / calibration vap-cpc-stereo gap=−0.0066）

## Don't go down these again（本轮新增证伪）

- **VAP/CPC 全量微调攻 BC** — 真分 0.6337，BC=0.222 打平 LGBM，head 对 BC |r|<0.04
- **任何纯音频 encoder 攻 BC**（HuBERT/MMS/mel/whisper）— 音频归纳偏置抓不住 backchannel 时机
- **增强 context 时序特征攻 BC** — baseline 46维已榨干，F1 顶 0.21=信息上限
- **cap1/滑窗 CV 调激进阈值** — 阈值铁律再验，T/I 提交用变体F 固定阈值（C.05/T.5/BC.75/I.65/NA.25）

## 之前已证伪（保留）

whisper LoRA/冻结 / LGBM+任何特征撞0.71墙 / Qwen3 mean-pool / 切片阈值变体B砸NA / rank集成C BC崩 / 滑窗CV估线上系统性误导

## 云主机（用户：24h 全程开着，不用是浪费）

- AutoDL 4090D 48GB，**全程开机**。`ssh -p 46379 root@connect.westd.seetacloud.com`，PATH=/root/miniconda3/bin
- VAP 仓库在云 `cloud/VAP`（用 `export VAP_ROOT=/root/audio-classifier/cloud/VAP`），数据在 `/root/audio-classifier/data/`
- 全量可用 8 卡 A100（用户）。本机限线程 OMP/MKL/VECLIB/OPENBLAS=4
- 0 残留进程（vap-full PID5193 已结束）

## 关键数字锚点

| 类 | 当前OOF F1 | test正例(SOTA) | 备注 |
|---|---|---|---|
| C | 0.971 | 974 | 饱和，阈值0.05 |
| NA | 0.797 | 949 | 较饱和，阈值0.25 |
| T | 0.542→**0.586**(text) | 504 | ★文本可提，未饱和 |
| I | 0.434→**0.488**(text) | 65 | ★文本可提，未饱和 |
| BC | 0.212 | 30 | 信息上限，暂搁置 |

## Ready-to-paste（下一步）

```bash
cd /Users/sujiangwen/sandbox/competitions-2026/Audio-Classifier
# 改 cycle_ti_text.py build() 用切片末采样后:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 tools/climb/cycle_ti_text.py --submit --folds 5 --run-dir tools/runs/climb/ti-text-sliced
# 看 test 分布是否回到 t≈504/i≈65
```
