# AutoDL 上云操作单 — whisper-large-v3 攻 BC

> 你做的:注册/充值/选卡/报备 + 贴 SSH 或贴日志回来。其余(环境/代码/提取/训练)我全包。
> 锚:cycle1 线上 0.7108(排22)。可信 CV=cap1 切片 0.656。BC=0.22 是瓶颈。
> 目标:whisper 把 BC 拉到 ~0.40 → 线上 ~0.747 → 进前 3。

---

## 0. 投钱前先想清楚(可信判断已就绪)

- 切片 CV 审计(`docs/status/2026-05-28-sliced-cv-audit.md`)已确认 **BC 是真瓶颈非 CV 假象**——三档采样都钉 0.22。
- 成本极低:全量提取估 **¥70–180**(RTX 4090 @ ¥2/hr)。**这是模型赌注不是钱的赌注。**
- 建议:**先充 ¥50 跑冒烟**(前 40 通,几块钱),看 whisper 帧在 40 通上 BC 有没有抬头;有信号再全量。

---

## 1. 你做:注册 + 实名 + 充值

1. autodl.com 注册,**实名认证**(绑手机/支付宝,国内 GPU 平台强制)
2. 充值 ¥50 起步(支付宝/微信)

## 2. 你做:选卡开机(照此选)

| 项 | 选什么 | 理由 |
|---|---|---|
| **地区** | 任意有 4090 库存的(优先"西北/内蒙"便宜区) | — |
| **GPU** | **RTX 4090 (24GB) × 1** | frozen whisper 提取 fp16 ≤16GB,4090 够且最省;A100 不值 |
| **镜像** | **PyTorch 2.7.1 / CUDA 12.1 / Python 3.12**(选最接近的) | 对齐本机 torch 2.7.1,MPS↔CUDA 一处 device 切换 |
| **数据盘** | ≥ 50GB | whisper 缓存 + train audio |
| **计费** | 按量(冒烟阶段);确认后可包日 | 冒烟便宜 |

开机后:
- 控制台开 **"学术资源加速"**(内置 HF/github 代理,`from_pretrained` 能直连;我的脚本也有 curl 兜底)
- 复制 **SSH 命令**(形如 `ssh -p XXXXX root@region.autodl.com`)备用

## 3. 你做(择一搭桥,我摸不到云实例)

- **方式 A(更稳,推荐)**:我把脚本写好(已就绪),你 ssh 上去 `git clone` 本仓库(或上传 cloud/ 目录),`bash cloud/run_cloud.sh smoke`,把输出贴回来给我看。
- **方式 B**:你把 SSH 连接信息贴给我(注意这等于给我云访问权,你决定)。

## 4. 你做:复赛报备(若用 whisper-large-v3 = 公开模型)

- **2026-06-10 前** 邮件 `xinyebei@xinye.com` 报备使用 whisper-large-v3(openai 公开模型)。
- 报备发件邮箱用**你的比赛注册邮箱**(对外身份,不是系统里的 Claude 账号邮箱)。

---

## 5. 数据上传(我已写好打包脚本)

本机先打包(我可代跑):
```bash
bash cloud/pack_data.sh                  # → data/cloud_upload/{meta,test_audio,train_audio}.tar*
# 冒烟只需 meta + test_audio + 前 40 通 train audio(小,先验证链路)
```
上传到 AutoDL:控制台"文件存储"网盘上传,或 `scp -P <port> data/cloud_upload/*.tar* root@<host>:/root/autodl-tmp/`
云端解包到 repo 根的 `data/` 下。

---

## 6. 云端跑(脚本已就绪,断点续跑)

```bash
# repo 根(git clone 或上传 cloud/+tools/climb/cycle_context.py)
bash cloud/run_cloud.sh smoke   # 前 40 通 train + 全 test，验证速度/BC 信号
# 冒烟有信号 → 全量
bash cloud/run_cloud.sh full
```
产物:`tools/runs/climb/cloud-whisper-h001/pred_test1.csv` + `cv_metrics.json`(含 cap1 切片 CV + BC F1)
下载 CSV 回本机 → 你手动提交公榜 → 贴回真分 → 我注入 climb calibration。

**断点续跑**:spot 被抢/中断,重跑 `run_cloud.sh extract` 只补未完成的通(`data/whisper_cache/<split>/_done/` 记进度)。

---

## 7. 复赛镜像(我已备 Dockerfile)

`cloud/Dockerfile`(FROM pytorch/pytorch:2.7.1-cuda12.1)= 复赛提交镜像基底,单一真相。
- AutoDL 零售:以此为规格,`pip install -r cloud/requirements.txt` 后存私有镜像,`pip freeze` 对齐。
- 若要提交镜像 1:1 一致 → 改用 RunPod(可直接 docker build+push 此镜像)。

---

## 关键产物清单(我已写好,在 `cloud/`)

| 文件 | 作用 |
|---|---|
| `Dockerfile` / `requirements.txt` | 复赛镜像 + 依赖锚定(torch 2.7.1) |
| `download_whisper.sh` | whisper-large-v3 curl 直下(绕 hf client) |
| `extract_whisper_cuda.py` | CUDA fp16 帧提取,断点续跑,PID+artifact 双信号 |
| `train_head_cuda.py` | 神经头(cross-attn over 帧)+ cap1 切片 CV + 出 CSV |
| `pack_data.sh` | 本机数据打包上传 |
| `run_cloud.sh` | 一键 smoke/full/extract/head |
