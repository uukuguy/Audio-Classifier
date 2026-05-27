"""H-V1: 最小 VAP 验证 — 双声道 mel 帧序列 + cross-attention 小头，攻 BC.

吸取教训：保帧序列不 pool（前6 cycle 栽在 pool）；双声道 cross-attention（BC 主导线索=对方在干什么）。
先用便宜 mel 编码器验证架构；BC 动了再换 Qwen2-Audio。先小规模(--convs N)验证。

架构: 每声道 mel[40,T] → conv frontend → [T',D]
      → 双声道 cross-attention (ch A query ch B) → 末帧/attn-pool → MLP → 5类
损失: BCE + 温和 pos_weight。

Usage: python tools/climb/cycle_vap_mel.py --convs 40 --epochs 6
"""
from __future__ import annotations
import argparse, glob, wave, sys, time
from pathlib import Path
import numpy as np
import torch, torchaudio
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

CTX,TGT,STRIDE,CHUNK_MS=375,25,5,80
SR8,SR16=8000,16000
CTX_SEC=8           # 预测点前 8s 音频（research: BC ~5s 够）
LAB={0:"C",1:"T",2:"BC",3:"I",4:"NA"}
SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)

_mel=torchaudio.transforms.MelSpectrogram(sample_rate=SR16,n_mels=40,n_fft=400,hop_length=160)

def read_stereo_slice(path, end_ms):
    """读 end_ms 前 CTX_SEC 秒的双声道，上采样16k → log-mel [2,40,T]."""
    with wave.open(path,'rb') as wf:
        sr=wf.getframerate(); n=wf.getnframes()
        end=int(end_ms/1000*sr); start=max(0,end-CTX_SEC*sr)
        wf.setpos(start); raw=wf.readframes(end-start)
    d=np.frombuffer(raw,dtype=np.int16)
    if len(d)<2: d=np.zeros(2,dtype=np.int16)
    d=d.reshape(-1,2).T.astype(np.float32)/32768.0
    wav=torch.tensor(d)
    if wav.shape[1] < CTX_SEC*sr:
        wav=torch.nn.functional.pad(wav,(CTX_SEC*sr-wav.shape[1],0))
    wav16=torchaudio.functional.resample(wav,sr,SR16)
    m=torch.log(_mel(wav16).clamp(min=1e-5))
    return m  # [2,40,T~800]

class VAPHead(nn.Module):
    def __init__(self, n_mels=40, d=128, heads=4):
        super().__init__()
        # 输入归一化（log-mel 值域大 [-11.5, 9]，不归一化→nan。per-mel-bin BN）
        self.in_norm=nn.BatchNorm1d(n_mels)
        # 每声道共享 conv frontend: mel[40,T] → [T',d]
        self.front=nn.Sequential(
            nn.Conv1d(n_mels,d,5,2,2), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d,d,5,2,2), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d,d,3,2,1), nn.BatchNorm1d(d), nn.GELU())  # /8 下采样
        self.cross=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.1)
        self.q=nn.Parameter(torch.randn(1,1,d)*0.02)
        self.head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,5))
    def encode_ch(self,mel):  # mel[B,40,T]→[B,T',d]
        return self.front(self.in_norm(mel)).transpose(1,2)
    def forward(self,mel2):   # mel2[B,2,40,T]
        a=self.encode_ch(mel2[:,0]); b=self.encode_ch(mel2[:,1])
        B=a.shape[0]; q=self.q.expand(B,-1,-1)
        # ch A attend ch B 和 反向（对方在干什么）
        ca,_=self.cross(q,b,b); cb,_=self.cross(q,a,a)
        return self.head(torch.cat([ca.squeeze(1),cb.squeeze(1)],-1))

def build(conv_ids, device, cache):
    X,Y,G=[],[],[]
    for gi,cid in enumerate(conv_ids):
        a=np.load(f"data/train/labels/{cid}.npy")
        wavpath=f"data/train/audio/{cid}.wav"
        ends=[e for e in range(CTX,a.shape[0]-TGT+1,STRIDE*4)]  # stride×4 降密度(验证够用)
        for e in ends:
            end_ms=e*CHUNK_MS
            m=cache.get((cid,end_ms))
            if m is None:
                m=read_stereo_slice(wavpath,end_ms); cache[(cid,end_ms)]=m
            fut=set(int(x) for x in a[e:e+TGT])
            X.append(m); Y.append([1 if k in fut else 0 for k in range(5)]); G.append(gi)
    return torch.stack(X), np.array(Y,dtype=np.float32), np.array(G)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--convs",type=int,default=40)
    ap.add_argument("--epochs",type=int,default=6)
    args=ap.parse_args()
    device="mps" if torch.backends.mps.is_available() else "cpu"
    ids=sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:args.convs]
    print(f"[vap] {len(ids)} convs, device={device}, 提取 mel 帧序列...",file=sys.stderr)
    t0=time.time(); cache={}
    X,Y,G=build(ids,device,cache)
    print(f"[vap] {len(X)} 窗, mel shape {tuple(X.shape)}, 提取 {time.time()-t0:.0f}s",file=sys.stderr)
    pw=torch.tensor([(len(Y)-Y[:,k].sum())/max(1,Y[:,k].sum()) for k in range(5)]).float().clamp(max=10).to(device)
    gkf=GroupKFold(3); oof=np.zeros((len(X),5))
    for fold,(tr,va) in enumerate(gkf.split(X.numpy().reshape(len(X),-1),Y[:,0],groups=G)):
        model=VAPHead().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        crit=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xtr=X[tr].to(device); ytr=torch.tensor(Y[tr]).to(device)
        bs=256
        for ep in range(args.epochs):
            model.train(); perm=torch.randperm(len(tr))
            for i in range(0,len(tr),bs):
                idx=perm[i:i+bs]; opt.zero_grad()
                loss=crit(model(Xtr[idx]),ytr[idx]); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad():
            oof[va]=torch.sigmoid(model(X[va].to(device))).cpu().numpy()
        print(f"[vap] fold {fold+1}/3 done",file=sys.stderr)
    f1s={}
    for k in range(5):
        bt,bf=0.5,-1; lo,hi=(0.05,0.25) if k==0 else (0.35,0.65)
        for t in np.linspace(lo,hi,13):
            f=f1_score(Y[:,k],(oof[:,k]>=t).astype(int),zero_division=0)
            if f>bf:bf,bt=f,t
        f1s[k]=bf
    print(f"[vap] 双声道mel+cross-attn: macro={np.mean(list(f1s.values())):.4f} | "+" ".join(f"{LAB[k]}={f1s[k]:.3f}" for k in range(5)))
    print(f"[vap] ★BC对照: context-only=0.227 → VAP-mel={f1s[2]:.3f}")

if __name__=="__main__":
    main()
