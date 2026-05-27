"""H-V2: VAP 公平对比 — 双声道 mel + cross-attn + context 手工特征融合.

关键问题：H-V1b 纯音频 BC=0.161 < 纯context 0.227，但对比不公平(不同输入)。
公平问题：音频+context 融合，BC 能否超纯 context 0.227？超→音频有增量信息，VAP值得放大(换Qwen2-Audio)；
不超→mel音频对BC无补充(或欠拟合)，需更强编码器或弃 VAP。

架构: [双声道mel→conv→cross-attn → 2d音频向量] + [context 80d手工特征] → concat → MLP → 5类
对照: --no_audio 消融(只context神经头) vs 默认(audio+context)

Usage: python tools/climb/cycle_vap_fusion.py --convs 40 --epochs 8 [--no_audio]
"""
from __future__ import annotations
import argparse, glob, wave, sys, time, json
from pathlib import Path
import numpy as np
import torch, torchaudio
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0,"tools/climb")
from cycle_context_v2 import featurize as ctxfeat

CTX,TGT,STRIDE,CHUNK_MS=375,25,5,80
SR16=16000; CTX_SEC=8
LAB={0:"C",1:"T",2:"BC",3:"I",4:"NA"}
SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)
_mel=torchaudio.transforms.MelSpectrogram(sample_rate=SR16,n_mels=40,n_fft=400,hop_length=160)

def read_mel(path,end_ms):
    with wave.open(path,'rb') as wf:
        sr=wf.getframerate(); n=wf.getnframes()
        end=int(end_ms/1000*sr); start=max(0,end-CTX_SEC*sr)
        wf.setpos(start); raw=wf.readframes(end-start)
    d=np.frombuffer(raw,dtype=np.int16)
    if len(d)<2: d=np.zeros(2,dtype=np.int16)
    d=d.reshape(-1,2).T.astype(np.float32)/32768.0
    wav=torch.tensor(d)
    if wav.shape[1]<CTX_SEC*sr: wav=torch.nn.functional.pad(wav,(CTX_SEC*sr-wav.shape[1],0))
    wav16=torchaudio.functional.resample(wav,sr,SR16)
    return torch.log(_mel(wav16).clamp(min=1e-5))

class FusionVAP(nn.Module):
    def __init__(self, ctx_dim=80, n_mels=40, d=128, heads=4, use_audio=True):
        super().__init__()
        self.use_audio=use_audio
        if use_audio:
            self.in_norm=nn.BatchNorm1d(n_mels)
            self.front=nn.Sequential(
                nn.Conv1d(n_mels,d,5,2,2),nn.BatchNorm1d(d),nn.GELU(),
                nn.Conv1d(d,d,5,2,2),nn.BatchNorm1d(d),nn.GELU(),
                nn.Conv1d(d,d,3,2,1),nn.BatchNorm1d(d),nn.GELU())
            self.cross=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.1)
            self.q=nn.Parameter(torch.randn(1,1,d)*0.02)
            fuse_in=ctx_dim+2*d
        else:
            fuse_in=ctx_dim
        self.ctx_norm=nn.BatchNorm1d(ctx_dim)
        self.head=nn.Sequential(nn.LayerNorm(fuse_in),nn.Linear(fuse_in,256),nn.GELU(),nn.Dropout(0.3),
                                nn.Linear(256,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,5))
    def enc(self,mel): return self.front(self.in_norm(mel)).transpose(1,2)
    def forward(self,ctx,mel2):
        c=self.ctx_norm(ctx)
        if self.use_audio:
            a=self.enc(mel2[:,0]); b=self.enc(mel2[:,1]); B=a.shape[0]; q=self.q.expand(B,-1,-1)
            ca,_=self.cross(q,b,b); cb,_=self.cross(q,a,a)
            feat=torch.cat([c,ca.squeeze(1),cb.squeeze(1)],-1)
        else:
            feat=c
        return self.head(feat)

def build(conv_ids,use_audio):
    Xc,Xm,Y,G=[],[],[],[]
    for gi,cid in enumerate(conv_ids):
        a=np.load(f"data/train/labels/{cid}.npy")
        wavpath=f"data/train/audio/{cid}.wav"
        for e in range(CTX,a.shape[0]-TGT+1,STRIDE*4):
            ctx=a[e-CTX:e].astype(int)
            Xc.append(ctxfeat(ctx))
            Xm.append(read_mel(wavpath,e*CHUNK_MS) if use_audio else torch.zeros(2,40,801))
            fut=set(int(x) for x in a[e:e+TGT])
            Y.append([1 if k in fut else 0 for k in range(5)]); G.append(gi)
    return np.array(Xc,dtype=np.float32), torch.stack(Xm), np.array(Y,dtype=np.float32), np.array(G)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--convs",type=int,default=40); ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--no_audio",action="store_true")
    args=ap.parse_args(); use_audio=not args.no_audio
    device="mps" if torch.backends.mps.is_available() else "cpu"
    ids=sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:args.convs]
    print(f"[vapf] {len(ids)}通 audio={use_audio} 提取...",file=sys.stderr)
    t0=time.time(); Xc,Xm,Y,G=build(ids,use_audio)
    print(f"[vapf] {len(Xc)}窗 ctx{Xc.shape[1]} mel{tuple(Xm.shape)} {time.time()-t0:.0f}s",file=sys.stderr)
    pw=torch.tensor([(len(Y)-Y[:,k].sum())/max(1,Y[:,k].sum()) for k in range(5)]).float().clamp(max=10).to(device)
    gkf=GroupKFold(3); oof=np.zeros((len(Xc),5))
    Xc_t=torch.tensor(Xc)
    for fold,(tr,va) in enumerate(gkf.split(Xc,Y[:,0],groups=G)):
        m=FusionVAP(ctx_dim=Xc.shape[1],use_audio=use_audio).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        crit=nn.BCEWithLogitsLoss(pos_weight=pw)
        ct=Xc_t[tr].to(device); mt=Xm[tr].to(device); yt=torch.tensor(Y[tr]).to(device)
        for ep in range(args.epochs):
            m.train(); perm=torch.randperm(len(tr))
            for i in range(0,len(tr),256):
                idx=perm[i:i+256]; opt.zero_grad()
                loss=crit(m(ct[idx],mt[idx]),yt[idx]); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad():
            oof[va]=torch.sigmoid(m(Xc_t[va].to(device),Xm[va].to(device))).cpu().numpy()
        print(f"[vapf] fold {fold+1}/3",file=sys.stderr)
    f1s={}
    for k in range(5):
        bf=-1; lo,hi=(0.05,0.25) if k==0 else (0.35,0.65)
        for t in np.linspace(lo,hi,13):
            f=f1_score(Y[:,k],(oof[:,k]>=t).astype(int),zero_division=0)
            if f>bf:bf=f
        f1s[k]=bf
    tag="ctx-only(消融)" if not use_audio else "ctx+VAP音频融合"
    print(f"[vapf] {tag}: macro={np.mean(list(f1s.values())):.4f} | "+" ".join(f"{LAB[k]}={f1s[k]:.3f}" for k in range(5)))
    print(f"[vapf] ★BC: {f1s[2]:.3f} (纯context基线 0.227)")

if __name__=="__main__":
    main()
