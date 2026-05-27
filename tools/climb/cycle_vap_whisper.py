"""H-V3: VAP whisper-small — 双声道 whisper-small 帧特征 + cross-attn + context 融合，攻 BC.

mel(H-V2)对BC无增量(0.198<0.227)。换 whisper-small(12层768维,SSL表征强于mel)。
本机实测 283ms/窗可行。large-v3 太慢(45h)，small 折中:比mel强、本机能跑。
有信号→上云跑 large-v3；无信号→whisper系对本机任务可能整体不够。

架构: 每声道 whisper-small encoder 帧序列[T,768] (冻结) → conv降采样 → cross-attn
      → 音频2向量 + context 80d → MLP → 5类
关键: whisper 帧特征预提取缓存(避免训练时重算)。

Usage: python tools/climb/cycle_vap_whisper.py --convs 20 --stride_mult 8 --epochs 10 [--no_audio]
"""
from __future__ import annotations
import argparse, glob, wave, sys, time
from pathlib import Path
import numpy as np
import torch, torchaudio
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0,"tools/climb")
from cycle_context_v2 import featurize as ctxfeat
from transformers import WhisperModel, WhisperFeatureExtractor

CTX,TGT,STRIDE,CHUNK_MS=375,25,5,80
SR16=16000; CTX_SEC=8
LAB={0:"C",1:"T",2:"BC",3:"I",4:"NA"}
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
WD="/Users/sujiangwen/.cache/manual_models/whisper-small"
DEV="mps" if torch.backends.mps.is_available() else "cpu"

_fe=WhisperFeatureExtractor.from_pretrained(WD)
_enc=None
def get_enc():
    global _enc
    if _enc is None:
        _enc=WhisperModel.from_pretrained(WD,dtype=torch.float32).encoder.to(DEV).eval()
    return _enc

@torch.no_grad()
def whisper_frames(wav16_ch_list):
    """list of 1D np arrays(16k) → [N, T_ds, 768] 帧序列(降采样到~150帧省内存)."""
    enc=get_enc()
    feats=_fe(wav16_ch_list,sampling_rate=SR16,return_tensors="pt")
    h=enc(feats.input_features.to(DEV,torch.float32)).last_hidden_state  # [N,1500,768]
    # 1500帧(30s)→只取末8s对应(末400帧) + 降采样到80帧省内存/算力
    tail=h[:,-400:,:]
    ds=torch.nn.functional.adaptive_avg_pool1d(tail.transpose(1,2),80).transpose(1,2)
    return ds.cpu().numpy()  # [N,80,768]

def build(conv_ids, use_audio, stride_mult):
    Xc,Xa,Y,G=[],[],[],[]
    for gi,cid in enumerate(conv_ids):
        a=np.load(f"data/train/labels/{cid}.npy")
        ends=list(range(CTX,a.shape[0]-TGT+1,STRIDE*stride_mult))
        # 批量提取该 conv 所有窗的 whisper 帧（双声道）
        if use_audio:
            with wave.open(f"data/train/audio/{cid}.wav",'rb') as wf:
                sr=wf.getframerate(); n=wf.getnframes(); full=wf.readframes(n)
            d=np.frombuffer(full,dtype=np.int16).reshape(-1,2).T.astype(np.float32)/32768.0
            ch_segs={0:[],1:[]}
            for e in ends:
                end=int(e*CHUNK_MS/1000*sr); start=max(0,end-CTX_SEC*sr)
                for ch in range(2):
                    seg=d[ch,start:end]
                    if len(seg)<CTX_SEC*sr: seg=np.pad(seg,(CTX_SEC*sr-len(seg),0))
                    w16=torchaudio.functional.resample(torch.tensor(seg),sr,SR16).numpy()
                    ch_segs[ch].append(w16)
            # batch 提取每声道
            fr0=whisper_frames(ch_segs[0]); fr1=whisper_frames(ch_segs[1])
        for i,e in enumerate(ends):
            Xc.append(ctxfeat(a[e-CTX:e].astype(int)))
            if use_audio:
                Xa.append(np.stack([fr0[i],fr1[i]]))  # [2,80,768]
            fut=set(int(x) for x in a[e:e+TGT])
            Y.append([1 if k in fut else 0 for k in range(5)]); G.append(gi)
        print(f"  conv {gi+1}/{len(conv_ids)} done",file=sys.stderr)
    Xa=np.array(Xa,dtype=np.float32) if use_audio else np.zeros((len(Xc),2,80,768),dtype=np.float32)
    return np.array(Xc,dtype=np.float32), Xa, np.array(Y,dtype=np.float32), np.array(G)

class WhisperVAP(nn.Module):
    def __init__(self, ctx_dim=80, wd=768, d=128, use_audio=True):
        super().__init__()
        self.use_audio=use_audio
        if use_audio:
            self.proj=nn.Sequential(nn.Linear(wd,d),nn.LayerNorm(d),nn.GELU())
            self.cross=nn.MultiheadAttention(d,4,batch_first=True,dropout=0.1)
            self.q=nn.Parameter(torch.randn(1,1,d)*0.02)
            fin=ctx_dim+2*d
        else: fin=ctx_dim
        self.cn=nn.BatchNorm1d(ctx_dim)
        self.head=nn.Sequential(nn.LayerNorm(fin),nn.Linear(fin,256),nn.GELU(),nn.Dropout(0.3),
                                nn.Linear(256,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,5))
    def forward(self,ctx,aud):
        c=self.cn(ctx)
        if self.use_audio:
            a=self.proj(aud[:,0]); b=self.proj(aud[:,1]); B=a.shape[0]; q=self.q.expand(B,-1,-1)
            ca,_=self.cross(q,b,b); cb,_=self.cross(q,a,a)
            feat=torch.cat([c,ca.squeeze(1),cb.squeeze(1)],-1)
        else: feat=c
        return self.head(feat)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--convs",type=int,default=20); ap.add_argument("--stride_mult",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--no_audio",action="store_true")
    args=ap.parse_args(); use_audio=not args.no_audio
    ids=sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:args.convs]
    print(f"[wvap] {len(ids)}通 audio={use_audio} 提取whisper帧...",file=sys.stderr)
    t0=time.time(); Xc,Xa,Y,G=build(ids,use_audio,args.stride_mult)
    print(f"[wvap] {len(Xc)}窗 ctx{Xc.shape[1]} aud{Xa.shape} {time.time()-t0:.0f}s",file=sys.stderr)
    pw=torch.tensor([(len(Y)-Y[:,k].sum())/max(1,Y[:,k].sum()) for k in range(5)]).float().clamp(max=10).to(DEV)
    gkf=GroupKFold(3); oof=np.zeros((len(Xc),5)); Xc_t=torch.tensor(Xc); Xa_t=torch.tensor(Xa)
    for fold,(tr,va) in enumerate(gkf.split(Xc,Y[:,0],groups=G)):
        m=WhisperVAP(ctx_dim=Xc.shape[1],use_audio=use_audio).to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); crit=nn.BCEWithLogitsLoss(pos_weight=pw)
        ct=Xc_t[tr].to(DEV); at=Xa_t[tr].to(DEV); yt=torch.tensor(Y[tr]).to(DEV)
        for ep in range(args.epochs):
            m.train(); perm=torch.randperm(len(tr))
            for i in range(0,len(tr),256):
                idx=perm[i:i+256]; opt.zero_grad()
                loss=crit(m(ct[idx],at[idx]),yt[idx]); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): oof[va]=torch.sigmoid(m(Xc_t[va].to(DEV),Xa_t[va].to(DEV))).cpu().numpy()
        print(f"[wvap] fold {fold+1}/3",file=sys.stderr)
    f1s={}
    for k in range(5):
        bf=-1; lo,hi=(0.05,0.25) if k==0 else (0.35,0.65)
        for t in np.linspace(lo,hi,13):
            f=f1_score(Y[:,k],(oof[:,k]>=t).astype(int),zero_division=0)
            if f>bf:bf=f
        f1s[k]=bf
    tag="ctx-only(消融)" if not use_audio else "ctx+whisper-small音频"
    print(f"[wvap] {tag}: macro={np.mean(list(f1s.values())):.4f} | "+" ".join(f"{LAB[k]}={f1s[k]:.3f}" for k in range(5)))
    print(f"[wvap] ★BC: {f1s[2]:.3f} (纯ctx基线0.227, mel融合0.198)")

if __name__=="__main__":
    main()
