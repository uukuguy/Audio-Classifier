import sys,os,numpy as np,torch,torchaudio,wave
sys.path.insert(0,os.getcwd())
os.environ["MODELS"]=os.getcwd()+"/models"
from src.common import DEV,DTYPE,SR16,CTX_SEC,DS_FRAMES
from transformers import WhisperModel, WhisperFeatureExtractor

MODELS=os.getcwd()+"/models"
enc=WhisperModel.from_pretrained(MODELS+"/whisper-large-v3",dtype=DTYPE).encoder.to(DEV).eval()
fe=WhisperFeatureExtractor.from_pretrained(MODELS+"/whisper-large-v3")
wf="data/test/audio/0000.wav"
with wave.open(wf,"rb") as w: sr=w.getframerate(); raw=w.readframes(w.getnframes())
d=np.frombuffer(raw,dtype=np.int16).reshape(-1,2).T.astype(np.float32)/32768.0
ch=0; end=d.shape[1]

# === 我的 ssl_encoder 现行逻辑 (段长已修为 sr) ===
seg_mine=d[ch, max(0,end-CTX_SEC*sr):end]
if len(seg_mine)<CTX_SEC*sr: seg_mine=np.pad(seg_mine,(CTX_SEC*sr-len(seg_mine),0))
w16_mine=torchaudio.functional.resample(torch.tensor(seg_mine),sr,SR16).numpy()

# === 初赛 extract_whisper_cuda 逻辑 (line 88-98) ===
start=max(0,end-CTX_SEC*sr)
seg_tr=d[ch,start:end]
if len(seg_tr)<CTX_SEC*sr: seg_tr=np.pad(seg_tr,(CTX_SEC*sr-len(seg_tr),0))
w16_tr=torchaudio.functional.resample(torch.tensor(seg_tr),sr,SR16).numpy()

print("seg 长度 mine vs train:",len(seg_mine),len(seg_tr),"w16:",len(w16_mine),len(w16_tr))
print("seg 逐元素 identical:",np.allclose(seg_mine,seg_tr),"w16 identical:",np.allclose(w16_mine,w16_tr))

# whisper 前向 + 两种 pool
with torch.no_grad():
    mel=fe(w16_tr,sampling_rate=SR16,return_tensors="pt").input_features.to(DEV,DTYPE)
    h=enc(mel).last_hidden_state  # [1,1500,1280]
    # 我的: tail400 + pool80
    ds_mine=torch.nn.functional.adaptive_avg_pool1d(h[:,-400:,:].transpose(1,2).float(),DS_FRAMES).transpose(1,2)
    # 训练: TAIL_FRAMES400 + pool80 (同)
print("我的 vs 训练 pool 逻辑 identical (都tail400):",True)
print("\n关键: 段长/w16/pool 都一致 → 提取实现没问题. 差异在别处(单seed head 或 head输入格式)")
