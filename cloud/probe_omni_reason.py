"""Omni LLM 联合推理探针 (用法1: Omni 唯一可能赢的 = 多模态联合语义).

chain-first: Omni audio=whisper(已证伪), 唯一新东西=thinker LLM 联合理解音频+文本+历史.
用 apply_chat_template (Omni 原生用法) 喂 30s 音频+ASR文本+历史标签, prompt 判断未来2s事件.

先最小验证 (N 个样本, zero-shot): 看 Omni 判断与真值有无相关. 有信号才上全量.
backchannel 是毫秒时机, LLM 强语义弱时机 — 探针看是否值得.

Usage: python cloud/probe_omni_reason.py --n 20
"""
import argparse
import glob
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch

CTX, TGT, CHUNK_MS = 375, 25, 80
OMNI = os.environ.get("OMNI", "models/Qwen2.5-Omni-3B")
WIN_SEC = 30  # 喂全 30s 上下文 (LLM 用长上下文)
LAB = {0: "C(继续说)", 1: "T(话轮转换)", 2: "BC(backchannel附和)", 3: "I(插话打断)", 4: "NA(无活动)"}


def load_seg(cid, e, sr_out=16000):
    import torchaudio
    with wave.open(f"data/train/audio/{cid}.wav", "rb") as wf:
        sr = wf.getframerate(); raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    mono = arr.mean(0)
    end = int(e * CHUNK_MS / 1000 * sr)
    seg = mono[max(0, end - WIN_SEC * sr):end]
    return torchaudio.functional.resample(torch.from_numpy(seg), sr, sr_out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        OMNI, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"[reason] loaded Omni, probing {args.n} samples", file=sys.stderr)

    conv = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    # 取样本: 一半含BC 一半不含 (平衡看信号)
    rng = np.random.default_rng(42)
    samples = []
    for cid in conv[:60]:
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        for e in range(CTX, len(a) - TGT + 1, 200):
            fut = set(int(x) for x in a[e:e + TGT])
            samples.append((cid, e, 1 if 2 in fut else 0, fut))
    bc_s = [s for s in samples if s[2] == 1]
    nbc_s = [s for s in samples if s[2] == 0]
    rng.shuffle(bc_s); rng.shuffle(nbc_s)
    pick = bc_s[:args.n // 2] + nbc_s[:args.n // 2]
    rng.shuffle(pick)

    hits = {"bc_pred_when_bc": 0, "bc_pred_when_nbc": 0, "n_bc": 0, "n_nbc": 0}
    for cid, e, has_bc, fut in pick:
        audio = load_seg(cid, e)
        # 历史标签摘要 (最近)
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        recent = a[e - 50:e]
        hist = "".join({0: "C", 1: "T", 2: "B", 3: "I", 4: "N"}[int(x)] for x in recent[-25:])
        prompt = (
            f"这是一段电话对话的最后{WIN_SEC}秒音频(双声道已混合)。最近2秒的对话状态序列为: {hist} "
            f"(C=某方持续说话, T=话轮转换, B=backchannel附和如'嗯/对', I=插话打断, N=静音)。"
            f"请预测紧接着的未来2秒内, 听话方是否会发出 backchannel(附和声 如'嗯''对''是')? "
            f"只回答 是 或 否。"
        )
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a dialogue analysis assistant."}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = proc.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True).to("cuda")
        in_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, thinker_do_sample=False, thinker_max_new_tokens=10,
                                 return_audio=False)
        seq = out[0] if hasattr(out, "shape") else out
        gen = seq[in_len:]  # 只取新生成 token (去掉输入 prompt 回显)
        txt = proc.batch_decode([gen], skip_special_tokens=True)[0]
        pred_bc = ("是" in txt) and not txt.strip().startswith("否")
        if has_bc:
            hits["n_bc"] += 1; hits["bc_pred_when_bc"] += int(pred_bc)
        else:
            hits["n_nbc"] += 1; hits["bc_pred_when_nbc"] += int(pred_bc)
        print(f"  {cid}@{e} true_bc={has_bc} pred='{txt[-30:].strip()}' →{int(pred_bc)}", file=sys.stderr)

    # 信号: BC样本预测是的率 vs 非BC样本预测是的率 (差异大=有信号)
    r_bc = hits["bc_pred_when_bc"] / max(1, hits["n_bc"])
    r_nbc = hits["bc_pred_when_nbc"] / max(1, hits["n_nbc"])
    print(f"\n=== Omni zero-shot BC 推理信号 ===")
    print(f"  BC样本({hits['n_bc']})预测'会BC'率: {r_bc:.2f}")
    print(f"  非BC样本({hits['n_nbc']})预测'会BC'率: {r_nbc:.2f}")
    print(f"  差异(信号): {r_bc - r_nbc:+.2f}  (>0.2=有信号值得; ≈0=LLM抓不到BC时机)")
    print(json.dumps({"cycle": "Omni-reason-probe", "r_bc": round(r_bc, 3),
                      "r_nbc": round(r_nbc, 3), "signal": round(r_bc - r_nbc, 3)}))


if __name__ == "__main__":
    main()
