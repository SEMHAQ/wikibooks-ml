"""
Phase 4b: Embedding 空间语言结构分析
目的：分析模型是否把不同语言分开，以及微调的影响

实验：
  1. Language separation score: 同语言 content 的平均余弦相似度 vs 不同语言的
  2. 微调前后对比：zero-shot vs finetuned 的语言分离程度
  3. 产出论文用的 embedding 可视化数据

这是论文 "Analysis" 章节的核心：展示模型学到的表示结构。
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import glob

SWITCH_DIR = "./data/switch"
RESULTS_DIR = "./experiments/baseline_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64
BATCH_SIZE = 512
N_PER_LANG = 1000  # sample per language for analysis

LANGS = ["en", "de", "fr", "es", "it", "pt"]


def cls_pooling(out, mask):
    return out.last_hidden_state[:, 0, :]


@torch.no_grad()
def embed_texts(texts, tokenizer, model):
    feats = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embed", leave=False):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.cuda.amp.autocast():
            out = model(**enc)
        feat = cls_pooling(out, enc["attention_mask"]).float()
        feats.append(feat.cpu())
    return torch.cat(feats, 0)


# ── Load content ──
print("=" * 60)
print("Embedding Space Language Analysis")
print("=" * 60)

df_content = pd.read_csv(f"{SWITCH_DIR}/content_0.csv").fillna({"title": "", "description": "", "text": ""})

# Sample N per language
sampled = []
for lang in LANGS:
    lang_df = df_content[df_content["language_t"] == lang].head(N_PER_LANG)
    sampled.append(lang_df)
df_sample = pd.concat(sampled, ignore_index=True)
print(f"Sampled: {len(df_sample)} content ({N_PER_LANG} per language)")

texts = []
for _, row in df_sample.iterrows():
    text = " ".join(str(row.get("text", "")).split(" ")[:32])
    texts.append(f"{row.get('title','')} # {row.get('description','')} # {text}")

langs = df_sample["language_t"].tolist()


def analyze_separation(feats, langs, label):
    """Compute intra-lingual vs inter-lingual cosine similarity."""
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.T

    n = len(langs)
    intra_sims, inter_sims = [], []
    for i in range(n):
        for j in range(i+1, n):
            s = sim[i, j].item()
            if langs[i] == langs[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    intra_mean = np.mean(intra_sims)
    inter_mean = np.mean(inter_sims)
    separation = intra_mean - inter_mean

    print(f"\n{label}:")
    print(f"  Intra-lingual sim (same lang):  {intra_mean:.4f}")
    print(f"  Inter-lingual sim (diff lang):  {inter_mean:.4f}")
    print(f"  Separation gap:                 {separation:.4f}")
    print(f"  → {'Languages are SEPARATED' if separation > 0.1 else 'Languages are MIXED (good cross-lingual alignment)'}")

    # Per-language centroid distance
    centroids = {}
    for lang in LANGS:
        mask = [l == lang for l in langs]
        if any(mask):
            centroids[lang] = feats[mask].mean(0)

    # Centroid similarity matrix
    print(f"\n  Language centroid similarity matrix:")
    header = "       " + "  ".join(f"{l:>5}" for l in LANGS)
    print(f"  {header}")
    for l1 in LANGS:
        if l1 not in centroids: continue
        row = f"  {l1:>4}  "
        for l2 in LANGS:
            if l2 not in centroids: continue
            c1, c2 = centroids[l1], centroids[l2]
            s = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
            row += f" {s:5.2f}"
        print(row)

    return {"intra_sim": intra_mean, "inter_sim": inter_mean,
            "separation": separation, "label": label}


# ── 1. Zero-shot model ──
print("\n--- Zero-shot paraphrase-mpnet ---")
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
feats_zs = embed_texts(texts, tokenizer, model)
res_zs = analyze_separation(feats_zs, langs, "Zero-shot")

# Save feats for visualization
np.save(f"{RESULTS_DIR}/feats_zeroshot.npy", feats_zs.numpy())
np.save(f"{RESULTS_DIR}/feats_langs.npy", np.array(langs))

del model
torch.cuda.empty_cache()

# ── 2. Finetuned model ──
print("\n--- Finetuned paraphrase-mpnet ---")
# Find the best checkpoint
ckpt_dirs = sorted(glob.glob("./experiments/cv_finetune/*/"))
if not ckpt_dirs:
    # fallback to main checkpoints
    ckpt_dirs = sorted(glob.glob("./checkpoints/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/*/"))
best_ckpt = None
best_score = -1
for d in ckpt_dirs:
    for f in glob.glob(f"{d}weights_e*0.6*.pth"):
        score = float(f.split("_")[-1].replace(".pth", ""))
        if score > best_score:
            best_score = score
            best_ckpt = f

print(f"Best checkpoint: {best_ckpt} (F2={best_score:.4f})")

if best_ckpt:
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    # Load finetuned weights
    state = torch.load(best_ckpt, map_location=DEVICE)
    # The saved state has model.transformer.* prefix, load into model
    new_state = {}
    for k, v in state.items():
        if k.startswith("transformer."):
            new_state[k[len("transformer."):]] = v
    model.load_state_dict(new_state, strict=False)

    feats_ft = embed_texts(texts, tokenizer, model)
    res_ft = analyze_separation(feats_ft, langs, "Finetuned")
    np.save(f"{RESULTS_DIR}/feats_finetuned.npy", feats_ft.numpy())

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("COMPARISON: Zero-shot vs Finetuned")
    print("=" * 60)
    print(f"{'':25s} {'Zero-shot':>12s} {'Finetuned':>12s} {'Δ':>10s}")
    print("-" * 60)
    for key in ["intra_sim", "inter_sim", "separation"]:
        delta = res_ft[key] - res_zs[key]
        print(f"{key:25s} {res_zs[key]:>12.4f} {res_ft[key]:>12.4f} {delta:>+10.4f}")
    print(f"\nInterpretation:")
    print(f"  If separation DECREASES after finetuning → model learns better")
    print(f"  cross-lingual alignment (languages mix in embedding space)")

# Save summary
results_df = pd.DataFrame([res_zs, res_ft]) if best_ckpt else pd.DataFrame([res_zs])
results_df.to_csv(f"{RESULTS_DIR}/embedding_analysis.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/embedding_analysis.csv")
