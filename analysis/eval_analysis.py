"""
Comprehensive analysis of the finetuned mpnet model:
1. Per-language F2 breakdown (for radar chart)
2. Margin hyperparameter sweep (inference-time, no retraining)
3. Error case study (best/worst topics)

Uses the best mpnet checkpoint (F2=0.6696).
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
sys.path.insert(0, '/root/learning_equality')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from retrieval.model import Net
import glob

DATA_DIR = "./data/wikibooks"
SWITCH_DIR = "./data/switch"
CV_DIR = "./data/cv_experiment"
RESULTS_DIR = "./experiments/analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 48
BATCH_SIZE = 512
LANGS = ["en", "de", "fr", "es", "it", "pt"]
LANG_NAMES = {"en": "English", "de": "German", "fr": "French",
              "es": "Spanish", "it": "Italian", "pt": "Portuguese"}

# ── Load data ──
print("Loading data...")
df_topics = pd.read_csv(f"{SWITCH_DIR}/topics_0.csv").fillna({"title": "", "description": ""})
df_content = pd.read_csv(f"{SWITCH_DIR}/content_0.csv").fillna({"title": "", "description": "", "text": ""})
df_corr = pd.read_csv(f"{DATA_DIR}/correlations.csv")
split = pd.read_csv(f"{CV_DIR}/topic_split.csv")
val_topics = set(split[split["fold"] == 0]["topic_id"].values)

gt_dict = {}
for _, row in df_corr.iterrows():
    gt_dict[row["topic_id"]] = set(str(row["content_ids"]).split())

topic_text = {}
for _, row in df_topics.iterrows():
    topic_text[row["id"]] = f"{row.get('channel','')} # {row.get('title','')} # {row.get('description','')}"
topic_lang = dict(zip(df_topics["id"], df_topics["language_t"]))

content_ids_all = df_content["id"].tolist()
content_text = {}
for _, row in df_content.iterrows():
    text = " ".join(str(row.get("text", "")).split(" ")[:32])
    content_text[row["id"]] = f"{row.get('title','')} # {row.get('description','')} # {text}"
content_lang = dict(zip(df_content["id"], df_content["language_t"]))

# ── Load finetuned model ──
print("Loading finetuned mpnet model...")
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Find best checkpoint
ckpts = glob.glob("./experiments/cv_finetune/*/weights_e10_*.pth") + \
        glob.glob("./experiments/cv_finetune/*/weights_e9_*.pth")
best_ckpt = max(ckpts, key=lambda f: float(f.split("_")[-1].replace(".pth", "")))
print(f"Using checkpoint: {best_ckpt}")

net = Net(transformer_name=model_name, pooling="cls", gradient_checkpointing=False)
net.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
net = net.to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)


@torch.no_grad()
def embed(texts):
    feats = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embed", leave=False):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.cuda.amp.autocast():
            feat = net(enc["input_ids"], enc["attention_mask"])  # returns pooled
        feats.append(F.normalize(feat.float(), dim=-1).cpu())
    return torch.cat(feats, 0)


# Embed val topics and all content
val_topic_ids = [t for t in val_topics if t in gt_dict and t in topic_text]
print(f"Val topics: {len(val_topic_ids)}")

print("Embedding topics...")
topic_feats = embed([topic_text[t] for t in val_topic_ids])
print("Embedding content...")
content_feats = embed([content_text[c] for c in content_ids_all])

# Group content by language
content_by_lang = defaultdict(list)
content_idx_by_lang = defaultdict(list)
for i, cid in enumerate(content_ids_all):
    lang = content_lang.get(cid, "en")
    content_by_lang[lang].append(cid)
    content_idx_by_lang[lang].append(i)


def f2_score(gt, pred):
    gt, pred = set(gt), set(pred)
    p = len(gt & pred) / len(pred) if pred else 0
    r = len(gt & pred) / len(gt) if gt else 0
    if 4*p + r == 0: return 0, p, r
    return (5*p*r)/(4*p+r), p, r


def retrieve(tid, lang, margin=0.16, cap=50):
    """Retrieve content for a topic with given margin."""
    t_idx = val_topic_ids.index(tid)
    c_indices = content_idx_by_lang.get(lang, [])
    if not c_indices: return []
    c_feats = content_feats[c_indices]
    c_ids = content_by_lang[lang]
    sim = (topic_feats[t_idx].unsqueeze(0) @ c_feats.T).squeeze(0)
    max_sim = sim.max()
    thresh = max_sim - margin * max_sim
    mask = sim >= thresh
    selected = [c_ids[j] for j in mask.nonzero(as_tuple=True)[0].tolist()][:cap]
    if not selected:
        selected = [c_ids[sim.argmax()]]
    return selected


# ════════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-language F2 breakdown
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ANALYSIS 1: Per-language F2 breakdown")
print("="*60)

per_lang = {}
for lang in LANGS:
    lang_topics = [t for t in val_topic_ids if topic_lang.get(t) == lang]
    f2_list = []
    for tid in lang_topics:
        pred = retrieve(tid, lang)
        f2, _, _ = f2_score(gt_dict[tid], pred)
        f2_list.append(f2)
    if f2_list:
        per_lang[lang] = {"f2": np.mean(f2_list), "n": len(f2_list)}
        print(f"  {lang} ({LANG_NAMES[lang]:10s}): F2={np.mean(f2_list):.4f} (n={len(f2_list)})")

pd.DataFrame(per_lang).T.to_csv(f"{RESULTS_DIR}/per_language_f2.csv")
print(f"Saved per_language_f2.csv")


# ════════════════════════════════════════════════════════════════
# ANALYSIS 2: Margin hyperparameter sweep
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ANALYSIS 2: Margin hyperparameter sweep")
print("="*60)

margins = [0.05, 0.10, 0.16, 0.20, 0.25, 0.30]
margin_results = {}
# Sample 2000 topics for speed
sample_topics = val_topic_ids[:2000]

for margin in margins:
    f2_list, p_list, r_list = [], [], []
    for tid in sample_topics:
        lang = topic_lang.get(tid, "en")
        pred = retrieve(tid, lang, margin=margin)
        f2, p, r = f2_score(gt_dict[tid], pred)
        f2_list.append(f2); p_list.append(p); r_list.append(r)
    margin_results[margin] = {"f2": np.mean(f2_list), "precision": np.mean(p_list), "recall": np.mean(r_list)}
    print(f"  margin={margin:.2f}: F2={np.mean(f2_list):.4f}  P={np.mean(p_list):.4f}  R={np.mean(r_list):.4f}")

pd.DataFrame(margin_results).T.to_csv(f"{RESULTS_DIR}/margin_sweep.csv")
print(f"Saved margin_sweep.csv")


# ════════════════════════════════════════════════════════════════
# ANALYSIS 3: Error case study
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ANALYSIS 3: Error case study")
print("="*60)

all_cases = []
for tid in tqdm(val_topic_ids, desc="Scoring all", leave=False):
    lang = topic_lang.get(tid, "en")
    pred = retrieve(tid, lang)
    f2, p, r = f2_score(gt_dict[tid], pred)
    # Get top-1 predicted and first GT for inspection
    t_idx = val_topic_ids.index(tid)
    c_indices = content_idx_by_lang.get(lang, [])
    c_feats = content_feats[c_indices]
    c_ids = content_by_lang[lang]
    sim = (topic_feats[t_idx].unsqueeze(0) @ c_feats.T).squeeze(0)
    top1 = c_ids[sim.argmax()]
    all_cases.append({
        "topic_id": tid, "lang": lang, "f2": f2, "precision": p, "recall": r,
        "n_gt": len(gt_dict[tid]),
        "topic_title": topic_text[tid][:80],
        "top1_pred": content_text.get(top1, "")[:80],
        "gt_example": content_text.get(next(iter(gt_dict[tid])), "")[:80] if gt_dict[tid] else "",
    })

cases_df = pd.DataFrame(all_cases)

# Worst cases (F2=0, i.e., complete failures)
worst = cases_df[cases_df["f2"] == 0].sort_values("n_gt", ascending=False).head(15)
print(f"\nWorst cases (F2=0): {len(cases_df[cases_df['f2']==0])} topics ({100*len(cases_df[cases_df['f2']==0])/len(cases_df):.1f}%)")
print("\nSample worst cases (most GT, but F2=0):")
for _, r in worst.head(8).iterrows():
    print(f"  [{r['lang']}] F2={r['f2']:.2f} n_gt={r['n_gt']}")
    print(f"    Topic: {r['topic_title']}")
    print(f"    Top1:  {r['top1_pred']}")
    print(f"    GT:    {r['gt_example']}")

# Best cases
best = cases_df[cases_df["f2"] >= 0.9].head(5)
print(f"\nBest cases (F2>=0.9): {len(cases_df[cases_df['f2']>=0.9])} topics ({100*len(cases_df[cases_df['f2']>=0.9])/len(cases_df):.1f}%)")

# Error analysis by language
print("\nError rate by language (F2=0 fraction):")
for lang in LANGS:
    lang_cases = cases_df[cases_df["lang"] == lang]
    if len(lang_cases) > 0:
        err_rate = (lang_cases["f2"] == 0).mean()
        print(f"  {lang}: {err_rate*100:.1f}% complete failures (n={len(lang_cases)})")

cases_df.to_csv(f"{RESULTS_DIR}/error_cases.csv", index=False)
worst.to_csv(f"{RESULTS_DIR}/worst_cases.csv", index=False)
print(f"\nSaved error_cases.csv, worst_cases.csv")

# Save embeddings for t-SNE
np.save(f"{RESULTS_DIR}/topic_feats_ft.npy", topic_feats.numpy())
np.save(f"{RESULTS_DIR}/content_feats_ft.npy", content_feats.numpy())
print("Saved feature arrays for visualization")
print("\n✅ All analyses complete!")
