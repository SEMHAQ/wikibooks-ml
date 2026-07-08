"""
Phase 4: Cross-lingual Retrieval Analysis
目的：测试模型是否真正学到跨语言语义对齐

实验设计：
  1. 同语言检索（intra-lingual）：英文 topic → 英文 content（baseline）
  2. 跨语言检索（cross-lingual）：英文 topic → 其他语言 content

  如果跨语言 F2 显著低于同语言 → 模型主要靠同语言匹配，没有真正跨语言对齐
  如果跨语言 F2 接近同语言 → 模型学到了语言无关的语义表示 🎯

用 zero-shot 预训练模型评估（不需要微调模型）。
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
import time

DATA_DIR = "./data/wikibooks"
SWITCH_DIR = "./data/switch"
CV_DIR = "./data/cv_experiment"
RESULTS_DIR = "./experiments/baseline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64
BATCH_SIZE = 512
MARGIN = 0.16
N_SAMPLE_TOPICS = 2000  # Sample for speed

LANGS = ["en", "de", "fr", "es", "it", "pt"]
LANG_NAMES = {"en": "English", "de": "German", "fr": "French",
              "es": "Spanish", "it": "Italian", "pt": "Portuguese"}


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
        feat = cls_pooling(out, enc["attention_mask"])
        feat = F.normalize(feat.float(), dim=-1)
        feats.append(feat.cpu())
    return torch.cat(feats, 0)


def f2_score(gt, pred):
    gt, pred = set(gt), set(pred)
    p = len(gt & pred) / len(pred) if pred else 0
    r = len(gt & pred) / len(gt) if gt else 0
    if 4*p + r == 0: return 0, p, r
    return (5*p*r)/(4*p+r), p, r


# ── Load data ──
print("=" * 70)
print("Cross-lingual Retrieval Analysis")
print("=" * 70)

df_topics = pd.read_csv(f"{SWITCH_DIR}/topics_0.csv").fillna({"title": "", "description": ""})
df_content = pd.read_csv(f"{SWITCH_DIR}/content_0.csv").fillna({"title": "", "description": "", "text": ""})
df_corr = pd.read_csv(f"{DATA_DIR}/correlations.csv")
split = pd.read_csv(f"{CV_DIR}/topic_split.csv")
val_topics = set(split[split["fold"] == 0]["topic_id"].values)

gt_dict = {}
for _, row in df_corr.iterrows():
    gt_dict[row["topic_id"]] = set(str(row["content_ids"]).split())

topic_text, topic_lang = {}, {}
for _, row in df_topics.iterrows():
    topic_text[row["id"]] = f"{row.get('channel','')} # {row.get('title','')} # {row.get('description','')}"
    topic_lang[row["id"]] = row["language_t"]

content_text, content_lang = {}, {}
for _, row in df_content.iterrows():
    text = " ".join(str(row.get("text", "")).split(" ")[:32])
    content_text[row["id"]] = f"{row.get('title','')} # {row.get('description','')} # {text}"
    content_lang[row["id"]] = row["language_t"]

# ── Embed with paraphrase-mpnet (zero-shot) ──
print("\nLoading model...")
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

# Sample val topics
val_with_gt = [t for t in val_topics if t in gt_dict and t in topic_text][:N_SAMPLE_TOPICS]
print(f"Eval topics: {len(val_with_gt)}")

# Embed all content (for cross-lingual search across ALL languages)
all_content_ids = df_content["id"].tolist()
print(f"Embedding {len(all_content_ids)} content items (all languages)...")
content_feats = embed_texts([content_text[c] for c in all_content_ids], tokenizer, model)

# Group content by language
content_by_lang = defaultdict(list)
content_idx_by_lang = defaultdict(list)
for i, cid in enumerate(all_content_ids):
    lang = content_lang.get(cid, "en")
    content_by_lang[lang].append(cid)
    content_idx_by_lang[lang].append(i)

# ── Cross-lingual experiment matrix ──
print("\n" + "=" * 70)
print("Cross-lingual Retrieval Matrix (F2 Score)")
print("Rows = Topic language, Cols = Content language pool searched")
print("=" * 70)

# Embed sampled topics
topic_feats_sample = embed_texts([topic_text[t] for t in val_with_gt], tokenizer, model)

# For each topic language, evaluate retrieval against each content language
results_matrix = np.zeros((len(LANGS), len(LANGS)))
counts_matrix = np.zeros((len(LANGS), len(LANGS)), dtype=int)

topic_id_to_feat_idx = {t: i for i, t in enumerate(val_with_gt)}

for t_lang_idx, t_lang in enumerate(LANGS):
    # Topics of this language
    t_ids = [t for t in val_with_gt if topic_lang.get(t) == t_lang]
    if not t_ids:
        continue
    t_feats = topic_feats_sample[[topic_id_to_feat_idx[t] for t in t_ids]]

    for c_lang_idx, c_lang in enumerate(LANGS):
        c_indices = content_idx_by_lang.get(c_lang, [])
        if not c_indices:
            continue
        c_feats = content_feats[c_indices]
        c_ids = content_by_lang[c_lang]

        # Score
        sim = t_feats @ c_feats.T  # [n_topics, n_content]

        f2_list = []
        for i, tid in enumerate(t_ids):
            gt = gt_dict.get(tid, set())
            if not gt:
                continue
            s = sim[i]
            max_s = s.max()
            thresh = max_s - MARGIN * max_s
            mask = s >= thresh
            selected = [c_ids[j] for j in mask.nonzero(as_tuple=True)[0].tolist()][:50]
            if not selected:
                selected = [c_ids[s.argmax()]]
            f2, _, _ = f2_score(gt, selected)
            f2_list.append(f2)

        if f2_list:
            results_matrix[t_lang_idx, c_lang_idx] = np.mean(f2_list)
            counts_matrix[t_lang_idx, c_lang_idx] = len(f2_list)

# Print matrix
header = "Topic\\Content | " + " | ".join(f"{l:^6}" for l in LANGS)
print(header)
print("-" * len(header))
for i, t_lang in enumerate(LANGS):
    row = f"    {t_lang:^9} | " + " | ".join(f"{results_matrix[i,j]:.3f} " for j in range(len(LANGS)))
    print(row)

# ── Summary: intra vs cross lingual ──
print("\n" + "=" * 70)
print("Summary: Intra-lingual vs Cross-lingual")
print("=" * 70)
intra_scores = [results_matrix[i, i] for i in range(len(LANGS)) if counts_matrix[i, i] > 0]
# cross = all off-diagonal with counts > 0
cross_scores = []
for i in range(len(LANGS)):
    for j in range(len(LANGS)):
        if i != j and counts_matrix[i, j] > 0:
            cross_scores.extend([results_matrix[i, j]])

print(f"Intra-lingual (topic & content same lang):")
print(f"  Mean F2: {np.mean(intra_scores):.4f}")
print(f"\nCross-lingual (topic & content different lang):")
print(f"  Mean F2: {np.mean(cross_scores):.4f}" if cross_scores else "  N/A (GT is always same-lang by construction)")

# Note: GT correlations are within same language (Wikibooks structure),
# so cross-lingual F2 will be ~0 by design. The interesting metric is RANKING quality.
print("\n" + "=" * 70)
print("Note: GT is intra-lingual by construction (Wikibooks hierarchy).")
print("Cross-lingual F2 ~0 is expected. Key metric: does the model RANK")
print("translated/related content highly even if not exact GT?")
print("=" * 70)

# Save
np.save(f"{RESULTS_DIR}/crosslingual_matrix.npy", results_matrix)
pd.DataFrame(results_matrix, index=LANGS, columns=LANGS).to_csv(f"{RESULTS_DIR}/crosslingual_matrix.csv")
print(f"\nSaved to {RESULTS_DIR}/crosslingual_matrix.csv")
