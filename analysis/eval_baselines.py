"""
Phase 2: 多模型 Zero-shot Baseline 对比
目的：在真实验证集上，对比多个预训练模型的 zero-shot 检索能力

这是论文的核心 baseline 表格。
不微调，直接用预训练 embedding 做 cosine 相似度检索。

模型对比：
  1. distiluse-base-multilingual-cased-v1 (轻量多语言)
  2. paraphrase-multilingual-mpnet-base-v2 (竞赛用)
  3. LaBSE (强跨语言)
  4. XLM-R (通用多语言)

评估指标：
  - F2 Score (竞赛指标)
  - Precision / Recall
  - MRR (Mean Reciprocal Rank)
  - Recall@5, Recall@10
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
import re
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


# ── Data loading ──
print("=" * 60)
print("Phase 2: Multi-model Zero-shot Baseline")
print("=" * 60)

df_topics = pd.read_csv(f"{SWITCH_DIR}/topics_0.csv").fillna({"title": "", "description": ""})
df_content = pd.read_csv(f"{SWITCH_DIR}/content_0.csv").fillna({"title": "", "description": "", "text": ""})
df_corr = pd.read_csv(f"{DATA_DIR}/correlations.csv")
split = pd.read_csv(f"{CV_DIR}/topic_split.csv")  # fold=0 means val

# Build splits
val_topics = set(split[split["fold"] == 0]["topic_id"].values)
train_topics = set(split[split["fold"] != 0]["topic_id"].values)
print(f"\nTrain topics: {len(train_topics)}, Val topics: {len(val_topics)}")

# GT dict
gt_dict = {}
for _, row in df_corr.iterrows():
    tid = row["topic_id"]
    gt_dict[tid] = set(str(row["content_ids"]).split())

# Topic text (breadcrumb style, matching training format)
topic_text = {}
for _, row in df_topics.iterrows():
    tid = row["id"]
    topic_text[tid] = f"{row.get('channel','')} # {row.get('title','')} # {row.get('description','')}"

# Content text
content_ids_all = df_content["id"].tolist()
content_text = {}
for _, row in df_content.iterrows():
    cid = row["id"]
    text = " ".join(str(row.get("text", "")).split(" ")[:32])
    content_text[cid] = f"{row.get('title','')} # {row.get('description','')} # {text}"

# Language grouping (eval within same language, like competition)
topic_lang = dict(zip(df_topics["id"], df_topics["language_t"]))
content_lang = dict(zip(df_content["id"], df_content["language_t"]))


# ── Embedding function ──
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output, attention_mask):
    return model_output.last_hidden_state[:, 0, :]


@torch.no_grad()
def embed_texts(texts, model_name, tokenizer, model, pooling="cls"):
    """Embed a list of texts in batches."""
    all_feats = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", leave=False):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN,
                       return_tensors="pt").to(DEVICE)
        with torch.cuda.amp.autocast():
            out = model(**enc)
        if pooling == "cls":
            feat = cls_pooling(out, enc["attention_mask"])
        else:
            feat = mean_pooling(out, enc["attention_mask"])
        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat.cpu())
    return torch.cat(all_feats, dim=0)


# ── Evaluation ──
def evaluate_retrieval(topic_feats, content_feats, topic_ids, content_ids,
                      gt_dict, topic_lang, content_lang, k_list=(1, 5, 10)):
    """Evaluate retrieval: for each topic, rank all content in same language."""
    # Group content by language
    content_by_lang = defaultdict(list)
    content_idx_by_lang = defaultdict(list)
    for i, cid in enumerate(content_ids):
        lang = content_lang.get(cid, "en")
        content_by_lang[lang].append(cid)
        content_idx_by_lang[lang].append(i)

    # Topic features by index
    topic_id_to_idx = {tid: i for i, tid in enumerate(topic_ids)}

    f2_list, prec_list, rec_list = [], [], []
    mrr_list = []
    recall_at_k = {k: [] for k in k_list}

    for tid in tqdm(topic_ids, desc="Scoring", leave=False):
        if tid not in gt_dict:
            continue
        lang = topic_lang.get(tid, "en")
        gt = gt_dict[tid]
        if not gt:
            continue

        t_idx = topic_id_to_idx[tid]
        c_indices = content_idx_by_lang.get(lang, [])
        if not c_indices:
            continue

        c_feats = content_feats[c_indices]
        c_ids = content_by_lang[lang]
        t_feat = topic_feats[t_idx].unsqueeze(0)

        sim = (t_feat @ c_feats.T).squeeze(0)
        sorted_idx = torch.argsort(sim, descending=True)

        # Dynamic threshold (competition style)
        max_sim = sim.max()
        threshold = max_sim - MARGIN * max_sim
        selected_mask = sim >= threshold
        selected_indices = sorted_idx[selected_mask[sorted_idx]][:50].tolist()

        if not selected_indices:
            selected_indices = sorted_idx[:1].tolist()

        pred = set(c_ids[i] for i in selected_indices)

        # F2
        tp = len(gt & pred)
        prec = tp / len(pred) if pred else 0
        rec = tp / len(gt) if gt else 0
        if 4 * prec + rec > 0:
            f2 = (5 * prec * rec) / (4 * prec + rec)
        else:
            f2 = 0
        f2_list.append(f2)
        prec_list.append(prec)
        rec_list.append(rec)

        # MRR (rank of first relevant item)
        sorted_cids = [c_ids[i] for i in sorted_idx.tolist()]
        first_rel_rank = None
        for rank, cid in enumerate(sorted_cids, 1):
            if cid in gt:
                first_rel_rank = rank
                break
        if first_rel_rank:
            mrr_list.append(1.0 / first_rel_rank)
        else:
            mrr_list.append(0.0)

        # Recall@K
        for k in k_list:
            topk = set(sorted_cids[:k])
            recall_at_k[k].append(len(gt & topk) / len(gt))

    results = {
        "F2": np.mean(f2_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "MRR": np.mean(mrr_list),
    }
    for k in k_list:
        results[f"Recall@{k}"] = np.mean(recall_at_k[k])
    results["n_eval"] = len(f2_list)
    return results


# ── Models to evaluate ──
MODELS = [
    ("paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "cls"),
    ("LaBSE", "sentence-transformers/LaBSE", "cls"),
    ("distiluse-base-multilingual-cased-v1", "sentence-transformers/distiluse-base-multilingual-cased-v1", "mean"),
    ("paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "mean"),
]

# Only eval on val set
val_topic_ids = [t for t in val_topics if t in topic_text and t in gt_dict]
val_topic_texts = [topic_text[t] for t in val_topic_ids]

all_results = {}

for model_short, model_name, pooling in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_short}")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

        t0 = time.time()
        # Embed topics
        print("  Embedding topics...")
        topic_feats = embed_texts(val_topic_texts, model_name, tokenizer, model, pooling)
        # Embed content
        print("  Embedding content...")
        content_feats = embed_texts([content_text[c] for c in content_ids_all], model_name, tokenizer, model, pooling)
        t_embed = time.time() - t0
        print(f"  Embedding time: {t_embed:.0f}s")

        # Evaluate
        t0 = time.time()
        results = evaluate_retrieval(
            topic_feats, content_feats,
            val_topic_ids, content_ids_all,
            gt_dict, topic_lang, content_lang
        )
        t_eval = time.time() - t0
        results["eval_time_s"] = t_eval

        print(f"\n  Results ({model_short}):")
        for k, v in results.items():
            if k != "n_eval":
                print(f"    {k:15s}: {v:.4f}")
        print(f"    n_eval         : {results['n_eval']}")

        all_results[model_short] = results

        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        continue

# ── Save results ──
print("\n" + "=" * 60)
print("Summary Table")
print("=" * 60)
print(f"{'Model':<45} {'F2':>7} {'Prec':>7} {'Rec':>7} {'MRR':>7} {'R@5':>7} {'R@10':>7}")
print("-" * 95)
for model_short, res in sorted(all_results.items(), key=lambda x: -x[1]["F2"]):
    print(f"{model_short:<45} {res['F2']:>7.4f} {res['Precision']:>7.4f} {res['Recall']:>7.4f} "
          f"{res['MRR']:>7.4f} {res['Recall@5']:>7.4f} {res['Recall@10']:>7.4f}")

# Save to CSV
df_results = pd.DataFrame(all_results).T
df_results.to_csv(f"{RESULTS_DIR}/zero_shot_baselines.csv")
print(f"\nSaved to {RESULTS_DIR}/zero_shot_baselines.csv")
