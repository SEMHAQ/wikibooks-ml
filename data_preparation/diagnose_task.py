"""
Phase 1: 诊断实验
目的：搞清楚 F2=0.974 是因为 (a) train=val 数据泄漏，还是 (b) 任务太简单

实验设计：
  1. 真正的 80/20 split：80% 的 topic 用于训练，20% 完全不参与训练做验证
  2. 在验证集上评估真实 F2
  3. 关键诊断：如果真实 F2 仍然 >0.9 → 任务太简单（标题字面匹配）
                如果真实 F2 掉到 0.5-0.7 → 任务有难度，之前的 0.974 是泄漏

  额外诊断：
  - "标题字面匹配" baseline：只用标题的 Jaccard/TF-IDF 相似度，不用深度学习
    如果这个 baseline 也能拿 0.8+ → 任务确实简单
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import re
import math

random.seed(42)
np.random.seed(42)

DATA_DIR = "./data/wikibooks"
SWITCH_DIR = "./data/switch"
CV_DIR = "./data/cv_experiment"
os.makedirs(CV_DIR, exist_ok=True)

print("=" * 60)
print("Phase 1: Task Diagnosis")
print("=" * 60)

# ── 1. Load all data ──
df_topics = pd.read_csv(f"{SWITCH_DIR}/topics_0.csv").fillna({"title": "", "description": ""})
df_content = pd.read_csv(f"{SWITCH_DIR}/content_0.csv").fillna({"title": "", "description": "", "text": ""})
df_corr = pd.read_csv(f"{DATA_DIR}/correlations.csv")

print(f"\nLoaded: {len(df_topics)} topics, {len(df_content)} content, {len(df_corr)} correlations")

# ── 2. Build topic hierarchy from Wikibooks structure ──
# Topics have 'channel' and 'parent' fields. Group by channel for splitting.
print("\n--- Topic distribution by language ---")
print(df_topics["language_t"].value_counts().to_string())

# ── 3. Stratified 80/20 split by language ──
train_topics = []
val_topics = []

for lang, group in df_topics.groupby("language_t"):
    topic_ids = group["id"].tolist()
    random.shuffle(topic_ids)
    n_val = max(1, int(len(topic_ids) * 0.2))
    val_topics.extend(topic_ids[:n_val])
    train_topics.extend(topic_ids[n_val:])

train_set = set(train_topics)
val_set = set(val_topics)
print(f"\n--- 80/20 Split ---")
print(f"Train topics: {len(train_set)}")
print(f"Val topics:   {len(val_set)}")

# ── 4. Build GT dict ──
gt_dict = {}
for _, row in df_corr.iterrows():
    tid = row["topic_id"]
    cids = str(row["content_ids"]).split()
    gt_dict[tid] = set(cids)

print(f"\nTopics with GT: {len(gt_dict)}")

# ── 5. Diagnostic 1: How leaky is the title match? ──
# If just matching topic title to content title gives high F2, task is too easy.
print("\n" + "=" * 60)
print("Diagnostic 1: Title-overlap baseline (no neural model)")
print("=" * 60)

# Build content title lookup
content_title = dict(zip(df_content["id"], df_content["title"]))
content_by_lang = defaultdict(list)
for _, row in df_content.iterrows():
    content_by_lang[row["language_t"]].append((row["id"], str(row["title"])))

topic_lang = dict(zip(df_topics["id"], df_topics["language_t"]))
topic_title = dict(zip(df_topics["id"], df_topics["title"]))


def title_jaccard(s1, s2):
    """Word-level Jaccard similarity."""
    w1 = set(re.findall(r'\w+', s1.lower()))
    w2 = set(re.findall(r'\w+', s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def f2_score(gt, pred):
    gt, pred = set(gt), set(pred)
    precision = len(gt & pred) / len(pred) if pred else 0.0
    recall = len(gt & pred) / len(gt) if gt else 0.0
    if 4 * precision + recall == 0:
        return 0.0, precision, recall
    f2 = (5 * precision * recall) / (4 * precision + recall)
    return f2, precision, recall


# Eval on val set using title-overlap matching
val_with_gt = [t for t in val_set if t in gt_dict]
print(f"Val topics with GT: {len(val_with_gt)}")

f2_list = []
for tid in val_with_gt[:2000]:  # sample for speed
    lang = topic_lang.get(tid, "en")
    t_title = topic_title.get(tid, "")
    gt = gt_dict[tid]

    # Score all content in same language by title overlap
    candidates = content_by_lang.get(lang, [])
    scored = [(cid, title_jaccard(t_title, c_title)) for cid, c_title in candidates]
    scored.sort(key=lambda x: -x[1])

    # Take top-K with margin (mimic the competition threshold)
    if not scored:
        continue
    max_score = scored[0][1]
    threshold = max_score - 0.16 * max_score
    selected = [cid for cid, s in scored if s >= threshold][:50]

    f2, _, _ = f2_score(gt, selected)
    f2_list.append(f2)

print(f"\nTitle-overlap baseline (val set):")
print(f"  F2 = {np.mean(f2_list):.4f}")
print(f"  (If this is >0.8, the task is too easy - titles alone solve it)")

# ── 6. Diagnostic 2: Cross-channel difficulty ──
print("\n" + "=" * 60)
print("Diagnostic 2: Per-channel analysis")
print("=" * 60)

# How many content per topic (avg)?
content_counts = [len(gt_dict[t]) for t in val_with_gt]
print(f"Avg content per topic: {np.mean(content_counts):.1f}")
print(f"Median content per topic: {np.median(content_counts):.0f}")

# Save split for training
split_df = pd.DataFrame({
    "topic_id": df_topics["id"],
    "fold": [0 if tid in val_set else -1 for tid in df_topics["id"]]  # fold=0 means val
})
split_df.to_csv(f"{CV_DIR}/topic_split.csv", index=False)
print(f"\nSaved split to {CV_DIR}/topic_split.csv")
print(f"\n{'='*60}")
print(f"NEXT: retrain with fold=0 as val (train_on_all=False) to get TRUE F2")
