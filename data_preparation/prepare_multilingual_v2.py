"""
Prepare multilingual Wikibooks - all languages in ONE pool (pool 0).
No language switching, but the model sees 6 languages simultaneously.
"""
import pandas as pd
import pickle
import os

DATA_DIR = "./data/wikibooks"
SWITCH_DIR = "./data/switch"

LANGUAGES = [
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
]

os.makedirs(SWITCH_DIR, exist_ok=True)

all_topics = []
all_content = []
all_correlations = []
topic2string = {}

for lang_code, lang_name in LANGUAGES:
    # Topics
    tf = f"{DATA_DIR}/topics_{lang_code}.csv"
    if not os.path.exists(tf):
        continue
    t = pd.read_csv(tf).fillna({"title": "", "description": ""})
    t["fold"] = 0
    t["language_t"] = lang_code
    all_topics.append(t)
    print(f"{lang_name}: {len(t)} topics", end="")

    # Content
    cf = f"{DATA_DIR}/content_{lang_code}.csv"
    c = pd.read_csv(cf).fillna({"title": "", "description": "", "text": ""})
    c["language_t"] = lang_code
    c["fold"] = ""
    all_content.append(c)
    print(f", {len(c)} content", end="")

    # Topic2string
    for _, row in t.iterrows():
        tid = row["id"]
        title = str(row.get("title", ""))
        desc = str(row.get("description", ""))
        channel = str(row.get("channel", ""))
        topic2string[tid] = f"{channel} # {title} # {desc}"

    # Correlations
    corf = f"{DATA_DIR}/correlations_{lang_code}.csv"
    if os.path.exists(corf):
        cr = pd.read_csv(corf)
        all_correlations.append(cr)
        print(f", {len(cr)} corr")
    else:
        print()

# Merge all
df_topics = pd.concat(all_topics, ignore_index=True)
df_content = pd.concat(all_content, ignore_index=True)
df_corr = pd.concat(all_correlations, ignore_index=True)

# Remove any topic-content pairs where content doesn't exist
existing_content = set(df_content["id"].values)
valid_rows = []
removed = 0
for _, row in df_corr.iterrows():
    cids = str(row["content_ids"]).split()
    valid = [c for c in cids if c in existing_content]
    if valid:
        valid_rows.append({"topic_id": row["topic_id"], "content_ids": " ".join(valid)})
    else:
        removed += 1

df_corr_clean = pd.DataFrame(valid_rows)
print(f"\nRemoved {removed} invalid correlations")

# Save to switch dir (pool 0)
df_topics.to_csv(f"{SWITCH_DIR}/topics_0.csv", index=False)
df_content.to_csv(f"{SWITCH_DIR}/content_0.csv", index=False)

# Save topic2string
with open(f"{SWITCH_DIR}/topic2string_0.pkl", "wb") as f:
    pickle.dump(topic2string, f)

# Save correlations
df_corr_clean.to_csv(f"{DATA_DIR}/correlations.csv", index=False)

# Verify
all_topic_ids = set(df_topics["id"].values)
all_content_ids = set(df_content["id"].values)
corr_topic_ids = set(df_corr_clean["topic_id"].values)

# Content IDs in correlations that don't exist in content
cids_in_corr = set()
for cids in df_corr_clean["content_ids"]:
    for c in str(cids).split():
        cids_in_corr.add(c)
missing = cids_in_corr - all_content_ids

print(f"\n=== Final Dataset ===")
print(f"Topics: {len(df_topics)} (unique: {len(all_topic_ids)})")
print(f"Content: {len(df_content)} (unique: {len(all_content_ids)})")
print(f"Correlations: {len(df_corr_clean)}")
print(f"Missing content_ids: {len(missing)}")

# Language distribution
print(f"\nTopics by language:")
for lang_code, _ in LANGUAGES:
    count = len(df_topics[df_topics["language_t"] == lang_code])
    print(f"  {lang_code}: {count}")
print(f"\nContent by language:")
for lang_code, _ in LANGUAGES:
    count = len(df_content[df_content["language_t"] == lang_code])
    print(f"  {lang_code}: {count}")

print(f"\nConfig: pool=(0,), train_on_all=True")
