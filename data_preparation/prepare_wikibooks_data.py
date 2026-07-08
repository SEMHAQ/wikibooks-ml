"""
Prepare Wikibooks data for training:
1. Copy CSVs to data/switch/ (as pool 0)
2. Generate topic2string_0.pkl
"""
import pandas as pd
import pickle
import os
import shutil

DATA_DIR = "./data/wikibooks"
SWITCH_DIR = "./data/switch"

def generate_topic2string(topics_csv):
    """Generate topic2string dict matching the original format."""
    df = pd.read_csv(topics_csv).fillna({"title": "", "description": ""})
    topic2string = {}
    for _, row in df.iterrows():
        tid = row["id"]
        # Format: "title # description" (matching breadcrumb style from original)
        title = row["title"] if pd.notna(row["title"]) else ""
        desc = row["description"] if pd.notna(row["description"]) else ""
        channel = row["channel"] if pd.notna(row["channel"]) else ""
        string = f"{channel} # {title} # {desc}"
        topic2string[tid] = string
    return topic2string, df


if __name__ == "__main__":
    os.makedirs(SWITCH_DIR, exist_ok=True)

    print("=== Generating topic2string for Wikibooks data ===")

    # Copy topics
    src = f"{DATA_DIR}/topics_en.csv"
    if os.path.exists(src):
        shutil.copy2(src, f"{SWITCH_DIR}/topics_0.csv")
        print(f"  Copied topics → {SWITCH_DIR}/topics_0.csv")

    # Copy content
    src = f"{DATA_DIR}/content_en.csv"
    if os.path.exists(src):
        shutil.copy2(src, f"{SWITCH_DIR}/content_0.csv")
        print(f"  Copied content → {SWITCH_DIR}/content_0.csv")

    # Copy correlations (renamed to match original)
    src = f"{DATA_DIR}/correlations_en.csv"
    if os.path.exists(src):
        shutil.copy2(src, f"{DATA_DIR}/correlations.csv")
        print(f"  Copied correlations → {DATA_DIR}/correlations.csv")

    # Generate topic2string
    topic2string, df_topics = generate_topic2string(f"{SWITCH_DIR}/topics_0.csv")
    with open(f"{SWITCH_DIR}/topic2string_0.pkl", "wb") as f:
        pickle.dump(topic2string, f)
    print(f"  Generated topic2string_0.pkl ({len(topic2string)} topics)")

    # Also save the fold column for compatibility
    df_topics["fold"] = -1  # All training (no CV split needed for now)
    df_topics.to_csv(f"{SWITCH_DIR}/topics_0.csv", index=False)

    # Create content with proper fold column
    df_content = pd.read_csv(f"{SWITCH_DIR}/content_0.csv")
    df_content["language_t"] = df_content["language"]  # For compatibility
    df_content["fold"] = ""
    df_content.to_csv(f"{SWITCH_DIR}/content_0.csv", index=False)

    print("\n✅ Done! Wikibooks data ready for training.")
    print(f"   Topics: {len(df_topics)}")
    print(f"   Content: {len(df_content)}")
    print(f"\n   Run: python train_fast.py")
