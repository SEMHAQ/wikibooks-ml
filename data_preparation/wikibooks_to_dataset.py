"""
Convert Wikibooks XML dump → topics.csv / content.csv / correlations.csv
Matching the format of the Learning Equality Kaggle competition.
"""
import bz2
import lxml.etree as ET
import pandas as pd
import re
import hashlib
import os
from tqdm import tqdm
from collections import defaultdict

WIKI_DUMPS = {
    "en": "enwikibooks-20260601-pages-articles-multistream.xml.bz2",
    "fr": "frwikibooks-20260601-pages-articles-multistream.xml.bz2",
    "es": "eswikibooks-20260601-pages-articles-multistream.xml.bz2",
    "de": "dewikibooks-20260601-pages-articles-multistream.xml.bz2",
    "pt": "ptwikibooks-20260601-pages-articles-multistream.xml.bz2",
    "it": "itwikibooks-20260601-pages-articles-multistream.xml.bz2",
}
DATA_DIR = "./data/wikibooks"
NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def wikitext_clean(text):
    if not text: return ""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[File:[^\]]*\]\]', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[[^\[\]]*\|([^\[\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\[\]]*)\]\]', r'\1', text)
    text = re.sub(r'={2,}\s*([^=]+)\s*={2,}', r'\1', text)
    text = re.sub(r"''+", '', text)
    text = re.sub(r'----+', '', text)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<(nowiki|pre|code|syntaxhighlight|source)[^>]*>.*?</\1>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def make_id(prefix, title):
    return f"{prefix}_{hashlib.md5(title.encode()).hexdigest()[:12]}"

def safe_str(val): return "" if val is None else str(val)


def parse_wikibooks_dump(filepath, lang):
    print(f"Parsing {filepath}...")
    all_topics = {}
    all_content = {}
    parent_child = []
    page_count = 0
    content_count = 0
    topics_seen = set()

    with bz2.open(filepath, 'rb') as f:
        # lxml iterparse - use tag filter for performance, lxml preserves text content
        context = ET.iterparse(f, events=('end',), tag=f'{NS}page')
        for event, elem in tqdm(context, desc="Pages"):
            page_count += 1

            # --- Namespace ---
            ns_elem = elem.find(f'{NS}ns')
            if ns_elem is None or ns_elem.text is None:
                elem.clear()
                continue
            try:
                ns = int(ns_elem.text.strip())
            except ValueError:
                elem.clear()
                continue
            if ns != 0:
                elem.clear()
                continue

            # --- Title ---
            title_elem = elem.find(f'{NS}title')
            if title_elem is None or title_elem.text is None:
                elem.clear()
                continue
            title = title_elem.text.strip()

            skip_prefixes = ("Wikibooks:", "Cookbook:", "Transwiki:", "Template:",
                           "Category:", "Help:", "User:", "Book:", "MediaWiki:", "File:")
            if any(title.startswith(p) for p in skip_prefixes) or title in ("Main Page", "Main_Page"):
                elem.clear()
                continue

            # --- Text ---
            rev_elem = elem.find(f'{NS}revision')
            if rev_elem is None:
                elem.clear()
                continue
            text_elem = rev_elem.find(f'{NS}text')
            raw_text = safe_str(text_elem.text if text_elem is not None else "")

            cleaned = wikitext_clean(raw_text)
            if len(cleaned) < 50:
                elem.clear()
                continue

            parts = title.split("/")
            depth = len(parts)
            channel = parts[0]

            # --- Topic for PARENT path ---
            if depth >= 2:
                parent_title = "/".join(parts[:-1])
                if parent_title not in topics_seen:
                    topics_seen.add(parent_title)
                    pt_parts = parent_title.split("/")
                    all_topics[parent_title] = {
                        "id": make_id("t", parent_title),
                        "title": pt_parts[-1],
                        "description": "",
                        "channel": pt_parts[0],
                        "language": lang,
                        "parent": make_id("t", "/".join(pt_parts[:-1])) if len(pt_parts) >= 2 else "",
                        "level": len(pt_parts) - 1,
                        "has_content": True,
                    }
            else:
                if channel not in topics_seen:
                    topics_seen.add(channel)
                    all_topics[channel] = {
                        "id": make_id("t", channel),
                        "title": channel,
                        "description": "", "channel": channel,
                        "language": lang, "parent": "", "level": 0, "has_content": True,
                    }

            # --- Content ---
            cid = make_id("c", title)
            all_content[cid] = {
                "id": cid, "title": parts[-1], "description": cleaned[:300],
                "text": cleaned[:2000], "language": lang, "kind": "document", "channel": channel,
            }

            # --- Correlation ---
            if depth >= 2:
                parent_child.append(("/".join(parts[:-1]), cid))
            else:
                parent_child.append((channel, cid))
            content_count += 1

            # Backfill topic description
            if title in all_topics:
                all_topics[title]["description"] = cleaned[:500]

            elem.clear()

    print(f"\nParsed: {page_count} total pages, {content_count} content pages")
    if not all_content:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    topic_path_to_id = {p: info["id"] for p, info in all_topics.items()}
    grp = defaultdict(list)
    unresolved = 0
    for pt, cid in parent_child:
        tid = topic_path_to_id.get(pt)
        if tid: grp[tid].append(cid)
        else: unresolved += 1

    corr = [{"topic_id": tid, "content_ids": " ".join(cids)} for tid, cids in grp.items()]
    print(f"  Topics: {len(all_topics)}, Content: {len(all_content)}, Correlations: {len(corr)} (unresolved: {unresolved})")
    return pd.DataFrame(list(all_topics.values())), pd.DataFrame(list(all_content.values())), pd.DataFrame(corr)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for lang, fn in WIKI_DUMPS.items():
        fp = f"./data/{fn}"
        if not os.path.exists(fp):
            print(f"Not found: {fp}")
            continue
        print(f"\n{'='*60}\nProcessing {lang}wikibooks...\n{'='*60}")
        dt, dc, dcr = parse_wikibooks_dump(fp, lang)
        if len(dc) == 0:
            print("No data. Skip.")
            continue
        dt.to_csv(f"{DATA_DIR}/topics_{lang}.csv", index=False)
        dc.to_csv(f"{DATA_DIR}/content_{lang}.csv", index=False)
        dcr.to_csv(f"{DATA_DIR}/correlations_{lang}.csv", index=False)
        print(f"  Saved: topics={len(dt)}, content={len(dc)}, correlations={len(dcr)}")

    # Combine
    print(f"\n{'='*60}\nCombining...")
    for prefix in ["topics", "content", "correlations"]:
        dfs = [pd.read_csv(f"{DATA_DIR}/{prefix}_{l}.csv") for l in WIKI_DUMPS if os.path.exists(f"{DATA_DIR}/{prefix}_{l}.csv")]
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(f"{DATA_DIR}/{prefix}.csv", index=False)
            print(f"  {prefix}.csv: {sum(len(d) for d in dfs)} rows")
    print("\nDone!")
