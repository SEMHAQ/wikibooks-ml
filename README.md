# Wikibooks-ML: A Multilingual Benchmark for Curriculum–Content Matching

This repository accompanies our Applied Sciences paper *"Wikibooks-ML: A Multilingual Benchmark for Curriculum–Content Matching with Curriculum-Aware Contrastive Learning."*

It provides:
- **Wikibooks-ML dataset** construction pipeline (6 languages, 49,998 topics, 211,137 content items)
- **InfoNCE contrastive learning** training code with multilingual sentence-transformer encoders
- **H-InfoNCE**: curriculum-aware hard-negative mining using the topic-tree hierarchy
- Full experiment suite: zero-shot baselines, backbone/length/margin ablations, per-language analysis, and embedding-space visualization

## Dataset: Wikibooks-ML

| Statistic | Value |
|-----------|-------|
| Languages | 6 (en, de, fr, es, it, pt) |
| Curriculum topics | 49,998 |
| Content items | 211,137 |
| Topic–content alignments | 50,203 |
| Source | Wikibooks XML dumps (CC BY-SA / GFDL) |

The dataset is derived entirely from open Wikibooks content. Topics and their alignments are extracted from the native page hierarchy (book → chapter → section), so no manual annotation is required.

## Repository structure

```
.
├── data_preparation/
│   ├── wikibooks_to_dataset.py     # XML dump → topics/content/correlations CSV
│   ├── prepare_multilingual_v2.py  # assemble 6 languages into a single pool
│   └── diagnose_task.py            # task-diffiness diagnosis + CV split
├── retrieval/                      # core model code
│   ├── model.py                    # dual-encoder Net (MeanPool / CLS)
│   ├── loss.py                     # InfoNCE
│   ├── dataset.py                  # contrastive dataset + smart batching
│   ├── trainer.py                  # training loop (FP16, grad accumulation)
│   └── evaluate.py                 # F2 / precision / recall scoring
├── training/
│   ├── train_cv.py                 # InfoNCE + dynamic hard-negative mining
│   ├── train_hierarchical.py       # H-InfoNCE (curriculum-aware negatives)
│   └── train_labse_v2.py           # LaBSE backbone variant
├── analysis/
│   ├── eval_baselines.py           # zero-shot multi-model evaluation
│   ├── eval_analysis.py            # per-language F2 + margin sweep + error cases
│   ├── analyze_embeddings.py       # intra/inter-lingual similarity
│   └── eval_crosslingual.py        # cross-lingual retrieval matrix
└── paper/
    ├── manuscript.tex              # MDPI Applied Sciences LaTeX source
    └── Definitions/                # figures + MDPI class files
```

## Quick start

### 1. Build the dataset

```bash
# Download Wikibooks dumps (example: English)
curl -O https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles-multistream.xml.bz2
# (repeat for fr/de/es/it/pt)

# Parse into topics/content/correlations
python data_preparation/wikibooks_to_dataset.py
python data_preparation/prepare_multilingual_v2.py
```

### 2. Train

```bash
# InfoNCE with dynamic hard-negative mining
python training/train_cv.py

# H-InfoNCE (curriculum-aware hard negatives)
python training/train_hierarchical.py
```

### 3. Evaluate

```bash
python analysis/eval_baselines.py     # zero-shot baselines
python analysis/eval_analysis.py      # per-language + margin + error analysis
```

## Key results

| Method | F2 |
|--------|-----|
| Zero-shot paraphrase-mpnet | 0.362 |
| Zero-shot LaBSE | 0.317 |
| Title-overlap (non-neural) | 0.539 |
| InfoNCE mpnet (max len 48) | 0.670 |
| **InfoNCE mpnet (max len 96, margin 0.10)** | **0.680** |
| LaBSE (max len 48) | 0.640 |
| H-InfoNCE (combined) | 0.669 |
| H-InfoNCE (curriculum only) | 0.620 |

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{peng2026wikibooksml,
  title={Wikibooks-ML: A Multilingual Benchmark for Curriculum--Content Matching with Curriculum-Aware Contrastive Learning},
  author={Peng, Donghai and Yu, Huanjie},
  journal={Applied Sciences},
  year={2026}
}
```

## Acknowledgements

This implementation builds on the InfoNCE contrastive framework and the sentence-transformer encoders. We thank the Wikibooks contributor community and the authors of prior curriculum-recommendation work whose open solutions informed our design.

## License

- **Code**: MIT
- **Dataset**: derived from Wikibooks (CC BY-SA 4.0 / GFDL)
