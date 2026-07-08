"""
Phase 3: H-InfoNCE — Curriculum-Aware Hard Negative Mining
论文核心方法创新

创新点：
  Baseline (InfoNCE + sim_sample):
    - 前4轮没有困难负样本（sim_sample 要等模型 warm up）
    - 第5轮才开始挖模型预测错的 content 作为 hard negative
    - 困难负样本质量依赖模型当前能力

  H-InfoNCE (我们的方法):
    - 从第1轮就用 Wikibooks 课程层次结构注入困难负样本
    - 兄弟节点（同 parent）的 content 是天然的高质量困难负样本：
      * 同一本书、同一章节级别 → 术语重叠、主题相关
      * 但不是正确答案 → 判别性强
    - 例: "有机化学/烷烃" 的正样本是烷烃内容，
          "有机化学/烯烃"(兄弟节点) 的内容是理想困难负样本

对比设计（公平消融）:
  - train_cv.py:        原始 InfoNCE + sim_sample     (F2 = 0.670)
  - train_hierarchical: + 层次化困难负样本 (epoch 1+)  (F2 = ?)
  - 配置完全相同，唯一变量是层次化负样本
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import torch
import time
import math
import shutil
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import AutoTokenizer

from retrieval.loss import InfoNCE
from retrieval.model import Net
from retrieval.trainer import train
from retrieval.utils import setup_system, Logger
from retrieval.evaluate import evaluate_val, evaluate_train
from retrieval.dataset import EqualDatasetTrain, EqualDatasetEval


@dataclass
class Configuration:
    transformer: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    pooling: str = 'cls'
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    proj = None
    margin: float = 0.16
    layers_to_keep = None

    transformer_teacher: str = 'sentence-transformers/LaBSE'
    use_teacher: bool = False
    pooling_teacher: str = 'cls'
    proj_teacher = None

    init_pool = 0
    pool = (0,)
    epoch_stop_switching: int = 36

    debug = None

    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    mixed_precision: bool = True
    gradient_accumulation: int = 2
    gradient_checkpointing: bool = True
    verbose: bool = True
    gpu_ids: tuple = (0,)

    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    zero_shot: bool = False

    clip_grad = 100.
    decay_exclue_bias: bool = False
    label_smoothing: float = 0.1

    lr: float = 0.0002
    scheduler: str = 'polynomial'
    warmup_epochs: int = 2
    lr_end: float = 0.00005

    language: str = 'all'
    fold: int = 0
    train_on_all: bool = False      # ★ 真实 holdout
    max_len: int = 48

    max_wrong: int = 128
    custom_sampling: bool = True
    sim_sample: bool = False          # ★ PURE hierarchical: no dynamic mining
    sim_sample_start: int = 99         # effectively disabled

    # ★★★ 创新点：层次化困难负样本 ★★★
    use_hierarchical: bool = True   # 启用课程层次结构困难负样本
    max_hier_neg: int = 8           # 每个 topic 最多注入的兄弟负样本数

    model_path: str = './experiments/hier_only_finetune'
    checkpoint_start = None
    checkpoint_teacher = None

    num_workers: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


config = Configuration()


def build_hierarchical_negatives(df_topics, df_corr, topic2content, max_per_topic=8):
    """
    构建 Curriculum-Aware 困难负样本。

    利用 Wikibooks 层次结构：
      sibling_map[topic] = [同 parent 的兄弟 topic]
      hierarchical_wrong[topic] = [(sibling_topic, sibling_content), ...]

    这些是天然的"似是而非"负样本：同书同章节级别，主题相关但非答案。
    """
    # 1. Build sibling map (topics sharing the same parent)
    parent2children = defaultdict(list)
    for _, row in df_topics.iterrows():
        parent = row.get("parent", "")
        if pd.notna(parent) and str(parent).strip():
            parent2children[str(parent)].append(row["id"])

    sibling_map = {}
    for parent, children in parent2children.items():
        if len(children) > 1:
            for child in children:
                sibling_map[child] = [c for c in children if c != child]

    # 2. Build hierarchical wrong dict
    # hierarchical_wrong[topic] = [(sibling_topic, sibling_content), ...]
    hierarchical_wrong = {}
    stats = {"topics_with_sib": 0, "total_pairs": 0}

    for topic, siblings in sibling_map.items():
        pairs = []
        for sib in siblings:
            sib_contents = topic2content.get(sib, [])
            for sc in sib_contents:
                pairs.append((sib, sc))
                if len(pairs) >= max_per_topic:
                    break
            if len(pairs) >= max_per_topic:
                break
        if pairs:
            hierarchical_wrong[topic] = pairs
            stats["topics_with_sib"] += 1
            stats["total_pairs"] += len(pairs)

    print(f"\n[Hierarchical Negatives]")
    print(f"  Parent groups with siblings: {sum(1 for v in parent2children.values() if len(v)>1)}")
    print(f"  Topics with hierarchical negs: {stats['topics_with_sib']}")
    print(f"  Total (topic, content) neg pairs: {stats['total_pairs']}")
    print(f"  Avg per topic: {stats['total_pairs']/max(stats['topics_with_sib'],1):.1f}")

    return hierarchical_wrong


def merge_wrong_dicts(hierarchical, dynamic):
    """Merge hierarchical (static) with dynamic (sim_sample) wrong dicts."""
    merged = defaultdict(list)
    if hierarchical:
        for k, v in hierarchical.items():
            merged[k].extend(v)
    if dynamic:
        for k, v in dynamic.items():
            merged[k].extend(v)
    return dict(merged) if merged else None


if __name__ == '__main__':
    model_path = '{}/{}'.format(config.model_path, time.strftime('%H%M%S'))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    script_path = os.path.abspath(__file__)
    shutil.copyfile(script_path, '{}/train.py'.format(model_path))
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed, cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # ── Inject CV split (same as baseline for fair comparison) ──
    print('\n[Injecting CV split into correlations.csv]')
    df_corr = pd.read_csv('./data/correlations.csv')
    split = pd.read_csv('./data/cv_experiment/topic_split.csv')
    topic_fold = dict(zip(split['topic_id'], split['fold']))
    def map_fold(x):
        f = topic_fold.get(x, -1)
        return 0 if f == 0 else 1
    df_corr['fold'] = df_corr['topic_id'].map(map_fold)
    df_corr.to_csv('./data/correlations.csv', index=False)
    print(f'  Train (fold=1): {(df_corr["fold"]==1).sum()}')
    print(f'  Val   (fold=0): {(df_corr["fold"]==0).sum()}')

    df_topics = pd.read_csv('./data/switch/topics_0.csv')
    df_topics['fold'] = df_topics['id'].map(map_fold)
    df_topics.to_csv('./data/switch/topics_0.csv', index=False)

    # ── Model ──
    print('\n{}[Model: {}]{}'.format(20*'-', config.transformer, 20*'-'))
    print(f'*** H-InfoNCE: Curriculum-Aware Hard Negative Mining ***')
    model = Net(transformer_name=config.transformer,
                gradient_checkpointing=config.gradient_checkpointing,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_dropout_prob=config.attention_dropout_prob,
                pooling=config.pooling, projection=config.proj)
    print(model.transformer.config)

    model = model.to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.transformer)

    # ── Data ──
    df_correlations = pd.read_csv('./data/correlations.csv')
    topics_arr = df_correlations['topic_id'].values
    content_arr = df_correlations['content_ids'].values
    gt_dict = dict()
    for i in range(len(topics_arr)):
        gt_dict[topics_arr[i]] = str(content_arr[i]).split(' ')

    df_correlations_train = df_correlations[df_correlations['fold'] != config.fold]

    train_dataset = EqualDatasetTrain(df_correlations=df_correlations_train,
                                      fold=config.fold, tokenizer=tokenizer,
                                      max_len=config.max_len, shuffle_batch_size=config.batch_size,
                                      pool=config.pool, init_pool=config.init_pool,
                                      train_on_all=config.train_on_all,
                                      language=config.language, debug=config.debug)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=not config.custom_sampling, num_workers=config.num_workers,
                              pin_memory=True, collate_fn=train_dataset.smart_batching_collate)
    print('\nTrain Pairs:', len(train_dataset))

    val_dataset_topic = EqualDatasetEval(mode='topic', typ='val', fold=config.fold,
                                         tokenizer=tokenizer, max_len=config.max_len,
                                         pool=config.pool, init_pool=config.init_pool,
                                         train_on_all=config.train_on_all,
                                         language=config.language, debug=config.debug)
    val_dataset_content = EqualDatasetEval(mode='content', typ='val', fold=config.fold,
                                           tokenizer=tokenizer, max_len=config.max_len,
                                           pool=config.pool, init_pool=config.init_pool,
                                           train_on_all=config.train_on_all,
                                           language=config.language, debug=config.debug)
    val_loader_topic = DataLoader(dataset=val_dataset_topic, batch_size=config.batch_size,
                                  shuffle=False, num_workers=config.num_workers,
                                  pin_memory=True, collate_fn=val_dataset_topic.smart_batching_collate)
    val_loader_content = DataLoader(dataset=val_dataset_content, batch_size=config.batch_size,
                                    shuffle=False, num_workers=config.num_workers,
                                    pin_memory=True, collate_fn=val_dataset_content.smart_batching_collate)
    print('Topics Val:', len(val_dataset_topic))
    print('Content Val:', len(val_dataset_content))

    train_dataset_topic = EqualDatasetEval(mode='topic', typ='train', fold=config.fold,
                                           tokenizer=tokenizer, max_len=config.max_len,
                                           pool=config.pool, init_pool=config.init_pool,
                                           train_on_all=config.train_on_all,
                                           language=config.language, debug=config.debug)
    train_dataset_content = EqualDatasetEval(mode='content', typ='train', fold=config.fold,
                                             tokenizer=tokenizer, max_len=config.max_len,
                                             pool=config.pool, init_pool=config.init_pool,
                                             train_on_all=config.train_on_all,
                                             language=config.language, debug=config.debug)
    train_loader_topic = DataLoader(dataset=train_dataset_topic, batch_size=config.batch_size,
                                    shuffle=False, num_workers=config.num_workers,
                                    pin_memory=True, collate_fn=train_dataset_topic.smart_batching_collate)
    train_loader_content = DataLoader(dataset=train_dataset_content, batch_size=config.batch_size,
                                      shuffle=False, num_workers=config.num_workers,
                                      pin_memory=True, collate_fn=train_dataset_content.smart_batching_collate)

    # ★★★ BUILD HIERARCHICAL NEGATIVES ★★★
    # topic2content from train dataset
    topic2content_train = {}
    for t, cset in train_dataset.topic2content.items():
        topic2content_train[t] = list(cset)

    hierarchical_wrong = None
    if config.use_hierarchical:
        hierarchical_wrong = build_hierarchical_negatives(
            df_topics, df_correlations_train, topic2content_train,
            max_per_topic=config.max_hier_neg
        )

    # ── Loss & optimizer ──
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn, device=config.device)
    scaler = GradScaler(init_scale=2.**10) if config.mixed_precision else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_steps = math.floor((len(train_loader) * config.epochs) / config.gradient_accumulation)
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_training_steps=train_steps,
                                                          lr_end=config.lr_end, power=1.5,
                                                          num_warmup_steps=warmup_steps)
    print('Train Steps:', train_steps, 'Warmup Steps:', warmup_steps)

    # ── Initial shuffle WITH hierarchical negatives from epoch 1 ──
    missing_pairs = None
    topic2wrong = None  # will be filled by sim_sample later
    if config.custom_sampling:
        # ★ KEY DIFFERENCE: inject hierarchical negatives from the very first epoch
        combined_wrong = merge_wrong_dicts(hierarchical_wrong, topic2wrong) if config.use_hierarchical else None
        train_loader.dataset.shuffle(missing_list=missing_pairs, wrong_dict=combined_wrong,
                                     max_wrong=config.max_wrong)

    best_score = 0
    for epoch in range(1, config.epochs + 1):
        print('\n{}[Epoch: {}]{}'.format(30*'-', epoch, 30*'-'))
        train_loss = train(config, model, dataloader=train_loader, loss_function=loss_function,
                          optimizer=optimizer, scheduler=scheduler, scaler=scaler, teacher=None)
        print('Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}'.format(epoch, train_loss, optimizer.param_groups[0]['lr']))

        print('\n{}[{}]{}'.format(30*'-', 'Evaluate (Val)', 30*'-'))
        f2, precision, recall = evaluate_val(config, model, reference_dataloader=val_loader_content,
                                             query_dataloader=val_loader_topic, gt_dict=gt_dict, cleanup=True)

        if f2 > best_score:
            best_score = f2
            best_checkpoint = '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, f2)
            torch.save(model.state_dict(), best_checkpoint)
            print(f'  *** New best: {best_score:.4f} ***')

        # sim_sample (dynamic hard negatives from model errors)
        if config.sim_sample and epoch >= config.sim_sample_start:
            print('\n{}[{}]{}'.format(30*'-', 'Evaluate (Train)', 30*'-'))
            missing_pairs, topic2wrong = evaluate_train(config=config, model=model,
                                                       reference_dataloader=train_loader_content,
                                                       query_dataloader=train_loader_topic, gt_dict=gt_dict,
                                                       content2topic=train_loader.dataset.content2topic, cleanup=True)

        # ★ Shuffle: MERGE hierarchical + dynamic negatives every epoch
        if config.custom_sampling:
            combined_wrong = merge_wrong_dicts(hierarchical_wrong, topic2wrong) if config.use_hierarchical else topic2wrong
            train_loader.dataset.shuffle(missing_list=missing_pairs, wrong_dict=combined_wrong,
                                         max_wrong=config.max_wrong)

    print(f'\n{"="*60}')
    print(f'H-InfoNCE Best F2: {best_score:.4f}')
    print(f'{"="*60}')
