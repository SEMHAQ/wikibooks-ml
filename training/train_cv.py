"""
Phase 2b: 真实 holdout 微调实验
目的：用真实验证集（20% topic 不参与训练）做 InfoNCE 微调，看真实 F2

关键：train_on_all=False，用 fold 列做 holdout
这样得到的 F2 才是论文可以用的真实结果。

设计：
  - 快速版：10 epoch（够看出微调效果，不用等40轮）
  - paraphrase-mpnet 模型
  - 真实 80/20 holdout
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
    # Model
    transformer: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    pooling: str = 'cls'
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    proj = None
    margin: float = 0.16
    layers_to_keep = None

    # Distillation
    transformer_teacher: str = 'sentence-transformers/LaBSE'
    use_teacher: bool = False
    pooling_teacher: str = 'cls'
    proj_teacher = None

    # Language Sampling - single pool (all 6 langs merged)
    init_pool = 0
    pool = (0,)
    epoch_stop_switching: int = 36

    debug = None

    # Training - FAST version (10 epochs to see real effect)
    seed: int = 42
    epochs: int = 10              # ★ 快速版，够看微调效果
    batch_size: int = 256
    mixed_precision: bool = True
    gradient_accumulation: int = 2
    gradient_checkpointing: bool = True
    verbose: bool = True
    gpu_ids: tuple = (0,)

    # Eval every epoch (we want to see the learning curve)
    eval_every_n_epoch: int = 1   # ★ 每轮都评估，看曲线
    normalize_features: bool = True
    zero_shot: bool = False

    clip_grad = 100.
    decay_exclue_bias: bool = False
    label_smoothing: float = 0.1

    lr: float = 0.0002
    scheduler: str = 'polynomial'
    warmup_epochs: int = 2
    lr_end: float = 0.00005

    # Data
    language: str = 'all'
    fold: int = 0                 # ★ val on fold 0
    train_on_all: bool = False    # ★★★ 关键：用 holdout，不用全部数据
    max_len: int = 48

    # Sampling
    max_wrong: int = 128
    custom_sampling: bool = True
    sim_sample: bool = True
    sim_sample_start: int = 4     # ★ start hard negative mining after warmup

    model_path: str = './experiments/cv_finetune'
    checkpoint_start = None
    checkpoint_teacher = None

    num_workers: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


config = Configuration()


if __name__ == '__main__':
    model_path = '{}/{}'.format(config.model_path, time.strftime('%H%M%S'))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Copy this script for reproducibility (handle being run from scripts/ dir)
    script_path = os.path.abspath(__file__)
    shutil.copyfile(script_path, '{}/train.py'.format(model_path))
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed, cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # ── Inject CV split into correlations ──
    # NOTE: dataset.py requires train fold to be != 0 AND != -1.
    # Our split uses fold=0 (val) and fold=-1 (train). Map train → fold=1.
    print('\n[Injecting CV split into correlations.csv]')
    df_corr = pd.read_csv('./data/correlations.csv')
    split = pd.read_csv('./data/cv_experiment/topic_split.csv')
    topic_fold = dict(zip(split['topic_id'], split['fold']))
    # val (fold=0) stays 0; train (fold=-1) becomes 1
    def map_fold(x):
        f = topic_fold.get(x, -1)
        return 0 if f == 0 else 1
    df_corr['fold'] = df_corr['topic_id'].map(map_fold)
    df_corr.to_csv('./data/correlations.csv', index=False)
    print(f'  Train (fold=1): {(df_corr["fold"]==1).sum()}')
    print(f'  Val   (fold=0): {(df_corr["fold"]==0).sum()}')

    # Also inject fold into topics_0.csv
    df_topics = pd.read_csv('./data/switch/topics_0.csv')
    df_topics['fold'] = df_topics['id'].map(map_fold)
    df_topics.to_csv('./data/switch/topics_0.csv', index=False)

    print('\n{}[Model: {}]{}'.format(20*'-', config.transformer, 20*'-'))
    model = Net(transformer_name=config.transformer,
                gradient_checkpointing=config.gradient_checkpointing,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_dropout_prob=config.attention_dropout_prob,
                pooling=config.pooling, projection=config.proj)
    print(model.transformer.config)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(config.device)

    teacher = None

    tokenizer = AutoTokenizer.from_pretrained(config.transformer)

    # ── Data ──
    df_correlations = pd.read_csv('./data/correlations.csv')
    topics = df_correlations['topic_id'].values
    content = df_correlations['content_ids'].values
    gt_dict = dict()
    for i in range(len(topics)):
        gt_dict[topics[i]] = str(content[i]).split(' ')

    # train_on_all=False → uses fold != config.fold for training
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

    # train loaders for sim_sample
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

    # Loss & optimizer
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

    # Zero-shot baseline before training
    if config.zero_shot:
        print('\n{}[{}]{}'.format(30*'-', 'Zero Shot', 30*'-'))
        evaluate_val(config, model, reference_dataloader=val_loader_content,
                     query_dataloader=val_loader_topic, gt_dict=gt_dict, cleanup=True)

    missing_pairs, topic2wrong = None, None
    if config.custom_sampling:
        train_loader.dataset.shuffle(missing_list=missing_pairs, wrong_dict=topic2wrong, max_wrong=config.max_wrong)

    best_score = 0
    for epoch in range(1, config.epochs + 1):
        print('\n{}[Epoch: {}]{}'.format(30*'-', epoch, 30*'-'))
        train_loss = train(config, model, dataloader=train_loader, loss_function=loss_function,
                          optimizer=optimizer, scheduler=scheduler, scaler=scaler, teacher=teacher)
        print('Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}'.format(epoch, train_loss, optimizer.param_groups[0]['lr']))

        print('\n{}[{}]{}'.format(30*'-', 'Evaluate (Val)', 30*'-'))
        f2, precision, recall = evaluate_val(config, model, reference_dataloader=val_loader_content,
                                             query_dataloader=val_loader_topic, gt_dict=gt_dict, cleanup=True)

        if f2 > best_score:
            best_score = f2
            best_checkpoint = '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, f2)
            torch.save(model.state_dict() if not (torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1)
                       else model.module.state_dict(), best_checkpoint)
            print(f'  *** New best: {best_score:.4f} ***')

        # sim_sample
        if config.sim_sample and epoch >= config.sim_sample_start:
            print('\n{}[{}]{}'.format(30*'-', 'Evaluate (Train)', 30*'-'))
            if len(config.pool) > 1:
                if epoch < config.epoch_stop_switching:
                    next_pool = 0 if epoch % 2 == 0 else config.pool[1:][epoch % len(config.pool[1:])]
                    train_loader_content.dataset.set_pool(next_pool)
                    train_loader_topic.dataset.set_pool(next_pool)
                    train_loader.dataset.set_pool(next_pool)
                else:
                    train_loader_content.dataset.set_pool(0)
                    train_loader_topic.dataset.set_pool(0)
                    train_loader.dataset.set_pool(0)
            missing_pairs, topic2wrong = evaluate_train(config=config, model=model,
                                                       reference_dataloader=train_loader_content,
                                                       query_dataloader=train_loader_topic, gt_dict=gt_dict,
                                                       content2topic=train_loader.dataset.content2topic, cleanup=True)

        if config.custom_sampling:
            train_loader.dataset.shuffle(missing_list=missing_pairs, wrong_dict=topic2wrong, max_wrong=config.max_wrong)

    print(f'\n{"="*60}\nBest F2: {best_score:.4f}\n{"="*60}')
