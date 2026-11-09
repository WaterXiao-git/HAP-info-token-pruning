"""Train a CompactVLM on CIFAR-10/100. Usage: python train.py --d 64 --dataset cifar10 --seed 0"""
import argparse, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEV, ROOT, load_cifar, build_vocab, ImageTextDS, CompactVLM,
                    set_seed, evaluate, jsave, C10_CLASSES, C100_CLASSES, PROMPT)


def get_class_names(dataset):
    return C10_CLASSES if dataset == 'cifar10' else C100_CLASSES


def run(d=64, depth=10, heads=4, dataset='cifar10', seed=0, epochs=15, bs=512, lr=3e-4, tag=''):
    set_seed(seed)
    tr_x, tr_y, te_x, te_y = load_cifar(dataset)
    names = get_class_names(dataset)
    vocab, w2i, tokenize = build_vocab(names)
    n_text = 1 + len(PROMPT.split())  # <cls> + 6 prompt words

    tr_ds = ImageTextDS(tr_x, tr_y, PROMPT)
    te_ds = ImageTextDS(te_x, te_y, PROMPT)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)

    model = CompactVLM(d=d, depth=depth, heads=heads, n_cls=len(names),
                       vocab_size=len(vocab), n_text=n_text, n_vis=64, dropout=0.0).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params_total = sum(p.numel() for p in model.parameters())

    best_acc, hist = 0., []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot_loss, n = 0., 0
        for imgs, labels, prompt in tr_ld:
            seq = torch.tensor([tokenize(PROMPT)] * imgs.shape[0], device=DEV)
            out = model.forward_full(seq, imgs.to(DEV))
            loss = F.cross_entropy(out['logits'], labels.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(labels); n += len(labels)
        sched.step()
        acc = evaluate(model, te_ld, tokenize, n_text)
        hist.append({'epoch': ep, 'loss': tot_loss / n, 'acc': acc})
        best_acc = max(best_acc, acc)
        print(f'[d={d} {dataset} s{seed}] ep{ep} loss {tot_loss/n:.4f} acc {acc:.4f}', flush=True)

    name = f'vlm_{dataset}_d{d}_L{depth}_s{seed}{tag}'
    torch.save({'model': model.state_dict(),
                'cfg': {'d': d, 'depth': depth, 'heads': heads, 'n_cls': len(names),
                        'vocab_size': len(vocab), 'n_text': n_text, 'n_vis': 64, 'patch': 4}},
               os.path.join(ROOT, 'ckpts', name + '.pt'))
    jsave(os.path.join(ROOT, 'results', f'train_{name}.json'),
          {'config': {'d': d, 'depth': depth, 'heads': heads, 'dataset': dataset, 'seed': seed,
                      'epochs': epochs, 'bs': bs, 'lr': lr},
           'n_params': n_params_total, 'best_acc': best_acc, 'final_acc': hist[-1]['acc'],
           'history': hist, 'train_minutes': (time.time() - t0) / 60})
    print(f'DONE {name}: params={n_params_total} best_acc={best_acc:.4f} '
          f'({(time.time()-t0)/60:.1f} min)', flush=True)
    return best_acc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dataset', type=str, default='cifar10')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--tag', type=str, default='')
    a = ap.parse_args()
    run(a.d, a.depth, a.heads, a.dataset, a.seed, a.epochs, tag=a.tag)
