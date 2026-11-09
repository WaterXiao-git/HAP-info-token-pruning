"""
Token pruning evaluation: baselines (random / attention / diversity) and HAP
(Horizon-Aware Pruning, learned per-layer token scorer trained by distillation,
applied progressively from the measured information horizon). Also HAP-info:
horizon-guided per-layer budget reallocation.
Usage: python prune_eval.py --dataset cifar10 --seed 0
"""
import argparse, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEV, ROOT, load_cifar, build_vocab, ImageTextDS, CompactVLM,
                    set_seed, jsave, PROMPT, model_flops)
from train import get_class_names
from measure import load_model

RATES = [0.25, 0.50, 0.75, 0.90]
N_TEXT = 1 + len(PROMPT.split())


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def collect_scores(model, seq, imgs, sampler_seed=0):
    """One full forward; returns per-layer importance orderings in original token space.
    attn_order[l]: [B, K] argsort (descending) of attention mass received by visual
    tokens at block l.  states[l]: [B, K, d] visual states entering block l."""
    out = model.forward_full(seq, imgs, collect_states=True)
    attns, states = out['attns'], out['vis_states']
    B, K = states[0].shape[0], states[0].shape[1]
    attn_mass = torch.stack([a[:, :, :N_TEXT, N_TEXT:].sum(dim=(1, 2)) for a in attns], 1)  # [B,L,K]
    return states, attn_mass


@torch.no_grad()
def kcenter_order(states, m_max):
    """Greedy k-center ordering per sample. states [B,K,d]; returns [B, m_max] indices."""
    B, K, d = states.shape
    s = F.normalize(states, dim=-1)
    dist = torch.bmm(s, s.transpose(1, 2))          # cosine sim
    dist = (1 - dist).clamp(min=0)                   # distance
    chosen = torch.zeros(B, m_max, dtype=torch.long, device=DEV)
    min_d = torch.full((B, K), 1e9, device=DEV)
    # first: token with max distance to the mean token (deterministic-ish start)
    mean_s = s.mean(1, keepdim=True)
    d0 = ((s - mean_s) ** 2).sum(-1)
    chosen[:, 0] = d0.argmax(-1)
    mask = torch.ones(B, K, device=DEV); mask[torch.arange(B), chosen[:, 0]] = 0
    for i in range(1, m_max):
        c = chosen[:, i - 1]
        min_d = torch.minimum(min_d, dist[torch.arange(B), c])
        score = min_d + (1 - mask) * (-1e9)
        chosen[:, i] = score.argmax(-1)
        mask[torch.arange(B), chosen[:, i]] = 0
    return chosen


def keep_sequence(K, lp, L, rate):
    """Progressive keep counts entering layers lp..L-1 at per-layer prune rate."""
    seq, m = [], K
    for l in range(lp, L):
        m = max(1, int(round(m * (1 - rate))))
        seq.append(m)
    return seq


def nested_spec(order_full, m_seq, lp, L, K):
    """order_full: numpy [B, K] per-sample ordering in ORIGINAL token space with
    nested prefixes (top-m subset of top-m' for m<m'). m_seq: keep counts entering
    layers lp..L-1. Returns prune_spec {layer: LongTensor [B, m]} whose indices are
    expressed in the CURRENT token space at that layer (as forward_full expects)."""
    B = order_full.shape[0]
    spec = {}
    cur = np.tile(np.arange(K, dtype=np.int64), (B, 1))  # original idx in current order
    rows = np.arange(B)[:, None]
    for li, l in enumerate(range(lp, L)):
        m = m_seq[li]
        keep_orig = order_full[:, :m].astype(np.int64)
        lookup = np.full((B, K), -1, dtype=np.int64)
        lookup[rows, cur] = np.arange(cur.shape[1], dtype=np.int64)[None, :]
        keep_pos = np.take_along_axis(lookup, keep_orig, axis=1)  # [B, m]
        spec[l] = torch.from_numpy(keep_pos).to(DEV)
        cur = keep_orig
    return spec


# ------------------------------------------------------------------ HAP gates
class GateBank(nn.Module):
    def __init__(self, d, L, lp):
        super().__init__()
        self.lp = lp
        self.gates = nn.ModuleList([nn.Linear(d, 1) for _ in range(L)])
        for g in self.gates:
            nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)

    def scores(self, l, vis):
        """vis: [B,K,d] -> raw scores [B,K] (higher = keep)."""
        return self.gates[l](vis).squeeze(-1)


def hap_train(model, gate_bank, tr_ld, tokenize, lp, rate_train, epochs=3, lr=1e-3,
              n_limit=12000, log=''):
    """Train gate_bank by distillation with progressive top-k pruning from layer lp."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(gate_bank.parameters(), lr=lr)
    K = model.n_vis
    L = model.depth
    seen, step = 0, 0
    for ep in range(epochs):
        for imgs, labels, prompt in tr_ld:
            imgs, labels = imgs.to(DEV), labels.to(DEV)
            B = imgs.shape[0]
            seq = torch.tensor([tokenize(PROMPT)] * B, device=DEV)
            with torch.no_grad():
                full = model.forward_full(seq, imgs)['logits']
            # manual gated progressive forward
            txt = model.tok(seq) + model.pos_text[:, :seq.shape[1]]
            vis = model.patch_embed(imgs).flatten(2).transpose(1, 2) + model.pos_vis
            m = K
            for l, blk in enumerate(model.blocks):
                if l >= lp:
                    sc = gate_bank.scores(l, vis)              # [B, K_cur]
                    m_new = max(1, int(round(m * (1 - rate_train))))
                    if m_new < m:
                        thr = sc.topk(m_new, dim=-1).values[:, -1:]
                        hard = (sc >= thr).float()
                        soft = torch.sigmoid(sc)
                        g = hard + soft - soft.detach()         # straight-through
                        vis = vis * g.unsqueeze(-1)
                        m = m_new
                ao = []
                x = torch.cat([txt, vis], dim=1)
                x = blk(x, attn_out=ao)
                txt, vis = x[:, :N_TEXT], x[:, N_TEXT:N_TEXT + vis.shape[1]]
            h = model.ln_f(txt[:, 0])
            logits = model.head(h)
            loss = F.cross_entropy(logits, labels) + \
                   F.kl_div(F.log_softmax(logits, -1), F.softmax(full, -1),
                            reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
            seen += B; step += 1
            if seen >= n_limit:
                break
        if seen >= n_limit:
            break
    return step


@torch.no_grad()
def hap_order(model, gate_bank, seq, imgs, lp):
    """Single nested ordering per sample (original token space): mean of z-scored
    per-layer gate scores over layers lp..L-1, computed on the unpruned model."""
    out = model.forward_full(seq, imgs, collect_states=True)
    states = out['vis_states']
    L, B, K = len(states), states[0].shape[0], states[0].shape[1]
    zs = []
    for l in range(lp, L):
        sc = gate_bank.scores(l, states[l])          # [B, K]
        sc = (sc - sc.mean(-1, keepdim=True)) / (sc.std(-1, keepdim=True) + 1e-6)
        zs.append(sc)
    order = torch.stack(zs, 0).mean(0).argsort(-1, descending=True)
    return order.cpu().numpy().astype(np.int64)


# ------------------------------------------------------------------ eval
@torch.no_grad()
def acc_with_prune(model, loader, tokenize, prune_builder, n_max=10000):
    model.eval()
    correct = total = 0
    for imgs, labels, prompt in loader:
        if total >= n_max:
            break
        B = imgs.shape[0]
        seq = torch.tensor([tokenize(PROMPT)] * B, device=DEV)
        spec = prune_builder(imgs, seq)
        out = model.forward_full(seq, imgs.to(DEV), prune_spec=spec)
        pred = out['logits'].argmax(-1).cpu()
        correct += (pred == labels).sum().item(); total += B
    return correct / total


def flops_ratio(model, lp, m_seq, K, n_text=N_TEXT):
    """Analytical FLOPs ratio vs full model for progressive pruning from lp."""
    full = model_flops(model, K, n_text=N_TEXT)
    d = model.d
    flops = model.patch ** 2 * 3 * d * (32 // model.patch) ** 2
    m = K
    ms = {}
    for i, l in enumerate(range(lp, model.depth)):
        ms[l] = m_seq[i]
    for l, blk in enumerate(range(model.depth)):
        T = n_text + (ms[l] if l in ms else K)
        flops += 3 * d * d * T + d * d * T + 2 * T * T * d + 8 * d * d * T
    flops += d * 10
    return flops / full


def run(dataset='cifar10', seed=0, d=64, depth=10, n_measure=500, tag=''):
    set_seed(seed)
    model = load_model(d, depth, dataset, seed, tag)
    names = get_class_names(dataset)
    vocab, w2i, tokenize = build_vocab(names)

    # horizon from measurement. horizon==L means "no permanent vanishing detected";
    # in that case we prune from the last layer (most conservative choice).
    meas = jload_safe(dataset, d, depth, seed, tag)
    horizon = meas['horizon'] if meas else max(1, depth // 2)

    tr_x, tr_y, te_x, te_y = load_cifar(dataset)
    tr_ds = ImageTextDS(tr_x, tr_y, PROMPT); te_ds = ImageTextDS(te_x, te_y, PROMPT)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)

    K, L = model.n_vis, model.depth
    horizon_eff = min(max(int(horizon), 1), L - 1)
    results = []

    def make_builder(method, lp, rate):
        m_seq = keep_sequence(K, lp, L, rate) if rate > 0 else None
        def builder(imgs, seq):
            imgs_d = imgs.to(DEV)
            B = imgs_d.shape[0]
            if rate == 0:
                return None
            if method == 'random':
                rng = np.random.RandomState(sampler_seed())
                order_full = np.stack([rng.permutation(K) for _ in range(B)]).astype(np.int64)
            elif method == 'attention':
                states, attn_mass = collect_scores(model, seq, imgs_d)
                # attention ranking measured at the first pruned layer; nested subsets
                order_full = attn_mass[:, lp].argsort(-1, descending=True
                                                      ).cpu().numpy().astype(np.int64)
            elif method == 'diversity':
                states, _ = collect_scores(model, seq, imgs_d)
                # k-center at the first pruned layer; prefix = nested subsets
                o = kcenter_order(states[lp], min(m_seq)).cpu().numpy()  # [B, m_min]
                is_chosen = np.zeros((B, K), dtype=bool)
                np.put_along_axis(is_chosen, o, True, axis=1)
                order_full = np.argsort(~is_chosen, axis=1, kind='stable').astype(np.int64)
            else:
                raise ValueError(method)
            return nested_spec(order_full, m_seq, lp, L, K)
        return builder

    # ---- HAP gate banks: one anchored at the (effective) horizon, one at layer 1
    banks = {'horizon': (GateBank(model.d, L, horizon_eff).to(DEV), horizon_eff),
             'layer1':  (GateBank(model.d, L, 1).to(DEV), 1)}
    for mode, (gb, lp0) in banks.items():
        hap_train(model, gb, tr_ld, tokenize, lp=lp0, rate_train=0.5,
                  log=f'{dataset} s{seed} {mode}')

    def sampler_seed():
        sampler_seed.s = getattr(sampler_seed, 's', 0) + 1
        return sampler_seed.s

    def hap_eval_acc(gb, lp, m_seq):
        correct = total = 0
        for imgs, labels, prompt in te_ld:
            imgs_d, B = imgs.to(DEV), imgs.shape[0]
            seq = torch.tensor([tokenize(PROMPT)] * B, device=DEV)
            order = hap_order(model, gb, seq, imgs_d, lp)  # [B, K] numpy
            spec = nested_spec(order, m_seq, lp, L, K)
            out = model.forward_full(seq, imgs_d, prune_spec=spec)
            pred = out['logits'].argmax(-1).cpu()
            correct += (pred == labels).sum().item(); total += B
        return correct / total

    # ---- baselines + HAP for both lp modes
    methods_bl = ['random', 'attention', 'diversity']
    for lp_mode in ['horizon', 'layer1']:
        lp = horizon_eff if lp_mode == 'horizon' else 1
        for method in methods_bl:
            for rate in RATES:
                acc = acc_with_prune(model, te_ld, tokenize, make_builder(method, lp, rate))
                fr = flops_ratio(model, lp, keep_sequence(K, lp, L, rate), K)
                results.append({'method': method, 'lp': lp, 'lp_mode': lp_mode,
                                'rate': rate, 'acc': acc, 'flops_ratio': fr})
                print(f'[{dataset} s{seed}] {method} lp={lp}({lp_mode}) r={rate} '
                      f'acc={acc:.4f} flops={fr:.3f}', flush=True)
        # full model reference
        results.append({'method': 'full', 'lp': lp, 'lp_mode': lp_mode, 'rate': 0.0,
                        'acc': acc_with_prune(model, te_ld, tokenize, make_builder('random', lp, 0)),
                        'flops_ratio': 1.0})

        # ---- HAP (learned ordering) + HAP-info (info-guided budget reallocation)
        gb = banks[lp_mode][0]
        for rate in RATES:
            m_seq = keep_sequence(K, lp, L, rate)
            acc = hap_eval_acc(gb, lp, m_seq)
            fr = flops_ratio(model, lp, m_seq, K)
            results.append({'method': 'hap', 'lp': lp, 'lp_mode': lp_mode,
                            'rate': rate, 'acc': acc, 'flops_ratio': fr})
            print(f'[{dataset} s{seed}] hap lp={lp}({lp_mode}) r={rate} '
                  f'acc={acc:.4f} flops={fr:.3f}', flush=True)

            if meas is not None:
                # info-modulated per-layer pruning rate: prune harder where token
                # info is low, softer where high. Monotone by construction.
                info = np.array(meas['info_mean'])
                norm = info / max(info.max(), 1e-12)
                m_seq_i = []
                m = K
                for l in range(lp, L):
                    rate_l = rate * float(np.clip(1.0 - norm[l], 0.0, 1.0))
                    m = max(1, int(round(m * (1 - rate_l))))
                    m_seq_i.append(m)
                acc_i = hap_eval_acc(gb, lp, m_seq_i)
                fr_i = flops_ratio(model, lp, m_seq_i, K)
                results.append({'method': 'hap_info', 'lp': lp, 'lp_mode': lp_mode,
                                'rate': rate, 'acc': acc_i, 'flops_ratio': fr_i})
                print(f'[{dataset} s{seed}] hap_info lp={lp}({lp_mode}) r={rate} '
                      f'acc={acc_i:.4f} flops={fr_i:.3f}', flush=True)

    out_name = f'prune_{dataset}_d{d}_L{depth}_s{seed}{tag}.json'
    jsave(os.path.join(ROOT, 'results', out_name),
          {'config': {'dataset': dataset, 'seed': seed, 'd': d, 'depth': depth,
                      'horizon': horizon}, 'results': results})
    print(f'DONE prune {out_name}', flush=True)


def jload_safe(dataset, d, depth, seed, tag):
    p = os.path.join(ROOT, 'results', f'measure_{dataset}_d{d}_L{depth}_s{seed}{tag}.json')
    if os.path.exists(p):
        return jload_local(p)
    return None


def jload_local(p):
    import json
    with open(p) as f:
        return json.load(f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='cifar10')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--tag', type=str, default='')
    a = ap.parse_args()
    run(a.dataset, a.seed, a.d, a.depth, tag=a.tag)
