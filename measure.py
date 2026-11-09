"""
Measure per-layer visual token information (leave-one-out protocol of Wang et al.
CVPR 2026), homogenization, attention mass, position controls, and estimate the
information horizon. Usage: python measure.py --d 64 --dataset cifar10 --seed 0
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEV, ROOT, load_cifar, build_vocab, ImageTextDS, CompactVLM,
                    set_seed, jsave, PROMPT)
from train import get_class_names

TAU = 0.05  # horizon threshold: normalized mean info below TAU => vanished


def load_model(d, depth, dataset, seed, tag=''):
    name = f'vlm_{dataset}_d{d}_L{depth}_s{seed}{tag}'
    ck = torch.load(os.path.join(ROOT, 'ckpts', name + '.pt'), map_location=DEV, weights_only=False)
    model = CompactVLM(**{k: v for k, v in ck['cfg'].items() if k != 'patch'}).to(DEV)
    model.load_state_dict(ck['model'])
    model.eval()
    return model


@torch.no_grad()
def measure(d=64, depth=10, dataset='cifar10', seed=0, n_samples=500, tag=''):
    set_seed(seed)
    model = load_model(d, depth, dataset, seed, tag)
    names = get_class_names(dataset)
    vocab, w2i, tokenize = build_vocab(names)
    n_text = 1 + len(PROMPT.split())
    K = model.n_vis
    L = model.depth

    _, _, te_x, te_y = load_cifar(dataset)
    # balanced subset of test samples
    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(len(names)):
        idxs.append(rng.choice(np.where(te_y == c)[0], n_samples // len(names), replace=False))
    idxs = np.concatenate(idxs); rng.shuffle(idxs)
    imgs = torch.tensor(te_x[idxs]).permute(0, 3, 1, 2).contiguous().to(DEV)
    labels = torch.tensor(te_y[idxs]).to(DEV)
    seq = torch.tensor([tokenize(PROMPT)] * len(idxs), device=DEV)
    B = len(idxs)

    # ---- text-only baseline probability
    out = model.forward_full(seq, imgs, prune_spec={0: torch.zeros(B, 0, dtype=torch.long, device=DEV)})
    p_text = out['logits'].softmax(-1)[torch.arange(B), labels]  # [B]

    # ---- full forward with collected states/attention
    out = model.forward_full(seq, imgs, collect_states=True)
    attns = out['attns']          # list of [B, heads, T, T]
    vis_states = out['vis_states']  # list of [B, K, d]

    # attention mass received by each visual token (summed over text-token queries)
    attn_mass = torch.stack([a[:, :, :n_text, n_text:].sum(dim=(1, 2)) for a in attns], 1)  # [B, L, K]

    # homogenization: mean pairwise cosine similarity among visual tokens at layer input
    homog = []
    for s in vis_states:
        s = F.normalize(s, dim=-1)
        sim = torch.bmm(s, s.transpose(1, 2))
        n = s.shape[1]
        off = sim.sum(dim=(1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)
        homog.append((off / (n * (n - 1))).cpu())
    homog = torch.stack(homog, 1)  # [B, L]

    # token norm per layer
    toknorm = torch.stack([s.norm(dim=-1).mean(-1) for s in vis_states], 1)  # [B, L]

    # ---- leave-one-out information per layer
    info_mean = np.zeros(L); info_std_tok = np.zeros(L)
    info_profile = np.zeros((L, K))       # averaged over samples: per-position info
    info_attn_corr = np.zeros(L); info_pos_corr = np.zeros(L)
    info_normcorr_attn = np.zeros(L)

    ar = torch.arange(B, device=DEV)
    for l in range(L):
        # batch: for each token k, keep only k at layer l -> B*K forwards in chunks
        probs = torch.zeros(B, K, device=DEV)
        chunk = max(1, (2 ** 17) // (B * max(1, 1)))
        for k0 in range(0, K, 8):
            ks = list(range(k0, min(k0 + 8, K)))
            keep = torch.tensor([[ [k] ] * B for k in ks], device=DEV)  # [8, B, 1]
            # flatten: batch dim = 8*B
            seqs = seq.repeat(len(ks), 1)
            ims = imgs.repeat(len(ks), 1, 1, 1)
            spec = {l: keep.reshape(len(ks) * B, 1)}
            o = model.forward_full(seqs, ims, prune_spec=spec)
            pr = o['logits'].softmax(-1)[ar.repeat(len(ks)), labels.repeat(len(ks))]
            probs[:, ks] = pr.reshape(len(ks), B).T
        info = (probs - p_text[:, None]).cpu().numpy()  # [B, K]
        info_mean[l] = info.mean()
        info_std_tok[l] = info.std(axis=1).mean()      # spatial variance of info per sample
        prof = info.mean(axis=0)
        info_profile[l] = prof
        am = attn_mass[:, l].mean(0).cpu().numpy()
        if np.std(prof) > 1e-9 and np.std(am) > 1e-9:
            info_attn_corr[l] = np.corrcoef(prof, am)[0, 1]
        info_pos_corr[l] = np.corrcoef(prof, np.arange(K))[0, 1] if np.std(prof) > 1e-9 else 0.

    # ---- horizon estimation (smoothed)
    def smooth(x, w=3):
        k = np.ones(w) / w
        return np.convolve(x, k, mode='same')
    norm = info_mean / max(info_mean.max(), 1e-12)
    sm = smooth(norm)
    horizon = L
    for l in range(L):
        if all(sm[l2] <= TAU for l2 in range(l, L)):
            horizon = l
            break
    # exponential decay fit on normalized curve
    ls = np.arange(1, L + 1)
    yy = np.clip(sm, 1e-6, None)
    bfit = np.polyfit(ls, np.log(yy), 1)[0] if np.all(np.isfinite(np.log(yy))) else None

    # ---- position-shuffle control: permute patch order at input, redo a cheap version
    # (layer-wise mean info with shuffled positions) on a smaller subset
    n_sub = 128
    imgs_s = imgs[:n_sub].clone()
    perm = torch.randperm(K, device=DEV)
    # permute by re-assigning patch embeddings: easiest is permuting images patches
    # (each patch is a distinct image crop). We permute the patch grid content.
    imgs_perm = permute_patches(imgs_s, perm)
    out_s = model.forward_full(seq[:n_sub], imgs_perm, collect_states=True)
    vs = out_s['vis_states']
    homog_perm = []
    for s in vs:
        s = F.normalize(s, dim=-1)
        sim = torch.bmm(s, s.transpose(1, 2))
        n = s.shape[1]
        off = sim.sum(dim=(1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)
        homog_perm.append((off / (n * (n - 1))).mean().item())
    # per-position info under permutation at the layer of max info contrast
    l_star = int(np.argmax(info_std_tok))
    prof_perm = loo_profile(model, seq[:n_sub], imgs_perm, labels[:n_sub], p_text[:n_sub], l_star, K)
    prof_orig = loo_profile(model, seq[:n_sub], imgs[:n_sub], labels[:n_sub], p_text[:n_sub], l_star, K)
    inv_perm = torch.argsort(perm).cpu().numpy()
    prof_realigned = prof_perm[inv_perm]  # undo permutation: content-aligned profile
    r_content = np.corrcoef(prof_orig, prof_realigned)[0, 1] if np.std(prof_realigned) > 1e-9 else 0.
    r_position = np.corrcoef(prof_orig, prof_perm)[0, 1] if np.std(prof_perm) > 1e-9 else 0.

    res = {
        'config': {'d': d, 'depth': depth, 'dataset': dataset, 'seed': seed,
                   'n_samples': int(B), 'tau': TAU},
        'layers': L, 'K': K,
        'info_mean': info_mean.tolist(),
        'info_std_tok': info_std_tok.tolist(),
        'info_profile': info_profile.tolist(),
        'info_attn_corr': info_attn_corr.tolist(),
        'info_pos_corr': info_pos_corr.tolist(),
        'attn_mass': attn_mass.mean(0).cpu().numpy().tolist(),
        'homog': homog.mean(0).cpu().numpy().tolist(),
        'homog_perm_shuffled': homog_perm,
        'toknorm': toknorm.mean(0).cpu().numpy().tolist(),
        'horizon': int(horizon),
        'decay_rate': float(bfit) if bfit is not None else None,
        'info_attn_corr_at_lstar': float(info_attn_corr[l_star]),
        'content_corr_shuffled': float(r_content),
        'position_corr_shuffled': float(r_position),
        'acc_subset': float((out['logits'].argmax(-1) == labels).float().mean().item()),
    }
    out_name = f'measure_{dataset}_d{d}_L{depth}_s{seed}{tag}.json'
    jsave(os.path.join(ROOT, 'results', out_name), res)
    print(f"DONE measure {out_name}: horizon={horizon}/{L} decay={res['decay_rate']:.3f} "
          f"content_corr={r_content:.3f} pos_corr={r_position:.3f} info_attn_corr@l*="
          f"{info_attn_corr[l_star]:.3f}", flush=True)
    return res


@torch.no_grad()
def loo_profile(model, seq, imgs, labels, p_text, layer, K, chunk=8):
    B = seq.shape[0]
    ar = torch.arange(B, device=DEV)
    probs = torch.zeros(B, K, device=DEV)
    for k0 in range(0, K, chunk):
        ks = list(range(k0, min(k0 + chunk, K)))
        keep = torch.tensor([[[k]] * B for k in ks], device=DEV)
        spec = {layer: keep.reshape(len(ks) * B, 1)}
        o = model.forward_full(seq.repeat(len(ks), 1), imgs.repeat(len(ks), 1, 1, 1), prune_spec=spec)
        pr = o['logits'].softmax(-1)[ar.repeat(len(ks)), labels.repeat(len(ks))]
        probs[:, ks] = pr.reshape(len(ks), B).T
    return (probs - p_text[:, None]).mean(0).cpu().numpy()


def permute_patches(imgs, perm):
    """Permute the 64 patch contents (8x8 grid of 4x4 patches) of images [B,3,32,32]."""
    B, C, H, W = imgs.shape
    p = 4
    patches = imgs.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, 64, C * p * p)
    patches = patches[:, perm]
    imgs2 = patches.reshape(B, 8, 8, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return imgs2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--dataset', type=str, default='cifar10')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='')
    a = ap.parse_args()
    measure(a.d, a.depth, a.dataset, a.seed, tag=a.tag)
