"""Recompute flops_ratio for all prune_*.json with the corrected n_text.
Accuracy values are unaffected; only the analytical FLOPs ratios are fixed."""
import os, json, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prune_eval import flops_ratio, keep_sequence, N_TEXT, RATES
from measure import load_model  # noqa: F401  (ensures import path works)
from common import DEV, ROOT, set_seed
import torch

# reuse the exact loader from measure.py
def get_model(dataset, d, depth, seed, tag=''):
    return load_model(d, depth, dataset, seed, tag)


def main():
    for p in sorted(glob.glob(os.path.join(ROOT, 'results', 'prune_*_s*.json'))):
        if '_smoke' in p:
            continue
        with open(p) as f:
            data = json.load(f)
        c = data['config']
        set_seed(c['seed'])
        model = get_model(c['dataset'], c['d'], c['depth'], c['seed'])
        K, L = model.n_vis, model.depth
        # info profile for hap_info reconstruction
        mp = os.path.join(ROOT, 'results',
                          f"measure_{c['dataset']}_d{c['d']}_L{c['depth']}_s{c['seed']}.json")
        info = np.array(json.load(open(mp))['info_mean'])
        norm = info / max(info.max(), 1e-12)
        n_fixed = 0
        for r in data['results']:
            lp = r['lp']; rate = r['rate']
            if r['method'] in ('hap_info',):
                m_seq = []
                m = K
                for l in range(lp, L):
                    rate_l = rate * float(np.clip(1.0 - norm[l], 0.0, 1.0))
                    m = max(1, int(round(m * (1 - rate_l))))
                    m_seq.append(m)
            else:
                m_seq = keep_sequence(K, lp, L, rate) if rate > 0 else [K] * (L - lp)
            new_fr = flops_ratio(model, lp, m_seq, K)
            if abs(new_fr - r['flops_ratio']) > 1e-6:
                n_fixed += 1
            r['flops_ratio'] = new_fr
        with open(p, 'w') as f:
            json.dump(data, f)
        print(f"fixed {n_fixed}/{len(data['results'])} entries in {os.path.basename(p)}")


if __name__ == '__main__':
    main()
