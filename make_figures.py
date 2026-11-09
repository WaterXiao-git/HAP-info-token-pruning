"""Generate all paper figures from results/*.json. Journal-quality styling."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.6,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
C = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377', '#66CCEE']
SCALES = [32, 48, 64]
LABEL_D = {32: 'W-32 (130K)', 48: 'W-48 (288K)', 64: 'W-64 (509K)'}
PARAMS = {32: 0.132, 48: 0.288, 64: 0.509}
LABEL_M = {'random': 'Random', 'attention': 'Attention', 'diversity': 'Diversity',
           'hap': 'HAP (ours)', 'hap_info': 'HAP-info (ours)', 'full': 'Full'}
DS_TITLE = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}


def jl(p):
    with open(p) as f:
        return json.load(f)


def fig_info_curves():
    """Layer-wise mean token info for 3 scales, both datasets; horizon markers
    only when a horizon within the network was detected."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, ds in zip(axes, ['cifar10', 'cifar100']):
        for i, d in enumerate(SCALES):
            p = os.path.join(RES, f'measure_{ds}_d{d}_L10_s0.json')
            if not os.path.exists(p):
                continue
            r = jl(p)
            info = np.array(r['info_mean']); info = info / info.max()
            L = len(info)
            ax.plot(np.arange(1, L + 1), info, '-o', color=C[i], ms=3,
                    label=LABEL_D[d])
            if r['horizon'] < L:   # within-network horizon
                ax.axvline(r['horizon'] + 0.5, color=C[i], ls=':', lw=0.9, alpha=0.7)
        ax.set_yscale('symlog', linthresh=0.02)
        ax.set_xlabel('Layer $\\ell$')
        ax.set_title(DS_TITLE[ds])
        ax.axhline(0.05, color='gray', lw=0.8, ls='--')
        ax.text(9.6, 0.056, '$\\tau$', fontsize=8, color='gray', ha='right')
        ax.set_xlim(0.8, 10.2)
    axes[0].set_ylabel('Normalized token info $\\bar{I}_\\ell$')
    axes[0].legend(frameon=False, loc='lower right')
    fig.savefig(os.path.join(FIG, 'fig_info_curves.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_info_curves.png'))
    plt.close(fig)


def fig_mechanism():
    """Homogenization vs normalized info (mechanism panel), d=64 both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, ds in zip(axes, ['cifar10', 'cifar100']):
        p = os.path.join(RES, f'measure_{ds}_d64_L10_s0.json')
        if not os.path.exists(p):
            continue
        r = jl(p)
        info = np.array(r['info_mean']); info /= info.max()
        hom = np.array(r['homog'])
        hom_perm = np.array(r['homog_perm_shuffled'])
        L = np.arange(1, len(info) + 1)
        ax.plot(L, info, '-o', color=C[0], ms=3, label='Token info (norm.)')
        ax.plot(L, hom, '-s', color=C[1], ms=3, label='Homogenization')
        ax.plot(L, hom_perm, '--^', color=C[1], ms=3, alpha=0.65,
                label='Homog. (content-shuffled ctrl)')
        ax.set_xlabel('Layer $\\ell$')
        ax.set_title(DS_TITLE[ds])
    axes[0].set_ylabel('Value')
    axes[0].legend(frameon=False, loc='lower right', fontsize=7)
    fig.savefig(os.path.join(FIG, 'fig_mechanism.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_mechanism.png'))
    plt.close(fig)


def fig_capacity():
    """Capacity scaling: final-layer token info (bits) and exponential growth rate b."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ds, mk in [('cifar10', '-o'), ('cifar100', '-s')]:
        fin, b, ps = [], [], []
        for d in SCALES:
            p = os.path.join(RES, f'measure_{ds}_d{d}_L10_s0.json')
            if not os.path.exists(p):
                continue
            r = jl(p)
            info = np.array(r['info_mean'])
            fin.append(info[-1])
            b.append(r['decay_rate'])
            ps.append(PARAMS[d])
        col = C[0] if ds == 'cifar10' else C[1]
        axes[0].plot(ps, fin, mk, color=col, label=DS_TITLE[ds])
        axes[1].plot(ps, b, mk, color=col, label=DS_TITLE[ds])
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('Parameters')
        ax.set_xticks(list(PARAMS.values()))
        ax.set_xticklabels(['130K', '290K', '509K'])
        ax.minorticks_off()
    axes[0].set_ylabel('Final-layer info $\\bar{I}_L$ (bits)')
    axes[1].set_ylabel('Growth rate $b$')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_capacity.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_capacity.png'))
    plt.close(fig)


def fig_pruning():
    """Accuracy vs FLOPs ratio for all methods with seed error bars (d=64)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, ds in zip(axes, ['cifar10', 'cifar100']):
        data = {}
        for s in [0, 1, 2]:
            p = os.path.join(RES, f'prune_{ds}_d64_L10_s{s}.json')
            if not os.path.exists(p):
                continue
            for r in jl(p)['results']:
                key = (r['method'], round(r['flops_ratio'], 3))
                data.setdefault(key, []).append(r['acc'])
        fulls = [a for (m, _), v in data.items() if m == 'full' for a in v]
        if fulls:
            ax.axhline(np.mean(fulls), color='gray', lw=0.9, ls='--')
            ax.text(0.025, np.mean(fulls) + 0.006, 'full model', fontsize=7.5,
                    ha='left', color='gray', transform=ax.get_yaxis_transform())
        for mi, m in enumerate(['random', 'attention', 'diversity', 'hap', 'hap_info']):
            frs, accs, errs = [], [], []
            rates = sorted({k[1] for k in data if k[0] == m})
            for fr in rates:
                acc = data.get((m, fr))
                if acc:
                    frs.append(fr); accs.append(np.mean(acc))
                    errs.append(np.std(acc) if len(acc) > 1 else 0)
            if frs:
                ax.errorbar(frs, accs, yerr=errs,
                            fmt='-^' if 'hap' in m else '-o',
                            color=C[mi], ms=3.5, capsize=2, lw=1.4,
                            label=LABEL_M[m], zorder=5 if 'hap' in m else 3)
        ax.set_xlabel('Compute (fraction of full FLOPs)')
        ax.set_ylabel('Test accuracy')
        ax.set_title(DS_TITLE[ds])
        ax.set_xlim(0.15, 1.05)
    axes[0].legend(frameon=False, loc='lower right')
    fig.savefig(os.path.join(FIG, 'fig_pruning.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_pruning.png'))
    plt.close(fig)


def fig_startlayer():
    """Accuracy vs pruning start layer (lp=1 vs lp=L-1) at fixed rate 0.5, d=64.
    Shows the dissociation: low shallow-layer LOO info but catastrophic early pruning."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    methods = ['random', 'attention', 'diversity']
    width = 0.26
    xs = np.arange(2)
    for mi, m in enumerate(methods):
        vals = []
        for lp_mode in ['layer1', 'horizon']:
            accs = []
            for s in [0, 1, 2]:
                p = os.path.join(RES, f'prune_cifar10_d64_L10_s{s}.json')
                if not os.path.exists(p):
                    continue
                for r in jl(p)['results']:
                    if r['method'] == m and r['lp_mode'] == lp_mode and r['rate'] == 0.5:
                        accs.append(r['acc'])
            vals.append(np.mean(accs) if accs else np.nan)
        ax.bar(xs + (mi - 1) * width, vals, width * 0.9, color=C[mi], label=LABEL_M[m])
    ax.set_xticks(xs)
    ax.set_xticklabels(['$\\ell_p=1$ (early)', '$\\ell_p=L{-}1$ (late)'])
    ax.set_ylabel('Test accuracy (rate = 0.5)')
    ax.set_title('CIFAR-10, W-64')
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG, 'fig_startlayer.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_startlayer.png'))
    plt.close(fig)


def fig_controls():
    """Position/attention correlation analysis (critique panel), averaged over
    d=64 seeds where available."""
    p = os.path.join(RES, 'measure_cifar10_d64_L10_s0.json')
    if not os.path.exists(p):
        return
    r = jl(p)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    L = np.arange(1, r['layers'] + 1)
    pos = [np.array(jl(os.path.join(RES, f'measure_cifar10_d64_L10_s{s}.json'))['info_pos_corr'])
           for s in [0, 1, 2] if os.path.exists(os.path.join(RES, f'measure_cifar10_d64_L10_s{s}.json'))]
    att = [np.array(jl(os.path.join(RES, f'measure_cifar10_d64_L10_s{s}.json'))['info_attn_corr'])
           for s in [0, 1, 2] if os.path.exists(os.path.join(RES, f'measure_cifar10_d64_L10_s{s}.json'))]
    ax.plot(L, np.mean(pos, 0), '-o', color=C[0], ms=3, label='info vs. position')
    ax.fill_between(L, np.mean(pos, 0) - np.std(pos, 0), np.mean(pos, 0) + np.std(pos, 0),
                    color=C[0], alpha=0.15, lw=0)
    ax.plot(L, np.mean(att, 0), '-s', color=C[1], ms=3, label='info vs. attention')
    ax.fill_between(L, np.mean(att, 0) - np.std(att, 0), np.mean(att, 0) + np.std(att, 0),
                    color=C[1], alpha=0.15, lw=0)
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_xlabel('Layer $\\ell$'); ax.set_ylabel('Pearson $r$')
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False, loc='lower left')
    fig.savefig(os.path.join(FIG, 'fig_controls.pdf'))
    fig.savefig(os.path.join(FIG, 'fig_controls.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig_info_curves(); fig_mechanism(); fig_capacity(); fig_pruning()
    fig_startlayer(); fig_controls()
    print('figures written to', FIG)
