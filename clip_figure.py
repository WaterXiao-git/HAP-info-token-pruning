"""Figure for the real-VLM (CLIP ViT-B/32) validation. Same styling as make_figures."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.6,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results_clip')
FIG = os.path.join(ROOT, 'figs')
C = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377', '#66CCEE']
LBL = {'random': 'Random', 'attention': 'Attention', 'diversity': 'Diversity',
       'hap': 'HAP (ours)', 'hap_info': 'HAP-info (ours)'}

prof = json.load(open(os.path.join(RES, 'clip_profile_cifar10.json')))
pru = json.load(open(os.path.join(RES, 'clip_prune_cifar10.json')))['results']
full = [r for r in pru if r['method'] == 'full'][0]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
# (a) LOO profile
info = np.array(prof['info_mean'])
L = np.arange(1, len(info) + 1)
axes[0].plot(L, info, '-o', color=C[0], ms=3.5)
tau = 0.05 * info.max()
axes[0].axhline(tau, color='gray', lw=0.8, ls='--')
axes[0].text(0.6, tau + 0.004, '$\\tau\\,\\bar I_{\\max}$', fontsize=8, color='gray')
axes[0].set_xlabel('Layer $\\ell$')
axes[0].set_ylabel('Mean LOO token info $\\bar I_\\ell$ (bits)')
axes[0].set_title('CLIP ViT-B/32: no horizon')
axes[0].set_xticks(L)
axes[0].set_ylim(0, 0.28)
# (b) accuracy vs FLOPs
marks = {'lp1': '-o', 'lp6': '-s'}
for mi, m in enumerate(['random', 'attention', 'diversity', 'hap', 'hap_info']):
    for lp, mk in [(1, '-o'), (6, '-s')]:
        pts = sorted([(r['flops_ratio'], r['acc']) for r in pru
                      if r['method'] == m and r.get('lp') == lp])
        if not pts:
            continue
        fr = [p[0] for p in pts]; ac = [p[1] for p in pts]
        lab = LBL[m] + ('' if lp == 1 else ' ($\\ell_p{=}6$)')
        axes[1].plot(fr, ac, mk, color=C[mi], ms=3.5, lw=1.2, label=lab if lp == 1 else None)
axes[1].axhline(full['acc'], color='gray', lw=0.9, ls='--')
axes[1].text(0.15, full['acc'] + 0.012, 'full model (90.6%)', fontsize=7.5, color='gray')
axes[1].set_xlabel('Compute (fraction of vision-encoder FLOPs)')
axes[1].set_ylabel('Zero-shot accuracy')
axes[1].set_title('CLIP zero-shot CIFAR-10')
axes[1].set_xlim(0.08, 0.65)
axes[1].set_ylim(0, 1.0)
axes[1].legend(frameon=False, loc='lower right', fontsize=6.5, ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_clip.pdf'))
fig.savefig(os.path.join(FIG, 'fig_clip.png'))
print('fig_clip written')
