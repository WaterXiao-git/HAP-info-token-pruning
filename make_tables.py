"""Generate LaTeX tables from results/*.json."""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')


def jl(p):
    with open(p) as f:
        return json.load(f)


def table_models():
    """Model configurations and accuracies."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, 'train_vlm_*_s0.json'))):
        if '_smoke' in p:
            continue
        r = jl(p)
        c = r['config']
        if c['seed'] != 0:
            continue
        rows.append((c['dataset'], c['d'], r['n_params'], r['best_acc']))
    order = {'cifar10': 0, 'cifar100': 1}
    rows.sort(key=lambda x: (order[x[0]], x[1]))
    lines = [
        '\\begin{table}[t]\\centering',
        '\\caption{Compact VLM configurations. Accuracy is best test accuracy over training.}',
        '\\label{tab:models}',
        '\\small',
        '\\begin{tabular}{llrr}\\toprule',
        'Dataset & Width $d$ & Params & Test acc. \\\\ \\midrule',
    ]
    for ds, d, np_, acc in rows:
        lines.append(f"{'CIFAR-10' if ds=='cifar10' else 'CIFAR-100'} & {d} & "
                     f"{np_/1000:.0f}K & {acc:.3f} \\\\")
    lines += ['\\bottomrule\\end{tabular}\\end{table}']
    return '\n'.join(lines)


def table_horizons():
    """Layer-wise info profile summary across scales (seed 0)."""
    lines = [
        '\\begin{table}[t]\\centering',
        '\\caption{Layer-wise leave-one-out token information across widths '
        '(depth $L{=}10$, $\\tau{=}0.05$). $\\ell^*$: information horizon (first layer from '
        'which normalized info stays below $\\tau$; $L$ = no within-network horizon). '
        '$b$: fitted exponential rate of the normalized info curve. $r_{\\text{cnt}}$, '
        '$r_{\\text{pos}}$: content- and position-shuffled control correlations with token info.}',
        '\\label{tab:horizons}',
        '\\small',
        '\\begin{tabular}{lcccccccc}\\toprule',
        ' & \\multicolumn{4}{c}{CIFAR-10} & \\multicolumn{4}{c}{CIFAR-100} \\\\',
        'Width $d$ & $\\ell^*$ & $b$ & $r_{\\text{cnt}}$ & $r_{\\text{pos}}$ & '
        '$\\ell^*$ & $b$ & $r_{\\text{cnt}}$ & $r_{\\text{pos}}$ \\\\ \\midrule',
    ]
    for d in [32, 48, 64]:
        row = [str(d)]
        for ds in ['cifar10', 'cifar100']:
            p = os.path.join(RES, f'measure_{ds}_d{d}_L10_s0.json')
            if os.path.exists(p):
                r = jl(p)
                row += [str(r['horizon']), f"{r['decay_rate']:.3f}",
                        f"{r['content_corr_shuffled']:.2f}", f"{r['position_corr_shuffled']:.2f}"]
            else:
                row += ['--', '--', '--', '--']
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule\\end{tabular}\\end{table}']
    return '\n'.join(lines)


def table_pruning():
    """Main pruning table: accuracy (mean+-std over seeds) at rates 0.5 and 0.9,
    for early (lp=1) and late (lp=horizon) pruning, both datasets, d=64."""
    RATES_SHOWN = [0.5, 0.9]
    METHODS = ['random', 'attention', 'diversity', 'hap', 'hap_info']

    def collect(ds):
        data = {}
        for s in [0, 1, 2]:
            p = os.path.join(RES, f'prune_{ds}_d64_L10_s{s}.json')
            if not os.path.exists(p):
                continue
            for r in jl(p)['results']:
                key = (r['method'], r['lp_mode'], round(r['rate'], 2))
                data.setdefault(key, []).append(r['acc'])
        return data

    full = {}
    for ds in ['cifar10', 'cifar100']:
        accs = []
        for s in [0, 1, 2]:
            p = os.path.join(RES, f'prune_{ds}_d64_L10_s{s}.json')
            if os.path.exists(p):
                accs += [r['acc'] for r in jl(p)['results'] if r['method'] == 'full']
        if accs:
            full[ds] = (np.mean(accs), np.std(accs))

    def cell(data, ds, m, mode, rate):
        v = data.get((m, mode, rate))
        if not v:
            return '--'
        if len(v) > 1:
            return f"{np.mean(v):.3f}$\\pm${np.std(v):.3f}"
        return f"{np.mean(v):.3f}"

    def _full_acc(ds):
        v = full.get(ds)
        return v[0] if v else float('nan')

    f10 = _full_acc('cifar10'); f100 = _full_acc('cifar100')
    lines = [
        '\\begin{table}[t]\\centering',
        '\\caption{Test accuracy at nominal per-layer pruning rates (width $d{=}64$, '
        'mean$\\pm$std over 3 seeds). Early: progressive pruning from layer 1; '
        'Late: from the effective horizon $\\ell_p{=}9$. HAP-info modulates per-layer '
        'rates by Eq.~\\eqref{eq:hapinfo}, so its realized compute differs; '
        'Fig.~\\ref{fig:pruning} compares at matched compute. Full-model reference: '
        f"CIFAR-10 {f10:.3f}, "
        f"CIFAR-100 {f100:.3f}.}}",
        '\\label{tab:pruning}', '\\small',
        '\\resizebox{\\linewidth}{!}{%',
        '\\begin{tabular}{lcccc|cccc}\\toprule',
        ' & \\multicolumn{4}{c|}{CIFAR-10} & \\multicolumn{4}{c}{CIFAR-100} \\\\',
        ' & \\multicolumn{2}{c}{Early} & \\multicolumn{2}{c|}{Late} & '
        '\\multicolumn{2}{c}{Early} & \\multicolumn{2}{c}{Late} \\\\',
        'Method & r=.5 & r=.9 & r=.5 & r=.9 & r=.5 & r=.9 & r=.5 & r=.9 \\\\ \\midrule',
    ]
    data10 = collect('cifar10'); data100 = collect('cifar100')
    for m in METHODS:
        row = [LABEL(m)]
        for data in [data10, data100]:
            for mode in ['layer1', 'horizon']:
                for rate in RATES_SHOWN:
                    row.append(cell(data, None, m, mode, rate))
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule\\end{tabular}}\\end{table}']
    return '\n'.join(lines)


def LABEL(m):
    return {'random': 'Random', 'attention': 'Attention', 'diversity': 'Diversity',
            'hap': 'HAP (ours)', 'hap_info': 'HAP-info (ours)'}[m]


def main():
    names = ['tab_models.tex', 'tab_horizons.tex', 'tab_pruning.tex']
    parts = [table_models(), table_horizons(), table_pruning()]
    for name, part in zip(names, parts):
        if part:
            with open(os.path.join(ROOT, name), 'w') as f:
                f.write(part)
    print('tables written:', [n for n, p in zip(names, parts) if p])


if __name__ == '__main__':
    main()
