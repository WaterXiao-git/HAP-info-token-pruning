"""Orchestrate all experiments. Stages: smoke | train | measure | prune | all"""
import argparse, os, subprocess, sys
PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

SCALES = [32, 48, 64]              # width scaling study
DATASETS = ['cifar10', 'cifar100']
SEEDS = [0, 1, 2]
EPOCHS = 15


def sh(args):
    args = [str(a) for a in args]
    print('>>', ' '.join(args), flush=True)
    r = subprocess.run([PY] + args)
    if r.returncode != 0:
        sys.exit(f'FAILED: {args}')


def stage_train(seeds=SEEDS, scales=SCALES, datasets=DATASETS, epochs=EPOCHS):
    # capacity study: all scales, seed 0; multi-seed: d=64 only
    jobs = [(ds, d, 0) for ds in datasets for d in scales]
    jobs += [(ds, 64, s) for ds in datasets for s in seeds[1:]]
    for ds, d, s in jobs:
        ck = os.path.join(ROOT, 'ckpts', f'vlm_{ds}_d{d}_L10_s{s}.pt')
        if os.path.exists(ck):
            print('skip train', ck); continue
        sh(['train.py', '--d', str(d), '--dataset', ds, '--seed', str(s),
            '--epochs', str(epochs)])


def stage_measure(scales=SCALES, datasets=DATASETS):
    jobs = [(ds, d, 0) for ds in datasets for d in scales]
    jobs += [(ds, 64, s) for ds in datasets for s in SEEDS[1:]]
    for ds, d, s in jobs:
        out = os.path.join(ROOT, 'results', f'measure_{ds}_d{d}_L10_s{s}.json')
        if os.path.exists(out):
            print('skip measure', out); continue
        sh(['measure.py', '--d', str(d), '--dataset', ds, '--seed', str(s)])


def stage_prune(seeds=SEEDS, datasets=DATASETS):
    for ds in datasets:
        for s in seeds:
            out = os.path.join(ROOT, 'results', f'prune_{ds}_d64_L10_s{s}.json')
            if os.path.exists(out):
                print('skip prune', out); continue
            sh(['prune_eval.py', '--dataset', ds, '--seed', str(s)])


def stage_smoke():
    sh(['train.py', '--d', 32, '--dataset', 'cifar10', '--seed', 0, '--epochs', 2,
        '--tag', '_smoke'])
    sh(['measure.py', '--d', 32, '--dataset', 'cifar10', '--seed', 0, '--tag', '_smoke'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['smoke', 'train', 'measure', 'prune', 'all'])
    a = ap.parse_args()
    if a.stage == 'smoke':
        stage_smoke()
    elif a.stage == 'train':
        stage_train()
    elif a.stage == 'measure':
        stage_measure()
    elif a.stage == 'prune':
        stage_prune()
    else:
        stage_smoke(); stage_train(); stage_measure(); stage_prune()
