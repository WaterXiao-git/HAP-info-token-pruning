"""Convert HF parquet CIFAR files to npz cache (one-time). Run once before training."""
import os, io, sys
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def convert(name):
    out = os.path.join(DATA, name + '.npz')
    if os.path.exists(out):
        print('exists', out); return
    dfs = {}
    for split in ['train', 'test']:
        p = os.path.join(DATA, f'{name}_{split}.parquet')
        dfs[split] = pd.read_parquet(p)
    def to_arrays(df):
        xs, ys = [], []
        col_img = 'img' if 'img' in df.columns else 'image'
        for i in range(len(df)):
            im = df.iloc[i][col_img]
            if isinstance(im, dict):
                b = im.get('bytes')
                from PIL import Image
                arr = np.array(Image.open(io.BytesIO(b)).convert('RGB'))
            else:
                arr = np.array(im).transpose(1, 2, 0)
            xs.append(arr.astype(np.uint8))
            ys.append(int(df.iloc[i]['label']))
        return np.stack(xs), np.array(ys)
    tr_x, tr_y = to_arrays(dfs['train'])
    te_x, te_y = to_arrays(dfs['test'])
    np.savez_compressed(out, tr_x=tr_x, tr_y=tr_y, te_x=te_x, te_y=te_y)
    print('saved', out, tr_x.shape, te_x.shape)


if __name__ == '__main__':
    for n in ['cifar10', 'cifar100']:
        convert(n)
