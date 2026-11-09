"""
Common utilities for the HAP paper experiments.
Compact vision-language model, CIFAR-10/100 data, token information measurement.
All experiments run on a single RTX 4060 (8 GB) GPU.
"""
import os, math, json, pickle, tarfile, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Device selection: CPU is faster than the (clock-throttled) dGPU on this machine.
# Set HAP_DEVICE=cuda to override.
DEV = torch.device(os.environ.get('HAP_DEVICE', 'cpu'))
if DEV.type == 'cuda':
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(max(1, os.cpu_count() or 8))
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
os.makedirs(DATA, exist_ok=True)

C10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
C100_CLASSES = ['aquatic mammals','fish','flowers','food containers','fruit and vegetables',
                'household electrical devices','household furniture','insects','large carnivores',
                'large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores',
                'medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees',
                'vehicles 1','vehicles 2']
# text template tokens (CIFAR-100 uses coarse labels, some are two words -> word-level vocab)
PROMPT = 'this is a photo of a'


# ---------------------------------------------------------------- data
def _load_cifar_folder(name, tarname):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        t = os.path.join(DATA, tarname)
        if not os.path.exists(t):
            raise FileNotFoundError(f'{tarname} not found in {DATA}; '
                                    'run prepare_data.py or download it first')
        with tarfile.open(t) as f:
            f.extractall(path=DATA)
    return p


def load_cifar(dataset='cifar10'):
    """Returns (train_images[N,32,32,3] float 0-1, train_labels, test_images, test_labels).
    Prefers npz cache produced by prepare_data.py; falls back to original pickles."""
    npz = os.path.join(DATA, f'{dataset}.npz')
    if os.path.exists(npz):
        z = np.load(npz)
        return (z['tr_x'].astype(np.float32) / 255., z['tr_y'],
                z['te_x'].astype(np.float32) / 255., z['te_y'])
    if dataset == 'cifar10':
        p = _load_cifar_folder('cifar-10-batches-py', 'cifar-10-python.tar.gz')
        def load(files, key):
            d, l = [], []
            for f in files:
                with open(os.path.join(p, f), 'rb') as h:
                    e = pickle.load(h, encoding='latin1')
                    d.append(e['data']); l.extend(e[key])
            return np.concatenate(d).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255., np.array(l)
        tr = load([f'data_batch_{i}' for i in range(1, 6)], 'labels')
        te = load(['test_batch'], 'labels')
    else:
        p = _load_cifar_folder('cifar-100-python', 'cifar-100-python.tar.gz')
        def load(files, key):
            d, l = [], []
            for f in files:
                with open(os.path.join(p, f), 'rb') as h:
                    e = pickle.load(h, encoding='latin1')
                    d.append(e['data']); v = e[key]
                    l.extend(v.tolist() if hasattr(v, 'tolist') else list(v))
            return np.concatenate(d).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255., np.array(l)
        tr = load(['train'], 'coarse_labels')
        te = load(['test'], 'coarse_labels')
    return tr[0], tr[1], te[0], te[1]


class ImageTextDS(Dataset):
    def __init__(self, images, labels, prompt):
        self.images = images; self.labels = labels; self.prompt = prompt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(np.ascontiguousarray(self.images[i].transpose(2, 0, 1)))
        return img, int(self.labels[i]), self.prompt


# ---------------------------------------------------------------- vocab
def build_vocab(class_names):
    words = PROMPT.split()
    for c in class_names:
        words += c.split()
    words = sorted(set(words))
    vocab = ['<pad>', '<cls>'] + words
    w2i = {w: i for i, w in enumerate(vocab)}
    def tokenize(prompt):
        seq = [1] + [w2i[w] for w in prompt.split()]  # <cls> then words
        return seq
    return vocab, w2i, tokenize


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_out=None):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if attn_out is not None:  # slow path: materialize attention weights
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
            att = att.softmax(dim=-1)
            attn_out.append(att.detach())
            y = att @ v
        else:  # fast fused path
            y = F.scaled_dot_product_attention(q, k, v)
        y = self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
        x = x + y
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class CompactVLM(nn.Module):
    """
    Joint multimodal transformer: [<cls>] + prompt words + visual patch tokens.
    Patch embedding is a strided conv; no separate visual encoder depth, so all
    measured "layers" are joint fusion+reasoning layers (as in the LOO protocol
    of Wang et al., CVPR 2026, which measures decoder-layer information).
    """
    def __init__(self, d=64, depth=10, heads=4, patch=4, n_cls=10, vocab_size=None,
                 n_text=6, n_vis=64, dropout=0.0):
        super().__init__()
        self.d = d; self.depth = depth; self.n_text = n_text; self.n_vis = n_vis; self.patch = patch
        self.tok = nn.Embedding(vocab_size, d)
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        self.pos_text = nn.Parameter(torch.randn(1, n_text, d) * 0.02)
        self.pos_vis = nn.Parameter(torch.randn(1, n_vis, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)

    def token_layout(self):
        return self.n_text, self.n_vis  # text slots (incl <cls> at index 0), visual slots

    def forward_full(self, seq, images, prune_spec=None, gates=None, collect_states=False):
        """
        seq: [B, n_text]; images: [B,3,32,32]
        prune_spec: {layer: LongTensor [B, K] indices of visual tokens to KEEP entering
                     that layer} (K may differ across layers but is fixed within a batch;
                     K=0 means text-only).
        gates: {layer: gate_scores [B, n_vis] in [0,1]} -- applied as soft multiplication
               on visual token states entering that layer.
        Returns logits [B, n_cls]; optional dict with per-layer visual states and attn.
        """
        B = seq.shape[0]
        txt = self.tok(seq) + self.pos_text[:, :seq.shape[1]]
        vis = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_vis
        states, attns = [], []
        for l, blk in enumerate(self.blocks):
            if prune_spec is not None and l in prune_spec:
                keep = prune_spec[l]  # [B, K]
                if keep.shape[1] == 0:
                    vis = vis[:, :0]
                else:
                    vis = torch.gather(vis, 1, keep.unsqueeze(-1).expand(-1, -1, vis.shape[-1]))
            if gates is not None and l in gates:
                g = gates[l]  # [B, n_vis] or [B, K] matching current vis count
                vis = vis * g.to(vis.dtype).unsqueeze(-1)
            if collect_states:
                states.append(vis.detach())
            ao = [] if collect_states else None
            x = torch.cat([txt, vis], dim=1)
            x = blk(x, attn_out=ao)
            if ao is not None:
                attns.append(ao[0])
            txt, vis = x[:, :self.n_text], x[:, self.n_text:self.n_text + vis.shape[1]]
        h = self.ln_f(txt[:, 0])
        logits = self.head(h)
        out = {'logits': logits}
        if collect_states:
            out['vis_states'] = states; out['attns'] = attns
        return out

    def forward_with_images(self, seq, images, **kw):
        return self.forward_full(seq, images, **kw)


# ---------------------------------------------------------------- flops
def model_flops(model, n_vis, n_text=6, pruned_layers=None, keep_frac=1.0):
    """Analytical training-free-forward FLOPs (multiply-adds) per sample."""
    d = model.d
    flops = 0
    # patch embed: k*k*3*d*(32/patch)^2
    k = model.patch
    flops += k * k * 3 * d * (32 // k) ** 2
    T_full = n_text + n_vis
    for l in range(model.depth):
        T = T_full
        if pruned_layers and l in pruned_layers:
            T = n_text + int(round(n_vis * keep_frac))
        flops += 3 * d * d * T            # qkv
        flops += d * d * T                # proj
        flops += 2 * T * T * d            # attention scores + weighted sum
        flops += 8 * d * d * T            # mlp (4d up, 4d down)
        n_vis_eff = T - n_text
    flops += d * 10                        # head
    return flops


# ---------------------------------------------------------------- utils
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(model, loader, tokenize_fn, n_text=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels, _prompt in loader:
            seq = torch.tensor([tokenize_fn(PROMPT)] * imgs.shape[0], device=DEV)
            out = model.forward_full(seq, imgs.to(DEV))
            pred = out['logits'].argmax(-1).cpu()
            correct += (pred == labels).sum().item(); total += len(labels)
    return correct / max(total, 1)


def jsave(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def jload(path):
    with open(path) as f:
        return json.load(f)
