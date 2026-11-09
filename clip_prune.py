"""Real-VLM experiments: CLIP ViT-B/32 zero-shot CIFAR-10 with vision-token pruning.

Stages:
  profile -- per-layer LOO token information of the CLIP vision encoder (real pretrained VLM)
  hap     -- train distilled linear scorers (per start layer)
  prune   -- zero-shot accuracy vs FLOPs/latency under progressive patch pruning

Run:  python clip_prune.py --stage all [--smoke]
Env:  HF_ENDPOINT=https://hf-mirror.com for model download in CN.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results_clip')
os.makedirs(OUT, exist_ok=True)
DEV = 'cpu'
MODEL_ID = 'openai/clip-vit-base-patch32'
LOCAL_DIR = os.path.expanduser('~/.cache/modelscope/hub/models/openai-mirror/clip-vit-base-patch32')
PROMPT = 'a photo of a {}'
CLASSES10 = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def jl(p): return json.load(open(p))
def save(p, obj): json.dump(obj, open(p,'w'))

def load_cifar10_test(n):
    import glob, pickle
    fs = glob.glob(os.path.join(ROOT, '..', 'data', 'cifar-10-batches-py', 'test_batch'))
    if not fs:
        fs = glob.glob(os.path.join(ROOT, '..', 'data', '**', 'test_batch'), recursive=True)
    assert fs, 'CIFAR-10 test_batch not found under paper/data'
    d = pickle.load(open(fs[0], 'rb'), encoding='bytes')
    x = torch.from_numpy(np.asarray(d[b'data'], dtype=np.uint8)).view(-1,3,32,32).float()/255.
    y = torch.tensor(d[b'labels'])
    return x[:n], y[:n]

def load_cifar10_train(n):
    import glob, pickle
    fs = sorted(glob.glob(os.path.join(ROOT, '..', 'data', 'cifar-10-batches-py', 'data_batch_*')))
    assert fs, 'CIFAR-10 train batches not found'
    xs, ys = [], []
    for f in fs:
        d = pickle.load(open(f,'rb'), encoding='bytes')
        xs.append(np.asarray(d[b'data'], dtype=np.uint8).reshape(-1,3,32,32))
        ys += list(d[b'labels'])
    x = torch.from_numpy(np.concatenate(xs)[:n]).float()/255.
    y = torch.tensor(ys[:n])
    return x, y

def build_model():
    src = LOCAL_DIR if os.path.isdir(LOCAL_DIR) else MODEL_ID
    m = CLIPModel.from_pretrained(src)
    p = CLIPProcessor.from_pretrained(src)
    m.eval().to(DEV)
    return m, p

@torch.no_grad()
def text_features(m, p):
    prompts = [PROMPT.format(c) for c in CLASSES10]
    inp = p(text=prompts, return_tensors='pt', padding=True).to(DEV)
    tf = m.get_text_features(**inp)
    return F.normalize(tf, dim=-1)

@torch.no_grad()
def vision_forward_trace(vm, px):
    """Return hidden states entering each encoder layer (list of L+1 tensors (1,T,d))."""
    hs = vm.pre_layrnorm(vm.embeddings(pixel_values=px.to(DEV)))
    trace = [hs]
    for layer in vm.encoder.layers:
        hs = layer(hs, None, None)[0]
        trace.append(hs)
    return trace

@torch.no_grad()
def run_layers(vm, hs, l_start, want_attn_layers=()):
    """Run encoder layers l_start..end. Returns final CLS image embedding (B,d) and
    dict {layer_idx: attn_weights (B,H,T,T)} for requested layers."""
    attns = {}
    h = hs
    for i in range(l_start, len(vm.encoder.layers)):
        out = vm.encoder.layers[i](h, None, None, output_attentions=i in want_attn_layers)
        h = out[0]
        if i in want_attn_layers:
            attns[i] = out[1]
    cls = vm.post_layernorm(h[:, 0, :])
    return cls, attns

def img_embed_from_cls(vm, model, cls_states):
    return model.visual_projection(cls_states)

def logits_from_embed(model, img_f, tf):
    return model.logit_scale.exp() * img_f @ tf.T

@torch.no_grad()
def stage_profile(m, p, tf, n_images=50, tag=''):
    vm = m.vision_model
    x, y = load_cifar10_test(n_images)
    K = None
    prof = []
    p_txt = None
    for i in range(x.shape[0]):
        px = p(images=x[i].mul(255).byte().permute(1,2,0).numpy(), return_tensors='pt').to(DEV)['pixel_values']
        trace = vision_forward_trace(vm, px)
        K = trace[0].shape[1]  # 197
        if p_txt is None:
            # null baseline: only CLS token through the whole encoder
            cls0, _ = run_layers(vm, trace[0][:, :1, :], 0)
            ef0 = img_embed_from_cls(vm, m, cls0)
            p_txt = F.softmax(logits_from_embed(m, ef0, tf), dim=-1)[0]
        for l in range(len(vm.encoder.layers)):
            ent = trace[l]                      # (1,T,d) entering layer l
            cls_tok = ent[:, :1, :].expand(K-1, -1, -1)   # (K-1,1,d)
            patch = ent[:, 1:, :].squeeze(0).unsqueeze(1) # (K-1,1,d)
            hs_var = torch.cat([cls_tok, patch], dim=1)   # (K-1,2,d) keep CLS + patch k
            cls_s, _ = run_layers(vm, hs_var, l)
            ef = img_embed_from_cls(vm, m, cls_s)
            pr = F.softmax(logits_from_embed(m, ef, tf), dim=-1)  # (K-1,C)
            info = pr - p_txt  # (K-1,C) I_l(v_k) per class; scalar info = value at true class
            prof.append(info[:, y[i]].cpu())  # I at true class for each patch
    prof = torch.stack(prof).view(x.shape[0], -1, K-1)  # (n_images, L, K-1)
    info_mean = prof.mean(dim=(0,2))   # (L,) mean over images and tokens
    res = {
        'model': MODEL_ID, 'n_images': int(x.shape[0]), 'K': int(K),
        'layers': int(len(vm.encoder.layers)),
        'info_mean': [float(v) for v in info_mean],
        'info_mean_per_layer_all': [[float(t) for t in prof[:,l,:].flatten()] for l in range(prof.shape[1])],
        'note': 'LOO on real CLIP ViT-B/32, zero-shot CIFAR-10; I = p(y|CLS+single patch) - p(y|CLS only), at true class',
    }
    save(os.path.join(OUT, f'clip_profile_cifar10_{tag}.json'), res)
    print('profile done:', [round(float(v),4) for v in info_mean])
    return res

@torch.no_grad()
def rank_scores(method, vm, ent, l_start, model=None, scorer=None, rng=None):
    """Return ranking of patch tokens (best first) at layer l_start."""
    T = ent.shape[1]
    if method == 'random':
        s = torch.rand(T-1)
        return torch.argsort(s, descending=True)
    if method == 'attention':
        _, attns = run_layers(vm, ent, l_start, want_attn_layers=(l_start,))
        a = attns[l_start].mean(dim=1)          # (1,T,T) avg heads
        s = a[0, 0, 1:]                          # attention from CLS to patches
        return torch.argsort(s, descending=True)
    if method == 'diversity':
        V = ent[0, 1:, :]                        # (K-1,d)
        Vn = F.normalize(V, dim=-1)
        chosen = [0]
        dmin = 1 - Vn @ Vn[0]
        for _ in range(T-2):
            j = int(torch.argmax(dmin))
            chosen.append(j)
            dmin = torch.minimum(dmin, 1 - Vn @ Vn[j])
        return torch.tensor(chosen)
    if method in ('hap', 'hap_info'):
        s = scorer(ent[0], l_start)              # (T,)
        return torch.argsort(s[1:], descending=True)
    raise ValueError(method)

@torch.no_grad()
def prune_forward(m, vm, px, l_start, keeps, method, scorer=None):
    """Progressive pruning: entering layer l, keep `keeps[l-l_start]` patches
    (re-ranked fresh among survivors at each layer, per-layer scoring)."""
    hs = vm.pre_layrnorm(vm.embeddings(pixel_values=px.to(DEV)))
    n = hs.shape[1] - 1
    for li, l in enumerate(range(l_start, len(vm.encoder.layers))):
        keep = keeps[li]
        if keep < n:
            order = rank_scores(method, vm, hs, l, scorer=scorer, rng=None)
            sel = order[:keep]
            hs = torch.cat([hs[:, :1, :], hs[:, 1:, :].squeeze(0)[sel].unsqueeze(0)], dim=1)
            n = keep
        hs = vm.encoder.layers[l](hs, None, None)[0]
    cls = vm.post_layernorm(hs[:, 0, :])
    return img_embed_from_cls(vm, m, cls)

def keep_schedule(P, l_start, rate, L, info_norm=None):
    n = P; keeps = []
    for l in range(l_start, L):
        r = rate if info_norm is None else rate * (1 - info_norm[l])
        keep = min(n, max(1, int(round(n * (1 - r)))))
        keeps.append(keep); n = keep
    return keeps

@torch.no_grad()
def eval_config(m, p, tf, x, y, l_start, rate, method, scorer=None, batch=16, info_norm=None):
    vm = m.vision_model
    L = len(vm.encoder.layers); P = x.shape[0] and 196
    correct = 0; lat = []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i+batch]
        preds = []; tb = time.time()
        for j in range(xb.shape[0]):
            px = p(images=xb[j].mul(255).byte().permute(1,2,0).numpy(), return_tensors='pt').to(DEV)['pixel_values']
            hs = vm.pre_layrnorm(vm.embeddings(pixel_values=px))
            if method == 'full' or rate == 0:
                cls_s, _ = run_layers(vm, hs, 0)
                ef = img_embed_from_cls(vm, m, cls_s)
            else:
                keeps = keep_schedule(hs.shape[1]-1, l_start, rate, L, info_norm=info_norm if method=='hap_info' else None)
                ef = prune_forward(m, vm, px, l_start, keeps, method, scorer=scorer)
            lg = logits_from_embed(m, ef, tf)
            preds.append(lg.argmax(dim=-1))
        pr = torch.cat(preds)
        correct += int((pr == y[i:i+batch].to(DEV)).sum())
        lat.append((time.time()-tb)/xb.shape[0])
    return correct/x.shape[0], float(np.mean(lat))

def block_flops(n, d=768):
    return 12*n*d*d + 2*n*n*d

@torch.no_grad()
def flops_ratio(l_start, keeps, L=12, P=196, d=768):
    full = sum(block_flops(P+1, d) for _ in range(L))
    s = 0.0; n = P+1
    for l in range(L):
        if l >= l_start:
            n = keeps[l-l_start] + 1
        s += block_flops(n, d)
    return s/full

class Scorer(torch.nn.Module):
    def __init__(self, L, d=768):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(L, d))
        self.b = torch.nn.Parameter(torch.zeros(L))
    def forward(self, hs, l):
        return hs @ self.w[l] + self.b[l]

def train_hap(m, p, tf, l_start, rate, n_train=500, epochs=1, tag=''):
    """Distill linear scorers with straight-through top-k on frozen CLIP."""
    vm = m.vision_model
    sc = Scorer(len(vm.encoder.layers)).to(DEV)
    opt = torch.optim.AdamW(sc.parameters(), lr=1e-3)
    x, y = load_cifar10_train(n_train)
    with torch.enable_grad():
        for ep in range(epochs):
            for i in range(x.shape[0]):
                px = p(images=x[i].mul(255).byte().permute(1,2,0).numpy(), return_tensors='pt').to(DEV)['pixel_values']
                with torch.no_grad():
                    hs_full = vm.pre_layrnorm(vm.embeddings(pixel_values=px))
                    cls_full, _ = run_layers(vm, hs_full, 0)
                    ef_full = img_embed_from_cls(vm, m, cls_full)
                    p_full = F.log_softmax(logits_from_embed(m, ef_full, tf), dim=-1)
                hs = hs_full
                n = hs.shape[1]-1
                for l in range(l_start, len(vm.encoder.layers)):
                    keep = max(1, int(round(n*(1-rate))))
                    s = sc(hs[0], l)           # (T,)
                    patches = hs[0,1:,:]
                    sp = s[1:]
                    soft = torch.sigmoid((sp - sp.median())/max(1e-3, sp.std()))
                    hard = torch.zeros_like(sp)
                    idx = torch.argsort(sp, descending=True)[:keep]
                    hard[idx] = 1.0
                    st = (hard - soft).detach() + soft
                    sel = torch.nonzero(hard > 0.5).flatten()
                    patches_k = (patches * st.unsqueeze(1)).index_select(0, sel)
                    hs = torch.cat([hs[:, :1, :], patches_k.unsqueeze(0)], dim=1)
                    n = keep
                    hs = vm.encoder.layers[l](hs, None, None)[0]
                cls = vm.post_layernorm(hs[:, 0, :])
                ef = img_embed_from_cls(vm, m, cls)
                logp = F.log_softmax(logits_from_embed(m, ef, tf), dim=-1)
                loss = F.kl_div(logp, p_full, reduction='batchmean') + F.nll_loss(logp, y[i:i+1].to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
    torch.save(sc.state_dict(), os.path.join(OUT, f'hap_scorer_lp{l_start}_{tag}.pt'))
    return sc

@torch.no_grad()
def stage_prune(m, p, tf, n_eval=500, lp_list=(1,6), rates=(0.5,0.75), smoke=False):
    x, y = load_cifar10_test(n_eval)
    vm = m.vision_model
    prof_f = os.path.join(OUT, 'clip_profile_cifar10.json')
    info_norm = None
    if os.path.exists(prof_f):
        im = jl(prof_f)['info_mean']
        mx = max(im)
        info_norm = [v/mx for v in im]
    results = []
    def run_one(method, lp, r, scorer=None):
        inn = info_norm if method == 'hap_info' else None
        acc, lat = eval_config(m, p, tf, x, y, lp, r, method, scorer=scorer, info_norm=inn)
        P = 196
        keeps = keep_schedule(P, lp, r, len(vm.encoder.layers), info_norm=inn)
        fr = flops_ratio(lp, keeps, L=len(vm.encoder.layers), P=P)
        results.append({'method': method, 'lp_mode': 'layer1' if lp==1 else 'layer6',
                        'lp': lp, 'rate': r, 'keeps': keeps, 'flops_ratio': fr,
                        'acc': acc, 'latency_ms': lat})
        print(f"  {method:10s} lp={lp} r={r}: acc={acc:.3f} flops={fr:.3f} lat={lat*1000:.0f}ms", flush=True)
    acc_full, lat_full = eval_config(m, p, tf, x, y, 0, 0.0, 'full')
    results.append({'method':'full','acc':acc_full,'latency_ms':lat_full})
    print(f'  full: acc={acc_full:.3f} lat={lat_full*1000:.0f}ms', flush=True)
    import psutil
    proc = psutil.Process()
    for lp in lp_list:
        for r in rates:
            for mth in ['random','attention','diversity','hap','hap_info']:
                if smoke and not (mth=='random' and r==rates[0]): continue
                scorer = None
                if mth in ('hap','hap_info'):
                    f = os.path.join(OUT, f'hap_scorer_lp{lp}_cifar10.pt')
                    if os.path.exists(f):
                        sc = Scorer(len(vm.encoder.layers)); sc.load_state_dict(torch.load(f)); sc.eval().to(DEV)
                    else:
                        sc = train_hap(m, p, tf, lp, r, tag='cifar10')
                    scorer = (lambda hs, l, sc=sc: sc(hs, l))
                mem0 = proc.memory_info().rss
                run_one(mth, lp, r, scorer=scorer)
                mem = proc.memory_info().rss
                results[-1]['peak_rss_mb'] = mem/1e6
    save(os.path.join(OUT, 'clip_prune_cifar10.json'), {'results': results, 'model': MODEL_ID})
    print('prune stage done')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['profile','prune','all'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    m, p = build_model()
    tf = text_features(m, p)
    if a.smoke:
        stage_profile(m, p, tf, n_images=2, tag='smoke')
        stage_prune(m, p, tf, n_eval=32, lp_list=(1,), rates=(0.5,), smoke=True)
    else:
        if a.stage in ('profile','all'):
            stage_profile(m, p, tf, n_images=50, tag='cifar10')
        if a.stage in ('prune','all'):
            stage_prune(m, p, tf, n_eval=500)
