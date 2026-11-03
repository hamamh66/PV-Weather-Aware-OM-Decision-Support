#!/usr/bin/env python
# PV monitoring evaluation: end-to-end fine-tuning on GPU (Digital Research Alliance of Canada, Fir)
# Author: H. Hamam.  Usage:  python pv_finetune.py   (outputs in ./out ; resumable)
import os, sys, json, time, glob, shutil, hashlib, random, math, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, 'data'); OUT = os.path.join(HERE, 'out'); DRIVE_OUT = OUT
for p in (LOCAL, OUT, OUT + '/runs', OUT + '/figures'): os.makedirs(p, exist_ok=True)
SRC = {'weather': 'jehanbhathena/weather-dataset', 'panel': 'pythonafroz/solar-panel-images'}
SEEDS = [0, 1, 2, 3, 4]
MODELS = ['SimpleCNN', 'ResNet-18', 'EfficientNet-B0', 'ViT-Tiny', 'MiT-B0', 'Swin-Tiny']
PROPOSED = {'weather': 'MiT-B0', 'panel': 'Swin-Tiny'}
IMG = 224; BATCH = 32; LR = 1e-4; MAX_EPOCHS = 30; PATIENCE = 4; MIN_DELTA = 0.002; VAL_FRAC = 0.15
TEST_FRAC = 0.30; SPLIT_SEED = 2026
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)
def save_small(name, obj):
    with open(f'{OUT}/{name}', 'w') as f: json.dump(obj, f, indent=1, default=float)
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
def files_of(d): return sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(EXT))
def class_root(path):
    best = None
    for d, subs, _ in os.walk(path):
        imgs = [s for s in subs if files_of(os.path.join(d, s))]
        if len(imgs) >= 5 and (best is None or len(imgs) > best[1]): best = (d, len(imgs))
    return best[0]
import kagglehub
ROOT = {}
for task, slug in SRC.items():
    t0 = time.time(); p = kagglehub.dataset_download(slug); ROOT[task] = class_root(p)
    cls = sorted(d for d in os.listdir(ROOT[task]) if os.path.isdir(os.path.join(ROOT[task], d)))
    n = {c: len(files_of(os.path.join(ROOT[task], c))) for c in cls}
    print(f'{task}: {slug} -> {ROOT[task]}: {sum(n.values())} images {n} ({time.time()-t0:.0f} s)', flush=True)

# ---------- Stage 0b: index, exact-duplicate check, image cache ----------
DATA = {}; md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
for task in SRC:
    root = ROOT[task]; classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = [(p, i) for i, c in enumerate(classes) for p in files_of(os.path.join(root, c))]   # includes images in sub-folders
    # exact duplicates: one copy is kept; images filed under two different classes (conflicting labels) are removed
    h = [md5(p) for p, _ in items]; labels = {}
    for x, (_, y) in zip(h, items): labels.setdefault(x, set()).add(y)
    n_raw = len(items); conflict = {x for x, ls in labels.items() if len(ls) > 1}; seen = set(); keep = []
    for x, it in zip(h, items):
        if x in conflict or x in seen: continue
        seen.add(x); keep.append(it)
    removed = dict(raw=n_raw, duplicate_copies_removed=n_raw - len(keep) - sum(1 for x, it in zip(h, items) if x in conflict),
                   conflicting_label_files_removed=sum(1 for x in h if x in conflict), conflicting_groups=len(conflict), kept=len(keep))
    print(f'  {task}: {removed}')
    items = keep; h = [md5(p) for p, _ in items]; first = {}; grp = [first.setdefault(x, k) for k, x in enumerate(h)]
    rng = np.random.default_rng(SPLIT_SEED); te_groups = set()
    for c in range(len(classes)):
        g = sorted({grp[k] for k, (_, y) in enumerate(items) if y == c}); g = list(rng.permutation(g))
        te_groups |= set(g[:int(round(TEST_FRAC * len(g)))])
    tr = [it for k, it in enumerate(items) if grp[k] not in te_groups]; te = [it for k, it in enumerate(items) if grp[k] in te_groups]
    print(f'{task}: classes {classes}')
    print(f'  development {len(tr)}, test {len(te)} (after de-duplication)')
    with open(f'{DRIVE_OUT}/split_{task}.csv', 'w') as f:
        f.write('file,label,part\n'); f.writelines(f'{os.path.relpath(p, root)},{classes[y]},train\n' for p, y in tr); f.writelines(f'{os.path.relpath(p, root)},{classes[y]},test\n' for p, y in te)
    DATA[task] = dict(classes=classes, train=tr, test=te, n_identical_train_test=0, dedup=removed)

def load_cache(task, part):
    key = hashlib.md5('|'.join(p for p, _ in DATA[task][part]).encode()).hexdigest()[:10]
    fn = f'{LOCAL}/cache_{task}_{part}_{len(DATA[task][part])}_{key}.pt'      # the name encodes the exact file list, so a stale cache is never reused
    if os.path.exists(fn): return torch.load(fn)
    items = DATA[task][part]; X = torch.empty((len(items), 3, IMG, IMG), dtype=torch.uint8); y = torch.tensor([c for _, c in items])
    t0 = time.time()
    for i, (p, _) in enumerate(items):
        im = Image.open(p).convert('RGB').resize((IMG, IMG), Image.BILINEAR)
        X[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        if (i + 1) % 1000 == 0 or i + 1 == len(items): print(f'  caching {task}/{part}: {i+1}/{len(items)} ({time.time()-t0:.0f} s)', flush=True)
    torch.save((X, y), fn); return X, y
CACHE = {(t, p): load_cache(t, p) for t in SRC for p in ('train', 'test')}
save_small('data_summary.json', {t: dict(source=SRC[t], test_frac=TEST_FRAC, split_seed=SPLIT_SEED, classes=DATA[t]['classes'], n_development=len(DATA[t]['train']), n_test=len(DATA[t]['test']),
          dedup=DATA[t]['dedup'],
          test_counts={c: int((CACHE[(t, 'test')][1] == i).sum()) for i, c in enumerate(DATA[t]['classes'])}) for t in SRC})

# ---------- Stage 1: per-seed stratified train/validation split of the provider train folder ----------
def split(task, seed):
    y = CACHE[(task, 'train')][1].numpy(); rng = np.random.default_rng(seed); tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); k = max(1, int(round(VAL_FRAC * len(idx))))
        va += idx[:k].tolist(); tr += idx[k:].tolist()
    return np.array(sorted(tr)), np.array(sorted(va))
for t in SRC:
    a, b = split(t, 0); print(f'{t}: seed-0 split train {len(a)} / validation {len(b)} / test {len(CACHE[(t, "test")][1])}')

# sample grid of the weather test images (four random images per class) for checking the labels
Xw, yw = CACHE[('weather', 'test')]; cw = DATA['weather']['classes']; rng = np.random.default_rng(1)
f, ax = plt.subplots(len(cw), 4, figsize=(8, 2 * len(cw)))
for a in ax.ravel(): a.axis('off')
for i, c in enumerate(cw):
    idx_c = np.where(yw.numpy() == i)[0]
    for j, k in enumerate(rng.choice(idx_c, min(4, len(idx_c)), replace=False)):
        ax[i, j].imshow(Xw[k].permute(1, 2, 0).numpy()); ax[i, j].axis('off')
        if j == 0: ax[i, j].set_title(c, loc='left', fontsize=9)
plt.tight_layout(); plt.savefig(f'{DRIVE_OUT}/figures/weather_samples_by_class.png', dpi=110); plt.close()
print('saved weather_samples_by_class.png')

# ---------- Stage 2: end-to-end fine-tuning ----------
import timm
from transformers import AutoModelForImageClassification
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
class SimpleCNN(nn.Module):
    def __init__(s, k):
        super().__init__(); ch = [3, 32, 64, 128, 256]; L = []
        for a, b in zip(ch[:-1], ch[1:]): L += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        s.f = nn.Sequential(*L); s.h = nn.Linear(256, k)
    def forward(s, x): return s.h(F.adaptive_avg_pool2d(s.f(x), 1).flatten(1))
class HF(nn.Module):
    def __init__(s, name, k):
        super().__init__(); s.m = AutoModelForImageClassification.from_pretrained(name, num_labels=k, ignore_mismatched_sizes=True)
    def forward(s, x): return s.m(pixel_values=x).logits
def build(name, k=11):
    if name == 'SimpleCNN': return SimpleCNN(k)
    if name == 'ResNet-18': return timm.create_model('resnet18', pretrained=True, num_classes=k)
    if name == 'EfficientNet-B0': return timm.create_model('efficientnet_b0', pretrained=True, num_classes=k)
    if name == 'ViT-Tiny': return timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=k)
    if name == 'MiT-B0': return HF('nvidia/mit-b0', k)
    if name == 'Swin-Tiny': return HF('microsoft/swin-tiny-patch4-window7-224', k)
def to_input(xb, train):
    x = xb.to(DEV, non_blocking=True).float() / 255.
    if train:
        m = torch.rand(x.shape[0], device=DEV) < 0.5; x[m] = x[m].flip(-1)
    return (x - MEAN.to(DEV)) / STD.to(DEV)
@torch.no_grad()
def predict(model, X):
    model.eval(); P = []
    for i in range(0, len(X), 128):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'): P.append(F.softmax(model(to_input(X[i:i+128], False)).float(), 1).cpu())
    return torch.cat(P).numpy()
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix, precision_recall_fscore_support
def metrics(y, P, k):
    p = P.argmax(1); Y = np.eye(k)[y]; conf = P.max(1); bins = np.linspace(0, 1, 16); ece = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): ece += m.mean() * abs((p[m] == y[m]).mean() - conf[m].mean())
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(k)), zero_division=0)
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p), macro_f1=f1_score(y, p, average='macro'),
                weighted_f1=f1_score(y, p, average='weighted'), log_loss=log_loss(y, np.clip(P, 1e-7, 1), labels=list(range(k))),
                brier=float(((P - Y) ** 2).sum(1).mean()), ece=float(ece), per_class_f1=f1.tolist(), per_class_recall=rc.tolist(),
                confusion=confusion_matrix(y, p, labels=list(range(k))).tolist())
def bootstrap_ci(y, p, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); acc, mf1 = [], []
    for _ in range(B):
        i = rng.integers(0, n, n); acc.append((p[i] == y[i]).mean()); mf1.append(f1_score(y[i], p[i], average='macro'))
    q = lambda a: [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
    return dict(accuracy_ci95=q(acc), macro_f1_ci95=q(mf1))

def run(task, name, seed):
    tag = f'{task}_{name}_s{seed}'; fj = f'{OUT}/runs/{tag}.json'
    if os.path.exists(fj): print(f'[skip] {tag} already done', flush=True); return json.load(open(fj))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.backends.cudnn.benchmark = True
    k = len(DATA[task]['classes']); Xtr_all, ytr_all = CACHE[(task, 'train')]; Xte, yte = CACHE[(task, 'test')]
    itr, iva = split(task, seed); Xtr, ytr, Xva, yva = Xtr_all[itr], ytr_all[itr], Xtr_all[iva], ytr_all[iva]
    model = build(name, k).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, best_ep, wait, hist = -1, None, 0, 0, []; t0 = time.time(); g = torch.Generator().manual_seed(seed)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); perm = torch.randperm(len(ytr), generator=g); tl = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i+BATCH]; x = to_input(Xtr[b], True); yb = ytr[b].to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'): loss = F.cross_entropy(model(x).float(), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item() * len(b)
        Pva = predict(model, Xva); vf1 = f1_score(yva.numpy(), Pva.argmax(1), average='macro')
        vloss = log_loss(yva.numpy(), np.clip(Pva, 1e-7, 1), labels=list(range(k)))
        hist.append(dict(epoch=ep, train_loss=tl / len(perm), val_loss=vloss, val_macro_f1=vf1))
        improved = vf1 > best + MIN_DELTA
        if improved: best, best_ep, wait = vf1, ep, 0; best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else: wait += 1
        print(f'  {tag} epoch {ep:2d}: train loss {tl/len(perm):.4f}  val loss {vloss:.4f}  val macro-F1 {vf1:.4f}{"  *" if improved else ""}  ({time.time()-t0:.0f} s)', flush=True)
        if wait >= PATIENCE: print(f'  early stop: no validation macro-F1 gain > {MIN_DELTA} for {PATIENCE} epochs; best epoch {best_ep}', flush=True); break
    model.load_state_dict(best_state); P = predict(model, Xte); y = yte.numpy()
    R = dict(task=task, model=name, seed=seed, best_epoch=best_ep, epochs_run=len(hist), train_time_s=time.time() - t0,
             n_train=len(itr), n_val=len(iva), n_test=len(y), history=hist, **metrics(y, P, k), **bootstrap_ci(y, P.argmax(1)))
    np.savez_compressed(f'{OUT}/runs/{tag}_probs.npz', P=P.astype(np.float16), y=y)
    json.dump(R, open(fj, 'w'), default=float)
    del model, opt; torch.cuda.empty_cache(); return R
jobs = [(t, m, s) for t in ('panel', 'weather') for m in MODELS for s in SEEDS]
T0 = time.time()
for j, (t, m, s) in enumerate(jobs, 1):
    print(f'=== [{j}/{len(jobs)}] {t} | {m} | seed {s}   (elapsed {(time.time()-T0)/60:.1f} min)', flush=True)
    r = run(t, m, s)
    print(f'    test accuracy {r["accuracy"]:.4f}  macro-F1 {r["macro_f1"]:.4f}  (best epoch {r["best_epoch"]})', flush=True)
print(f'all runs finished in {(time.time()-T0)/60:.1f} min', flush=True)

# ---------- Stage 3: efficiency (parameters, FLOPs, latency) ----------
from torch.utils.flop_counter import FlopCounterMode
EFF = {}
for m in MODELS:
    net = build(m, 11).to(DEV).eval(); x = torch.randn(1, 3, IMG, IMG, device=DEV)
    with torch.no_grad(), FlopCounterMode(display=False) as fc: net(x)
    flops = fc.get_total_flops()
    def lat(dev, n):
        nn_ = net.to(dev); xx = x.to(dev); ts = []
        with torch.no_grad():
            for i in range(n + 10):
                if dev == 'cuda': torch.cuda.synchronize()
                t = time.perf_counter(); nn_(xx)
                if dev == 'cuda': torch.cuda.synchronize()
                if i >= 10: ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts))
    EFF[m] = dict(params_M=sum(p.numel() for p in net.parameters()) / 1e6, GFLOPs=flops / 1e9,
                  gpu_ms_batch1=lat('cuda', 100) if DEV == 'cuda' else None, cpu_ms_batch1=lat('cpu', 20))
    print(m, {k: (round(v, 2) if v is not None else None) for k, v in EFF[m].items()}, flush=True)
    del net; torch.cuda.empty_cache()
EFF['_device'] = dict(gpu=torch.cuda.get_device_name(0) if DEV == 'cuda' else None, cpu_threads=torch.get_num_threads(),
                      note='full classifier with 11 outputs; FLOPs counted by torch.utils.flop_counter for one 224x224 image (a multiply-add counts as two FLOPs)')
save_small('efficiency.json', EFF)

# ---------- Stage 3b: aggregation and paired comparisons ----------
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
RUNS = {(r['task'], r['model'], r['seed']): r for r in (json.load(open(f)) for f in glob.glob(f'{DRIVE_OUT}/runs/*_s[0-9].json'))}
PROBS = {k: np.load(f'{DRIVE_OUT}/runs/{k[0]}_{k[1]}_s{k[2]}_probs.npz') for k in RUNS}
KEYS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'log_loss', 'brier', 'ece', 'best_epoch', 'train_time_s']
SUM = {}
for t in SRC:
    SUM[t] = {}
    for m in MODELS:
        rs = [RUNS[(t, m, s)] for s in SEEDS if (t, m, s) in RUNS]
        if not rs: continue
        SUM[t][m] = {k: dict(mean=float(np.mean([r[k] for r in rs])), sd=float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0) for k in KEYS}
        SUM[t][m]['n_seeds'] = len(rs); SUM[t][m]['per_class_f1_mean'] = np.mean([r['per_class_f1'] for r in rs], 0).tolist()
        SUM[t][m]['seed0_ci95'] = dict(accuracy=rs[0]['accuracy_ci95'], macro_f1=rs[0]['macro_f1_ci95'])
    ref = PROPOSED[t]
    for m in MODELS:
        if m == ref or m not in SUM[t]: continue
        a = [RUNS[(t, ref, s)]['macro_f1'] for s in SEEDS]; b = [RUNS[(t, m, s)]['macro_f1'] for s in SEEDS]
        tt = stats.ttest_rel(a, b); ps = []
        for s in SEEDS:
            pa = PROBS[(t, ref, s)]['P'].argmax(1); pb = PROBS[(t, m, s)]['P'].argmax(1); y = PROBS[(t, ref, s)]['y']
            ca, cb = pa == y, pb == y; tab = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
            ps.append(float(mcnemar(tab, exact=True).pvalue))
        SUM[t][m]['vs_proposed'] = dict(mean_diff_macro_f1=float(np.mean(a) - np.mean(b)), paired_t_p=float(tt.pvalue),
                                        mcnemar_p_per_seed=ps, seeds_with_mcnemar_p_below_0_05=int(sum(p < 0.05 for p in ps)))
    print(f'\n{t}: test metrics, mean ± SD over {len(SEEDS)} seeds')
    for m, v in sorted(SUM[t].items(), key=lambda kv: -kv[1]['macro_f1']['mean']):
        line = f"  {m:16s} acc {v['accuracy']['mean']:.4f}±{v['accuracy']['sd']:.4f}  macro-F1 {v['macro_f1']['mean']:.4f}±{v['macro_f1']['sd']:.4f}  bal.acc {v['balanced_accuracy']['mean']:.4f}  logloss {v['log_loss']['mean']:.3f}  Brier {v['brier']['mean']:.3f}  ECE {v['ece']['mean']:.3f}"
        if 'vs_proposed' in v: line += f"  | Δ(prop−this) {v['vs_proposed']['mean_diff_macro_f1']:+.4f}, paired-t p {v['vs_proposed']['paired_t_p']:.3g}, McNemar p<0.05 in {v['vs_proposed']['seeds_with_mcnemar_p_below_0_05']}/5 seeds"
        print(line)
save_small('summary.json', SUM)

# ---------- Stage 4: label audit and figures ----------
AUD = {}
for t in SRC:
    ref = PROPOSED[t]; y = PROBS[(t, ref, 0)]['y']; cls = DATA[t]['classes']
    wrong_all = np.all([PROBS[(t, ref, s)]['P'].argmax(1) != y for s in SEEDS], 0)
    idx = np.where(wrong_all)[0]; Pm = np.mean([PROBS[(t, ref, s)]['P'].astype(float) for s in SEEDS], 0)
    AUD[t] = [dict(file=os.path.relpath(DATA[t]['test'][i][0], LOCAL), label=cls[y[i]], predicted=cls[Pm[i].argmax()], mean_conf=float(Pm[i].max())) for i in idx]
    print(f'{t}: {len(idx)} test images misclassified by all {len(SEEDS)} seeds of {ref}')
    if len(idx):
        n = min(24, len(idx)); sel = idx[np.argsort(-Pm[idx].max(1))][:n]; X = CACHE[(t, 'test')][0]
        f, ax = plt.subplots(int(math.ceil(n / 6)), 6, figsize=(12, 2.3 * math.ceil(n / 6))); ax = np.atleast_2d(ax)
        for a in ax.ravel(): a.axis('off')
        for a, i in zip(ax.ravel(), sel):
            a.imshow(X[i].permute(1, 2, 0).numpy()); a.set_title(f'label {cls[y[i]]}\npred {cls[Pm[i].argmax()]} ({Pm[i].max():.2f})', fontsize=7)
        plt.tight_layout(); plt.savefig(f'{DRIVE_OUT}/figures/{t}_consistently_misclassified.png', dpi=110); plt.close()
save_small('label_audit.json', AUD)
for t in SRC:   # confusion matrix (mean over seeds) and validation curves of the proposed models
    ref = PROPOSED[t]; cls = DATA[t]['classes']; cm = np.mean([RUNS[(t, ref, s)]['confusion'] for s in SEEDS], 0)
    cmn = cm / cm.sum(1, keepdims=True); f, a = plt.subplots(figsize=(0.55 * len(cls) + 3, 0.55 * len(cls) + 2.5))
    a.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)): a.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if cmn[i, j] > .5 else 'black')
    a.set_xticks(range(len(cls))); a.set_xticklabels(cls, rotation=45, ha='right', fontsize=8); a.set_yticks(range(len(cls))); a.set_yticklabels(cls, fontsize=8)
    a.set_xlabel('Predicted'); a.set_ylabel('True'); a.set_title(f'{ref}, {t}: row-normalised confusion (mean of {len(SEEDS)} seeds)', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{DRIVE_OUT}/figures/{t}_{ref}_confusion.png', dpi=150); plt.close()
    h = RUNS[(t, ref, 0)]['history']; f, a = plt.subplots(1, 2, figsize=(9, 3.2))
    a[0].plot([e['epoch'] for e in h], [e['train_loss'] for e in h], label='train'); a[0].plot([e['epoch'] for e in h], [e['val_loss'] for e in h], label='validation')
    a[0].set_xlabel('Epoch'); a[0].set_ylabel('Cross-entropy loss'); a[0].legend()
    a[1].plot([e['epoch'] for e in h], [e['val_macro_f1'] for e in h]); a[1].axvline(RUNS[(t, ref, 0)]['best_epoch'], ls='--', c='gray')
    a[1].set_xlabel('Epoch'); a[1].set_ylabel('Validation macro-F1'); plt.suptitle(f'{ref}, {t}, seed 0 (dashed: selected epoch)', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{DRIVE_OUT}/figures/{t}_{ref}_curves.png', dpi=150); plt.close()
f, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for a, t in zip(ax, SRC):
    ms = [m for m in MODELS if m in SUM[t]]; mu = [SUM[t][m]['macro_f1']['mean'] for m in ms]; sd = [SUM[t][m]['macro_f1']['sd'] for m in ms]
    a.bar(ms, mu, yerr=sd, capsize=3, color=['#c05621' if m == PROPOSED[t] else '#4a5568' for m in ms]); a.set_ylim(min(mu) - 0.1, 1.0)
    a.set_title(f'{t}: test macro-F1 (mean ± SD, {len(SEEDS)} seeds)', fontsize=9); a.tick_params(axis='x', rotation=30, labelsize=8)
plt.tight_layout(); plt.savefig(f'{DRIVE_OUT}/figures/baselines_macro_f1.png', dpi=150); plt.close(); print('figures saved')

# ---------- Stage 5: decision matrix (SRM): audit, weather dependence, error propagation ----------
SRM_V1 = json.loads(r"""{"Dew-Bird_drop": {"Human": "Light cleaning if needed", "IoT": "MODERATE WARNING → Slight moisture and bird droppings detected; monitor panels", "AI": "Auto signal: Suggest cleaning schedule", "Risk": "LOW"}, "Dew-Clean": {"Human": "No action", "IoT": "NORMAL → Moisture and irradiance sensors report normal conditions", "AI": "Maintain normal operation", "Risk": "LOW"}, "Dew-Dusty": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Dust detected; monitor output", "AI": "Auto: Postpone cleaning", "Risk": "MEDIUM"}, "Dew-Electrical_damage": {"Human": "Inspect panel immediately", "IoT": "ALERT → Voltage irregularities detected; moisture may cause short-circuit", "AI": "Auto: Isolate affected string", "Risk": "HIGH"}, "Dew-Physical_Damage": {"Human": "Inspect panel", "IoT": "CAUTION → Minor physical damage detected by sensors", "AI": "Auto: Send risk alert", "Risk": "MEDIUM"}, "Dew-Snow_Covered": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Slight snow detected; monitor panels", "AI": "Auto: Postpone cleaning", "Risk": "MEDIUM"}, "FogSmog-Bird_drop": {"Human": "Optional inspection", "IoT": "MODERATE WARNING → Humidity and low visibility; monitor bird droppings", "AI": "Auto: Adjust expected power output", "Risk": "MEDIUM"}, "FogSmog-Clean": {"Human": "No action", "IoT": "NORMAL → Humidity and irradiance normal", "AI": "Maintain baseline operation", "Risk": "LOW"}, "FogSmog-Dusty": {"Human": "Optional cleaning", "IoT": "MODERATE WARNING → Dust detected; monitor output", "AI": "Auto: Adjust expected output", "Risk": "MEDIUM"}, "FogSmog-Electrical_damage": {"Human": "Inspection required", "IoT": "ALERT → Voltage/current anomalies detected", "AI": "Auto: Isolate affected string", "Risk": "HIGH"}, "FogSmog-Physical_Damage": {"Human": "Inspect panel", "IoT": "ALERT → Stress sensors triggered", "AI": "Auto: Send high-risk alert", "Risk": "HIGH"}, "FogSmog-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Ice accumulation detected", "AI": "Auto: Wait strategy", "Risk": "MEDIUM"}, "Frost-Bird_drop": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Frost and minor bird droppings detected", "AI": "Auto: Wait-for-melt", "Risk": "MEDIUM"}, "Frost-Clean": {"Human": "No action", "IoT": "NORMAL → Frost detected but safe; monitor panels", "AI": "Auto: Wait-for-melt", "Risk": "LOW"}, "Frost-Dusty": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Frost + dust; monitor sensors", "AI": "Auto: Wait-for-melt", "Risk": "MEDIUM"}, "Frost-Electrical_damage": {"Human": "Inspect after thaw", "IoT": "ALERT → Voltage anomalies detected", "AI": "Auto: Isolate string", "Risk": "HIGH"}, "Frost-Physical_Damage": {"Human": "Inspect after thaw", "IoT": "ALERT → Structural stress detected", "AI": "Auto: Send high-risk alert", "Risk": "HIGH"}, "Frost-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Snow detected", "AI": "Auto: Wait-for-melt", "Risk": "MEDIUM"}, "Glaze-Bird_drop": {"Human": "Optional cleaning", "IoT": "MODERATE WARNING → Ice accumulation detected; monitor panels", "AI": "Auto: Temporary shutdown", "Risk": "MEDIUM"}, "Glaze-Clean": {"Human": "No action", "IoT": "MODERATE WARNING → Ice detected", "AI": "Auto: Temporary inverter shutdown", "Risk": "MEDIUM"}, "Glaze-Dusty": {"Human": "Optional cleaning", "IoT": "MODERATE WARNING → Ice + dust detected", "AI": "Auto: Temporary shutdown", "Risk": "MEDIUM"}, "Glaze-Electrical_damage": {"Human": "Repair required", "IoT": "ALERT → Ground/fault detected", "AI": "Auto: Isolate circuits", "Risk": "HIGH"}, "Glaze-Physical_Damage": {"Human": "Inspect panel", "IoT": "ALERT → Tilt/ice sensors triggered", "AI": "Auto: Shutdown inverter", "Risk": "HIGH"}, "Glaze-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Ice detected", "AI": "Auto: Temporary shutdown", "Risk": "MEDIUM"}, "Hail-Bird_drop": {"Human": "Optional inspection", "IoT": "MODERATE WARNING → Impact detected by sensors", "AI": "Auto: Log output deviation", "Risk": "MEDIUM"}, "Hail-Clean": {"Human": "Optional inspection", "IoT": "MODERATE WARNING → Monitor output post-storm", "AI": "Auto: Post-storm assessment", "Risk": "MEDIUM"}, "Hail-Dusty": {"Human": "Inspect after storm", "IoT": "ALERT → Dust & impact detected", "AI": "Auto: Derate output", "Risk": "HIGH"}, "Hail-Electrical_damage": {"Human": "Inspection required", "IoT": "ALERT → Voltage & impact sensors triggered", "AI": "Auto: Disconnect panel", "Risk": "HIGH"}, "Hail-Physical_Damage": {"Human": "Replace panels", "IoT": "ALERT → Impact sensors triggered", "AI": "Auto: Disconnect damaged panels", "Risk": "HIGH"}, "Hail-Snow_Covered": {"Human": "Optional removal", "IoT": "MODERATE WARNING → Snow/impact detected", "AI": "Auto: Derate output", "Risk": "MEDIUM"}, "Lightning-Bird_drop": {"Human": "Optional inspection", "IoT": "ALERT → Surge sensors triggered", "AI": "Auto: Health check", "Risk": "HIGH"}, "Lightning-Clean": {"Human": "No action", "IoT": "MODERATE WARNING → Surge sensors normal", "AI": "Auto: Health check", "Risk": "MEDIUM"}, "Lightning-Dusty": {"Human": "Optional inspection", "IoT": "MODERATE WARNING → Surge + dust sensors triggered", "AI": "Auto: Health check", "Risk": "MEDIUM"}, "Lightning-Electrical_damage": {"Human": "Immediate inspection", "IoT": "ALERT → Surge detected", "AI": "Auto: Disconnect grid", "Risk": "HIGH"}, "Lightning-Physical_Damage": {"Human": "Inspect panel", "IoT": "ALERT → Voltage/surge sensor triggered", "AI": "Auto: Isolate circuits", "Risk": "HIGH"}, "Lightning-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Surge sensor normal", "AI": "Auto: Health check", "Risk": "MEDIUM"}, "Rain-Bird_drop": {"Human": "Optional cleaning", "IoT": "NORMAL → Rain sensor reports moisture; panels safe", "AI": "Auto: Evaluate cleaning effect", "Risk": "LOW"}, "Rain-Clean": {"Human": "No action", "IoT": "NORMAL → Rainwater detected, panels functioning", "AI": "Maintain forecast model", "Risk": "LOW"}, "Rain-Dusty": {"Human": "Skip cleaning", "IoT": "MODERATE WARNING → Dust + rain; monitor panels", "AI": "Auto: Cancel cleaning", "Risk": "MEDIUM"}, "Rain-Electrical_damage": {"Human": "Inspect if anomaly", "IoT": "ALERT → Voltage anomaly detected", "AI": "Auto: Isolate string", "Risk": "HIGH"}, "Rain-Physical_Damage": {"Human": "Inspect panel", "IoT": "ALERT → Stress sensor triggered", "AI": "Auto: Send alert", "Risk": "HIGH"}, "Rain-Snow_Covered": {"Human": "Optional wait", "IoT": "MODERATE WARNING → Rain + snow sensor", "AI": "Auto: Wait-for-clear", "Risk": "MEDIUM"}, "Rainbow-Bird_drop": {"Human": "No action", "IoT": "NORMAL → No risk detected", "AI": "Monitor only", "Risk": "LOW"}, "Rainbow-Clean": {"Human": "No action", "IoT": "NORMAL → No risk detected", "AI": "Monitor only", "Risk": "LOW"}, "Rainbow-Dusty": {"Human": "Optional cleaning", "IoT": "MODERATE WARNING → Minor dust detected", "AI": "Auto: Suggest cleaning", "Risk": "MEDIUM"}, "Rainbow-Electrical_damage": {"Human": "Inspect panel", "IoT": "ALERT → Minor anomalies detected", "AI": "Auto: Isolate string", "Risk": "HIGH"}, "Rainbow-Physical_Damage": {"Human": "Inspect panel", "IoT": "ALERT → Sensors triggered", "AI": "Auto: Alert", "Risk": "HIGH"}, "Rainbow-Snow_Covered": {"Human": "Optional wait", "IoT": "MODERATE WARNING → Snow detected", "AI": "Auto: Wait-for-clear", "Risk": "MEDIUM"}, "Rime-Bird_drop": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Frost + minor debris", "AI": "Auto: Wait-for-melt", "Risk": "MEDIUM"}, "Rime-Clean": {"Human": "No action", "IoT": "NORMAL → Frost detected; safe operation", "AI": "Maintain normal operation", "Risk": "LOW"}, "Rime-Dusty": {"Human": "Optional cleaning after thaw", "IoT": "MODERATE WARNING → Frost + dust", "AI": "Auto: Wait-for-clear", "Risk": "MEDIUM"}, "Rime-Electrical_damage": {"Human": "Inspect after thaw", "IoT": "ALERT → Voltage anomaly detected", "AI": "Auto: Isolate string", "Risk": "HIGH"}, "Rime-Physical_Damage": {"Human": "Inspect after thaw", "IoT": "ALERT → Structural stress detected", "AI": "Auto: Send alert", "Risk": "HIGH"}, "Rime-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Snow detected", "AI": "Auto: Wait-for-clear", "Risk": "MEDIUM"}, "Sandstorm-Bird_drop": {"Human": "Inspect panels", "IoT": "MODERATE WARNING → Dust & particles detected", "AI": "Auto: Reduce output", "Risk": "MEDIUM"}, "Sandstorm-Clean": {"Human": "Optional cleaning later", "IoT": "MODERATE WARNING → Dust detected", "AI": "Auto: Reduce output", "Risk": "MEDIUM"}, "Sandstorm-Dusty": {"Human": "Skip cleaning", "IoT": "MODERATE WARNING → High dust detected", "AI": "Auto: Derate output", "Risk": "MEDIUM"}, "Sandstorm-Electrical_damage": {"Human": "Inspect after storm", "IoT": "ALERT → Voltage anomalies detected", "AI": "Auto: Isolate string", "Risk": "HIGH"}, "Sandstorm-Physical_Damage": {"Human": "Inspect panels", "IoT": "ALERT → Structural stress detected", "AI": "Auto: Send alert", "Risk": "HIGH"}, "Sandstorm-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Dust + snow detected", "AI": "Auto: Derate output", "Risk": "MEDIUM"}, "Snow-Bird_drop": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Snow + bird droppings detected", "AI": "Auto: Wait-for-melt", "Risk": "MEDIUM"}, "Snow-Clean": {"Human": "No action", "IoT": "NORMAL → Snow detected; panels safe", "AI": "Auto: Wait-for-clear", "Risk": "LOW"}, "Snow-Dusty": {"Human": "Delay cleaning", "IoT": "MODERATE WARNING → Snow + dust detected", "AI": "Auto: Wait-for-clear", "Risk": "MEDIUM"}, "Snow-Electrical_damage": {"Human": "Inspect after thaw", "IoT": "ALERT → Voltage anomaly detected", "AI": "Auto: Isolate string", "Risk": "HIGH"}, "Snow-Physical_Damage": {"Human": "Inspect panel", "IoT": "ALERT → Structural sensors triggered", "AI": "Auto: Send alert", "Risk": "HIGH"}, "Snow-Snow_Covered": {"Human": "No action", "IoT": "MODERATE WARNING → Snow detected", "AI": "Auto: Wait-for-clear", "Risk": "MEDIUM"}}""")
W = ['Dew', 'FogSmog', 'Frost', 'Glaze', 'Hail', 'Lightning', 'Rain', 'Rainbow', 'Rime', 'Sandstorm', 'Snow']
PNL = ['Bird_drop', 'Clean', 'Dusty', 'Electrical_damage', 'Physical_Damage', 'Snow_Covered']
RISK = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
# safety principles used for the audit (sources cited in the article)
#  P1 lightning: no outdoor or on-site work during a thunderstorm; resume 30 min after the last thunder (OSHA/NOAA FS-3863)
#  P2 electrical damage: work only after isolation / de-energisation by qualified personnel (NFPA 70E; IEC 62446-2)
#  P3 ice: no manual cleaning of iced, frosted or snow-loaded modules (engineering judgement: slip, fall and glass-damage hazard)
#  P4 hail and sandstorm: no on-site work while the event is ongoing (engineering judgement)
ICE = {'Glaze', 'Rime', 'Frost', 'Snow'}; STORM = {'Hail', 'Sandstorm'}
def flags_from_text(h, ai=''):
    h = h.lower(); later = any(w in h for w in ('after', 'delay', 'skip', 'wait', 'later', 'no action', 'no on-site', 'postpone'))
    onsite = (not later) and any(w in h for w in ('inspect', 'clean', 'repair', 'replace', 'removal', 'inspection'))
    clean = (not later) and 'clean' in h
    iso = any(w in (h + ' ' + ai.lower()) for w in ('isolat', 'disconnect', 'de-energ', 'shutdown', 'shut down'))
    return dict(on_site_work_now=onsite, cleaning_now=clean, isolate_or_disconnect=iso)
def violations(w, p, fl):
    v = []
    if w == 'Lightning' and (fl['on_site_work_now'] or fl['cleaning_now']): v.append('P1')
    if p == 'Electrical_damage' and not fl['isolate_or_disconnect']: v.append('P2')
    if w in ICE and fl['cleaning_now']: v.append('P3')
    if w in STORM and fl['on_site_work_now']: v.append('P4')
    return v
# SRM v2: the v1 rules that violate a principle are corrected; risk levels are unchanged
FIX = {'Lightning-Bird_drop': 'No on-site work; inspect after the storm (30 min after the last thunder)',
       'Lightning-Dusty': 'No on-site work; inspect after the storm (30 min after the last thunder)',
       'Lightning-Physical_Damage': 'No on-site work; inspect after the storm (30 min after the last thunder)',
       'Lightning-Electrical_damage': 'No on-site work; remote isolation; de-energised inspection by qualified personnel after the storm',
       'Glaze-Bird_drop': 'Delay cleaning until the ice has melted', 'Glaze-Dusty': 'Delay cleaning until the ice has melted',
       'Sandstorm-Bird_drop': 'Inspect after the storm', 'Sandstorm-Physical_Damage': 'Inspect after the storm',
       'Hail-Bird_drop': 'Optional inspection after the storm', 'Hail-Clean': 'Optional inspection after the storm',
       'Hail-Electrical_damage': 'De-energised inspection by qualified personnel after the storm', 'Hail-Physical_Damage': 'Replace damaged panels after the storm',
       'Hail-Snow_Covered': 'Optional removal after the storm'}
SRM_V2 = {k: dict(v, Human=FIX.get(k, v['Human'])) for k, v in SRM_V1.items()}
AUDIT = {}
for name, S in (('v1', SRM_V1), ('v2', SRM_V2)):
    rows = []
    for w in W:
        for p in PNL:
            r = S[f'{w}-{p}']; fl = flags_from_text(r['Human'], r['AI']); v = violations(w, p, fl)
            rows.append(dict(rule=f'{w}-{p}', human=r['Human'], ai=r['AI'], risk=r['Risk'], violations=v))
    AUDIT[name] = dict(n_rules=len(rows), n_rules_violating=sum(bool(r['violations']) for r in rows),
                       by_principle={P: sum(P in r['violations'] for r in rows) for P in ('P1', 'P2', 'P3', 'P4')}, rules=rows)
    print(f'SRM {name}: {AUDIT[name]["n_rules_violating"]} of 66 rules violate a stated principle {AUDIT[name]["by_principle"]}')
for r in AUDIT['v1']['rules']:
    if r['violations']: print(f"   v1 {r['rule']:28s} {r['violations']}  '{r['human']}'  ->  '{SRM_V2[r['rule']]['Human']}'")

# weather dependence: how many decisions change with the weather for a fixed panel state
DEC = lambda S, w, p: (S[f'{w}-{p}']['Risk'], S[f'{w}-{p}']['Human'], S[f'{w}-{p}']['AI'])
dep = {}
for p in PNL:
    ds = [DEC(SRM_V2, w, p) for w in W]; rs = [d[0] for d in ds]
    mode_dec = max(set(ds), key=ds.count); mode_risk = max(set(rs), key=rs.count)
    dep[p] = dict(distinct_decisions=len(set(ds)), distinct_risk_levels=len(set(rs)),
                  weathers_where_panel_only_decision_differs=sum(d != mode_dec for d in ds),
                  weathers_where_panel_only_risk_differs=sum(r != mode_risk for r in rs),
                  weathers_where_panel_only_risk_is_lower=sum(RISK[mode_risk] < RISK[r] for r in rs))
tot = {k: sum(v[k] for v in dep.values()) for k in ('weathers_where_panel_only_decision_differs', 'weathers_where_panel_only_risk_differs', 'weathers_where_panel_only_risk_is_lower')}
print('weather dependence (SRM v2):', tot, 'of 66 scenarios')

# propagation of classification errors to decisions (random pairing of a weather test image and a panel test image)
def propagate(seed, n=20000, S=SRM_V2):
    rng = np.random.default_rng(100 + seed)
    Pw = PROBS[('weather', PROPOSED['weather'], seed)]; Pp = PROBS[('panel', PROPOSED['panel'], seed)]
    iw = rng.integers(0, len(Pw['y']), n); ip = rng.integers(0, len(Pp['y']), n)
    cw_, cp_ = DATA['weather']['classes'], DATA['panel']['classes']
    norm = lambda s: {x.lower(): x for x in W}.get(s.lower().replace('-', '').replace('_', ''), s)
    wt = [norm(cw_[k]) for k in Pw['y'][iw]]; wp = [norm(cw_[k]) for k in Pw['P'][iw].argmax(1)]
    normp = lambda s: {x.lower(): x for x in PNL}.get(s.lower().replace('-', '_'), s)
    pt = [normp(cp_[k]) for k in Pp['y'][ip]]; pp = [normp(cp_[k]) for k in Pp['P'][ip].argmax(1)]
    dt = [DEC(S, a, b) for a, b in zip(wt, pt)]; dp = [DEC(S, a, b) for a, b in zip(wp, pp)]
    mode = {p: max(set([DEC(S, w, p) for w in W]), key=[DEC(S, w, p) for w in W].count) for p in PNL}
    d_panel_only = [mode[b] for b in pp]
    return dict(both_correct=float(np.mean([(a == b) and (c == d) for a, b, c, d in zip(wt, wp, pt, pp)])),
                decision_exact=float(np.mean([a == b for a, b in zip(dt, dp)])), risk_agree=float(np.mean([a[0] == b[0] for a, b in zip(dt, dp)])),
                risk_under=float(np.mean([RISK[b[0]] < RISK[a[0]] for a, b in zip(dt, dp)])),
                panel_only_decision_exact=float(np.mean([a == b for a, b in zip(dt, d_panel_only)])),
                panel_only_risk_under=float(np.mean([RISK[b[0]] < RISK[a[0]] for a, b in zip(dt, d_panel_only)])))
PROP = [propagate(s) for s in SEEDS]
PROPS = {k: dict(mean=float(np.mean([p[k] for p in PROP])), sd=float(np.std([p[k] for p in PROP], ddof=1))) for k in PROP[0]}
for k, v in PROPS.items(): print(f'  {k:28s} {v["mean"]:.4f} ± {v["sd"]:.4f}')
save_small('srm_analysis.json', dict(audit=AUDIT, weather_dependence=dict(per_panel=dep, totals=tot), propagation=PROPS, srm_v2=SRM_V2))

def _final():
    # ---------- Final: one file with everything the article reports ----------
    ALL = {}
    for f in ('data_summary.json', 'summary.json', 'efficiency.json', 'label_audit.json', 'srm_analysis.json', 'llm_eval.json'):
        if os.path.exists(f'{OUT}/{f}'): ALL[f[:-5]] = json.load(open(f'{OUT}/{f}'))
    ALL['protocol'] = dict(seeds=SEEDS, img=IMG, batch=BATCH, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA, val_frac=VAL_FRAC, test_frac=TEST_FRAC, split_seed=SPLIT_SEED,
                           training='end-to-end fine-tuning, ImageNet-pretrained (SimpleCNN from scratch)', augmentation='random horizontal flip', normalisation='ImageNet mean/std',
                           optimizer='Adam', precision='bfloat16 autocast', selection='epoch with the best validation macro-F1; early stop after PATIENCE epochs without a gain > MIN_DELTA')
    ALL['environment'] = dict(torch=torch.__version__, timm=timm.__version__, python=sys.version.split()[0])
    save_small('ALL_RESULTS.json', ALL)
    print('Done:', OUT + '/ALL_RESULTS.json')

_final()

