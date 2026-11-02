import os, sys, subprocess, json, time, random, math, glob, hashlib, shutil, re, gc, copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

def pip_install(packages):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + packages, check=True)
pip_install(['open_clip_torch', 'ftfy', 'regex', 'scikit-learn', 'pandas', 'tqdm'])
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from IPython.display import display
import open_clip
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError as exc:
    raise RuntimeError('Bu notebook Google Colab için hazırlanmıştır.') from exc

@dataclass
class CFG:
    run_tag: str = 'avion_applenet_capstone_protocol_matched_v1'
    protocol_version: str = 'capstone_clip_disjoint5fold_validation_free_v11'
    support_namespace: str = 'capstone_disjoint5fold_support5_v1'
    drive_root: str = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'
    data_root: str = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/data'
    feature_cache_dir: str = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/feature_cache/cache_capstone_preproj_v1'
    output_root: str = ''
    local_cache: str = '/content/capstone_pm_cache'
    datasets: list = field(default_factory=lambda: ['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT'])
    expected_counts: dict = field(default_factory=lambda: {'NWPU_RESISC45': 31500, 'PatternNet': 30400, 'AID': 10000, 'UCMerced': 2100, 'WHU_RS19': 1005, 'EuroSAT': 27000})
    expected_class_counts: dict = field(default_factory=lambda: {'NWPU_RESISC45': 45, 'PatternNet': 38, 'AID': 30, 'UCMerced': 21, 'WHU_RS19': 19, 'EuroSAT': 10})
    primary_backbone: str = 'openclip_b32'
    arch: str = 'ViT-B-32'
    pretrained_candidates: list = field(default_factory=lambda: ['laion2b_s34b_b79k', 'openai'])
    n_outer_folds: int = 5
    fold_seed: int = 20260717
    shots: list = field(default_factory=lambda: [1, 2, 4, 8, 16])
    support_seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    methods: list = field(default_factory=lambda: ['AVION-PM', 'APPLeNet-PM'])
    avion_epochs: int = 80
    applenet_epochs: int = 80
    batch_size: int = 64
    eval_chunk: int = 2048
    weight_decay: float = 0.001
    avion_lr: float = 0.0002
    applenet_lr: float = 0.0005
    applenet_anticorr: float = 0.05
    ece_bins: int = 15
    require_existing_capstone_cache: bool = True
    require_manifest_match: bool = True
    force_rerun: bool = False
    save_model_checkpoints: bool = False
    run_mode: str = 'paper'
cfg = CFG()
cfg.output_root = os.path.join(cfg.drive_root, f'outputs_{cfg.run_tag}')
if cfg.run_mode == 'smoke':
    cfg.datasets = ['NWPU_RESISC45']
    cfg.shots = [1]
    cfg.support_seeds = [0]
    cfg.n_outer_folds = 1
    cfg.avion_epochs = 2
    cfg.applenet_epochs = 2
elif cfg.run_mode != 'paper':
    raise ValueError("run_mode yalnızca 'paper' veya 'smoke' olabilir")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
    raise RuntimeError('Colab çalışma zamanından GPU seçin: Runtime > Change runtime type > GPU')
for p in [cfg.output_root, cfg.local_cache, os.path.join(cfg.output_root, 'tables'), os.path.join(cfg.output_root, 'checkpoints')]:
    os.makedirs(p, exist_ok=True)
print('Device:', torch.cuda.get_device_name(0))
print('Output:', cfg.output_root)
print('Protocol:', cfg.protocol_version)
print('Target validation: DISABLED')
print('Methods:', cfg.methods)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def set_seed(seed):
    seed = int(seed) % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stable_int(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

def norm_name(s):
    return ''.join((ch for ch in str(s).lower() if ch.isalnum()))

def prettify(name):
    return ' '.join(str(name).replace('_', ' ').replace('-', ' ').strip().lower().split())

def resolve_dataset_folder(name):
    direct = os.path.join(cfg.data_root, name)
    if os.path.isdir(direct):
        return direct
    aliases = {'nwpu_resisc45': ['nwpu-resisc45', 'nwpu_resisc45', 'resisc45'], 'ucmerced': ['ucmerced', 'ucmercedlanduse', 'ucm'], 'whu_rs19': ['whu_rs19', 'whurs19', 'whu-rs19', 'whu19'], 'eurosat': ['eurosat', 'eurosatrgb', 'eurosat_rgb']}
    candidates = [d for d in sorted(os.listdir(cfg.data_root)) if os.path.isdir(os.path.join(cfg.data_root, d))]
    wanted = {norm_name(name)} | {norm_name(x) for x in aliases.get(name.lower(), [])}
    for d in candidates:
        if norm_name(d) in wanted:
            return os.path.join(cfg.data_root, d)
    raise FileNotFoundError(f'Veri seti klasörü bulunamadı: {name}')

def flat_filename_class(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    cls = re.sub('[_\\-\\s]*\\d+$', '', stem).strip('_- ')
    return cls or stem

def classification_groups(root):
    image_paths = sorted((p for p in glob.iglob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(p) and p.lower().endswith(IMG_EXT)))
    if not image_paths:
        return {}
    by_parent = defaultdict(list)
    for p in image_paths:
        by_parent[os.path.dirname(p)].append(p)
    if len(by_parent) > 1:
        groups = defaultdict(list)
        for parent, files in sorted(by_parent.items()):
            groups[os.path.basename(parent)].extend(files)
        return {k: sorted(v) for k, v in groups.items()}
    only_files = next(iter(by_parent.values()))
    by_prefix = defaultdict(list)
    for p in only_files:
        by_prefix[flat_filename_class(p)].append(p)
    return {k: sorted(v) for k, v in by_prefix.items()}

def scan_dataset(root, dataset_name):
    groups = classification_groups(root)
    if not groups:
        raise RuntimeError(f'Görüntü bulunamadı: {root}')
    class_names = sorted(groups)
    if len(class_names) != cfg.expected_class_counts[dataset_name]:
        raise RuntimeError(f'{dataset_name}: {len(class_names)} sınıf bulundu, beklenen {cfg.expected_class_counts[dataset_name]}')
    samples = []
    for label, cls in enumerate(class_names):
        samples.extend(((p, label) for p in groups[cls]))
    if len(samples) < cfg.expected_counts[dataset_name]:
        raise RuntimeError(f'{dataset_name}: {len(samples)} < {cfg.expected_counts[dataset_name]}')
    return (samples, class_names)
DATA_AT = {name: resolve_dataset_folder(name) for name in cfg.datasets}
for k, v in DATA_AT.items():
    print(f'{k:>16}: {v}')

class ImagePathDataset(Dataset):

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert('RGB'))

class CLIPVisualCacheWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
        self.proj = getattr(self.visual, 'proj', None)
        try:
            self.visual.output_tokens = True
        except Exception:
            pass

    def forward(self, x):
        out = self.visual(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            pooled, tokens = (out[0], out[1])
        else:
            pooled, tokens = (out, None)
        if tokens is not None and tokens.shape[-1] != pooled.shape[-1]:
            if isinstance(self.proj, nn.Linear):
                tokens = self.proj(tokens)
            elif self.proj is not None:
                tokens = tokens @ self.proj
        return (pooled, tokens)

def load_openclip():
    errors = []
    for tag in cfg.pretrained_candidates:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(cfg.arch, pretrained=tag)
            tokenizer = open_clip.get_tokenizer(cfg.arch)
            model = model.to(DEVICE).eval()
            for p in model.parameters():
                p.requires_grad = False
            print('Loaded OpenCLIP:', cfg.arch, tag)
            return (model, preprocess, tokenizer, tag)
        except Exception as exc:
            errors.append(f'{tag}: {type(exc).__name__}: {exc}')
    raise RuntimeError('OpenCLIP yüklenemedi: ' + ' | '.join(errors))
clip_model, preprocess, tokenizer, LOADED_PRETRAINED = load_openclip()
visual_wrapper = CLIPVisualCacheWrapper(clip_model).to(DEVICE).eval()
with torch.no_grad():
    EMBED_DIM = int(clip_model.encode_text(tokenizer(['a satellite image']).to(DEVICE)).shape[-1])
LOGIT_SCALE = float(clip_model.logit_scale.exp().detach().cpu())
print('Embedding dim:', EMBED_DIM, '| logit scale:', LOGIT_SCALE)
PROMPT_TEMPLATES = ['a satellite photo of a {}.', 'an aerial image of a {}.', 'a remote sensing image of {}.', 'a high resolution satellite image of a {}.', 'a top view of {}.']

@torch.no_grad()
def encode_texts(texts, batch_size=256):
    outputs = []
    for s in range(0, len(texts), batch_size):
        tok = tokenizer(texts[s:s + batch_size]).to(DEVICE)
        outputs.append(F.normalize(clip_model.encode_text(tok).float(), dim=-1).cpu())
    return torch.cat(outputs, 0)

@torch.no_grad()
def class_text_features(class_names):
    rows = []
    for cls in class_names:
        texts = [t.format(prettify(cls)) for t in PROMPT_TEMPLATES]
        rows.append(F.normalize(encode_texts(texts).mean(0), dim=-1))
    return torch.stack(rows, 0)

def cache_path(name):
    return os.path.join(cfg.feature_cache_dir, f'{cfg.primary_backbone}__{name}__capstone_preproj_v1__dataset.pt')

@torch.no_grad()
def build_cache(name, samples, class_names):
    paths = [p for p, _ in samples]
    y = torch.tensor([v for _, v in samples], dtype=torch.long)
    loader = DataLoader(ImagePathDataset(paths, preprocess), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    gs, ts = ([], [])
    for x in tqdm(loader, desc=f'encode {name}'):
        with torch.autocast('cuda', enabled=True):
            g, t = visual_wrapper(x.to(DEVICE, non_blocking=True))
        if t is None:
            raise RuntimeError('APPLeNet-PM için patch token cache üretilemedi')
        gs.append(g.half().cpu())
        ts.append(t.half().cpu())
    blob = {'g': torch.cat(gs), 't': torch.cat(ts), 'y': y, 'paths': paths, 'n': len(paths), 'has_tokens': True, 'path_digest': hashlib.sha256('\n'.join(paths).encode()).hexdigest()}
    os.makedirs(cfg.feature_cache_dir, exist_ok=True)
    torch.save(blob, cache_path(name))
    return blob
CACHES = {}
CLASS_NAMES = {}
BASE_TEXT = {}
for name in cfg.datasets:
    samples, class_names = scan_dataset(DATA_AT[name], name)
    cp = cache_path(name)
    if os.path.isfile(cp):
        print('[CAPSTONE cache]', name, cp)
        blob = torch.load(cp, map_location='cpu', weights_only=False)
    elif cfg.require_existing_capstone_cache:
        raise FileNotFoundError(f'CAPSTONE cache bulunamadı: {cp}\nÖnce ana CAPSTONE notebookunun feature-cache hücrelerini çalıştırın.')
    else:
        blob = build_cache(name, samples, class_names)
    if int(blob.get('n', len(blob['y']))) != cfg.expected_counts[name]:
        raise RuntimeError(f"{name}: cache n={blob.get('n')} beklenen={cfg.expected_counts[name]}")
    if int(torch.unique(blob['y']).numel()) != cfg.expected_class_counts[name]:
        raise RuntimeError(f'{name}: cache sınıf sayısı uyuşmuyor')
    if blob.get('t') is None:
        raise RuntimeError(f'{name}: APPLeNet-PM için token cache yok')
    CACHES[name] = {'g': blob['g'].contiguous(), 't': blob['t'].contiguous(), 'y': blob['y'].long().contiguous(), 'paths': blob.get('paths', []), 'path_digest': blob.get('path_digest')}
    CLASS_NAMES[name] = class_names
    BASE_TEXT[name] = class_text_features(class_names)
    print(name, 'g', tuple(CACHES[name]['g'].shape), 't', tuple(CACHES[name]['t'].shape))
gc.collect()
torch.cuda.empty_cache()
FULL_N_OUTER_FOLDS = 5

def build_disjoint_folds(labels):
    labels = np.asarray(labels, dtype=int)
    skf = StratifiedKFold(n_splits=FULL_N_OUTER_FOLDS, shuffle=True, random_state=cfg.fold_seed)
    out = {}
    for oid, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        out[oid] = {'train_pool': np.asarray(train_idx, dtype=np.int64), 'test': np.asarray(test_idx, dtype=np.int64)}
    all_test = np.concatenate([out[k]['test'] for k in sorted(out)])
    assert len(all_test) == len(labels) and len(np.unique(all_test)) == len(labels)
    return out
SPLITS = {name: build_disjoint_folds(CACHES[name]['y'].numpy()) for name in cfg.datasets}

def support_indices(name, K, seed, outer_id):
    pool = SPLITS[name][int(outer_id)]['train_pool']
    y = CACHES[name]['y'].numpy()
    classes = sorted(np.unique(y))
    rng = np.random.RandomState(stable_int(cfg.support_namespace, name, int(outer_id), int(K), int(seed), classes))
    out = []
    for c in classes:
        cand = pool[y[pool] == c].copy()
        if len(cand) < K:
            raise RuntimeError(f'{name}: class {c}: {len(cand)} < K={K}')
        rng.shuffle(cand)
        out.extend(cand[:K].tolist())
    return np.asarray(out, dtype=np.int64)

def array_sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()

def locate_capstone_manifest():
    exact = os.path.join(cfg.drive_root, 'outputs_q1_capstone_clip_v11_full_external', f'split_manifest_{cfg.support_namespace}.json')
    candidates = []
    if os.path.isfile(exact):
        candidates.append(exact)
    candidates += glob.glob(os.path.join(cfg.drive_root, 'outputs_*', f'split_manifest_{cfg.support_namespace}.json'))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0] if candidates else None
MANIFEST_PATH = locate_capstone_manifest()
MANIFEST_AUDIT = []
if MANIFEST_PATH:
    manifest = json.load(open(MANIFEST_PATH, encoding='utf-8'))
    for name in cfg.datasets:
        for oid in range(FULL_N_OUTER_FOLDS):
            expected = manifest['datasets'][name][str(oid)]
            split = SPLITS[name][oid]
            checks = {'train_pool': array_sha(split['train_pool']) == expected['train_pool_sha256'], 'test': array_sha(split['test']) == expected['test_sha256']}
            for K in [1, 2, 4, 8, 16]:
                for seed in [0, 1, 2, 3, 4]:
                    key = f'K{K}_s{seed}'
                    checks[key] = array_sha(support_indices(name, K, seed, oid)) == expected['supports'][key]
            MANIFEST_AUDIT.append({'dataset': name, 'fold': oid, 'all_match': all(checks.values()), 'failed': [k for k, v in checks.items() if not v]})
else:
    manifest = None
manifest_ok = bool(MANIFEST_AUDIT) and all((x['all_match'] for x in MANIFEST_AUDIT))
print('CAPSTONE manifest:', MANIFEST_PATH)
print('Manifest birebir eşleşme:', manifest_ok)
if cfg.require_manifest_match and (not manifest_ok):
    raise RuntimeError("CAPSTONE split manifesti bulunamadı veya hash'ler birebir eşleşmedi. Makale karşılaştırması için koşu durduruldu.")
PROTOCOL_PAYLOAD = {'protocol_version': cfg.protocol_version, 'support_namespace': cfg.support_namespace, 'fold_seed': cfg.fold_seed, 'n_outer_folds': FULL_N_OUTER_FOLDS, 'shots': [1, 2, 4, 8, 16], 'support_seeds': [0, 1, 2, 3, 4], 'datasets': ['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT'], 'backbone': cfg.primary_backbone, 'loaded_pretrained': LOADED_PRETRAINED, 'target_validation': False, 'manifest_path': MANIFEST_PATH, 'manifest_verified': manifest_ok, 'implementations': {'AVION-PM': 'protocol-matched feature-level reimplementation', 'APPLeNet-PM': 'protocol-matched token-conditioned reimplementation'}}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL_PAYLOAD, sort_keys=True).encode()).hexdigest()[:16]
json.dump({'protocol': PROTOCOL_PAYLOAD, 'audit': MANIFEST_AUDIT}, open(os.path.join(cfg.output_root, 'protocol_audit.json'), 'w'), indent=2)
print('Protocol hash:', PROTOCOL_HASH)

def expected_calibration_error(probs, y, n_bins=None):
    n_bins = n_bins or cfg.ece_bins
    conf, pred = probs.max(1)
    correct = pred.eq(y)
    ece = torch.tensor(0.0, device=probs.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].float().mean()).abs()
    return float(ece.detach().cpu())

def metrics_from_logits(logits, y):
    probs = logits.softmax(-1)
    pred = probs.argmax(-1)
    yc = y.detach().cpu().numpy()
    pc = pred.detach().cpu().numpy()
    return {'Accuracy': float(accuracy_score(yc, pc)), 'MacroF1': float(f1_score(yc, pc, average='macro')), 'BalancedAccuracy': float(balanced_accuracy_score(yc, pc)), 'NLL': float(F.cross_entropy(logits, y).detach().cpu()), 'ECE': expected_calibration_error(probs, y), 'N': int(len(y))}

class ResidualMLP(nn.Module):

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(16, dim // reduction)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)

class GatedResidual(nn.Module):

    def __init__(self, dim, reduction=4, init_gate=0.1):
        super().__init__()
        self.block = ResidualMLP(dim, reduction)
        init = float(np.log(init_gate / (1 - init_gate)))
        self.gate_logit = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x):
        return x + self.gate() * self.block(x)

def n_trainable(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def indexed(cache, key, idx):
    i = torch.as_tensor(idx, dtype=torch.long)
    return cache[key][i]

def iterate_minibatches(n, batch_size, generator):
    order = torch.randperm(n, generator=generator)
    for s in range(0, n, batch_size):
        yield order[s:s + batch_size]
AVION_TEMPLATE_BANK = ['an overhead satellite view of {}.', 'a high-resolution aerial scene showing {}.', 'remote sensing imagery of {} viewed from above.', 'a top-down image containing the characteristic structures of {}.', 'an aerial photograph highlighting the spatial layout of {}.', 'a satellite scene with visual patterns typical of {}.', 'an overhead view showing the land-cover context of {}.', 'a remote sensing scene depicting {} and its surrounding structures.', 'a nadir-view aerial image of {}.', 'a satellite image emphasizing the geometry and texture of {}.', 'an aerial scene where {} is identifiable from its spatial arrangement.', 'an overhead remote sensing photograph centered on {}.', 'a high-altitude image showing the distinctive appearance of {}.', 'a top-view scene with objects and surfaces associated with {}.', 'satellite imagery capturing the broader context of {}.', 'an aerial land-use scene categorized as {}.']
AVION_DESCRIPTION_ROOT = os.path.join(cfg.drive_root, 'external_official_baselines', 'avion_data', 'teacher_descriptions')
AVION_CANDIDATE_CACHE = {}

def load_avion_descriptions(name, class_names):
    candidates = []
    source = 'deterministic_remote_sensing_templates'
    json_candidates = [os.path.join(AVION_DESCRIPTION_ROOT, f'{name}.json'), os.path.join(AVION_DESCRIPTION_ROOT, f'{name.lower()}.json')]
    external = None
    for p in json_candidates:
        if os.path.isfile(p):
            external = json.load(open(p, encoding='utf-8'))
            source = p
            break
    for cls in class_names:
        key_options = [cls, prettify(cls), norm_name(cls)]
        texts = None
        if isinstance(external, dict):
            for k in key_options:
                if k in external and isinstance(external[k], list):
                    texts = [str(x).strip() for x in external[k] if str(x).strip()]
                    break
        if not texts or len(texts) < 5:
            texts = [t.format(prettify(cls)) for t in AVION_TEMPLATE_BANK]
        candidates.append(texts)
    return (candidates, source)

@torch.no_grad()
def avion_candidate_embeddings(name):
    if name in AVION_CANDIDATE_CACHE:
        return AVION_CANDIDATE_CACHE[name]
    descriptions, source = load_avion_descriptions(name, CLASS_NAMES[name])
    rows = []
    for texts in descriptions:
        rows.append(encode_texts(texts))
    J = min((len(x) for x in rows))
    emb = torch.stack([x[:J] for x in rows], 0)
    AVION_CANDIDATE_CACHE[name] = (emb, source, J)
    return (emb, source, J)

@torch.no_grad()
def build_avion_teacher(name, support_idx, beta=10.0, mad_threshold=3.0):
    cache = CACHES[name]
    y = cache['y']
    g = F.normalize(indexed(cache, 'g', support_idx).float(), dim=-1)
    ys = y[torch.as_tensor(support_idx, dtype=torch.long)]
    C = len(CLASS_NAMES[name])
    visual = []
    for c in range(C):
        visual.append(F.normalize(g[ys == c].mean(0), dim=-1))
    visual = torch.stack(visual, 0)
    cand, source, J = avion_candidate_embeddings(name)
    sim = torch.einsum('cd,cjd->cj', visual, cand)
    proto = []
    kept_counts = []
    for c in range(C):
        s = sim[c]
        med = s.median()
        mad = (s - med).abs().median().clamp_min(1e-08)
        robust_z = 0.6745 * (s - med) / mad
        keep = robust_z >= -mad_threshold
        if int(keep.sum()) < 3:
            keep = torch.ones_like(keep, dtype=torch.bool)
        weights = torch.softmax(beta * s[keep], dim=0)
        proto.append(F.normalize((weights[:, None] * cand[c, keep]).sum(0), dim=-1))
        kept_counts.append(int(keep.sum()))
    return (visual, torch.stack(proto, 0), {'teacher_source': source, 'candidate_count': J, 'mean_kept_candidates': float(np.mean(kept_counts))})

class AvionPM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.visual_prompt = GatedResidual(dim, reduction=4, init_gate=0.1)
        self.text_prompt = GatedResidual(dim, reduction=4, init_gate=0.1)
        self.logit_scale = LOGIT_SCALE

    def encode_image(self, g):
        return F.normalize(self.visual_prompt(g), dim=-1)

    def encode_text(self, t):
        return F.normalize(self.text_prompt(t), dim=-1)

    def logits(self, g, base_text):
        return self.logit_scale * self.encode_image(g) @ self.encode_text(base_text).t()

def train_avion_pm(name, K, seed, fold):
    run_seed = stable_int('AVION-PM', name, K, seed, fold, PROTOCOL_HASH)
    set_seed(run_seed)
    support = support_indices(name, K, seed, fold)
    cache = CACHES[name]
    x = indexed(cache, 'g', support).float().to(DEVICE)
    y = indexed(cache, 'y', support).long().to(DEVICE)
    base = BASE_TEXT[name].float().to(DEVICE)
    _, teacher_text, teacher_info = build_avion_teacher(name, support)
    teacher_text = teacher_text.float().to(DEVICE)
    frozen_x = F.normalize(x, dim=-1)
    teacher_logits = LOGIT_SCALE * frozen_x @ teacher_text.t()
    model = AvionPM(EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.avion_lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device='cpu').manual_seed(run_seed)
    temperature = 2.0
    for _ in range(cfg.avion_epochs):
        for bi in iterate_minibatches(len(y), cfg.batch_size, gen):
            bi = bi.to(DEVICE)
            sx = model.encode_image(x[bi])
            st = model.encode_text(base)
            logits = LOGIT_SCALE * sx @ st.t()
            task = F.cross_entropy(logits, y[bi])
            img_align = (1 - (sx * frozen_x[bi]).sum(-1)).mean()
            text_align = (1 - (st * teacher_text).sum(-1)).mean()
            kd = F.kl_div(F.log_softmax(logits / temperature, dim=-1), F.softmax(teacher_logits[bi] / temperature, dim=-1), reduction='batchmean') * temperature ** 2
            loss = task + 0.5 * img_align + 0.5 * text_align + 1.0 * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return (model, {**teacher_info, 'run_seed': run_seed, 'epochs': cfg.avion_epochs})

@torch.no_grad()
def evaluate_avion_pm(model, name, test_idx):
    cache = CACHES[name]
    base = BASE_TEXT[name].float().to(DEVICE)
    logits_all = []
    y_all = []
    ids = torch.as_tensor(test_idx, dtype=torch.long)
    for s in range(0, len(ids), cfg.eval_chunk):
        ii = ids[s:s + cfg.eval_chunk]
        g = cache['g'][ii].float().to(DEVICE, non_blocking=True)
        logits_all.append(model.logits(g, base).cpu())
        y_all.append(cache['y'][ii].long().cpu())
    return metrics_from_logits(torch.cat(logits_all), torch.cat(y_all))

class AppleNetPM(nn.Module):

    def __init__(self, base_text, dim, n_visual_tokens=4):
        super().__init__()
        self.register_buffer('base_text', base_text.clone())
        self.n_visual_tokens = n_visual_tokens
        hidden = max(64, dim // 2)
        self.content = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.style = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.inject = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_visual_tokens * dim))
        self.shared_context = nn.Parameter(torch.zeros(1, base_text.shape[0], dim))
        self.prompt_scale = nn.Parameter(torch.tensor(0.1))
        self.logit_scale = LOGIT_SCALE

    def forward(self, g, tokens, style_stats, return_reg=False):
        g = F.normalize(g, dim=-1)
        tokens = tokens.float()
        token_mean = tokens.mean(1)
        token_std = tokens.std(1, unbiased=False)
        content = self.content(torch.cat([g, token_mean, token_std], dim=-1))
        style = self.style(style_stats).expand(len(g), -1)
        visual_tokens = self.inject(content + style).reshape(len(g), self.n_visual_tokens, g.shape[-1])
        visual_tokens = F.normalize(visual_tokens, dim=-1)
        pooled = F.normalize(visual_tokens.mean(1), dim=-1)
        affinity = torch.einsum('bd,cd->bc', pooled, self.base_text)
        delta = torch.tanh(affinity).unsqueeze(-1) * pooled.unsqueeze(1)
        class_text = F.normalize(self.base_text.unsqueeze(0) + self.shared_context + self.prompt_scale.tanh() * delta, dim=-1)
        logits = self.logit_scale * torch.einsum('bd,bcd->bc', g, class_text)
        if not return_reg:
            return logits
        gram = torch.bmm(visual_tokens, visual_tokens.transpose(1, 2))
        eye = torch.eye(self.n_visual_tokens, device=gram.device).unsqueeze(0)
        anticorr = ((gram - eye) * (1 - eye)).pow(2).mean()
        return (logits, anticorr)

def train_applenet_pm(name, K, seed, fold):
    run_seed = stable_int('APPLeNet-PM', name, K, seed, fold, PROTOCOL_HASH)
    set_seed(run_seed)
    support = support_indices(name, K, seed, fold)
    cache = CACHES[name]
    x = indexed(cache, 'g', support).float().to(DEVICE)
    tokens = indexed(cache, 't', support).float().to(DEVICE)
    y = indexed(cache, 'y', support).long().to(DEVICE)
    style_stats = torch.cat([F.normalize(x, dim=-1).mean(0), F.normalize(x, dim=-1).std(0, unbiased=False)], dim=0).unsqueeze(0)
    model = AppleNetPM(BASE_TEXT[name].float().to(DEVICE), EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.applenet_lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device='cpu').manual_seed(run_seed)
    for _ in range(cfg.applenet_epochs):
        for bi in iterate_minibatches(len(y), cfg.batch_size, gen):
            bi = bi.to(DEVICE)
            logits, anticorr = model(x[bi], tokens[bi], style_stats, return_reg=True)
            loss = F.cross_entropy(logits, y[bi]) + cfg.applenet_anticorr * anticorr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return (model, style_stats.detach(), {'run_seed': run_seed, 'epochs': cfg.applenet_epochs, 'n_visual_tokens': model.n_visual_tokens, 'uses_final_patch_tokens': True})

@torch.no_grad()
def evaluate_applenet_pm(model, style_stats, name, test_idx):
    cache = CACHES[name]
    ids = torch.as_tensor(test_idx, dtype=torch.long)
    logits_all = []
    y_all = []
    for s in range(0, len(ids), cfg.eval_chunk):
        ii = ids[s:s + cfg.eval_chunk]
        g = cache['g'][ii].float().to(DEVICE, non_blocking=True)
        t = cache['t'][ii].float().to(DEVICE, non_blocking=True)
        logits_all.append(model(g, t, style_stats).cpu())
        y_all.append(cache['y'][ii].long().cpu())
    return metrics_from_logits(torch.cat(logits_all), torch.cat(y_all))
STATE_PATH = os.path.join(cfg.output_root, 'state_protocol_matched.json')

def load_state():
    if not os.path.isfile(STATE_PATH):
        return {'__protocol_hash__': PROTOCOL_HASH, 'runs': {}}
    state = json.load(open(STATE_PATH, encoding='utf-8'))
    if state.get('__protocol_hash__') != PROTOCOL_HASH:
        raise RuntimeError(f"Eski state protokolü farklı: {state.get('__protocol_hash__')} != {PROTOCOL_HASH}. Yeni run_tag kullanın.")
    state.setdefault('runs', {})
    return state

def save_state(state):
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_PATH)
STATE = load_state()

def run_key(method, name, K, fold, seed):
    return f'{method}/{name}/K{K}/f{fold}/s{seed}'

def run_one(method, name, K, fold, seed):
    key = run_key(method, name, K, fold, seed)
    if not cfg.force_rerun and key in STATE['runs'] and (STATE['runs'][key].get('status') == 'completed'):
        return STATE['runs'][key]
    test_idx = SPLITS[name][fold]['test']
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    model = None
    try:
        if method == 'AVION-PM':
            model, info = train_avion_pm(name, K, seed, fold)
            metrics = evaluate_avion_pm(model, name, test_idx)
            fidelity = 'Selective teacher text prototypes and tri-aspect alignment are retained; deep transformer prompt injection is represented by feature-level gated residual prompts.'
        elif method == 'APPLeNet-PM':
            model, style_stats, info = train_applenet_pm(name, K, seed, fold)
            metrics = evaluate_applenet_pm(model, style_stats, name, test_idx)
            fidelity = 'Image-conditioned token injection and anti-correlation are retained; CAPSTONE final-layer patch tokens replace the official multi-layer hook stack.'
        else:
            raise KeyError(method)
        torch.cuda.synchronize()
        row = {'status': 'completed', 'backbone': cfg.primary_backbone, 'dataset': name, 'K': int(K), 'fold': int(fold), 'support_seed': int(seed), 'method': method, **metrics, **info, 'train_eval_s': time.time() - t0, 'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024 ** 2, 'trainable_params': n_trainable(model), 'protocol_hash': PROTOCOL_HASH, 'target_validation': False, 'directly_comparable_protocol': True, 'official_repository_unchanged': False, 'implementation_label': 'protocol-matched reimplementation', 'fidelity_note': fidelity}
        if cfg.save_model_checkpoints:
            ck = os.path.join(cfg.output_root, 'checkpoints', re.sub('[^A-Za-z0-9_.-]', '_', key) + '.pt')
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'row': row}, ck)
    except Exception as exc:
        row = {'status': 'failed', 'backbone': cfg.primary_backbone, 'dataset': name, 'K': int(K), 'fold': int(fold), 'support_seed': int(seed), 'method': method, 'Accuracy': np.nan, 'MacroF1': np.nan, 'BalancedAccuracy': np.nan, 'NLL': np.nan, 'ECE': np.nan, 'N': 0, 'reason': f'{type(exc).__name__}: {exc}', 'protocol_hash': PROTOCOL_HASH}
        print('[FAILED]', key, row['reason'])
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    STATE['runs'][key] = row
    save_state(STATE)
    return row
print('Resume tamamlanmış koşu:', sum((v.get('status') == 'completed' for v in STATE['runs'].values())))
folds_to_run = [0] if cfg.run_mode == 'smoke' else list(range(FULL_N_OUTER_FOLDS))
jobs = [(method, name, K, fold, seed) for name in cfg.datasets for K in cfg.shots for fold in folds_to_run for seed in cfg.support_seeds for method in cfg.methods]
print('Toplam görev:', len(jobs))
rows = []
for method, name, K, fold, seed in tqdm(jobs, desc='AVION-PM + APPLeNet-PM'):
    rows.append(run_one(method, name, K, fold, seed))
raw_pm = pd.DataFrame(list(STATE['runs'].values()))
raw_path = os.path.join(cfg.output_root, 'tables', 'raw_protocol_matched_external.csv')
raw_pm.to_csv(raw_path, index=False)
print('Raw results:', raw_path)
display(raw_pm.groupby(['method', 'status']).size().reset_index(name='count'))
completed = raw_pm[raw_pm.status == 'completed'].copy()
metrics = ['Accuracy', 'MacroF1', 'BalancedAccuracy', 'NLL', 'ECE']
agg = {m: ['mean', 'std', 'count'] for m in metrics}
summary = completed.groupby(['backbone', 'dataset', 'K', 'method'], dropna=False).agg(agg)
summary.columns = [f'{a}_{b}' for a, b in summary.columns]
summary = summary.reset_index()
summary_path = os.path.join(cfg.output_root, 'tables', 'table_protocol_matched_summary.csv')
summary.to_csv(summary_path, index=False)

def locate_capstone_raw():
    exact = os.path.join(cfg.drive_root, 'outputs_q1_capstone_clip_v11_full_external', 'tables', 'raw_primary_classification.csv')
    candidates = []
    if os.path.isfile(exact):
        candidates.append(exact)
    candidates += glob.glob(os.path.join(cfg.drive_root, 'outputs_*', 'tables', 'raw_primary_classification.csv'))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0] if candidates else None
CAPSTONE_RAW_PATH = locate_capstone_raw()
if CAPSTONE_RAW_PATH:
    capstone_raw = pd.read_csv(CAPSTONE_RAW_PATH)
    combined = pd.concat([capstone_raw, completed], ignore_index=True, sort=False)
    combined_path = os.path.join(cfg.output_root, 'tables', 'combined_raw_capstone_pm.csv')
    combined.to_csv(combined_path, index=False)
    focus = combined[combined.method.isin(['CAPSTONE-CLIP', 'AVION-PM', 'APPLeNet-PM'])].copy()
    focus_summary = focus.groupby(['dataset', 'K', 'method'], dropna=False).agg(Accuracy_mean=('Accuracy', 'mean'), Accuracy_std=('Accuracy', 'std'), MacroF1_mean=('MacroF1', 'mean'), MacroF1_std=('MacroF1', 'std'), ECE_mean=('ECE', 'mean'), n_runs=('Accuracy', 'count')).reset_index()
    focus_path = os.path.join(cfg.output_root, 'tables', 'table_capstone_vs_avionpm_applenetpm.csv')
    focus_summary.to_csv(focus_path, index=False)
    print('CAPSTONE raw:', CAPSTONE_RAW_PATH)
    print('Combined:', combined_path)
    print('Main comparison:', focus_path)
    display(focus_summary.head(30))
else:
    print('CAPSTONE raw_primary_classification.csv bulunamadı; PM özeti yine kaydedildi.')
method_audit = pd.DataFrame([{'method': 'AVION-PM', 'same_capstone_split': manifest_ok, 'same_shots': True, 'same_support_seeds': True, 'same_backbone_key': True, 'target_validation': False, 'official_repository_unchanged': False, 'allowed_table_label': 'AVION-PM (protocol-matched reimplementation)', 'not_allowed_label': 'Official AVION'}, {'method': 'APPLeNet-PM', 'same_capstone_split': manifest_ok, 'same_shots': True, 'same_support_seeds': True, 'same_backbone_key': True, 'target_validation': False, 'official_repository_unchanged': False, 'allowed_table_label': 'APPLeNet-PM (protocol-matched reimplementation)', 'not_allowed_label': 'Official APPLeNet'}])
method_audit.to_csv(os.path.join(cfg.output_root, 'tables', 'method_fidelity_audit.csv'), index=False)
readme = f"# AVION-PM + APPLeNet-PM run summary\n\n- Protocol: {cfg.protocol_version}\n- Protocol hash: {PROTOCOL_HASH}\n- CAPSTONE manifest verified: {manifest_ok}\n- Backbone key: {cfg.primary_backbone}\n- Loaded OpenCLIP weights: {LOADED_PRETRAINED}\n- Target-domain validation: No\n- Completed runs: {int((raw_pm.status == 'completed').sum())}\n- Failed runs: {int((raw_pm.status == 'failed').sum())}\n\n## Manuscript rule\nUse the labels `AVION-PM (protocol-matched reimplementation)` and\n`APPLeNet-PM (protocol-matched reimplementation)`. Do not describe these rows\nas unchanged official-code results.\n"
open(os.path.join(cfg.output_root, 'RUN_SUMMARY.md'), 'w', encoding='utf-8').write(readme)
archive_base = f'/content/{cfg.run_tag}'
zip_path = shutil.make_archive(archive_base, 'zip', cfg.output_root)
print('Summary:', summary_path)
print('ZIP:', zip_path)
