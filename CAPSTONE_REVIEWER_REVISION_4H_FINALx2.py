import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
RUN_PROFILE = 'paper_full'

@dataclass
class CFG:
    run_tag: str = 'reviewer_revision_4h_v1'
    protocol_version: str = 'capstone_reviewer_revision_4h_v1'
    support_namespace: str = 'capstone_disjoint5fold_support5_v1'
    resume_from_run_tag: Optional[str] = None
    drive_root: str = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'
    data_root: str = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/data'
    out_root: str = ''
    local_out: str = ''
    local_data: str = '/content/data_local'
    cache_dirname: str = 'cache_capstone_preproj_v1'
    datasets: List[str] = field(default_factory=lambda: ['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT'])
    expected_counts: Dict[str, int] = field(default_factory=lambda: {'NWPU_RESISC45': 31500, 'PatternNet': 30400, 'AID': 10000, 'UCMerced': 2100, 'WHU_RS19': 1005, 'EuroSAT': 27000})
    expected_class_counts: Dict[str, int] = field(default_factory=lambda: {'NWPU_RESISC45': 45, 'PatternNet': 38, 'AID': 30, 'UCMerced': 21, 'WHU_RS19': 19, 'EuroSAT': 10})
    strict_dataset_integrity: bool = True
    auto_exclude_exact_duplicates: bool = True
    auto_exclude_filename_artifacts: bool = True
    caption_datasets: List[str] = field(default_factory=lambda: ['RSICD', 'RSITMD'])
    caption_dataset: str = 'RSICD'
    external_caption_dataset: str = 'RSITMD'
    require_second_caption_dataset: bool = True
    primary_backbone: str = 'openclip_b32'
    backbone_specs: List[Dict[str, Any]] = field(default_factory=lambda: [{'key': 'openclip_b32', 'label': 'OpenCLIP ViT-B/32', 'kind': 'openclip', 'arch': 'ViT-B-32', 'pretrained_candidates': ['laion2b_s34b_b79k', 'openai'], 'cache_tokens': True, 'cache_preproj': True}, {'key': 'remoteclip_b32', 'label': 'RemoteCLIP ViT-B/32', 'kind': 'checkpoint', 'arch': 'ViT-B-32', 'hf_repo': 'chendelong/RemoteCLIP', 'hf_filename': 'RemoteCLIP-ViT-B-32.pt', 'cache_tokens': True, 'cache_preproj': True}, {'key': 'georsclip_b32', 'label': 'GeoRSCLIP ViT-B/32', 'kind': 'checkpoint', 'arch': 'ViT-B-32', 'hf_repo': 'Zilun/GeoRSCLIP', 'hf_filename': 'ckpt/RS5M_ViT-B-32.pt', 'cache_tokens': True, 'cache_preproj': True}, {'key': 'openclip_b16', 'label': 'OpenCLIP ViT-B/16', 'kind': 'openclip', 'arch': 'ViT-B-16', 'pretrained_candidates': ['laion2b_s34b_b88k', 'openai'], 'cache_tokens': False, 'cache_preproj': True}])
    n_outer_folds: int = 5
    fold_seed: int = 20260717
    split_seed: int = 20260717
    shots: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    support_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    max_epochs: int = 80
    caption_epochs: int = 60
    sequential_caption_epochs: int = 40
    sequential_support_epochs: int = 40
    prolip_epochs: int = 300
    rla_epochs: int = 300
    hoso_epochs: int = 200
    tipf_epochs: int = 80
    coop_epochs: int = 80
    taskres_epochs: int = 100
    batch_size: int = 64
    caption_batch_size: int = 64
    caption_steps_per_epoch: int = 20
    weight_decay: float = 0.001
    eval_chunk: int = 8192
    presets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {'CLIP-Adapter': {'lr': 0.002, 'res': 0.2}, 'Tip-Adapter': {'alpha': 1.0, 'beta': 5.5}, 'Tip-Adapter-F': {'alpha': 1.0, 'beta': 5.5, 'lr': 0.0005}, 'CoOp': {'lr': 0.005}, 'TaskRes': {'lr': 0.005, 'alpha': 0.5}, 'RLA': {'lr': 0.001, 'lambda_rule': '1/K'}, 'ProLIP': {'lr': 0.001, 'lambda_rule': '1/K'}, 'HOSO-Adapter': {'adapter_lr': 0.002, 'alpha_lr': 0.1, 'res': 0.2}, 'Shared-Caption': {'lr': 0.0002, 'lambda_cap': 0.5}, 'CAPSTONE-CLIP': {'lr': 0.0002, 'lambda_cap': 0.5, 'lambda_shared_cls': 0.25, 'lambda_decor': 0.01, 'lambda_cons': 0.05, 'lambda_gate': 0.0001, 'caption_scale': 1.0, 'caption_mode': 'joint'}})
    caption_scales: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    caption_control_datasets: List[str] = field(default_factory=lambda: ['NWPU_RESISC45', 'AID', 'EuroSAT'])
    caption_control_shots: List[int] = field(default_factory=lambda: [1, 4, 16])
    caption_controls: List[str] = field(default_factory=lambda: ['no_caption', 'matched_budget', 'shuffled_caption', 'image_only', 'joint_full', 'sequential_full'])
    primary_methods: List[str] = field(default_factory=lambda: ['Zero-shot', 'LinearProbe', 'CLIP-Adapter', 'Tip-Adapter', 'Tip-Adapter-F', 'CoOp', 'TaskRes', 'RLA', 'ProLIP', 'HOSO-Adapter', 'Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP'])
    cross_backbone_methods: List[str] = field(default_factory=lambda: ['Zero-shot', 'RLA', 'ProLIP', 'Shared-Caption', 'CAPSTONE-CLIP'])
    cross_backbone_keys: List[str] = field(default_factory=lambda: ['openclip_b32', 'remoteclip_b32', 'georsclip_b32'])
    cross_backbone_shots: List[int] = field(default_factory=lambda: [1, 4, 16])
    base_to_new_datasets: List[str] = field(default_factory=lambda: ['NWPU_RESISC45', 'PatternNet', 'AID', 'EuroSAT'])
    base_to_new_shots: List[int] = field(default_factory=lambda: [4, 16])
    cross_dataset_source: str = 'NWPU_RESISC45'
    cross_dataset_shot: int = 4
    encode_batch: int = 384
    encode_workers: int = 0
    shard_size: int = 4096
    cache_to_drive: bool = True
    stage_from_zip: bool = True
    gpu_resident: bool = True
    retrieval_max_images: Optional[int] = None
    inferential_alpha: float = 0.05
    bootstrap_repetitions: int = 10000
    classification_noninferiority_margin: float = 0.01
    retrieval_noninferiority_margin: float = 0.01
    ece_bins: int = 15
    run_primary: bool = True
    run_caption_controls: bool = True
    run_retrieval: bool = True
    run_base_to_new: bool = True
    run_cross_dataset: bool = True
    run_cross_backbone: bool = True
    run_efficiency: bool = True
    run_near_duplicate_audit: bool = True
    run_architecture_ablations: bool = True
    run_class_diagnostics: bool = True
    ablation_datasets: List[str] = field(default_factory=lambda: ['NWPU_RESISC45', 'AID', 'EuroSAT'])
    ablation_shots: List[int] = field(default_factory=lambda: [1, 4, 16])
    diagnostic_shot: int = 4
    seed: int = 42
    smoke_per_class: int = 20
    SMOKE_TEST: bool = False
cfg = CFG()
if RUN_PROFILE == 'smoke':
    cfg.run_tag += '_smoke'
    cfg.SMOKE_TEST = True
    cfg.datasets = ['NWPU_RESISC45']
    cfg.expected_counts = {}
    cfg.expected_class_counts = {}
    cfg.caption_datasets = ['RSICD']
    cfg.require_second_caption_dataset = False
    cfg.n_outer_folds = 2
    cfg.shots = [1, 2]
    cfg.support_seeds = [0]
    cfg.primary_methods = ['Zero-shot', 'RLA', 'ProLIP', 'Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP']
    cfg.max_epochs = 2
    cfg.caption_epochs = 2
    cfg.prolip_epochs = 5
    cfg.rla_epochs = 5
    cfg.hoso_epochs = 5
    cfg.tipf_epochs = 2
    cfg.coop_epochs = 2
    cfg.taskres_epochs = 3
    cfg.caption_steps_per_epoch = 1
    cfg.caption_control_datasets = cfg.datasets
    cfg.caption_control_shots = [1]
    cfg.caption_scales = [0.25]
    cfg.caption_controls = ['no_caption', 'joint_full']
    cfg.ablation_datasets = cfg.datasets
    cfg.ablation_shots = [1]
    cfg.run_class_diagnostics = False
    cfg.cross_backbone_keys = [cfg.primary_backbone]
    cfg.cross_backbone_shots = [1]
    cfg.base_to_new_datasets = cfg.datasets
    cfg.base_to_new_shots = [1]
    cfg.retrieval_max_images = 80
elif RUN_PROFILE == 'paper_core':
    cfg.run_caption_controls = False
    cfg.run_base_to_new = False
    cfg.run_cross_dataset = False
    cfg.run_cross_backbone = False
    cfg.run_architecture_ablations = False
    cfg.run_class_diagnostics = False
elif RUN_PROFILE != 'paper_full':
    raise ValueError('RUN_PROFILE must be smoke, paper_core, or paper_full')
cfg.out_root = os.path.join(cfg.drive_root, f'outputs_{cfg.run_tag}')
base_local = '/content' if os.path.isdir('/content') else os.getcwd()
cfg.local_out = os.path.join(base_local, f'capstone_outputs_{cfg.run_tag}')
print('Model: CAPSTONE-CLIP')
print('Profile:', RUN_PROFILE)
print('Protocol:', cfg.protocol_version)
print('Target-domain validation: DISABLED')
print('Disjoint outer folds:', cfg.n_outer_folds, '| support seeds:', cfg.support_seeds)
print('Datasets:', cfg.datasets)
print('Backbones:', cfg.cross_backbone_keys)
print('Output:', cfg.out_root)
import sys, subprocess

def pip_install(pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=False)
pip_install(['open_clip_torch', 'ftfy', 'regex', 'scikit-learn', 'scipy', 'pandas', 'matplotlib', 'tqdm', 'huggingface_hub', 'statsmodels', 'psutil'])
print('Dependencies installed.')
import os, glob, json, time, random, math, shutil, warnings, itertools, hashlib, copy, gc, re
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
import psutil
from tqdm.auto import tqdm
T0_SESSION = time.time()
TIMING = {'staging_s': 0.0, 'feature_encoding_s': 0.0, 'support_only_diagnostics_s': 0.0, 'final_training_s': 0.0, 'final_evaluation_s': 0.0, 'retrieval_s': 0.0, 'generalization_s': 0.0, 'cross_backbone_s': 0.0}

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(cfg.seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = torch.cuda.is_available()
HW_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
IN_COLAB = False
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    print('Colab dışı ortam: Drive atlandı.')
DRIVE_MOUNT = '/content/drive'
DRIVE_RETRY_ERRNOS = {5, 107, 116}

def _is_drive_path(path):
    try:
        p = os.path.abspath(os.fspath(path))
        return p == DRIVE_MOUNT or p.startswith(DRIVE_MOUNT + os.sep)
    except Exception:
        return False

def _drive_probe():
    if not IN_COLAB:
        return True
    probe_dir = os.path.join(DRIVE_MOUNT, 'MyDrive')
    if not os.path.isdir(probe_dir):
        return False
    probe = os.path.join(probe_dir, '.xgeoclip_drive_probe.tmp')
    try:
        with open(probe, 'wb') as f:
            f.write(b'ok')
        with open(probe, 'rb') as f:
            ok = f.read() == b'ok'
        try:
            os.remove(probe)
        except OSError:
            pass
        return bool(ok)
    except OSError:
        return False

def ensure_drive_ready(force_remount=False):
    if not IN_COLAB:
        return True
    if not force_remount and _drive_probe():
        return True
    print('[Drive] Bağlantı sağlıksız; yeniden bağlanıyor...')
    try:
        drive.flush_and_unmount()
    except Exception:
        pass
    time.sleep(2)
    drive.mount(DRIVE_MOUNT, force_remount=True)
    time.sleep(2)
    if not _drive_probe():
        raise RuntimeError("Google Drive yeniden bağlandı ancak okuma/yazma testi başarısız. Runtime'ı yeniden başlatıp mount hücresini tekrar çalıştırın.")
    print('[Drive] Yeniden bağlantı başarılı.')
    return True

def safe_isfile(path, retries=3):
    for attempt in range(retries):
        try:
            return os.path.isfile(path)
        except OSError as exc:
            if not _is_drive_path(path) or attempt == retries - 1:
                raise
            print(f'[Drive retry {attempt + 1}/{retries}] isfile: {exc}')
            ensure_drive_ready(force_remount=True)
            time.sleep(1.5 * (attempt + 1))
    return False

def safe_makedirs(path, exist_ok=True, retries=3, required=True):
    for attempt in range(retries):
        try:
            os.makedirs(path, exist_ok=exist_ok)
            return True
        except OSError as exc:
            if not _is_drive_path(path):
                raise
            if attempt < retries - 1:
                print(f'[Drive retry {attempt + 1}/{retries}] makedirs: {exc}')
                ensure_drive_ready(force_remount=True)
                time.sleep(1.5 * (attempt + 1))
                continue
            if required:
                raise RuntimeError(f'Drive klasörü oluşturulamadı: {path}') from exc
            print(f'[Drive warning] Klasör oluşturma atlandı: {path} | {exc}')
            return False

def safe_copy2(src, dst, retries=3, required=True):
    for attempt in range(retries):
        try:
            parent = os.path.dirname(dst)
            if parent:
                safe_makedirs(parent, exist_ok=True, retries=retries, required=required)
            shutil.copy2(src, dst)
            return dst
        except FileNotFoundError:
            if required:
                raise
            return None
        except OSError as exc:
            drive_related = _is_drive_path(src) or _is_drive_path(dst)
            if not drive_related:
                raise
            if attempt < retries - 1:
                print(f'[Drive retry {attempt + 1}/{retries}] {os.path.basename(os.fspath(src))}: {exc}')
                ensure_drive_ready(force_remount=True)
                time.sleep(1.5 * (attempt + 1))
                continue
            if required:
                raise RuntimeError(f'Drive dosya aktarımı {retries} denemeden sonra başarısız: {src} -> {dst}') from exc
            print(f'[Drive warning] Cache aktarımı atlandı; yerel olarak yeniden üretilecek: {src}')
            return None

def safe_copytree(src, dst, retries=3, required=True, **kwargs):
    for attempt in range(retries):
        try:
            safe_makedirs(os.path.dirname(dst), exist_ok=True, retries=retries, required=required)
            return shutil.copytree(src, dst, **kwargs)
        except OSError as exc:
            drive_related = _is_drive_path(src) or _is_drive_path(dst)
            if not drive_related:
                raise
            if attempt < retries - 1:
                print(f'[Drive retry {attempt + 1}/{retries}] copytree: {exc}')
                ensure_drive_ready(force_remount=True)
                time.sleep(1.5 * (attempt + 1))
                continue
            if required:
                raise RuntimeError(f'Drive klasör aktarımı başarısız: {src} -> {dst}') from exc
            print(f'[Drive warning] Klasör senkronizasyonu atlandı: {dst}')
            return None
if IN_COLAB:
    ensure_drive_ready()
    if not os.path.isdir(cfg.data_root):
        raise RuntimeError(f'Veri klasörü yok: {cfg.data_root}')
FIG_DIR = os.path.join(cfg.local_out, 'figures')
TAB_DIR = os.path.join(cfg.local_out, 'tables')
MODEL_DIR = os.path.join(cfg.local_out, 'checkpoints')
CACHE_BASE = '/content/xgeoclip_cache' if os.path.isdir('/content') else os.path.join(cfg.local_out, '_cache')
CACHE_DIR = os.path.join(CACHE_BASE, cfg.cache_dirname)
DRIVE_CACHE = os.path.join(cfg.drive_root, 'feature_cache', cfg.cache_dirname)
LEGACY_CACHE = None
for d in (cfg.local_out, FIG_DIR, TAB_DIR, MODEL_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)
if IN_COLAB:
    safe_makedirs(cfg.out_root, exist_ok=True)
protocol_payload = {'version': cfg.protocol_version, 'model': 'CAPSTONE-CLIP', 'support_namespace': cfg.support_namespace, 'n_outer_folds': cfg.n_outer_folds, 'fold_seed': cfg.fold_seed, 'target_validation': False, 'selection_policy': 'prespecified universal settings; no target-domain validation', 'datasets': cfg.datasets, 'expected_counts': cfg.expected_counts, 'caption_datasets': cfg.caption_datasets, 'backbones': cfg.backbone_specs, 'shots': cfg.shots, 'support_seeds': cfg.support_seeds, 'presets': cfg.presets, 'caption_controls': cfg.caption_controls, 'caption_scales': cfg.caption_scales, 'integrity': {'strict': cfg.strict_dataset_integrity, 'exact_duplicates': cfg.auto_exclude_exact_duplicates, 'filename_artifacts': cfg.auto_exclude_filename_artifacts}}
PROTOCOL_HASH = hashlib.sha256(json.dumps(protocol_payload, sort_keys=True).encode()).hexdigest()[:16]
STATE_LOCAL = os.path.join(cfg.local_out, f'state_{cfg.run_tag}.json')
STATE_DRIVE = os.path.join(cfg.out_root, os.path.basename(STATE_LOCAL))

def _read_json(path):
    for attempt in range(3):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if IN_COLAB and _is_drive_path(path) and (attempt < 2):
                print(f'[Drive retry {attempt + 1}/3] state read: {exc}')
                ensure_drive_ready(force_remount=True)
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None
STATE = _read_json(STATE_DRIVE) or _read_json(STATE_LOCAL) or {}
old_hash = STATE.get('__protocol_hash__')
if old_hash and old_hash != PROTOCOL_HASH:
    raise RuntimeError(f'State protocol mismatch: {old_hash} != {PROTOCOL_HASH}. Yeni bir run_tag kullanın veya eski output klasörünü silin.')

def _safe_v1_state_key(key):
    if key.startswith('__'):
        return False
    if 'AID' in key:
        return False
    if key.startswith('cross_backbone_result/'):
        return False
    return True
MIGRATION_REPORT = {'requested_source': cfg.resume_from_run_tag, 'source_found': False, 'migrated_units': 0, 'skipped_aid_or_bundled': 0, 'checkpoints_copied': 0}
completed_now = sum((1 for v in STATE.values() if isinstance(v, dict) and v.get('done')))
if False and completed_now == 0 and IN_COLAB and cfg.resume_from_run_tag:
    src_root = os.path.join(cfg.drive_root, f'outputs_{cfg.resume_from_run_tag}')
    src_state = os.path.join(src_root, f'state_{cfg.resume_from_run_tag}.json')
    previous = _read_json(src_state)
    if previous:
        MIGRATION_REPORT['source_found'] = True
        for key, value in previous.items():
            if _safe_v1_state_key(key) and isinstance(value, dict) and value.get('done'):
                STATE[key] = value
                MIGRATION_REPORT['migrated_units'] += 1
            elif not key.startswith('__'):
                MIGRATION_REPORT['skipped_aid_or_bundled'] += 1
        src_ckpt = os.path.join(src_root, 'checkpoints')
        dst_ckpt = os.path.join(cfg.out_root, 'checkpoints')
        if os.path.isdir(src_ckpt):
            os.makedirs(dst_ckpt, exist_ok=True)
            for p in glob.glob(os.path.join(src_ckpt, '*.pt')):
                if '__AID__' in os.path.basename(p):
                    continue
                dst = os.path.join(dst_ckpt, os.path.basename(p))
                if not os.path.isfile(dst):
                    safe_copy2(p, dst, required=False)
                    MIGRATION_REPORT['checkpoints_copied'] += 1
STATE['__protocol_hash__'] = PROTOCOL_HASH
STATE['__protocol_version__'] = cfg.protocol_version
STATE['__support_namespace__'] = cfg.support_namespace
STATE['__migration_report__'] = MIGRATION_REPORT

def save_state():
    with open(STATE_LOCAL, 'w', encoding='utf-8') as f:
        json.dump(STATE, f, indent=2, default=str)
    if IN_COLAB and cfg.cache_to_drive:
        safe_makedirs(cfg.out_root, exist_ok=True, required=False)
        safe_copy2(STATE_LOCAL, STATE_DRIVE, required=False)

def mark_done(key, payload=None):
    STATE[key] = {'done': True, 'data': payload}
    save_state()

def is_done(key):
    return isinstance(STATE.get(key), dict) and bool(STATE[key].get('done'))

def state_data(key):
    return STATE.get(key, {}).get('data') if isinstance(STATE.get(key), dict) else None
save_state()

def stable_int(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

def mean_std(xs):
    a = np.asarray([float(x) for x in xs if np.isfinite(float(x))], dtype=float)
    if len(a) == 0:
        return (float('nan'), float('nan'))
    if len(a) == 1:
        return (float(a.mean()), 0.0)
    return (float(a.mean()), float(a.std(ddof=1)))

def gpu_free_bytes():
    return torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0

def maybe_gpu(t, reserve=4000000000.0, max_bytes=400000000.0):
    nbytes = t.element_size() * t.nelement()
    if cfg.gpu_resident and torch.cuda.is_available() and (nbytes <= max_bytes) and (gpu_free_bytes() > nbytes + reserve):
        return t.to(DEVICE, non_blocking=True)
    return t

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print('Device:', DEVICE, '|', HW_NAME, '| protocol hash:', PROTOCOL_HASH)
print('Migration:', MIGRATION_REPORT)
print('Completed resumable units:', sum((1 for v in STATE.values() if isinstance(v, dict) and v.get('done'))))
_t0 = time.time()
os.makedirs(cfg.local_data, exist_ok=True)

def _norm(s):
    return ''.join((ch for ch in s.lower() if ch.isalnum()))

def resolve_dataset_folder(name, data_root):
    if os.path.isdir(os.path.join(data_root, name)) or not os.path.isdir(data_root):
        return name
    want = _norm(name)
    cands = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    for d in cands:
        if _norm(d) == want:
            return d
    aliases = {'rsitmd': ['rsitmd', 'rsitmdataset'], 'ucmerced': ['ucmerced', 'ucmercedlanduse', 'ucm'], 'whurs19': ['whurs19', 'whu19'], 'eurosat': ['eurosat', 'eurosatrgb', 'eurosatallbands']}
    for d in cands:
        nd = _norm(d)
        if nd.startswith(want) or want.startswith(nd) or nd in aliases.get(want, []):
            return d
    return name

def _has_files(root):
    return os.path.isdir(root) and next((f for f in glob.iglob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(f)), None) is not None

def stage_dataset(cfg_name):
    folder = resolve_dataset_folder(cfg_name, cfg.data_root)
    local_tgt = os.path.join(cfg.local_data, folder)
    if _has_files(local_tgt):
        return (local_tgt, 'local')
    if cfg.stage_from_zip:
        for zname in (folder + '.zip', cfg_name + '.zip'):
            z = os.path.join(cfg.data_root, zname)
            if os.path.isfile(z):
                print('Unzipping', zname)
                shutil.unpack_archive(z, cfg.local_data)
                if _has_files(local_tgt):
                    return (local_tgt, 'zip->local')
                for d in os.listdir(cfg.local_data):
                    if _norm(d) == _norm(folder) and _has_files(os.path.join(cfg.local_data, d)):
                        return (os.path.join(cfg.local_data, d), 'zip->local')
    drive_dir = os.path.join(cfg.data_root, folder)
    if _has_files(drive_dir):
        return (drive_dir, 'Drive direct')
    return (drive_dir, 'NOT FOUND')
DATA_AT = {}
for name in cfg.datasets + cfg.caption_datasets:
    path, how = stage_dataset(name)
    DATA_AT[name] = path
    print(f'[stage] {name:>16}: {how} -> {path}')
missing_cls = [n for n in cfg.datasets if not _has_files(DATA_AT[n])]
if missing_cls:
    raise RuntimeError(f'Eksik sınıflandırma veri kümeleri: {missing_cls}')
missing_cap = [n for n in cfg.caption_datasets if not _has_files(DATA_AT[n])]
if cfg.require_second_caption_dataset and missing_cap:
    raise RuntimeError(f'Eksik caption veri kümeleri: {missing_cap}. Q1 final koşusu için RSICD ve RSITMD gereklidir.')
if missing_cap:
    print('WARNING: caption corpora skipped:', missing_cap)
TIMING['staging_s'] += time.time() - _t0
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
DATA_INTEGRITY = {}
AUDITED_GROUPS_CACHE = {}

def prettify(name):
    return ' '.join(name.replace('_', ' ').replace('-', ' ').strip().lower().split())

def _flat_filename_class(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    cls = re.sub('[_\\-\\s]*\\d+$', '', stem).strip('_- ')
    return cls or stem

def _classification_groups(root):
    if not os.path.isdir(root):
        return {}
    image_paths = sorted((p for p in glob.iglob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(p) and p.lower().endswith(IMG_EXT)))
    if not image_paths:
        return {}
    by_parent = {}
    for p in image_paths:
        by_parent.setdefault(os.path.dirname(p), []).append(p)
    if len(by_parent) > 1:
        groups = {}
        for parent, files in sorted(by_parent.items()):
            cls = os.path.basename(parent)
            groups.setdefault(cls, []).extend(files)
        return {k: sorted(v) for k, v in groups.items()}
    only_files = next(iter(by_parent.values()))
    by_prefix = {}
    for p in only_files:
        cls = _flat_filename_class(p)
        by_prefix.setdefault(cls, []).append(p)
    if len(by_prefix) > 1:
        return {k: sorted(v) for k, v in by_prefix.items()}
    parent = next(iter(by_parent))
    return {os.path.basename(parent): sorted(only_files)}

def _sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _canonical_filename_matches_class(cls, path):
    cls_n = _norm(cls)
    stem_n = _norm(os.path.splitext(os.path.basename(path))[0])
    return bool(cls_n and stem_n.startswith(cls_n))

def _obvious_image_artifact(path):
    stem = _norm(os.path.splitext(os.path.basename(path))[0])
    tokens = ('preview', 'thumbnail', 'thumb', 'cover', 'readme', 'logo', 'sample', 'screenshot')
    return any((t in stem for t in tokens))

def _exact_duplicate_exclusions(paths, max_remove):
    if max_remove <= 0:
        return []
    by_size = defaultdict(list)
    for p in paths:
        try:
            by_size[os.path.getsize(p)].append(p)
        except OSError:
            pass
    excluded = []
    for size, group in sorted(by_size.items()):
        if len(group) < 2:
            continue
        by_hash = defaultdict(list)
        for p in sorted(group):
            by_hash[_sha256_file(p)].append(p)
        for _, dup_group in sorted(by_hash.items()):
            if len(dup_group) > 1:
                excluded.extend(sorted(dup_group)[1:])
                if len(excluded) >= max_remove:
                    return excluded[:max_remove]
    return excluded[:max_remove]

def _repair_dataset_groups(groups, dataset_name, root):
    cache_key = (dataset_name, os.path.abspath(root))
    if cache_key in AUDITED_GROUPS_CACHE:
        return {k: list(v) for k, v in AUDITED_GROUPS_CACHE[cache_key].items()}
    expected_total = cfg.expected_counts.get(dataset_name)
    expected_classes = cfg.expected_class_counts.get(dataset_name)
    initial_counts = {c: len(v) for c, v in sorted(groups.items())}
    initial_total = sum(initial_counts.values())
    excluded = []
    if expected_classes is not None and len(groups) != expected_classes:
        raise RuntimeError(f'{dataset_name}: discovered {len(groups)} classes, expected {expected_classes}. root={root}; discovered={list(sorted(groups))[:20]}')
    if expected_total is not None and initial_total > expected_total:
        need = initial_total - expected_total
        all_items = [(cls, p) for cls, files in groups.items() for p in files]
        if cfg.auto_exclude_filename_artifacts:
            obvious = [(c, p, 'obvious_artifact_filename') for c, p in all_items if _obvious_image_artifact(p)]
            for item in obvious[:need]:
                excluded.append(item)
            need -= min(need, len(obvious))
        if need > 0 and cfg.auto_exclude_exact_duplicates:
            remaining_paths = [p for _, p in all_items if p not in {x[1] for x in excluded}]
            dup_paths = _exact_duplicate_exclusions(remaining_paths, need)
            path_to_class = {p: c for c, p in all_items}
            for p in dup_paths:
                excluded.append((path_to_class[p], p, 'exact_byte_duplicate'))
            need -= len(dup_paths)
        if need > 0 and cfg.auto_exclude_filename_artifacts:
            excluded_paths = {x[1] for x in excluded}
            mismatch = [(c, p, 'filename_not_matching_class') for c, p in all_items if p not in excluded_paths and (not _canonical_filename_matches_class(c, p))]
            if len(mismatch) == need:
                excluded.extend(mismatch)
                need = 0
        if need > 0:
            class_counts = sorted(initial_counts.items(), key=lambda x: (-x[1], x[0]))
            raise RuntimeError(f'{dataset_name}: {initial_total} images found, expected {expected_total}. Automatic logical repair could not identify {need} extra file(s). Largest classes={class_counts[:10]}. No source file was modified.')
    excluded_paths = {p for _, p, _ in excluded}
    repaired = {c: [p for p in files if p not in excluded_paths] for c, files in groups.items()}
    repaired_counts = {c: len(v) for c, v in sorted(repaired.items())}
    repaired_total = sum(repaired_counts.values())
    if expected_total is not None and cfg.strict_dataset_integrity and (repaired_total != expected_total):
        raise RuntimeError(f'{dataset_name}: integrity check failed after logical repair: {repaired_total} != {expected_total}')
    report = {'dataset': dataset_name, 'root': root, 'expected_images': expected_total, 'expected_classes': expected_classes, 'initial_images': initial_total, 'final_images': repaired_total, 'initial_class_counts': initial_counts, 'final_class_counts': repaired_counts, 'excluded': [{'class': c, 'path': os.path.relpath(p, root).replace('\\', '/'), 'reason': reason, 'sha256': _sha256_file(p) if os.path.isfile(p) else None} for c, p, reason in excluded], 'source_files_modified': False, 'pass': (expected_total is None or repaired_total == expected_total) and (expected_classes is None or len(repaired) == expected_classes)}
    DATA_INTEGRITY[dataset_name] = report
    if excluded:
        print(f'[integrity repair] {dataset_name}: excluded {len(excluded)} logical file(s):')
        for x in report['excluded']:
            print('  ', x)
    AUDITED_GROUPS_CACHE[cache_key] = {k: list(v) for k, v in repaired.items()}
    return repaired

def find_class_dirs(root):
    groups = _classification_groups(root)
    out = {}
    for cls, files in groups.items():
        parents = {os.path.dirname(p) for p in files}
        if len(parents) == 1:
            out[cls] = next(iter(parents))
    return out

def scan_dataset_at(root, smoke_per_class=None, dataset_name=None):
    groups = _classification_groups(root)
    if not groups:
        raise RuntimeError(f'No class-organized images found under {root}')
    groups = _repair_dataset_groups(groups, dataset_name, root)
    class_names = sorted(groups)
    prompts = [prettify(c) for c in class_names]
    samples = []
    for idx, cls in enumerate(class_names):
        files = groups[cls]
        if smoke_per_class:
            files = files[:smoke_per_class]
        samples.extend(((p, idx) for p in files))
    if not samples:
        raise RuntimeError(f'No usable class-organized images found under {root}')
    return (samples, class_names, prompts)

def build_caption_splits_at(root, dataset_name, json_search_dirs=None):
    splits = {'train': [], 'val': [], 'test': []}
    if not _has_files(root):
        return (splits, None)
    search_dirs = [root] + [d for d in json_search_dirs or [] if d and os.path.isdir(d)]
    candidates = []
    for d in search_dirs:
        candidates += glob.glob(os.path.join(d, '**', '*.json'), recursive=True)
    candidates = list(dict.fromkeys(candidates))
    target = _norm(dataset_name)
    candidates.sort(key=lambda p: (0 if target in _norm(os.path.basename(p)) else 1, len(p)))
    records, used = ([], None)
    for j in candidates:
        try:
            data = json.load(open(j, encoding='utf-8'))
            images = data.get('images', []) if isinstance(data, dict) else []
            temp = []
            for im in images:
                fn = im.get('filename') or im.get('file_name') or im.get('imgid')
                sent = im.get('sentences', [])
                caps = []
                for s in sent:
                    if isinstance(s, dict):
                        txt = s.get('raw') or s.get('sentence') or s.get('caption') or ''
                    else:
                        txt = str(s)
                    txt = txt.strip()
                    if txt:
                        caps.append(txt)
                sp = str(im.get('split') or 'train').lower()
                if sp in ('validation', 'valid', 'dev'):
                    sp = 'val'
                if sp not in splits:
                    sp = 'train'
                if fn and caps:
                    temp.append((str(fn), caps, sp))
            if temp:
                records, used = (temp, j)
                break
        except Exception:
            continue
    path_by_name = {os.path.basename(f): f for f in glob.glob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(f) and f.lower().endswith(IMG_EXT)}
    for fn, caps, sp in records:
        p = path_by_name.get(os.path.basename(fn))
        if p:
            splits[sp].append((p, caps))
    if splits['train'] and (not splits['val']):
        rng = np.random.RandomState(cfg.split_seed + stable_int(dataset_name, 'caption-val'))
        order = np.arange(len(splits['train']))
        rng.shuffle(order)
        n_val = max(1, int(round(0.1 * len(order))))
        original = splits['train']
        splits['val'] = [original[i] for i in order[:n_val]]
        splits['train'] = [original[i] for i in order[n_val:]]
        print(f'[{dataset_name}] annotation had no val split; deterministic 90/10 train/val derived from train only.')
    return (splits, used)

class ImagePathDS(Dataset):

    def __init__(self, paths, preprocess, flip=False):
        self.paths, self.preprocess, self.flip = (paths, preprocess, flip)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        x = self.preprocess(img)
        return torch.flip(x, [-1]) if self.flip else x
import open_clip
from huggingface_hub import hf_hub_download
BACKBONE_SPECS = {x['key']: x for x in cfg.backbone_specs}
ACTIVE_BACKBONE_KEY = None
ACTIVE_SPEC = None
ACTIVE_CACHE_TOKENS = True
ACTIVE_CACHE_PREPROJ = True
clip_model = preprocess = tokenizer = backbone_visual = None
EMBED_DIM = PREPROJ_DIM = None
VISUAL_PROJ0 = None

def _apply_visual_proj(x, proj):
    if proj is None:
        return x
    if isinstance(proj, nn.Linear):
        return proj(x)
    return x @ proj

class CLIPVisualCacheWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
        self.proj = getattr(self.visual, 'proj', None)
        try:
            self.visual.output_tokens = True
        except Exception:
            pass

    def _reduce_hook_output(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x is None:
            return None
        if x.ndim == 3:
            pool_type = str(getattr(self.visual, 'global_pool', 'token')).lower()
            if 'avg' in pool_type and x.shape[1] > 1:
                x = x[:, 1:].mean(1)
            else:
                x = x[:, 0]
        return x

    def forward(self, x):
        captured = {}
        handle = None
        ln_post = getattr(self.visual, 'ln_post', None)
        if ln_post is not None:
            handle = ln_post.register_forward_hook(lambda module, inp, out: captured.__setitem__('preproj', out))
        try:
            out = self.visual(x)
        finally:
            if handle is not None:
                handle.remove()
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            pooled, tokens = (out[0], out[1])
        else:
            pooled, tokens = (out, None)
        preproj = self._reduce_hook_output(captured.get('preproj'))
        if preproj is None:
            proj = self.proj.weight.t() if isinstance(self.proj, nn.Linear) else self.proj
            in_dim = int(proj.shape[0]) if proj is not None else int(pooled.shape[-1])
            if pooled.shape[-1] == in_dim:
                preproj = pooled
        if tokens is not None and tokens.shape[-1] != pooled.shape[-1]:
            tokens = _apply_visual_proj(tokens, self.proj)
        return (pooled, tokens, preproj)

def _load_openclip(spec):
    errors = []
    for tag in spec.get('pretrained_candidates', [None]):
        try:
            model, _, pp = open_clip.create_model_and_transforms(spec['arch'], pretrained=tag)
            return (model, pp, tag)
        except Exception as e:
            errors.append(f'{tag}: {type(e).__name__}: {e}')
    raise RuntimeError('OpenCLIP load failed: ' + ' | '.join(errors))

def _normalize_checkpoint_state(ckpt):
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt = ckpt['model']
    out = {}
    for k, v in ckpt.items():
        for prefix in ('module.', 'model.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out

def load_backbone(spec):
    if spec['kind'] == 'openclip':
        model, pp, loaded_tag = _load_openclip(spec)
    elif spec['kind'] == 'checkpoint':
        model, _, pp = open_clip.create_model_and_transforms(spec['arch'], pretrained=None)
        ckpt_path = hf_hub_download(spec['hf_repo'], spec['hf_filename'], cache_dir=os.path.join(CACHE_BASE, 'hf'))
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        msg = model.load_state_dict(_normalize_checkpoint_state(ckpt), strict=False)
        if len(msg.missing_keys) > 10 or len(msg.unexpected_keys) > 10:
            print('Checkpoint load warning:', msg)
        loaded_tag = spec['hf_filename']
    else:
        raise ValueError(spec['kind'])
    tok = open_clip.get_tokenizer(spec['arch'])
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        d = int(model.encode_text(tok(['a satellite image']).to(DEVICE)).shape[-1])
    visual = CLIPVisualCacheWrapper(model).to(DEVICE).eval()
    proj = getattr(model.visual, 'proj', None)
    if isinstance(proj, nn.Linear):
        proj0 = proj.weight.detach().t().float().cpu()
    elif proj is not None:
        proj0 = proj.detach().float().cpu()
    else:
        proj0 = None
    pre_dim = int(proj0.shape[0]) if proj0 is not None else d
    return (model, pp, tok, visual, d, pre_dim, proj0, loaded_tag)

def activate_backbone(key):
    global ACTIVE_BACKBONE_KEY, ACTIVE_SPEC, ACTIVE_CACHE_TOKENS, ACTIVE_CACHE_PREPROJ
    global clip_model, preprocess, tokenizer, backbone_visual, EMBED_DIM, PREPROJ_DIM, VISUAL_PROJ0
    spec = BACKBONE_SPECS[key]
    free_gpu()
    clip_model, preprocess, tokenizer, backbone_visual, EMBED_DIM, PREPROJ_DIM, VISUAL_PROJ0, loaded = load_backbone(spec)
    ACTIVE_BACKBONE_KEY, ACTIVE_SPEC = (key, spec)
    ACTIVE_CACHE_TOKENS = bool(spec.get('cache_tokens', True))
    ACTIVE_CACHE_PREPROJ = bool(spec.get('cache_preproj', True))
    print(f"Activated {key}: {spec['label']} | loaded={loaded} | D={EMBED_DIM} | pre-D={PREPROJ_DIM}")
PROMPT_TEMPLATES = ['a satellite photo of a {}.', 'an aerial image of a {}.', 'a remote sensing image of {}.', 'a high resolution satellite image of a {}.', 'a top view of {}.']

@torch.no_grad()
def raw_prompt_embeddings(prompts):
    out = []
    for name in prompts:
        tok = tokenizer([t.format(name) for t in PROMPT_TEMPLATES]).to(DEVICE)
        out.append(clip_model.encode_text(tok).float().cpu())
    return torch.stack(out, 0)
activate_backbone(cfg.primary_backbone)

def _cache_fname(name, suffix='dataset'):
    return f'{ACTIVE_BACKBONE_KEY}__{name}__capstone_preproj_v1__{suffix}.pt'

def drive_mirror(local_path):
    if not (IN_COLAB and cfg.cache_to_drive):
        return None
    safe_makedirs(DRIVE_CACHE, exist_ok=True, required=False)
    return safe_copy2(local_path, os.path.join(DRIVE_CACHE, os.path.basename(local_path)), required=False)

def drive_fetch(fname):
    local = os.path.join(CACHE_DIR, fname)
    if os.path.isfile(local):
        return local
    src = os.path.join(DRIVE_CACHE, fname)
    try:
        exists = safe_isfile(src)
    except OSError:
        exists = False
    if not exists:
        return None
    copied = safe_copy2(src, local, required=False)
    return local if copied and os.path.isfile(local) else None

@torch.no_grad()
def encode_paths(paths, flip=False, desc='encode', batch=None, keep_tokens=None, keep_preproj=None):
    t0 = time.time()
    keep_tokens = ACTIVE_CACHE_TOKENS if keep_tokens is None else bool(keep_tokens)
    keep_preproj = ACTIVE_CACHE_PREPROJ if keep_preproj is None else bool(keep_preproj)
    ds = ImagePathDS(paths, preprocess, flip=flip)
    loader = DataLoader(ds, batch_size=batch or cfg.encode_batch, shuffle=False, num_workers=cfg.encode_workers, pin_memory=torch.cuda.is_available())
    gs, ts, hs = ([], [], [])
    for x in tqdm(loader, desc=desc, leave=False):
        with torch.autocast('cuda', enabled=AMP):
            g, t, h = backbone_visual(x.to(DEVICE, non_blocking=True))
        gs.append(g.half().cpu())
        if keep_tokens:
            if t is None:
                raise RuntimeError(f'{ACTIVE_BACKBONE_KEY} did not expose patch tokens')
            ts.append(t.half().cpu())
        if keep_preproj:
            if h is None:
                raise RuntimeError(f'{ACTIVE_BACKBONE_KEY}: pre-projection feature unavailable; Full ProLIP cannot be cached')
            hs.append(h.half().cpu())
    TIMING['feature_encoding_s'] += time.time() - t0
    return (torch.cat(gs), torch.cat(ts) if ts else None, torch.cat(hs) if hs else None)
FEATS = {}
CLASS_U = {}
CAPTION_BANKS = {}

def _path_digest(paths):
    return hashlib.sha256('\n'.join(paths).encode()).hexdigest()

def get_dataset_cache(name):
    cache_key = (ACTIVE_BACKBONE_KEY, name)
    if cache_key in FEATS:
        return FEATS[cache_key]
    samples, class_names, prompts = scan_dataset_at(DATA_AT[name], cfg.smoke_per_class if RUN_PROFILE == 'smoke' else None, dataset_name=name)
    paths = [p for p, _ in samples]
    y = torch.tensor([v for _, v in samples], dtype=torch.long)
    digest = _path_digest(paths)
    exp = cfg.expected_counts.get(name)
    if exp and len(samples) != exp:
        raise RuntimeError(f'{name}: {len(samples)} != expected {exp}')
    fname = _cache_fname(name)
    lp = drive_fetch(fname)
    blob = None
    if lp:
        b = torch.load(lp, map_location='cpu', weights_only=False)
        ok = b.get('n') == len(samples) and b.get('path_digest') == digest and (bool(b.get('has_tokens')) == ACTIVE_CACHE_TOKENS) and (bool(b.get('has_preproj')) == ACTIVE_CACHE_PREPROJ) and (int(torch.unique(b.get('y')).numel()) == len(class_names))
        if ok:
            blob = b
        else:
            print(f'[cache invalidated] {ACTIVE_BACKBONE_KEY}:{name}')
    if blob is None:
        gs, ts, hs = ([], [], [])
        for s in range(math.ceil(len(paths) / cfg.shard_size)):
            lo, hi = (s * cfg.shard_size, min(len(paths), (s + 1) * cfg.shard_size))
            sf = _cache_fname(f'{name}_shard{s}', 'features')
            sp = drive_fetch(sf)
            sb = None
            if sp:
                q = torch.load(sp, map_location='cpu', weights_only=False)
                ok = len(q.get('g', [])) == hi - lo
                ok = ok and (not ACTIVE_CACHE_TOKENS or q.get('t') is not None)
                ok = ok and (not ACTIVE_CACHE_PREPROJ or q.get('h') is not None)
                if ok:
                    sb = q
            if sb is None:
                g, t, h = encode_paths(paths[lo:hi], desc=f'{ACTIVE_BACKBONE_KEY}:{name} {s + 1}')
                sb = {'g': g, 't': t, 'h': h}
                sp = os.path.join(CACHE_DIR, sf)
                torch.save(sb, sp)
                drive_mirror(sp)
            gs.append(sb['g'])
            if ACTIVE_CACHE_TOKENS:
                ts.append(sb['t'])
            if ACTIVE_CACHE_PREPROJ:
                hs.append(sb['h'])
        blob = {'g': torch.cat(gs), 't': torch.cat(ts) if ts else None, 'h': torch.cat(hs) if hs else None, 'y': y, 'paths': paths, 'n': len(paths), 'has_tokens': ACTIVE_CACHE_TOKENS, 'has_preproj': ACTIVE_CACHE_PREPROJ, 'path_digest': digest}
        lp = os.path.join(CACHE_DIR, fname)
        torch.save(blob, lp)
        drive_mirror(lp)
    t = blob.get('t')
    if t is None:
        t = blob['g'].unsqueeze(1)
    out = {'g': maybe_gpu(blob['g']), 't': maybe_gpu(t), 'h': maybe_gpu(blob['h']) if blob.get('h') is not None else None, 'y': y, 'paths': paths, 'class_names': class_names, 'prompts': prompts, 'true_tokens': bool(blob.get('has_tokens', False)), 'path_digest': digest}
    FEATS[cache_key] = out
    print(f"[cache] {ACTIVE_BACKBONE_KEY}:{name}: {len(paths)} | tokens={out['true_tokens']} | preproj={out['h'] is not None}")
    return out
RESULTS = {'protocol_hash': PROTOCOL_HASH, 'protocol_version': cfg.protocol_version, 'model': 'CAPSTONE-CLIP', 'data_integrity': {}, 'datasets': {}, 'classification_runs': [], 'caption_control_runs': [], 'retrieval_runs': [], 'ablation_runs': [], 'class_diagnostics': [], 'base_to_new_runs': [], 'cross_dataset_runs': [], 'cross_backbone_runs': [], 'efficiency': [], 'statistics': [], 'external_baselines': [], 'near_duplicate_audit': [], 'claim_audit': [], 'timing': {}}
AVAILABLE = list(cfg.datasets)
for name in AVAILABLE:
    fc = get_dataset_cache(name)
    report = copy.deepcopy(DATA_INTEGRITY.get(name, {}))
    report['used_images'] = len(fc['paths'])
    report['used_path_digest'] = fc['path_digest']
    RESULTS['data_integrity'][name] = report
    RESULTS['datasets'][name] = {'n_images': len(fc['paths']), 'n_classes': len(fc['class_names']), 'class_names': fc['class_names'], 'path_digest': fc['path_digest']}
CLASS_U = {n: raw_prompt_embeddings(get_dataset_cache(n)['prompts']).to(DEVICE) for n in AVAILABLE}
print('Primary caches ready:', {n: len(get_dataset_cache(n)['paths']) for n in AVAILABLE})
if 'AVAILABLE' not in globals():
    AVAILABLE = list(cfg.datasets)
    print('[recovery] AVAILABLE was missing; restored from cfg.datasets.')
if not AVAILABLE:
    raise RuntimeError('No classification dataset is configured in cfg.datasets.')
SPLITS = {}
ACTIVE_OUTER_ID = 0
TEST_UNLOCKED = False
MANIFEST_PATH = os.path.join(cfg.local_out, f'split_manifest_{cfg.support_namespace}.json')

def build_disjoint_folds(labels):
    labels = np.asarray(labels, dtype=int)
    skf = StratifiedKFold(n_splits=cfg.n_outer_folds, shuffle=True, random_state=cfg.fold_seed)
    out = {}
    for oid, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        out[oid] = {'train_pool': np.asarray(train_idx, dtype=np.int64), 'test': np.asarray(test_idx, dtype=np.int64)}
    all_test = np.concatenate([out[k]['test'] for k in sorted(out)])
    assert len(all_test) == len(labels) and len(np.unique(all_test)) == len(labels)
    return out
for name in AVAILABLE:
    try:
        SPLITS[name] = build_disjoint_folds(get_dataset_cache(name)['y'].numpy())
    except Exception as e:
        raise RuntimeError(f'Failed to initialize disjoint folds for {name}. Run the frozen-feature-cache cell immediately above this section and resolve its first error before continuing.') from e

def set_outer_split(outer_id):
    global ACTIVE_OUTER_ID
    outer_id = int(outer_id)
    if outer_id not in range(cfg.n_outer_folds):
        raise IndexError(outer_id)
    ACTIVE_OUTER_ID = outer_id
    return outer_id

def train_pool_indices(name, outer_id=None):
    oid = ACTIVE_OUTER_ID if outer_id is None else int(outer_id)
    return SPLITS[name][oid]['train_pool']

def test_indices(name, outer_id=None):
    if not TEST_UNLOCKED:
        raise RuntimeError('Test locked; only final-evaluation sections may unlock it')
    oid = ACTIVE_OUTER_ID if outer_id is None else int(outer_id)
    return SPLITS[name][oid]['test']

def val_indices(name):
    raise RuntimeError('Target-domain validation is disabled. Use prespecified settings only.')

def support_indices(name, K, seed, outer_id=None, class_subset=None):
    oid = ACTIVE_OUTER_ID if outer_id is None else int(outer_id)
    pool = train_pool_indices(name, oid)
    y = get_dataset_cache(name)['y'].numpy()
    classes = sorted(np.unique(y)) if class_subset is None else sorted(map(int, class_subset))
    rng = np.random.RandomState(stable_int(cfg.support_namespace, name, oid, int(K), int(seed), classes))
    out = []
    for c in classes:
        cand = pool[y[pool] == c].copy()
        if len(cand) < K:
            raise RuntimeError(f'{name}: class {c} has {len(cand)} < K={K}')
        rng.shuffle(cand)
        out.extend(cand[:K].tolist())
    return np.asarray(out, dtype=np.int64)

def hoso_support_split(name, K, seed, outer_id=None):
    if int(K) < 2:
        raise ValueError('HOSO requires K>=2')
    full = support_indices(name, K, seed, outer_id)
    y = get_dataset_cache(name)['y'].numpy()
    rng = np.random.RandomState(stable_int('hoso', name, outer_id, K, seed))
    tr, hold = ([], [])
    for c in sorted(np.unique(y[full])):
        ids = full[y[full] == c].copy()
        rng.shuffle(ids)
        hold.append(int(ids[0]))
        tr.extend(ids[1:].tolist())
    return (np.asarray(tr), np.asarray(hold))
manifest = {'protocol_version': cfg.protocol_version, 'support_namespace': cfg.support_namespace, 'target_validation': False, 'fold_seed': cfg.fold_seed, 'n_outer_folds': cfg.n_outer_folds, 'support_seeds': cfg.support_seeds, 'shots': cfg.shots, 'datasets': {}}
for name in AVAILABLE:
    manifest['datasets'][name] = {}
    for oid in range(cfg.n_outer_folds):
        sp = SPLITS[name][oid]
        manifest['datasets'][name][str(oid)] = {'train_pool_count': int(len(sp['train_pool'])), 'test_count': int(len(sp['test'])), 'train_pool_sha256': hashlib.sha256(sp['train_pool'].tobytes()).hexdigest(), 'test_sha256': hashlib.sha256(sp['test'].tobytes()).hexdigest(), 'supports': {f'K{K}_s{s}': hashlib.sha256(support_indices(name, K, s, oid).tobytes()).hexdigest() for K in cfg.shots for s in cfg.support_seeds}}
with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
if IN_COLAB and cfg.cache_to_drive:
    safe_copy2(MANIFEST_PATH, os.path.join(cfg.out_root, os.path.basename(MANIFEST_PATH)), required=False)
print('Disjoint split manifest:', MANIFEST_PATH)

@torch.no_grad()
def encode_caption_bank(pairs, desc):
    if not pairs:
        return None
    paths = [p for p, _ in pairs]
    g, t, _ = encode_paths(paths, desc=desc + ' images', keep_preproj=False)
    caps, starts, counts, owner = ([], [], [], [])
    for i, (_, cs) in enumerate(pairs):
        use = cs[:5]
        starts.append(len(caps))
        counts.append(len(use))
        caps.extend(use)
        owner.extend([i] * len(use))
    us = []
    for s in tqdm(range(0, len(caps), 256), desc=desc + ' captions', leave=False):
        tok = tokenizer(caps[s:s + 256]).to(DEVICE)
        with torch.autocast('cuda', enabled=AMP):
            us.append(clip_model.encode_text(tok).half().cpu())
    return {'g': g, 't': t if t is not None else g.unsqueeze(1), 'u': torch.cat(us), 'cap_start': torch.tensor(starts), 'cap_count': torch.tensor(counts), 'cap_owner': torch.tensor(owner), 'n_img': len(pairs), 'captions': caps}

def load_caption_banks(dataset_names=None):
    global CAPTION_BANKS
    dataset_names = dataset_names or [d for d in cfg.caption_datasets if _has_files(DATA_AT.get(d, ''))]
    for dsname in dataset_names:
        key = (ACTIVE_BACKBONE_KEY, dsname)
        if key in CAPTION_BANKS:
            continue
        drive_dir = os.path.join(cfg.data_root, resolve_dataset_folder(dsname, cfg.data_root))
        splits, used = build_caption_splits_at(DATA_AT[dsname], dsname, [drive_dir, cfg.data_root])
        if not cfg.SMOKE_TEST and any((len(splits[k]) == 0 for k in ('train', 'val', 'test'))):
            raise RuntimeError(f'{dsname} train/val/test incomplete. annotation={used}')
        want = {k: len(splits[k]) for k in ('train', 'val', 'test')}
        fname = _cache_fname(dsname, 'caption_bank')
        lp = drive_fetch(fname)
        banks = None
        if lp:
            b = torch.load(lp, map_location='cpu', weights_only=False)
            if b.get('sizes') == want and bool(b.get('has_tokens')) == ACTIVE_CACHE_TOKENS:
                banks = b['bank']
        if banks is None:
            banks = {k: encode_caption_bank(splits[k], f'{ACTIVE_BACKBONE_KEY}:{dsname}:{k}') for k in ('train', 'val', 'test')}
            lp = os.path.join(CACHE_DIR, fname)
            torch.save({'sizes': want, 'bank': banks, 'has_tokens': ACTIVE_CACHE_TOKENS}, lp)
            drive_mirror(lp)
        CAPTION_BANKS[key] = banks
        print(f'[caption cache] {ACTIVE_BACKBONE_KEY}:{dsname}', want, 'annotation=', used)
    return CAPTION_BANKS

def cap_bank(dsname, split):
    b = CAPTION_BANKS[ACTIVE_BACKBONE_KEY, dsname][split]
    return b
load_caption_banks()

def subset_caption_bank(bank, fraction=1.0, max_images=None, seed=0):
    n = bank['n_img']
    take = n if fraction >= 1 else max(1, int(round(n * float(fraction))))
    if max_images is not None:
        take = min(take, int(max_images))
    rng = np.random.RandomState(stable_int('caption_subset', ACTIVE_BACKBONE_KEY, seed, n, fraction, max_images))
    ids = np.arange(n)
    rng.shuffle(ids)
    ids = np.sort(ids[:take])
    gs = bank['g'][ids]
    ts = bank['t'][ids]
    caps = []
    starts = []
    counts = []
    owners = []
    us = []
    for new_i, old_i in enumerate(ids):
        st = int(bank['cap_start'][old_i])
        ct = int(bank['cap_count'][old_i])
        sel = torch.arange(st, st + ct)
        starts.append(len(caps))
        counts.append(ct)
        caps.extend([bank['captions'][j] for j in sel.tolist()])
        owners.extend([new_i] * ct)
        us.append(bank['u'][sel])
    return {'g': gs, 't': ts, 'u': torch.cat(us), 'cap_start': torch.tensor(starts), 'cap_count': torch.tensor(counts), 'cap_owner': torch.tensor(owners), 'n_img': len(ids), 'captions': caps, 'source_ids': torch.tensor(ids)}

def shuffled_caption_bank(bank, seed=0):
    out = {k: v.clone() if torch.is_tensor(v) else list(v) if isinstance(v, list) else v for k, v in bank.items()}
    gen = torch.Generator().manual_seed(stable_int('shuffle_caption', seed, bank['n_img']))
    perm = torch.randperm(len(out['u']), generator=gen)
    out['u'] = out['u'][perm]
    out['captions'] = [out['captions'][i] for i in perm.tolist()]
    return out

def merge_caption_banks(banks):
    gs = []
    ts = []
    us = []
    starts = []
    counts = []
    owners = []
    caps = []
    img_off = 0
    cap_off = 0
    for b in banks:
        gs.append(b['g'])
        ts.append(b['t'])
        us.append(b['u'])
        starts.extend((b['cap_start'] + cap_off).tolist())
        counts.extend(b['cap_count'].tolist())
        owners.extend((b['cap_owner'] + img_off).tolist())
        caps.extend(b['captions'])
        img_off += b['n_img']
        cap_off += len(b['u'])
    return {'g': torch.cat(gs), 't': torch.cat(ts), 'u': torch.cat(us), 'cap_start': torch.tensor(starts), 'cap_count': torch.tensor(counts), 'cap_owner': torch.tensor(owners), 'n_img': img_off, 'captions': caps}

class ResidualMLP(nn.Module):

    def __init__(self, dim, reduction=4, dropout=0.0):
        super().__init__()
        h = max(8, dim // reduction)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))

    def forward(self, x):
        return self.net(x)

class GatedResidual(nn.Module):

    def __init__(self, dim, reduction=4, init_gate=0.2):
        super().__init__()
        self.block = ResidualMLP(dim, reduction)
        init = float(np.log(init_gate / (1 - init_gate)))
        self.gate_logit = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def residual(self, x):
        return self.block(x)

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x):
        return x + self.gate() * self.residual(x)

class SharedCaptionAdapter(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.img = GatedResidual(dim)
        self.txt = GatedResidual(dim)
        self.register_buffer('logit_scale', clip_model.logit_scale.detach().clone())

    def encode_image(self, g):
        return F.normalize(self.img(g), dim=-1)

    def encode_text(self, u):
        return F.normalize(self.txt(u), dim=-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

class WidthGatedResidual(nn.Module):

    def __init__(self, dim, hidden, init_gate=0.2):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        init = float(np.log(init_gate / (1 - init_gate)))
        self.gate_logit = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def residual(self, x):
        return self.block(x)

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x):
        return x + self.gate() * self.residual(x)

class CapacityMatchedSharedCaptionAdapter(nn.Module):

    def __init__(self, dim):
        super().__init__()
        hidden = max(8, dim // 2)
        self.img = WidthGatedResidual(dim, hidden)
        self.txt = WidthGatedResidual(dim, hidden)
        self.register_buffer('logit_scale', clip_model.logit_scale.detach().clone())

    def encode_image(self, g):
        return F.normalize(self.img(g), dim=-1)

    def encode_text(self, u):
        return F.normalize(self.txt(u), dim=-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

class CAPSTONECLIP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.shared_img = GatedResidual(dim, reduction=4, init_gate=0.2)
        self.shared_txt = GatedResidual(dim, reduction=4, init_gate=0.2)
        self.cls_img = GatedResidual(dim, reduction=8, init_gate=0.1)
        self.cls_txt = GatedResidual(dim, reduction=8, init_gate=0.1)
        self.ret_img = GatedResidual(dim, reduction=8, init_gate=0.1)
        self.ret_txt = GatedResidual(dim, reduction=8, init_gate=0.1)
        self.register_buffer('logit_scale', clip_model.logit_scale.detach().clone())

    def shared_image_raw(self, g):
        return self.shared_img(g)

    def shared_text_raw(self, u):
        return self.shared_txt(u)

    def encode_shared_image(self, g):
        return F.normalize(self.shared_image_raw(g), dim=-1)

    def encode_shared_text(self, u):
        return F.normalize(self.shared_text_raw(u), dim=-1)

    def encode_class_image(self, g):
        return F.normalize(self.cls_img(self.shared_image_raw(g)), dim=-1)

    def encode_class_text(self, u):
        return F.normalize(self.cls_txt(self.shared_text_raw(u)), dim=-1)

    def encode_ret_image(self, g):
        return F.normalize(self.ret_img(self.shared_image_raw(g)), dim=-1)

    def encode_ret_text(self, u):
        return F.normalize(self.ret_txt(self.shared_text_raw(u)), dim=-1)

    def private_residuals(self, g):
        z = self.shared_image_raw(g)
        return (self.cls_img.gate() * self.cls_img.residual(z), self.ret_img.gate() * self.ret_img.residual(z))

    def private_decorrelation(self, g):
        rc, rr = self.private_residuals(g)
        rc = F.normalize(rc, dim=-1)
        rr = F.normalize(rr, dim=-1)
        return (rc * rr).sum(-1).pow(2).mean()

    def gate_regularizer(self):
        gates = [m.gate() for m in (self.shared_img, self.shared_txt, self.cls_img, self.cls_txt, self.ret_img, self.ret_txt)]
        return torch.stack(gates).pow(2).mean()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

class RLAHead(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        return F.normalize(x @ self.W, dim=-1)

    def identity_penalty(self):
        I = torch.eye(self.W.shape[0], device=self.W.device, dtype=self.W.dtype)
        return (self.W - I).pow(2).sum()

class ProLIPProjector(nn.Module):

    def __init__(self, W0):
        super().__init__()
        self.register_buffer('W0', W0.clone())
        self.W = nn.Parameter(W0.clone())

    def forward(self, h):
        return F.normalize(h @ self.W, dim=-1)

    def zero_shot_penalty(self):
        return (self.W - self.W0).pow(2).sum()

class CLIPAdapter(nn.Module):

    def __init__(self, dim, res=0.2, reduction=4):
        super().__init__()
        h = max(8, dim // reduction)
        self.res = float(res)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, dim), nn.ReLU())

    def forward(self, x):
        return F.normalize((1 - self.res) * x + self.res * self.mlp(x), dim=-1)

class HOSOAdapter(nn.Module):

    def __init__(self, dim, reduction=4):
        super().__init__()
        h = max(8, dim // reduction)
        self.adapter = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, dim), nn.ReLU())
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit) * 0.8 + 0.1

    def forward(self, x):
        a = self.alpha()
        return F.normalize((1 - a) * x + a * F.normalize(self.adapter(x), dim=-1), dim=-1)

class TaskResModel(nn.Module):

    def __init__(self, class_features, alpha=0.5):
        super().__init__()
        self.register_buffer('base', class_features.clone())
        self.residual = nn.Parameter(torch.zeros_like(class_features))
        self.alpha = alpha

    def class_features(self):
        return F.normalize(self.base + self.alpha * self.residual, dim=-1)

def _text_tower(model):
    return getattr(model, 'text', model) if hasattr(getattr(model, 'text', None), 'token_embedding') else model

class OpenCLIPTextEncoderFromEmbeddings(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.m = _text_tower(model)
        miss = [x for x in ['token_embedding', 'positional_embedding', 'transformer', 'ln_final'] if not hasattr(self.m, x)]
        if miss:
            raise RuntimeError(f'CoOp unsupported; missing {miss}')

    def forward(self, prompt_emb, tokenized):
        tr = self.m.transformer
        cast = tr.get_cast_dtype() if hasattr(tr, 'get_cast_dtype') else prompt_emb.dtype
        x = prompt_emb.to(cast) + self.m.positional_embedding[:prompt_emb.shape[1]].to(prompt_emb.device, dtype=cast)
        mask = getattr(self.m, 'attn_mask', None)
        try:
            x = tr(x, attn_mask=mask)
        except TypeError:
            x = tr(x)
        x = self.m.ln_final(x)
        eot = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot]
        proj = getattr(self.m, 'text_projection', None)
        if proj is not None:
            x = proj(x) if isinstance(proj, nn.Linear) else x @ proj
        return F.normalize(x.float(), dim=-1)

class CoOpPromptLearner(nn.Module):

    def __init__(self, class_names, n_ctx=4):
        super().__init__()
        self.n_ctx = n_ctx
        tower = _text_tower(clip_model)
        prompts = [' '.join(['X'] * n_ctx) + ' ' + prettify(c) + '.' for c in class_names]
        tok = tokenizer(prompts).to(DEVICE)
        with torch.no_grad():
            emb = tower.token_embedding(tok)
        self.ctx = nn.Parameter(torch.empty(n_ctx, emb.shape[-1], device=DEVICE))
        nn.init.normal_(self.ctx, std=0.02)
        self.register_buffer('prefix', emb[:, :1, :].detach())
        self.register_buffer('suffix', emb[:, 1 + n_ctx:, :].detach())
        self.register_buffer('tokenized', tok.detach())

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(len(self.prefix), -1, -1)
        return (torch.cat([self.prefix, ctx, self.suffix], 1), self.tokenized)

class CoOpModel(nn.Module):

    def __init__(self, class_names, n_ctx=4):
        super().__init__()
        self.prompt = CoOpPromptLearner(class_names, n_ctx)
        self.encoder = OpenCLIPTextEncoderFromEmbeddings(clip_model)

    def class_features(self):
        p, t = self.prompt()
        return self.encoder(p, t)

def n_trainable(m):
    return sum((p.numel() for p in m.parameters() if p.requires_grad))
print('Shared-Caption params:', n_trainable(SharedCaptionAdapter(EMBED_DIM)))
print('Capacity-Matched-Shared params:', n_trainable(CapacityMatchedSharedCaptionAdapter(EMBED_DIM)))
print('CAPSTONE-CLIP params:', n_trainable(CAPSTONECLIP(EMBED_DIM)))
print('RLA params:', n_trainable(RLAHead(EMBED_DIM)))
if VISUAL_PROJ0 is not None:
    print('ProLIP params:', n_trainable(ProLIPProjector(VISUAL_PROJ0.to(DEVICE))))
set_seed(123)
B, D = (7, EMBED_DIM)
x = torch.randn(B, D, device=DEVICE)
u = torch.randn(B, D, device=DEVICE)
net = CAPSTONECLIP(D).to(DEVICE)
zc = net.encode_class_image(x)
zt = net.encode_class_text(u)
zr = net.encode_ret_image(x)
zur = net.encode_ret_text(u)
assert zc.shape == zt.shape == zr.shape == zur.shape == (B, D)
assert torch.allclose(zc.norm(dim=-1), torch.ones(B, device=DEVICE), atol=0.0001)
loss = (zc @ zt.t()).mean() + net.private_decorrelation(x) + net.gate_regularizer()
loss.backward()
assert all((torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
if VISUAL_PROJ0 is not None:
    p = ProLIPProjector(VISUAL_PROJ0.to(DEVICE))
    h = torch.randn(B, VISUAL_PROJ0.shape[0], device=DEVICE)
    assert p(h).shape == (B, D)
print('Synthetic architecture tests: PASS')
free_gpu()

def _slice(fc, idx):
    i = torch.as_tensor(idx, dtype=torch.long)
    return (fc['g'][i].float().to(DEVICE), fc['t'][i].float().to(DEVICE), fc['y'][i].to(DEVICE))

def frozen_class_features(name):
    return F.normalize(CLASS_U[name].mean(1), dim=-1)

def class_features_for(model, name, kind='frozen', class_subset=None):
    u = CLASS_U[name]
    C, T, D = u.shape
    flat = u.reshape(-1, D)
    if kind == 'capstone':
        z = model.encode_class_text(flat)
    elif kind == 'shared':
        z = model.encode_text(flat)
    else:
        z = F.normalize(flat, dim=-1)
    cf = F.normalize(z.reshape(C, T, D).mean(1), dim=-1)
    if class_subset is not None:
        cf = cf[torch.as_tensor(class_subset, dtype=torch.long, device=cf.device)]
    return cf

def expected_calibration_error(probs, y, n_bins=None):
    n_bins = n_bins or cfg.ece_bins
    conf, pred = probs.max(1)
    correct = pred.eq(y)
    ece = torch.tensor(0.0, device=probs.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs()
    return float(ece.cpu())

def metrics_from_logits(logits, y):
    probs = logits.softmax(-1)
    pred = probs.argmax(-1)
    yc = y.detach().cpu().numpy()
    pc = pred.detach().cpu().numpy()
    return {'Accuracy': float(accuracy_score(yc, pc)), 'MacroF1': float(f1_score(yc, pc, average='macro')), 'BalancedAccuracy': float(balanced_accuracy_score(yc, pc)), 'NLL': float(F.cross_entropy(logits, y).detach().cpu()), 'ECE': expected_calibration_error(probs, y), 'N': int(len(y))}

@torch.no_grad()
def evaluate_image_encoder(name, idx, image_encoder, class_features, label_subset=None):
    fc = get_dataset_cache(name)
    idx = torch.as_tensor(idx, dtype=torch.long)
    all_logits = []
    all_y = []
    label_map = None if label_subset is None else {int(c): i for i, c in enumerate(label_subset)}
    scale = clip_model.logit_scale.exp().detach()
    for s in range(0, len(idx), cfg.eval_chunk):
        ii = idx[s:s + cfg.eval_chunk]
        g = fc['g'][ii].float().to(DEVICE)
        y = fc['y'][ii].to(DEVICE)
        if label_map is not None:
            y = torch.as_tensor([label_map[int(v)] for v in y.detach().cpu().tolist()], dtype=torch.long, device=DEVICE)
        all_logits.append(scale * image_encoder(g) @ class_features.t())
        all_y.append(y)
    return metrics_from_logits(torch.cat(all_logits), torch.cat(all_y))

@torch.no_grad()
def evaluate_capstone(model, name, idx, class_subset=None, task='class'):
    cf = class_features_for(model, name, 'capstone', class_subset)
    enc = model.encode_class_image if task == 'class' else model.encode_ret_image
    return evaluate_image_encoder(name, idx, enc, cf, label_subset=class_subset)

@torch.no_grad()
def evaluate_shared(model, name, idx):
    return evaluate_image_encoder(name, idx, model.encode_image, class_features_for(model, name, 'shared'))

def sample_caption_batch(bank, B, generator=None):
    n = bank['n_img']
    B = min(int(B), n)
    img_idx = torch.randperm(n, generator=generator)[:B]
    offs = (torch.rand(B, generator=generator) * bank['cap_count'][img_idx].float()).long()
    cap_idx = bank['cap_start'][img_idx] + offs
    return (bank['g'][img_idx].float().to(DEVICE), bank['t'][img_idx].float().to(DEVICE), bank['u'][cap_idx].float().to(DEVICE))

def symmetric_clip_loss(image_z, text_z, logit_scale):
    target = torch.arange(len(image_z), device=image_z.device)
    sim = logit_scale.exp() * image_z @ text_z.t()
    return 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))

def feature_view(x, p=0.15, noise=0.01):
    mask = (torch.rand_like(x) > p).float() / (1 - p)
    return x * mask + noise * torch.randn_like(x)

def support_batch(name, K, seed, outer_id, class_subset=None):
    idx = support_indices(name, K, seed, outer_id, class_subset)
    return (_slice(get_dataset_cache(name), idx), idx)

def _cpu_state(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

def ckpt_path(*parts):
    safe = '__'.join((re.sub('[^A-Za-z0-9_.-]+', '-', str(x)) for x in parts))
    return os.path.join(MODEL_DIR, safe + '.pt')

def save_checkpoint(model, path, info):
    torch.save({'state_dict': _cpu_state(model), 'info': info, 'protocol_hash': PROTOCOL_HASH}, path)
    if IN_COLAB and cfg.cache_to_drive:
        safe_copy2(path, os.path.join(cfg.out_root, 'checkpoints', os.path.basename(path)), required=False)

def train_shared_caption(name, K, seed, outer_id, caption_dataset=None):
    hp = cfg.presets['Shared-Caption']
    set_seed(stable_int('shared', name, K, seed, outer_id, ACTIVE_BACKBONE_KEY))
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    net = SharedCaptionAdapter(EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=hp['lr'], weight_decay=cfg.weight_decay)
    bank = cap_bank(caption_dataset or cfg.caption_dataset, 'train')
    for ep in range(cfg.caption_epochs):
        steps = max(1, cfg.caption_steps_per_epoch)
        for _ in range(steps):
            ii = torch.randint(0, len(y), (min(cfg.batch_size, len(y)),), device=DEVICE)
            cf = class_features_for(net, name, 'shared')
            cls = F.cross_entropy(net.logit_scale.exp() * net.encode_image(g[ii]) @ cf.t(), y[ii])
            cg, _, cu = sample_caption_batch(bank, cfg.caption_batch_size)
            ret = symmetric_clip_loss(net.encode_image(cg), net.encode_text(cu), net.logit_scale)
            opt.zero_grad()
            (cls + hp['lambda_cap'] * ret).backward()
            opt.step()
    return (net, {'selection': 'prespecified', 'external_images': bank['n_img'], 'caption_mode': 'joint'})

def train_capacity_matched_shared(name, K, seed, outer_id, caption_dataset=None):
    hp = cfg.presets['Shared-Caption']
    set_seed(stable_int('capacity_shared', name, K, seed, outer_id, ACTIVE_BACKBONE_KEY))
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    net = CapacityMatchedSharedCaptionAdapter(EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=hp['lr'], weight_decay=cfg.weight_decay)
    bank = cap_bank(caption_dataset or cfg.caption_dataset, 'train')
    for ep in range(cfg.caption_epochs):
        for _ in range(max(1, cfg.caption_steps_per_epoch)):
            ii = torch.randint(0, len(y), (min(cfg.batch_size, len(y)),), device=DEVICE)
            cf = class_features_for(net, name, 'shared')
            cls = F.cross_entropy(net.logit_scale.exp() * net.encode_image(g[ii]) @ cf.t(), y[ii])
            cg, _, cu = sample_caption_batch(bank, cfg.caption_batch_size)
            ret = symmetric_clip_loss(net.encode_image(cg), net.encode_text(cu), net.logit_scale)
            opt.zero_grad()
            (cls + hp['lambda_cap'] * ret).backward()
            opt.step()
    ratio = n_trainable(net) / max(1, n_trainable(CAPSTONECLIP(EMBED_DIM)))
    return (net, {'selection': 'prespecified', 'external_images': bank['n_img'], 'caption_mode': 'joint', 'capacity_ratio_to_capstone': ratio})

def train_capstone(name, K, seed, outer_id, caption_dataset=None, control='joint_full', caption_scale=1.0, class_subset=None, hp_overrides=None):
    hp = dict(cfg.presets['CAPSTONE-CLIP'])
    hp.update(hp_overrides or {})
    set_seed(stable_int('capstone', name, K, seed, outer_id, control, caption_scale, ACTIVE_BACKBONE_KEY, class_subset))
    (g, _, y), support_idx = support_batch(name, K, seed, outer_id, class_subset)
    classes = sorted(np.unique(get_dataset_cache(name)['y'][support_idx].numpy()))
    y_map = {c: i for i, c in enumerate(classes)}
    y_local = torch.tensor([y_map[int(v)] for v in y.detach().cpu()], device=DEVICE)
    net = CAPSTONECLIP(EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=hp['lr'], weight_decay=cfg.weight_decay)
    base_bank = cap_bank(caption_dataset or cfg.caption_dataset, 'train')
    max_images = len(support_idx) if control == 'matched_budget' else None
    bank = subset_caption_bank(base_bank, fraction=caption_scale, max_images=max_images, seed=stable_int(name, K, seed, outer_id))
    if control == 'shuffled_caption':
        bank = shuffled_caption_bank(bank, seed)
    use_caption = control not in ('no_caption',)
    image_only = control == 'image_only'
    sequential = control == 'sequential_full'
    cf_base = CLASS_U[name][torch.as_tensor(classes, dtype=torch.long, device=DEVICE)]

    def classification_step():
        ii = torch.randint(0, len(y_local), (min(cfg.batch_size, len(y_local)),), device=DEVICE)
        C, T, D = cf_base.shape
        flat = cf_base.reshape(-1, D)
        cf_cls = F.normalize(net.encode_class_text(flat).reshape(C, T, D).mean(1), dim=-1)
        cf_shared = F.normalize(net.encode_shared_text(flat).reshape(C, T, D).mean(1), dim=-1)
        logits = net.logit_scale.exp() * net.encode_class_image(g[ii]) @ cf_cls.t()
        logits_s = net.logit_scale.exp() * net.encode_shared_image(g[ii]) @ cf_shared.t()
        l_cls = F.cross_entropy(logits, y_local[ii])
        l_shared = F.cross_entropy(logits_s, y_local[ii])
        l_cons = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits_s.detach(), dim=-1), reduction='batchmean')
        l_dec = net.private_decorrelation(g[ii])
        return l_cls + hp['lambda_shared_cls'] * l_shared + hp['lambda_cons'] * l_cons + hp['lambda_decor'] * l_dec

    def auxiliary_step():
        cg, _, cu = sample_caption_batch(bank, cfg.caption_batch_size)
        if image_only:
            z1 = net.encode_ret_image(feature_view(cg))
            z2 = net.encode_ret_image(feature_view(cg))
            return symmetric_clip_loss(z1, z2, net.logit_scale)
        return symmetric_clip_loss(net.encode_ret_image(cg), net.encode_ret_text(cu), net.logit_scale)
    if sequential and use_caption:
        for ep in range(cfg.sequential_caption_epochs):
            for _ in range(max(1, cfg.caption_steps_per_epoch)):
                loss = hp['lambda_cap'] * auxiliary_step() + hp['lambda_gate'] * net.gate_regularizer()
                opt.zero_grad()
                loss.backward()
                opt.step()
        for ep in range(cfg.sequential_support_epochs):
            loss = classification_step() + hp['lambda_gate'] * net.gate_regularizer()
            opt.zero_grad()
            loss.backward()
            opt.step()
    else:
        for ep in range(cfg.caption_epochs if use_caption else cfg.max_epochs):
            steps = max(1, cfg.caption_steps_per_epoch if use_caption else math.ceil(len(y_local) / cfg.batch_size))
            for _ in range(steps):
                loss = classification_step()
                if use_caption:
                    loss = loss + hp['lambda_cap'] * auxiliary_step()
                loss = loss + hp['lambda_gate'] * net.gate_regularizer()
                opt.zero_grad()
                loss.backward()
                opt.step()
    info = {'selection': 'prespecified_no_target_validation', 'control': control, 'caption_scale': caption_scale, 'support_images': len(support_idx), 'external_images': bank['n_img'] if use_caption else 0, 'external_to_support_ratio': bank['n_img'] / len(support_idx) if use_caption else 0.0, 'classes': classes, 'caption_dataset': caption_dataset or cfg.caption_dataset}
    return (net, info)

def train_rla(name, K, seed, outer_id):
    hp = cfg.presets['RLA']
    set_seed(stable_int('rla', name, K, seed, outer_id))
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    x = F.normalize(g, dim=-1)
    cf = frozen_class_features(name)
    m = RLAHead(EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=hp['lr'], weight_decay=0)
    lam = 1 / max(1, K)
    for _ in range(cfg.rla_epochs):
        loss = F.cross_entropy(clip_model.logit_scale.exp() * m(x) @ cf.t(), y) + lam * m.identity_penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (m, {'lambda': lam, 'selection': '1/K_validation_free'})

def train_prolip(name, K, seed, outer_id):
    fc = get_dataset_cache(name)
    if fc['h'] is None or VISUAL_PROJ0 is None:
        return (None, {'available': False, 'reason': 'pre-projection feature/projector unavailable'})
    hp = cfg.presets['ProLIP']
    idx = support_indices(name, K, seed, outer_id)
    h = fc['h'][idx].float().to(DEVICE)
    y = fc['y'][idx].to(DEVICE)
    m = ProLIPProjector(VISUAL_PROJ0.to(DEVICE))
    opt = torch.optim.AdamW(m.parameters(), lr=hp['lr'], weight_decay=0)
    cf = frozen_class_features(name)
    lam = 1 / max(1, K)
    for _ in range(cfg.prolip_epochs):
        loss = F.cross_entropy(clip_model.logit_scale.exp() * m(h) @ cf.t(), y) + lam * m.zero_shot_penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (m, {'available': True, 'lambda': lam, 'selection': 'official_style_1/K_validation_free'})

@torch.no_grad()
def eval_prolip(m, name, idx):
    fc = get_dataset_cache(name)
    i = torch.as_tensor(idx, dtype=torch.long)
    h = fc['h'][i].float().to(DEVICE)
    y = fc['y'][i].to(DEVICE)
    return metrics_from_logits(clip_model.logit_scale.exp() * m(h) @ frozen_class_features(name).t(), y)

def train_taskres(name, K, seed, outer_id):
    hp = cfg.presets['TaskRes']
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    x = F.normalize(g, dim=-1)
    m = TaskResModel(frozen_class_features(name), hp['alpha']).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=hp['lr'], weight_decay=cfg.weight_decay)
    for _ in range(cfg.taskres_epochs):
        loss = F.cross_entropy(clip_model.logit_scale.exp() * x @ m.class_features().t(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (m, {'selection': 'fixed_preset'})

def train_clip_adapter(name, K, seed, outer_id):
    hp = cfg.presets['CLIP-Adapter']
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    x = F.normalize(g, dim=-1)
    m = CLIPAdapter(EMBED_DIM, hp['res']).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=hp['lr'], weight_decay=cfg.weight_decay)
    cf = frozen_class_features(name)
    for _ in range(cfg.max_epochs):
        loss = F.cross_entropy(clip_model.logit_scale.exp() * m(x) @ cf.t(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (m, {'selection': 'fixed_preset'})

def build_tip(name, K, seed, outer_id):
    hp = cfg.presets['Tip-Adapter']
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    return (F.normalize(g, dim=-1), y, hp)

@torch.no_grad()
def eval_tip(name, idx, keys, labels, hp):
    fc = get_dataset_cache(name)
    i = torch.as_tensor(idx, dtype=torch.long)
    x = F.normalize(fc['g'][i].float().to(DEVICE), dim=-1)
    y = fc['y'][i].to(DEVICE)
    clip_logits = clip_model.logit_scale.exp() * x @ frozen_class_features(name).t()
    aff = x @ keys.t()
    cache = torch.exp(-hp['beta'] * (1 - aff))
    onehot = F.one_hot(labels, num_classes=len(fc['class_names'])).float()
    return metrics_from_logits(clip_logits + hp['alpha'] * cache @ onehot, y)

def train_tipf(name, K, seed, outer_id):
    keys, labels, hp = build_tip(name, K, seed, outer_id)
    keys = nn.Parameter(keys.clone())
    opt = torch.optim.AdamW([keys], lr=cfg.presets['Tip-Adapter-F']['lr'])
    cf = frozen_class_features(name)
    onehot = F.one_hot(labels, num_classes=len(cf)).float()
    x0 = keys.detach().clone()
    for _ in range(cfg.tipf_epochs):
        x = F.normalize(x0, dim=-1)
        k = F.normalize(keys, dim=-1)
        aff = x @ k.t()
        logits = clip_model.logit_scale.exp() * x @ cf.t() + hp['alpha'] * torch.exp(-hp['beta'] * (1 - aff)) @ onehot
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (F.normalize(keys.detach(), dim=-1), labels, {'selection': 'fixed_preset'})

def train_hoso(name, K, seed, outer_id):
    if K < 2:
        return (None, {'available': False, 'reason': 'K<2'})
    hp = cfg.presets['HOSO-Adapter']
    tr, hold = hoso_support_split(name, K, seed, outer_id)
    fc = get_dataset_cache(name)
    x = F.normalize(fc['g'][tr].float().to(DEVICE), dim=-1)
    y = fc['y'][tr].to(DEVICE)
    xh = F.normalize(fc['g'][hold].float().to(DEVICE), dim=-1)
    yh = fc['y'][hold].to(DEVICE)
    m = HOSOAdapter(EMBED_DIM).to(DEVICE)
    oa = torch.optim.SGD(m.adapter.parameters(), lr=hp['adapter_lr'], momentum=0.9)
    og = torch.optim.SGD([m.alpha_logit], lr=hp['alpha_lr'])
    cf = frozen_class_features(name)
    for _ in range(cfg.hoso_epochs):
        oa.zero_grad()
        F.cross_entropy(clip_model.logit_scale.exp() * m(x) @ cf.t(), y).backward()
        oa.step()
        for p in m.adapter.parameters():
            p.requires_grad_(False)
        og.zero_grad()
        F.cross_entropy(clip_model.logit_scale.exp() * m(xh) @ cf.t(), yh).backward()
        og.step()
        for p in m.adapter.parameters():
            p.requires_grad_(True)
    return (m, {'available': True, 'alpha': float(m.alpha().detach().cpu())})

def train_coop(name, K, seed, outer_id):
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    x = F.normalize(g, dim=-1)
    m = CoOpModel(get_dataset_cache(name)['class_names'], 4).to(DEVICE)
    opt = torch.optim.AdamW(m.prompt.parameters(), lr=cfg.presets['CoOp']['lr'])
    for _ in range(cfg.coop_epochs):
        loss = F.cross_entropy(clip_model.logit_scale.exp() * x @ m.class_features().t(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (m, {'selection': 'fixed_preset'})

@torch.no_grad()
def eval_coop(m, name, idx):
    return evaluate_image_encoder(name, idx, lambda x: F.normalize(x, dim=-1), m.class_features())

def fit_linear_probe(name, K, seed, outer_id):
    fc = get_dataset_cache(name)
    idx = support_indices(name, K, seed, outer_id)
    x = F.normalize(fc['g'][idx].float(), dim=-1).cpu().numpy()
    y = fc['y'][idx].numpy()
    return LogisticRegression(C=1.0, max_iter=3000, multi_class='auto', random_state=stable_int(name, K, seed, outer_id)).fit(x, y)

@torch.no_grad()
def eval_linear_probe(clf, name, idx):
    fc = get_dataset_cache(name)
    i = torch.as_tensor(idx, dtype=torch.long)
    x = F.normalize(fc['g'][i].float(), dim=-1).cpu().numpy()
    y = fc['y'][i].numpy()
    probs = torch.tensor(clf.predict_proba(x), dtype=torch.float32, device=DEVICE)
    return metrics_from_logits(torch.log(probs.clamp_min(1e-12)), torch.tensor(y, device=DEVICE))
from pathlib import Path
import time, json, os, glob, hashlib, math, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
REVISION_START = time.time()
SOFT_BUDGET_HOURS = 4.5
HARD_BUDGET_HOURS = 4.9
SOFT_BUDGET_S = SOFT_BUDGET_HOURS * 3600
HARD_BUDGET_S = HARD_BUDGET_HOURS * 3600
REV_ROOT = Path(cfg.drive_root) / 'reviewer_revision_4h_v1'
REV_TABLES = REV_ROOT / 'tables'
REV_FIGS = REV_ROOT / 'figures'
REV_STATE = REV_ROOT / 'state'
for p in [REV_ROOT, REV_TABLES, REV_FIGS, REV_STATE]:
    p.mkdir(parents=True, exist_ok=True)
REV_WIDTHS = [32, 64, 128, 256, 512]
REV_WIDTH_K = [1, 4, 16]
REV_WIDTH_SEEDS = [0, 1]
REV_EXPOSURE_K = [1, 4, 16]
REV_EXPOSURE_SEEDS = [0, 1]
REV_EXPOSURE_CONTROLS = ['no_caption', 'image_only', 'shuffled_caption', 'joint_full']
REV_CAPTION_K = 4
REV_CAPTION_SEEDS = [0]
REV_CAPTION_CORPORA = ['RSICD', 'RSITMD']
REV_AUDIT_THRESHOLDS = [0.99, 0.995, 0.999]
REV_MAX_VERIFIED_PAIRS_PER_DATASET = 100

def elapsed_h():
    return (time.time() - REVISION_START) / 3600.0

def remaining_h():
    return max(0.0, HARD_BUDGET_HOURS - elapsed_h())

def budget_ok(min_remaining_minutes=3):
    return time.time() - REVISION_START < HARD_BUDGET_S - min_remaining_minutes * 60

def stage_ok(min_remaining_minutes):
    return remaining_h() * 60 >= min_remaining_minutes

def atomic_csv_append(row, path, key_cols):
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    new = pd.DataFrame([row])
    if len(df):
        for c in key_cols:
            if c not in df.columns:
                df[c] = np.nan
        mask = np.ones(len(df), dtype=bool)
        for c in key_cols:
            mask &= df[c].astype(str).values == str(row.get(c))
        df = df.loc[~mask].copy()
    out = pd.concat([df, new], ignore_index=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return out

def row_done(path, key):
    path = Path(path)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        if df.empty:
            return False
        mask = np.ones(len(df), dtype=bool)
        for k, v in key.items():
            if k not in df.columns:
                return False
            mask &= df[k].astype(str).values == str(v)
        return bool(mask.any())
    except Exception:
        return False

def latest_existing_table(filename):
    candidates = []
    for pattern in [str(Path(cfg.drive_root) / 'outputs_*' / 'tables' / filename), str(Path(cfg.drive_root) / '**' / filename)]:
        candidates.extend(glob.glob(pattern, recursive=True))
    candidates = [p for p in candidates if 'reviewer_revision_4h_v1' not in p]
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]
PRIMARY_EXISTING_PATH = latest_existing_table('raw_primary_classification.csv')
CONTROLS_EXISTING_PATH = latest_existing_table('raw_caption_controls.csv')
PRIMARY_EXISTING = pd.read_csv(PRIMARY_EXISTING_PATH) if PRIMARY_EXISTING_PATH else pd.DataFrame()
CONTROLS_EXISTING = pd.read_csv(CONTROLS_EXISTING_PATH) if CONTROLS_EXISTING_PATH else pd.DataFrame()
print('Revision output:', REV_ROOT)
print('Existing primary:', PRIMARY_EXISTING_PATH)
print('Existing controls:', CONTROLS_EXISTING_PATH)
print(f'Budget: soft={SOFT_BUDGET_HOURS:.2f} h, hard={HARD_BUDGET_HOURS:.2f} h')
try:
    import imagehash
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ImageHash'])
    import imagehash
from PIL import Image

def _caption_train_paths(dsname):
    drive_dir = os.path.join(cfg.data_root, resolve_dataset_folder(dsname, cfg.data_root))
    splits, used = build_caption_splits_at(DATA_AT[dsname], dsname, [drive_dir, cfg.data_root])
    return ([p for p, _ in splits['train']], used)

def _sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _phash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert('RGB'))

@torch.no_grad()
def exhaustive_caption_target_feature_audit(caption_ds, target_ds, q_chunk=256, t_chunk=4096):
    bank = cap_bank(caption_ds, 'train')
    q_cpu = bank['g'].float()
    tgt = get_dataset_cache(target_ds)
    t_cpu = tgt['g'].float()
    best_score = torch.full((len(q_cpu),), -2.0, dtype=torch.float32)
    best_index = torch.full((len(q_cpu),), -1, dtype=torch.long)
    for qs in range(0, len(q_cpu), q_chunk):
        qe = min(len(q_cpu), qs + q_chunk)
        q = F.normalize(q_cpu[qs:qe].to(DEVICE), dim=-1)
        local_best = torch.full((qe - qs,), -2.0, device=DEVICE)
        local_idx = torch.full((qe - qs,), -1, dtype=torch.long, device=DEVICE)
        for ts in range(0, len(t_cpu), t_chunk):
            te = min(len(t_cpu), ts + t_chunk)
            t = F.normalize(t_cpu[ts:te].to(DEVICE), dim=-1)
            vals, pos = (q @ t.T).max(dim=1)
            improve = vals > local_best
            local_best[improve] = vals[improve]
            local_idx[improve] = pos[improve] + ts
        best_score[qs:qe] = local_best.detach().cpu()
        best_index[qs:qe] = local_idx.detach().cpu()
        free_gpu()
    cap_paths, annotation_source = _caption_train_paths(caption_ds)
    target_paths = list(tgt['paths'])
    if len(cap_paths) != len(q_cpu):
        print(f'[audit warning] path count mismatch for {caption_ds}: {len(cap_paths)} vs {len(q_cpu)}')
        cap_paths = [None] * len(q_cpu)
    summary = {'caption_dataset': caption_ds, 'target_dataset': target_ds, 'n_caption_train_images': int(len(q_cpu)), 'n_target_images': int(len(t_cpu)), 'annotation_source': str(annotation_source), 'feature_search_exhaustive': True, 'max_similarity': float(best_score.max())}
    for th in REV_AUDIT_THRESHOLDS:
        summary[f'n_ge_{th:.3f}'] = int((best_score >= th).sum())
        summary[f'frac_ge_{th:.3f}'] = float((best_score >= th).float().mean())
    order = torch.argsort(best_score, descending=True)
    flagged = torch.where(best_score >= min(REV_AUDIT_THRESHOLDS))[0]
    candidate_ids = flagged[:REV_MAX_VERIFIED_PAIRS_PER_DATASET]
    if len(candidate_ids) < min(20, len(order)):
        candidate_ids = order[:min(20, len(order))]
    pairs = []
    for qi in candidate_ids.tolist():
        ti = int(best_index[qi])
        cp = cap_paths[qi] if qi < len(cap_paths) else None
        tp = target_paths[ti] if ti < len(target_paths) else None
        row = {'caption_dataset': caption_ds, 'target_dataset': target_ds, 'caption_index': qi, 'target_index': ti, 'cosine_similarity': float(best_score[qi]), 'caption_path': cp, 'target_path': tp, 'sha256_equal': np.nan, 'phash_distance': np.nan}
        try:
            if cp and tp and os.path.isfile(cp) and os.path.isfile(tp):
                row['sha256_equal'] = bool(_sha256_file(cp) == _sha256_file(tp))
                row['phash_distance'] = int(_phash(cp) - _phash(tp))
        except Exception as e:
            row['verification_error'] = f'{type(e).__name__}: {e}'
        pairs.append(row)
    return (summary, pairs)
AUDIT_SUMMARY_PATH = REV_TABLES / 'cross_corpus_overlap_summary.csv'
AUDIT_PAIRS_PATH = REV_TABLES / 'cross_corpus_overlap_candidates.csv'
audit_summaries = []
audit_pairs = []
for caption_ds in [d for d in REV_CAPTION_CORPORA if (ACTIVE_BACKBONE_KEY, d) in CAPTION_BANKS]:
    for target_ds in cfg.datasets:
        key = {'caption_dataset': caption_ds, 'target_dataset': target_ds}
        if row_done(AUDIT_SUMMARY_PATH, key):
            continue
        if not budget_ok(5):
            print('Hard budget guard reached during overlap audit.')
            break
        print(f'[overlap audit] {caption_ds} -> {target_ds}')
        s, p = exhaustive_caption_target_feature_audit(caption_ds, target_ds)
        atomic_csv_append(s, AUDIT_SUMMARY_PATH, ['caption_dataset', 'target_dataset'])
        if p:
            old = pd.read_csv(AUDIT_PAIRS_PATH) if AUDIT_PAIRS_PATH.exists() else pd.DataFrame()
            add = pd.DataFrame(p)
            if len(old):
                old = old[~((old.caption_dataset == caption_ds) & (old.target_dataset == target_ds))]
            pd.concat([old, add], ignore_index=True).to_csv(AUDIT_PAIRS_PATH, index=False)
        audit_summaries.append(s)
        audit_pairs.extend(p)
print('Overlap audit complete/updated:', AUDIT_SUMMARY_PATH)

class RevisionWidthSharedAdapter(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()
        self.hidden = int(hidden)
        self.img = WidthGatedResidual(dim, self.hidden, init_gate=0.2)
        self.txt = WidthGatedResidual(dim, self.hidden, init_gate=0.2)
        self.register_buffer('logit_scale', clip_model.logit_scale.detach().clone())

    def encode_image(self, g):
        return F.normalize(self.img(g), dim=-1)

    def encode_text(self, u):
        return F.normalize(self.txt(u), dim=-1)

def train_revision_width_shared(name, K, seed, outer_id, hidden, caption_dataset='RSICD'):
    hp = cfg.presets['Shared-Caption']
    set_seed(stable_int('reviewer_width', name, K, seed, outer_id, hidden, ACTIVE_BACKBONE_KEY))
    (g, _, y), _ = support_batch(name, K, seed, outer_id)
    net = RevisionWidthSharedAdapter(EMBED_DIM, hidden).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=hp['lr'], weight_decay=cfg.weight_decay)
    bank = cap_bank(caption_dataset, 'train')
    t0 = time.time()
    for ep in range(cfg.caption_epochs):
        for _ in range(max(1, cfg.caption_steps_per_epoch)):
            ii = torch.randint(0, len(y), (min(cfg.batch_size, len(y)),), device=DEVICE)
            cf = class_features_for(net, name, 'shared')
            cls = F.cross_entropy(net.logit_scale.exp() * net.encode_image(g[ii]) @ cf.T, y[ii])
            cg, _, cu = sample_caption_batch(bank, cfg.caption_batch_size)
            ret = symmetric_clip_loss(net.encode_image(cg), net.encode_text(cu), net.logit_scale)
            opt.zero_grad()
            (cls + hp['lambda_cap'] * ret).backward()
            opt.step()
    return (net, time.time() - t0)

def existing_width_rows(primary):
    rows = []
    if primary is None or primary.empty:
        return pd.DataFrame()
    method_to_width = {'Shared-Caption': 128, 'CAPSTONE-CLIP-Shared': 128, 'Capacity-Matched-Shared': 256, 'CAPSTONE-CLIP-CM': 256}
    for method, width in method_to_width.items():
        q = primary[primary.method.astype(str) == method].copy()
        if q.empty:
            continue
        q['width'] = width
        q['source'] = 'existing_primary'
        rows.append(q)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
WIDTH_PATH = REV_TABLES / 'adapter_width_sweep_raw.csv'
reuse = existing_width_rows(PRIMARY_EXISTING)
if len(reuse):
    required = ['dataset', 'K', 'fold', 'support_seed', 'width', 'Accuracy']
    if set(required).issubset(reuse.columns):
        keep = [c for c in reuse.columns if c in set(required + ['MacroF1', 'BalancedAccuracy', 'NLL', 'ECE', 'source'])]
        cur = pd.read_csv(WIDTH_PATH) if WIDTH_PATH.exists() else pd.DataFrame()
        add = reuse[keep].copy()
        if len(cur):
            merged = pd.concat([cur, add], ignore_index=True)
            merged = merged.drop_duplicates(['dataset', 'K', 'fold', 'support_seed', 'width'], keep='first')
        else:
            merged = add.drop_duplicates(['dataset', 'K', 'fold', 'support_seed', 'width'])
        merged.to_csv(WIDTH_PATH, index=False)

def width_param_count(hidden):
    m = RevisionWidthSharedAdapter(EMBED_DIM, hidden)
    return int(sum((p.numel() for p in m.parameters() if p.requires_grad)))
width_order = [32, 64, 128, 256, 512]
for hidden in width_order:
    for name in cfg.datasets:
        for K in REV_WIDTH_K:
            for oid in range(cfg.n_outer_folds):
                for seed in REV_WIDTH_SEEDS:
                    key = {'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'width': hidden}
                    if row_done(WIDTH_PATH, key):
                        continue
                    if not budget_ok(8):
                        print('Hard budget guard reached in width sweep.')
                        break
                    print(f'[width] h={hidden} {name} K={K} f={oid} s={seed} | elapsed={elapsed_h():.2f}h')
                    try:
                        model, train_s = train_revision_width_shared(name, K, seed, oid, hidden, 'RSICD')
                        met = evaluate_shared(model, name, test_indices(name, oid))
                        row = {**key, 'backbone': ACTIVE_BACKBONE_KEY, 'trainable_params': width_param_count(hidden), 'train_seconds': float(train_s), 'source': 'new_revision_run', **met}
                    except Exception as e:
                        row = {**key, 'backbone': ACTIVE_BACKBONE_KEY, 'Accuracy': np.nan, 'available': False, 'reason': f'{type(e).__name__}: {e}'}
                    atomic_csv_append(row, WIDTH_PATH, ['dataset', 'K', 'fold', 'support_seed', 'width'])
                    free_gpu()
print('Width sweep file:', WIDTH_PATH)
EXPOSURE_PATH = REV_TABLES / 'all6_exposure_controls_raw.csv'
if CONTROLS_EXISTING is not None and len(CONTROLS_EXISTING):
    q = CONTROLS_EXISTING.copy()
    if 'control' in q.columns:
        q = q[q.control.isin(REV_EXPOSURE_CONTROLS)].copy()
    if len(q):
        q['source'] = 'existing_caption_controls'
        cur = pd.read_csv(EXPOSURE_PATH) if EXPOSURE_PATH.exists() else pd.DataFrame()
        merged = pd.concat([cur, q], ignore_index=True) if len(cur) else q
        keys = ['dataset', 'K', 'fold', 'support_seed', 'control']
        present = [k for k in keys if k in merged.columns]
        if len(present) == len(keys):
            merged = merged.drop_duplicates(keys, keep='first')
        merged.to_csv(EXPOSURE_PATH, index=False)
existing_ds = set()
if EXPOSURE_PATH.exists():
    try:
        existing_ds = set(pd.read_csv(EXPOSURE_PATH).dataset.astype(str).unique())
    except Exception:
        pass
ordered_datasets = [d for d in cfg.datasets if d not in existing_ds] + [d for d in cfg.datasets if d in existing_ds]
for name in ordered_datasets:
    for K in REV_EXPOSURE_K:
        for oid in range(cfg.n_outer_folds):
            for seed in REV_EXPOSURE_SEEDS:
                for control in REV_EXPOSURE_CONTROLS:
                    key = {'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'control': control}
                    if row_done(EXPOSURE_PATH, key):
                        continue
                    if not budget_ok(8):
                        print('Hard budget guard reached in exposure expansion.')
                        break
                    print(f'[exposure] {name} K={K} f={oid} s={seed} {control} | elapsed={elapsed_h():.2f}h')
                    try:
                        model, info = train_capstone(name, K, seed, oid, 'RSICD', control, 1.0)
                        met = evaluate_capstone(model, name, test_indices(name, oid))
                        row = {**key, 'backbone': ACTIVE_BACKBONE_KEY, 'caption_corpus': 'RSICD', 'source': 'new_revision_run', **met, **info}
                    except Exception as e:
                        row = {**key, 'backbone': ACTIVE_BACKBONE_KEY, 'Accuracy': np.nan, 'available': False, 'reason': f'{type(e).__name__}: {e}'}
                    atomic_csv_append(row, EXPOSURE_PATH, ['dataset', 'K', 'fold', 'support_seed', 'control'])
                    free_gpu()
print('Exposure-control file:', EXPOSURE_PATH)
CAPTION_ROBUSTNESS_PATH = REV_TABLES / 'caption_corpus_robustness_raw.csv'
if stage_ok(20) and all(((ACTIVE_BACKBONE_KEY, d) in CAPTION_BANKS for d in REV_CAPTION_CORPORA)):
    for corpus in REV_CAPTION_CORPORA:
        for name in cfg.datasets:
            for oid in range(cfg.n_outer_folds):
                for seed in REV_CAPTION_SEEDS:
                    key = {'caption_corpus': corpus, 'dataset': name, 'K': REV_CAPTION_K, 'fold': oid, 'support_seed': seed}
                    if row_done(CAPTION_ROBUSTNESS_PATH, key):
                        continue
                    if not budget_ok(6):
                        break
                    print(f'[caption robustness] {corpus} {name} f={oid} s={seed}')
                    try:
                        model, info = train_capstone(name, REV_CAPTION_K, seed, oid, corpus, 'joint_full', 1.0)
                        met = evaluate_capstone(model, name, test_indices(name, oid))
                        row = {**key, 'backbone': ACTIVE_BACKBONE_KEY, 'source': 'new_revision_run', **met, **info}
                    except Exception as e:
                        row = {**key, 'Accuracy': np.nan, 'available': False, 'reason': f'{type(e).__name__}: {e}'}
                    atomic_csv_append(row, CAPTION_ROBUSTNESS_PATH, ['caption_corpus', 'dataset', 'K', 'fold', 'support_seed'])
                    free_gpu()
else:
    print(f'Caption-corpus stage skipped: remaining={remaining_h():.2f}h or RSITMD bank unavailable.')
from itertools import product

def _metric_to_pp(series):
    s = pd.to_numeric(series, errors='coerce')
    if s.abs().max() <= 1.5:
        return 100 * s
    return s

def exact_sign_flip_p(dataset_effects):
    d = np.asarray(dataset_effects, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan
    obs = abs(d.mean())
    count = 0
    total = 2 ** n
    for bits in product([-1, 1], repeat=n):
        v = d * np.asarray(bits)
        if abs(v.mean()) >= obs - 1e-12:
            count += 1
    return count / total

def dataset_bootstrap_ci(dataset_effects, B=20000, seed=20261005):
    d = np.asarray(dataset_effects, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(B)])
    return tuple(np.quantile(vals, [0.025, 0.975]))
width_df = pd.read_csv(WIDTH_PATH) if WIDTH_PATH.exists() else pd.DataFrame()
if len(width_df):
    width_df['Accuracy_pp'] = _metric_to_pp(width_df['Accuracy'])
    if 'ECE' in width_df:
        width_df['ECE_pp'] = _metric_to_pp(width_df['ECE'])
    width_summary = width_df.dropna(subset=['Accuracy_pp']).groupby(['width'], as_index=False).agg(Accuracy_pp=('Accuracy_pp', 'mean'), Accuracy_sd_pp=('Accuracy_pp', 'std'), ECE_pp=('ECE_pp', 'mean') if 'ECE_pp' in width_df else ('Accuracy_pp', 'size'), n=('Accuracy_pp', 'size'))
    width_summary['trainable_params'] = width_summary.width.map(width_param_count)
    width_summary.to_csv(REV_TABLES / 'adapter_width_sweep_summary.csv', index=False)
    foldw = width_df.groupby(['dataset', 'K', 'fold', 'width'], as_index=False).Accuracy_pp.mean()
    a = foldw[foldw.width == 512].rename(columns={'Accuracy_pp': 'a'})
    b = foldw[foldw.width == 128].rename(columns={'Accuracy_pp': 'b'})
    m = a.merge(b, on=['dataset', 'K', 'fold'])
    if len(m):
        ds_eff = m.assign(delta=m.a - m.b).groupby('dataset').delta.mean()
        lo, hi = dataset_bootstrap_ci(ds_eff.values)
        width_stats = pd.DataFrame([{'comparison': 'width512_minus_width128', 'mean_delta_pp': float(ds_eff.mean()), 'dataset_boot_ci_low_pp': lo, 'dataset_boot_ci_high_pp': hi, 'exact_sign_flip_p': exact_sign_flip_p(ds_eff.values), 'n_datasets': len(ds_eff)}])
        width_stats.to_csv(REV_TABLES / 'adapter_width_dataset_level_stats.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(width_summary.width, width_summary.Accuracy_pp, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Shared adapter hidden width')
    ax.set_ylabel('Mean accuracy (%)')
    ax.set_title('Adapter-width sweep under fixed supervision')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REV_FIGS / 'adapter_width_accuracy.png', dpi=220)
    plt.close(fig)
    if 'ECE_pp' in width_summary:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(width_summary.width, width_summary.ECE_pp, marker='o')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Shared adapter hidden width')
        ax.set_ylabel('Mean ECE (%)')
        ax.set_title('Calibration versus adapter width')
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(REV_FIGS / 'adapter_width_ece.png', dpi=220)
        plt.close(fig)
exp_df = pd.read_csv(EXPOSURE_PATH) if EXPOSURE_PATH.exists() else pd.DataFrame()
if len(exp_df):
    exp_df['Accuracy_pp'] = _metric_to_pp(exp_df['Accuracy'])
    if 'ECE' in exp_df:
        exp_df['ECE_pp'] = _metric_to_pp(exp_df['ECE'])
    exp_summary = exp_df.dropna(subset=['Accuracy_pp']).groupby(['dataset', 'K', 'control'], as_index=False).agg(Accuracy_pp=('Accuracy_pp', 'mean'), ECE_pp=('ECE_pp', 'mean') if 'ECE_pp' in exp_df else ('Accuracy_pp', 'size'), n=('Accuracy_pp', 'size'))
    exp_summary.to_csv(REV_TABLES / 'all6_exposure_controls_summary.csv', index=False)
    macro = exp_summary.groupby(['K', 'control'], as_index=False).agg(Accuracy_pp=('Accuracy_pp', 'mean'), ECE_pp=('ECE_pp', 'mean'))
    macro.to_csv(REV_TABLES / 'all6_exposure_controls_macro.csv', index=False)
    stat_rows = []
    folde = exp_df.groupby(['dataset', 'K', 'fold', 'control'], as_index=False).Accuracy_pp.mean()
    for ref in ['image_only', 'shuffled_caption', 'no_caption']:
        a = folde[folde.control == 'joint_full'].rename(columns={'Accuracy_pp': 'a'})
        b = folde[folde.control == ref].rename(columns={'Accuracy_pp': 'b'})
        mm = a.merge(b, on=['dataset', 'K', 'fold'])
        if len(mm):
            dse = mm.assign(delta=mm.a - mm.b).groupby('dataset').delta.mean()
            lo, hi = dataset_bootstrap_ci(dse.values)
            stat_rows.append({'comparison': f'joint_full_minus_{ref}', 'mean_delta_pp': float(dse.mean()), 'dataset_boot_ci_low_pp': lo, 'dataset_boot_ci_high_pp': hi, 'exact_sign_flip_p': exact_sign_flip_p(dse.values), 'n_datasets': len(dse)})
    pd.DataFrame(stat_rows).to_csv(REV_TABLES / 'all6_exposure_dataset_level_stats.csv', index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for ctrl, g in macro.groupby('control'):
        ax.plot(g.K, g.Accuracy_pp, marker='o', label=ctrl)
    ax.set_xlabel('Shots per class')
    ax.set_ylabel('Six-dataset mean accuracy (%)')
    ax.set_title('Expanded image-exposure controls')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REV_FIGS / 'all6_exposure_controls.png', dpi=220)
    plt.close(fig)
if PRIMARY_EXISTING is not None and len(PRIMARY_EXISTING) and set(['dataset', 'K', 'fold', 'support_seed', 'method', 'Accuracy']).issubset(PRIMARY_EXISTING.columns):
    q = PRIMARY_EXISTING.copy()
    q['Accuracy_pp'] = _metric_to_pp(q['Accuracy'])
    seed_var = q.groupby(['dataset', 'K', 'fold', 'method'], as_index=False).agg(seed_sd_pp=('Accuracy_pp', 'std'), seed_range_pp=('Accuracy_pp', lambda x: float(np.nanmax(x) - np.nanmin(x))), n_seeds=('support_seed', 'nunique'))
    seed_var.to_csv(REV_TABLES / 'empirical_support_seed_variability.csv', index=False)
    selected = seed_var[seed_var.method.astype(str).isin(['CAPSTONE-CLIP', 'CAPSTONE-CLIP-Full', 'Shared-Caption', 'CAPSTONE-CLIP-Shared', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP-CM', 'ProLIP', 'RLA'])]
    if len(selected):
        desc = pd.DataFrame([{'median_seed_sd_pp': float(selected.seed_sd_pp.median()), 'p75_seed_sd_pp': float(selected.seed_sd_pp.quantile(0.75)), 'p95_seed_sd_pp': float(selected.seed_sd_pp.quantile(0.95)), 'median_seed_range_pp': float(selected.seed_range_pp.median()), 'p95_seed_range_pp': float(selected.seed_range_pp.quantile(0.95)), 'n_cells': len(selected)}])
        desc.to_csv(REV_TABLES / 'empirical_variability_scale_summary.csv', index=False)
cap_df = pd.read_csv(CAPTION_ROBUSTNESS_PATH) if CAPTION_ROBUSTNESS_PATH.exists() else pd.DataFrame()
if len(cap_df):
    cap_df['Accuracy_pp'] = _metric_to_pp(cap_df['Accuracy'])
    cap_summary = cap_df.groupby(['dataset', 'caption_corpus'], as_index=False).agg(Accuracy_pp=('Accuracy_pp', 'mean'), ECE=('ECE', 'mean') if 'ECE' in cap_df else ('Accuracy_pp', 'size'), n=('Accuracy_pp', 'size'))
    cap_summary.to_csv(REV_TABLES / 'caption_corpus_robustness_summary.csv', index=False)
lines = ['# CAPSTONE reviewer-revision experiment report', '', f'- Wall-clock used: {elapsed_h():.2f} h', f'- Hard budget: {HARD_BUDGET_HOURS:.2f} h', f'- Backbone: {ACTIVE_BACKBONE_KEY}', '- Target validation: not used', '']
if AUDIT_SUMMARY_PATH.exists():
    a = pd.read_csv(AUDIT_SUMMARY_PATH)
    lines += ['## R1-#4 — Cross-corpus overlap audit', f'- Completed caption→target combinations: {len(a)} / {len(REV_CAPTION_CORPORA) * len(cfg.datasets)}', f'- Maximum observed frozen-feature cosine similarity: {a.max_similarity.max():.6f}', '']
if (REV_TABLES / 'adapter_width_sweep_summary.csv').exists():
    w = pd.read_csv(REV_TABLES / 'adapter_width_sweep_summary.csv')
    best = w.loc[w.Accuracy_pp.idxmax()]
    lines += ['## R1-#7 — Adapter-width sweep', f'- Widths represented: {sorted(w.width.astype(int).unique().tolist())}', f'- Best macro accuracy in completed sweep: width={int(best.width)}, {best.Accuracy_pp:.3f}%', '- Interpret only over the tested width interval; do not claim capacity is universally irrelevant.', '']
if (REV_TABLES / 'all6_exposure_controls_summary.csv').exists():
    e = pd.read_csv(REV_TABLES / 'all6_exposure_controls_summary.csv')
    lines += ['## R1-#5/#6 — Exposure controls', f'- Datasets represented: {e.dataset.nunique()} / {len(cfg.datasets)}', f'- Controls represented: {sorted(e.control.unique().tolist())}', '- The shuffled-caption control is objective-matched and isolates correspondence more cleanly than the image-only control.', '']
if (REV_TABLES / 'empirical_variability_scale_summary.csv').exists():
    v = pd.read_csv(REV_TABLES / 'empirical_variability_scale_summary.csv').iloc[0]
    lines += ['## R1-#9 — Empirical variability scale', f'- Median support-seed SD: {v.median_seed_sd_pp:.3f} pp', f'- 95th percentile support-seed SD: {v.p95_seed_sd_pp:.3f} pp', '- Use this only as descriptive empirical context; it is not a formally calibrated equivalence margin.', '']
if (REV_TABLES / 'caption_corpus_robustness_summary.csv').exists():
    c = pd.read_csv(REV_TABLES / 'caption_corpus_robustness_summary.csv')
    lines += ['## R1-#12 — Caption-corpus robustness', f'- Datasets represented: {c.dataset.nunique()}', f'- Corpora represented: {sorted(c.caption_corpus.unique().tolist())}', '']
lines += ['## Files', f'- Tables: `{REV_TABLES}`', f'- Figures: `{REV_FIGS}`', '', 'The response letter should quote the actual completed rows and should not claim completion for stages stopped by the wall-clock guard.']
(REV_ROOT / 'REVIEWER_REVISION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
import shutil
zip_path = shutil.make_archive(str(REV_ROOT), 'zip', str(REV_ROOT))
print('\n'.join(lines))
print('Revision ZIP:', zip_path)
