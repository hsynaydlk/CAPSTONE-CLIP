import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
RUN_PROFILE = 'paper_full'

@dataclass
class CFG:
    run_tag: str = 'reviewer_revision_final_clean_20261006'
    protocol_version: str = 'capstone_reviewer_revision_final_clean_v1'
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
cfg.run_tag = 'final_29_ekim_ucmerced'
cfg.datasets = ['UCMerced']
cfg.expected_counts = {'UCMerced': 2100}
cfg.expected_class_counts = {'UCMerced': 21}
cfg.caption_datasets = ['RSITMD']
cfg.caption_dataset = 'RSITMD'
cfg.external_caption_dataset = 'RSITMD'
cfg.require_second_caption_dataset = True
cfg.shots = [4]
cfg.support_seeds = [0]
cfg.caption_control_datasets = []
cfg.ablation_datasets = []
for _flag in ('run_primary', 'run_caption_controls', 'run_retrieval', 'run_base_to_new', 'run_cross_dataset', 'run_cross_backbone', 'run_efficiency', 'run_near_duplicate_audit', 'run_architecture_ablations', 'run_class_diagnostics'):
    setattr(cfg, _flag, False)
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
import os, json, hashlib, itertools, math, time, csv, traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps, ImageDraw
try:
    import imagehash
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ImageHash'])
    import imagehash
from scipy import stats as scipy_stats
TARGET = 'UCMerced'
CAPTION = 'RSITMD'
FINAL_K = 4
FINAL_SUPPORT_SEED = 0
BASE_REV = Path(cfg.drive_root) / 'reviewer_revision_final_clean_20261006' / 'tables'
FINAL_ROOT = Path(cfg.drive_root) / 'reviewer_final_29_ekim'
FINAL_TABLES = FINAL_ROOT / 'tables'
FINAL_FIGS = FINAL_ROOT / 'figures'
for folder in (FINAL_ROOT, FINAL_TABLES, FINAL_FIGS):
    folder.mkdir(parents=True, exist_ok=True)
BASE_ROBUSTNESS = BASE_REV / 'caption_corpus_robustness_raw.csv'
BASE_ADAPTER = BASE_REV / 'adapter_width_sweep_raw.csv'
BASE_EXPOSURE = BASE_REV / 'all6_exposure_controls_raw.csv'
EXISTING_AUDIT = BASE_REV / 'cross_corpus_overlap_candidates.csv'
assert TARGET in SPLITS and (ACTIVE_BACKBONE_KEY, CAPTION) in CAPTION_BANKS
assert cfg.n_outer_folds == 5 and cfg.fold_seed == 20260717
assert cfg.support_namespace == 'capstone_disjoint5fold_support5_v1'
assert BASE_ROBUSTNESS.is_file(), f'Missing initial RSITMD baseline: {BASE_ROBUSTNESS}'
assert all(((FINAL_K, FINAL_SUPPORT_SEED) == (4, 0) for _ in [0]))
print('Output:', FINAL_ROOT)
print('Baseline:', BASE_ROBUSTNESS)
print('Fold/test lock:', TEST_UNLOCKED, ' | target validation: disabled')

class BKNode:
    __slots__ = ('value', 'ids', 'kids')

    def __init__(self, value, idx):
        self.value = int(value)
        self.ids = [idx]
        self.kids = {}

class HammingBKTree:

    def __init__(self):
        self.root = None

    def add(self, value, idx):
        value = int(value)
        if self.root is None:
            self.root = BKNode(value, idx)
            return
        node = self.root
        while True:
            d = (node.value ^ value).bit_count()
            if d == 0:
                node.ids.append(idx)
                return
            if d not in node.kids:
                node.kids[d] = BKNode(value, idx)
                return
            node = node.kids[d]

    def query(self, value, radius):
        value = int(value)
        hits = []
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            d = (node.value ^ value).bit_count()
            if d <= radius:
                hits.extend(((j, d) for j in node.ids))
            stack.extend((ch for gap, ch in node.kids.items() if d - radius <= gap <= d + radius))
        return hits

def inspect_image(path):
    assert os.path.isfile(path), f'Missing image: {path}'
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        ph = int(str(imagehash.phash(im)), 16)
    return ph

def stream_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
caption_path_root = os.path.join(cfg.data_root, resolve_dataset_folder(CAPTION, cfg.data_root))
caption_splits, caption_annotation = build_caption_splits_at(DATA_AT[CAPTION], CAPTION, [caption_path_root, cfg.data_root])
caption_paths = [str(p) for p, _ in caption_splits['train']]
target_fc = get_dataset_cache(TARGET)
target_paths = list(map(str, target_fc['paths']))
original_bank = cap_bank(CAPTION, 'train')
assert len(caption_paths) == int(original_bank['n_img']), 'Caption bank / annotation length mismatch'
assert len(target_paths) == len(target_fc['g']), 'Target cache / paths length mismatch'
assert len(set(caption_paths)) == len(caption_paths), 'Duplicate caption paths: verify ordering'
assert len(set(target_paths)) == len(target_paths), 'Duplicate target paths: verify ordering'
print('Caption train images:', len(caption_paths), ' | Target images:', len(target_paths))
print('Caption annotation:', caption_annotation)

@torch.no_grad()
def verify_caption_cache_alignment(paths, bank, samples=12, tolerance=0.985):
    ids = np.linspace(0, len(paths) - 1, min(samples, len(paths)), dtype=int)
    gs, _, _ = encode_paths([paths[int(i)] for i in ids], 'validate_caption_index_alignment', keep_preproj=False)
    cache = bank['g'][ids].float()
    alignment = F.cosine_similarity(F.normalize(gs.float(), dim=-1), F.normalize(cache, dim=-1)).cpu().numpy()
    print('Caption index-alignment similarities:', np.round(alignment, 6))
    if not np.isfinite(alignment).all() or float(np.min(alignment)) < tolerance:
        raise RuntimeError('Caption paths and cached embedding indices may be misaligned. Stop before exclusion.')
    return alignment
alignment_scores = verify_caption_cache_alignment(caption_paths, original_bank)
HASH_CSV = FINAL_TABLES / 'rsitmd_ucmerced_image_hashes.csv'
HASH_VERSION = 'phash64_exif_v1'
prior = pd.read_csv(HASH_CSV, dtype={'phash64': str, 'sha256': str}) if HASH_CSV.is_file() else pd.DataFrame()
existing = {}
if not prior.empty and set(['file_path', 'sha256', 'phash64', 'hash_version']).issubset(prior.columns):
    existing = {str(r.file_path): r for r in prior.itertuples(index=False) if str(r.hash_version) == HASH_VERSION}
rows = []
for corpus, paths in [('RSITMD_train', caption_paths), ('UCMerced_all', target_paths)]:
    for idx, path in enumerate(paths):
        old_hash = existing.get(path)
        if old_hash is not None and os.path.isfile(path) and (int(os.path.getsize(path)) == int(getattr(old_hash, 'size_bytes', -1))):
            d = {'source': corpus, 'index': idx, 'file_path': path, 'sha256': str(old_hash.sha256), 'phash64': str(old_hash.phash64), 'size_bytes': int(old_hash.size_bytes), 'hash_version': HASH_VERSION}
        else:
            d = {'source': corpus, 'index': idx, 'file_path': path, 'sha256': stream_sha256(path), 'phash64': f'{inspect_image(path):016x}', 'size_bytes': int(os.path.getsize(path)), 'hash_version': HASH_VERSION}
        rows.append(d)
    pd.DataFrame(rows).to_csv(HASH_CSV.with_suffix('.tmp'), index=False)
    os.replace(HASH_CSV.with_suffix('.tmp'), HASH_CSV)
hashes = pd.DataFrame(rows)
assert len(hashes) == len(caption_paths) + len(target_paths)
assert hashes['sha256'].notna().all() and hashes['phash64'].notna().all()
print('Hashed images:', len(hashes), '→', HASH_CSV)
tg = hashes[hashes.source == 'UCMerced_all'].reset_index(drop=True)
cp = hashes[hashes.source == 'RSITMD_train'].reset_index(drop=True)
assert tg['index'].tolist() == list(range(len(target_paths)))
assert cp['index'].tolist() == list(range(len(caption_paths)))
tree = HammingBKTree()
for r in tg.itertuples():
    tree.add(int(r.phash64, 16), int(r.index))
sha_targets = defaultdict(list)
for r in tg.itertuples():
    sha_targets[str(r.sha256)].append(int(r.index))
pair_map = {}
for r in cp.itertuples():
    ci = int(r.index)
    for ti, d in tree.query(int(r.phash64, 16), 4):
        pair_map[ci, ti] = min(d, pair_map.get((ci, ti), 99))
    for ti in sha_targets.get(str(r.sha256), []):
        pair_map[ci, ti] = 0
cm = F.normalize(original_bank['g'].float(), dim=-1)
tm = F.normalize(target_fc['g'].float(), dim=-1)
pair_rows = []
for (ci, ti), distance in sorted(pair_map.items()):
    pair_rows.append({'caption_index': ci, 'target_index': ti, 'caption_path': caption_paths[ci], 'target_path': target_paths[ti], 'phash_distance': int(distance), 'sha256_equal': bool(cp.iloc[ci].sha256 == tg.iloc[ti].sha256), 'cosine_similarity': float(torch.dot(cm[ci], tm[ti])), 'source': 'full_all_images_phash_bktree'})
pairs_df = pd.DataFrame(pair_rows, columns=['caption_index', 'target_index', 'caption_path', 'target_path', 'phash_distance', 'sha256_equal', 'cosine_similarity', 'source'])
PAIR_PATH = FINAL_TABLES / 'rsitmd_ucmerced_all_phash_candidates.csv'
pairs_df.to_csv(PAIR_PATH, index=False)
remove_ids = sorted(set(pairs_df['caption_index'].astype(int).tolist())) if len(pairs_df) else []
assert all((0 <= i < len(caption_paths) for i in remove_ids))
assert 0 < len(caption_paths) - len(remove_ids), 'No caption images would remain after exclusion'
print('Candidate pairs:', len(pairs_df), ' | removed unique caption images:', len(remove_ids))
print('Unique caption train images:', len(cp), ' | remaining:', len(cp) - len(remove_ids))
if len(pairs_df):
    print('pHash distances:', pairs_df.phash_distance.value_counts().sort_index().to_dict())
    display(pairs_df.sort_values(['phash_distance', 'cosine_similarity'], ascending=[True, False]).head(20))
print('Sensitivity / excluded RSITMD images: pHash<=0:', pairs_df.loc[pairs_df.phash_distance <= 0, 'caption_index'].nunique() if len(pairs_df) else 0, 'pHash<=2:', pairs_df.loc[pairs_df.phash_distance <= 2, 'caption_index'].nunique() if len(pairs_df) else 0, 'pHash<=4:', len(remove_ids))
CONTACT_PATH = FINAL_FIGS / 'rsitmd_ucmerced_near_duplicate_contact_sheet.jpg'
if len(pairs_df):
    best = pairs_df.sort_values(['phash_distance', 'cosine_similarity'], ascending=[True, False]).drop_duplicates('caption_index')
    best = best.head(24)
    thumb_w, thumb_h = (210, 150)
    tile_w, tile_h = (440, 195)
    columns = 2
    sheet = Image.new('RGB', (tile_w * columns, tile_h * math.ceil(len(best) / columns)), 'white')
    painter = ImageDraw.Draw(sheet)
    for n, r in enumerate(best.itertuples()):
        x = n % columns * tile_w
        y = n // columns * tile_h
        for path, dx in [(r.caption_path, 0), (r.target_path, thumb_w + 12)]:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((thumb_w, thumb_h))
                sheet.paste(im, (x + dx, y + 24))
        painter.text((x + 4, y + 2), f'caption#{r.caption_index} → target#{r.target_index} pH={r.phash_distance} cos={r.cosine_similarity:.4f}', fill='black')
    sheet.save(CONTACT_PATH, quality=88)
    display(sheet.resize((min(sheet.width, 850), round(sheet.height * min(sheet.width, 850) / sheet.width))))
    print('Contact sheet:', CONTACT_PATH)
else:
    print('No candidates; no contact sheet required.')
raw_base = pd.read_csv(BASE_ROBUSTNESS)
required = {'caption_corpus', 'dataset', 'K', 'fold', 'support_seed', 'Accuracy'}
assert required.issubset(raw_base.columns), f'Missing baseline columns: {required - set(raw_base.columns)}'
mask = (raw_base['caption_corpus'].astype(str) == CAPTION) & (raw_base['dataset'].astype(str) == TARGET)
mask &= pd.to_numeric(raw_base['K'], errors='coerce') == FINAL_K
mask &= pd.to_numeric(raw_base['support_seed'], errors='coerce') == FINAL_SUPPORT_SEED
baseline = raw_base.loc[mask].copy()
baseline['fold'] = pd.to_numeric(baseline['fold'], errors='coerce')
baseline['Accuracy'] = pd.to_numeric(baseline['Accuracy'], errors='coerce')
baseline = baseline[np.isfinite(baseline['Accuracy'])].sort_values('fold')
assert len(baseline) == 5 and set(baseline.fold.astype(int)) == set(range(5)), f"Incomplete 5-fold RSITMD baseline: {baseline[['fold', 'Accuracy']].to_string(index=False)}"
assert not baseline.duplicated('fold').any()
for fold in range(5):
    assert len(test_indices(TARGET, fold)) if TEST_UNLOCKED else len(SPLITS[TARGET][fold]['test'])
    assert len(support_indices(TARGET, FINAL_K, FINAL_SUPPORT_SEED, fold)) == 21 * FINAL_K
print('Baseline mean acc:', round(float(baseline.Accuracy.mean()) * (100 if baseline.Accuracy.max() <= 1.5 else 1), 4), 'pp')
print('Baseline folds:', baseline[['fold', 'Accuracy']].to_string(index=False))

def filter_caption_bank(bank, drop_indices):
    removed = set(map(int, drop_indices))
    keep = [i for i in range(int(bank['n_img'])) if i not in removed]
    assert len(keep) > 0
    starts = []
    counts = []
    owners = []
    caps = []
    texts = []
    for new_i, old_i in enumerate(keep):
        start = int(bank['cap_start'][old_i])
        num = int(bank['cap_count'][old_i])
        assert num > 0
        selection = list(range(start, start + num))
        starts.append(len(caps))
        counts.append(num)
        caps.extend((bank['captions'][j] for j in selection))
        owners.extend([new_i] * num)
        texts.append(bank['u'][selection])
    out = {'g': bank['g'][keep], 't': bank['t'][keep], 'u': torch.cat(texts, dim=0), 'cap_start': torch.tensor(starts), 'cap_count': torch.tensor(counts), 'cap_owner': torch.tensor(owners), 'n_img': len(keep), 'captions': caps, 'source_ids': torch.tensor(keep, dtype=torch.long)}
    assert len(out['g']) == len(out['t']) == out['n_img']
    assert len(out['u']) == len(out['captions']) == len(out['cap_owner'])
    assert int(out['cap_count'].sum()) == len(out['u'])
    return out
clean_bank = filter_caption_bank(original_bank, remove_ids)
removed_manifest = {'source_corpus': CAPTION, 'target_dataset': TARGET, 'original_images': len(caption_paths), 'removed_images': len(remove_ids), 'remaining_images': clean_bank['n_img'], 'caption_indices_removed': remove_ids, 'caption_image_paths_removed': [caption_paths[x] for x in remove_ids], 'candidate_file': str(PAIR_PATH), 'pHash_radius': 4, 'policy': 'exclude all RSITMD train images having >=1 UCMerced match with 64-bit phash<=4 OR equal SHA256', 'protocol': 'same existing 5 disjoint folds, K4, support seed0, frozen CLIP, fixed optimizer'}
manifest_hash = hashlib.sha256(json.dumps(removed_manifest, sort_keys=True).encode()).hexdigest()
removed_manifest['manifest_sha256'] = manifest_hash
manifest_file = FINAL_TABLES / 'rsitmd_ucmerced_exclusion_manifest.json'
if manifest_file.exists():
    prior = json.loads(manifest_file.read_text())
    assert prior.get('manifest_sha256') == manifest_hash, 'Existing final run has a different exclusion definition; use new versioned folder'
else:
    manifest_file.write_text(json.dumps(removed_manifest, indent=2, ensure_ascii=False))
print('Exclusion manifest:', manifest_file, 'sha256:', manifest_hash[:12])
from contextlib import contextmanager

@contextmanager
def final_test_eval_context():
    global TEST_UNLOCKED
    prev = TEST_UNLOCKED
    TEST_UNLOCKED = True
    try:
        yield
    finally:
        TEST_UNLOCKED = prev

def atomic_save_dataframe(df, path):
    tmp = Path(str(path) + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
FINAL_RAW = FINAL_TABLES / 'rsitmd_ucmerced_decontaminated_k4_fold_raw.csv'
RUN_ACTUAL_TRAINING = True
run_config = {'manifest_sha256': manifest_hash, 'model': 'CAPSTONE-CLIP', 'backbone': ACTIVE_BACKBONE_KEY, 'protocol_hash': PROTOCOL_HASH, 'fold_seed': cfg.fold_seed, 'support_namespace': cfg.support_namespace, 'K': FINAL_K, 'support_seed': FINAL_SUPPORT_SEED, 'training_presets': cfg.presets['CAPSTONE-CLIP'], 'caption_epochs': cfg.caption_epochs, 'caption_steps_per_epoch': cfg.caption_steps_per_epoch, 'weight_decay': cfg.weight_decay}
run_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
run_config['run_config_sha256'] = run_hash
run_config_path = FINAL_TABLES / 'rsitmd_ucmerced_run_config.json'
if run_config_path.exists():
    assert json.loads(run_config_path.read_text()).get('run_config_sha256') == run_hash, 'Run configuration changed: preserve old run, use a versioned folder'
else:
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
print('Training: ', RUN_ACTUAL_TRAINING, ' | output:', FINAL_RAW)
if RUN_ACTUAL_TRAINING:
    bank_key = (ACTIVE_BACKBONE_KEY, CAPTION)
    assert CAPTION_BANKS[bank_key]['train'] is original_bank, 'Original caption bank unexpectedly changed'
    all_rows = pd.read_csv(FINAL_RAW) if FINAL_RAW.is_file() else pd.DataFrame()
    for fold in range(cfg.n_outer_folds):
        completed = False
        if not all_rows.empty and {'fold', 'Accuracy', 'run_config_sha256'}.issubset(all_rows.columns):
            hit = all_rows[(pd.to_numeric(all_rows.fold, errors='coerce') == fold) & (all_rows.run_config_sha256.astype(str) == run_hash)]
            completed = bool(len(hit) == 1 and np.isfinite(pd.to_numeric(hit.Accuracy, errors='coerce').iloc[0]))
        if completed:
            print(f'Fold {fold}: restored from completed CSV row')
            continue
        assert not TEST_UNLOCKED
        print(f'Running fold {fold}/4, K={FINAL_K}, seed={FINAL_SUPPORT_SEED}')
        t0 = time.perf_counter()
        try:
            CAPTION_BANKS[bank_key]['train'] = clean_bank
            net, info = train_capstone(TARGET, FINAL_K, FINAL_SUPPORT_SEED, fold, CAPTION, 'joint_full', 1.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            with final_test_eval_context():
                met = evaluate_capstone(net, TARGET, test_indices(TARGET, fold))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            evaluation_s = time.perf_counter() - t1
            assert np.isfinite(float(met['Accuracy']))
            row = {'dataset': TARGET, 'caption_corpus': CAPTION, 'K': FINAL_K, 'fold': fold, 'support_seed': FINAL_SUPPORT_SEED, 'condition': 'rsitmd_decontaminated_phash4', 'removed_caption_images': len(remove_ids), 'remaining_caption_images': clean_bank['n_img'], 'training_seconds': training_s, 'evaluation_seconds': evaluation_s, 'run_config_sha256': run_hash, 'manifest_sha256': manifest_hash, **met, **info}
            old = pd.read_csv(FINAL_RAW) if FINAL_RAW.is_file() else pd.DataFrame()
            if not old.empty and 'fold' in old.columns:
                old = old[pd.to_numeric(old.fold, errors='coerce') != fold]
            new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            atomic_save_dataframe(new, FINAL_RAW)
            all_rows = new
            print('Saved fold', fold, 'accuracy:', met['Accuracy'], 'training_s:', round(training_s, 1))
        finally:
            CAPTION_BANKS[bank_key]['train'] = original_bank
            if 'net' in locals():
                del net
            free_gpu()
    print('All folds available:', FINAL_RAW)
else:
    print('Dry run: no models have been trained; only the exclusion manifest was created')
if FINAL_RAW.is_file():
    clean = pd.read_csv(FINAL_RAW)
    clean = clean[(pd.to_numeric(clean.K, errors='coerce') == FINAL_K) & (pd.to_numeric(clean.support_seed, errors='coerce') == FINAL_SUPPORT_SEED)]
    assert len(clean) == 5 and set(clean.fold.astype(int)) == set(range(5))
    assert clean.run_config_sha256.astype(str).eq(run_hash).all()
    base = baseline[['fold', 'Accuracy']].rename(columns={'Accuracy': 'original_accuracy'}).copy()
    decon = clean[['fold', 'Accuracy']].rename(columns={'Accuracy': 'decontaminated_accuracy'}).copy()
    comp = base.merge(decon, on='fold', validate='one_to_one')
    scale = 100 if max(comp.original_accuracy.max(), comp.decontaminated_accuracy.max()) <= 1.5 else 1
    comp['original_acc_pp'] = comp.original_accuracy * scale
    comp['decontaminated_acc_pp'] = comp.decontaminated_accuracy * scale
    comp['delta_pp'] = comp.decontaminated_acc_pp - comp.original_acc_pp
    cmp_file = FINAL_TABLES / 'rsitmd_ucmerced_decontamination_fold_comparison.csv'
    comp.to_csv(cmp_file, index=False)
    summ = {'n_folds': 5, 'K': 4, 'seed': 0, 'removed_caption_images': len(remove_ids), 'original_accuracy_pp': float(comp.original_acc_pp.mean()), 'decontaminated_accuracy_pp': float(comp.decontaminated_acc_pp.mean()), 'delta_accuracy_pp': float(comp.delta_pp.mean()), 'min_fold_delta_pp': float(comp.delta_pp.min()), 'max_fold_delta_pp': float(comp.delta_pp.max()), 'interpretation': 'secondary within-dataset robustness check; not independent dataset-level significance'}
    (FINAL_TABLES / 'rsitmd_ucmerced_decontamination_summary.json').write_text(json.dumps(summ, indent=2))
    display(comp[['fold', 'original_acc_pp', 'decontaminated_acc_pp', 'delta_pp']])
    print('Average Δ accuracy:', round(summ['delta_accuracy_pp'], 4), 'pp')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    x = comp.fold.astype(int).values
    plt.plot(x, comp.original_acc_pp.values, 'o-', label='Original RSITMD')
    plt.plot(x, comp.decontaminated_acc_pp.values, 's--', label='Decontaminated RSITMD')
    plt.xticks(x)
    plt.xlabel('Held-out UCMerced fold')
    plt.ylabel('Accuracy (%)')
    plt.title('RSITMD→UCMerced: matched 5-fold sensitivity (K=4, seed=0)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FINAL_FIGS / 'rsitmd_ucmerced_fold_accuracy.png', dpi=200)
    plt.show()
else:
    print('No decontamination model results yet. Execute Section 4 in Colab to produce fold comparison.')
PRIMARY_CANDIDATES = [Path(cfg.drive_root) / 'outputs_q1_capstone_clip_v11_full_external' / 'tables' / 'raw_primary_classification.csv', Path(cfg.drive_root) / 'outputs_capstone_paper_postprocess_v1' / 'tables' / 'raw_primary_classification.csv']
PRIMARY_CANDIDATES.extend(Path(cfg.drive_root).glob('outputs_*/tables/raw_primary_classification.csv'))
primary_file = next((p for p in PRIMARY_CANDIDATES if p.is_file()), None)
print('Source primary results:', primary_file)
if primary_file is None:
    print('WARNING: full primary results unavailable; no Friedman numbers will be inferred.')
else:
    primary = pd.read_csv(primary_file)
    print('Columns:', list(primary.columns), ' | rows:', len(primary))
    display(primary.head(3))
if primary_file is not None:

    def column(df, names):
        return next((col for col in names if col in df.columns), None)
    dscol = column(primary, ['dataset', 'Dataset', 'data'])
    shotcol = column(primary, ['K', 'shot', 'shots', 'Shots'])
    methodcol = column(primary, ['method', 'Method', 'model'])
    acccol = column(primary, ['Accuracy', 'accuracy', 'accuracy_mean', 'Acc'])
    need = {'dataset': dscol, 'shots': shotcol, 'method': methodcol, 'accuracy': acccol}
    assert all(need.values()), f'Cannot identify primary columns: {need}'
    data = primary.rename(columns={dscol: 'dataset', shotcol: 'K', methodcol: 'method', acccol: 'Accuracy'}).copy()
    data['K'] = pd.to_numeric(data.K, errors='coerce')
    data['Accuracy'] = pd.to_numeric(data.Accuracy, errors='coerce')
    data = data[data['dataset'].isin(['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT']) & data.K.isin([1, 2, 4, 8, 16])]
    data = data[np.isfinite(data.Accuracy)].copy()
    assert len(data) > 0, 'Primary results empty after strict filters'
    means = data.groupby(['dataset', 'K', 'method'], as_index=False).Accuracy.mean()
    bydataset = means.groupby(['dataset', 'method'], as_index=False).Accuracy.mean()
    pvt = bydataset.pivot(index='dataset', columns='method', values='Accuracy')
    valid = pvt.columns[pvt.notna().all()].tolist()
    pvt = pvt.loc[:, valid]
    counts = means.groupby(['dataset', 'method']).K.nunique().unstack('method')
    valid = [m for m in valid if counts[m].eq(5).all()]
    pvt = pvt.loc[:, valid]
    assert pvt.shape[0] == 6 and len(valid) >= 3, f'Insufficient complete six-dataset methods: {pvt.shape}'
    assert np.isfinite(pvt.to_numpy()).all()
    from scipy.stats import friedmanchisquare, rankdata
    score, pvalue = friedmanchisquare(*(pvt[m].to_numpy() for m in valid))
    ranks = pvt.rank(axis=1, ascending=False, method='average')
    ranks.mean().sort_values().rename('mean_rank').to_csv(FINAL_TABLES / 'dataset_block_friedman_mean_ranks.csv')
    pvt.to_csv(FINAL_TABLES / 'dataset_block_friedman_accuracy.csv')
    out = {'n_independent_blocks': 6, 'n_methods': len(valid), 'n_shots_per_dataset': 5, 'statistic': float(score), 'p_value': float(pvalue), 'methods': valid, 'interpretation': 'exploratory only; dataset-level clustering honored; small n; no confirmatory independence claim'}
    (FINAL_TABLES / 'dataset_block_friedman_summary.json').write_text(json.dumps(out, indent=2))
    print('Dataset-block Friedman n=6; methods:', len(valid), 'chi2=', round(float(score), 4), 'p=', round(float(pvalue), 6))
    display(ranks.mean().sort_values().head(15))
ADAPTER_STATS = BASE_REV / 'adapter_width_dataset_level_stats.csv'
ADAPTER_SUM = BASE_REV / 'adapter_width_sweep_summary.csv'
if ADAPTER_STATS.is_file() and ADAPTER_SUM.is_file():
    width_stats = pd.read_csv(ADAPTER_STATS)
    width_summary = pd.read_csv(ADAPTER_SUM)
    display(width_stats)
    display(width_summary)
    width_stats.to_csv(FINAL_TABLES / 'reviewer2_adapter_capacity_statistics_reference.csv', index=False)
    width_summary.to_csv(FINAL_TABLES / 'reviewer2_adapter_capacity_sweep_reference.csv', index=False)
    (FINAL_TABLES / 'adapter_capacity_interpretation.txt').write_text('Absence of significant improvement is not equivalence. Conclusions are restricted to the tested adapter family, widths (32-512), backbones, datasets, and support protocol.\n')
else:
    print('Adapter sweep summary files unavailable; do not report estimated values.')
budget_rows = []
for method in cfg.primary_methods:
    direct_epochs = {'CAPSTONE-CLIP': cfg.caption_epochs, 'Shared-Caption': cfg.caption_epochs, 'Capacity-Matched-Shared': cfg.caption_epochs, 'ProLIP': cfg.prolip_epochs, 'RLA': cfg.rla_epochs, 'HOSO-Adapter': cfg.hoso_epochs, 'Tip-Adapter-F': cfg.tipf_epochs, 'CoOp': cfg.coop_epochs, 'TaskRes': cfg.taskres_epochs}
    budget_rows.append({'method': method, 'configured_epochs': direct_epochs.get(method, np.nan), 'caption_steps_per_epoch': cfg.caption_steps_per_epoch if method in ('CAPSTONE-CLIP', 'Shared-Caption', 'Capacity-Matched-Shared') else np.nan, 'training_budget_identical': 'No', 'wall_time_source': 'refer to original Table 12 and logged hardware conditions', 'limitation': 'Not a general hardware-matched efficiency comparison'})
budget = pd.DataFrame(budget_rows)
budget.to_csv(FINAL_TABLES / 'reviewer2_runtime_budget_inventory.csv', index=False)
display(budget)
report = []
report.append('# IEEE Access CAPSTONE — Reviewer 2 final-revision evidence report')
report.append('Generated from CAPSTONECLIP_final_29_ekim.ipynb; only executed rows are evidence.')
report.append('## 1 — Friedman dependence')
fp = FINAL_TABLES / 'dataset_block_friedman_summary.json'
if fp.is_file():
    d = json.loads(fp.read_text())
    report.append(f"Dataset-block exploratory Friedman: n={d['n_independent_blocks']}, methods={d['n_methods']}, p={d['p_value']:.6g}.")
else:
    report.append('INCOMPLETE: 6-block Friedman sensitivity not yet computed.')
report.append('## 2 — Runtime fairness')
report.append('Reported wall-clock times describe the implemented, non-identical budgets and do not establish generally superior computational efficiency.')
report.append('## 3 — Adapter capacity')
if ADAPTER_STATS.is_file():
    d = pd.read_csv(ADAPTER_STATS)
    report.append(d.to_string(index=False))
else:
    report.append('INCOMPLETE: no adapter-width summary source.')
report.append('The absence of a significant gain is not evidence of equivalence or a definitive saturation point.')
report.append('## 4 — RSITMD–UCMerced cross-corpus overlap')
report.append(f'Full SHA/pHash audit: {len(caption_paths)} RSITMD train images × {len(target_paths)} UCMerced images; candidates={len(pairs_df)}; removed source images={len(remove_ids)} (pHash radius <=4 or exact SHA).')
sf = FINAL_TABLES / 'rsitmd_ucmerced_decontamination_summary.json'
if sf.is_file():
    d = json.loads(sf.read_text())
    report.append(f"Original mean accuracy={d['original_accuracy_pp']:.4f}%; decontaminated mean accuracy={d['decontaminated_accuracy_pp']:.4f}%; matched five-fold delta={d['delta_accuracy_pp']:+.4f} pp.")
else:
    report.append('INCOMPLETE: matched five-fold clean retraining has not finished.')
report.append('Caution: This is a secondary robustness experiment; remaining unknown overlap and dataset-specific representativeness limit generalization.')
report.append('## Suggested English manuscript sentences (verify placement)')
report.append('**Friedman:** "Because the five shot levels from the same dataset are not independent, the original 30-condition Friedman analysis is treated as exploratory. A complementary dataset-block sensitivity analysis aggregates within dataset and uses only six dataset-level blocks; it is not interpreted as independent confirmatory evidence."')
report.append('**Runtime:** "The reported training and evaluation times characterize our implemented configurations under non-identical training budgets; they should not be interpreted as a general computational efficiency ranking."')
report.append('**Capacity:** "Within the tested adapter family and bottleneck widths (32–512), widening beyond 128 did not show a consistent accuracy improvement; the analysis does not establish statistical equivalence or a universal saturation point."')
report.append('**Overlap:** "A full perceptual-hash audit and a matched five-fold RSITMD-caption decontamination sensitivity analysis were performed. The resulting differences are reported descriptively, and remaining overlap uncertainty limits inference from this secondary robustness check."')
report.append('## Publication checklist')
report.append('- Confirm code/data alignment and inspect saved contact sheet for false matches.')
report.append('- Confirm five fold CSV results and final summary exist. Verify no failure row was silently considered complete.')
report.append('- Verify dataset-block Friedman uses exactly six independent dataset blocks and all five shots per method.')
report.append('- Keep original methods, author list, and references unchanged after acceptance.')
report.append('- Align main paper, supplementary, response, figures and LaTeX compilation before final IEEE submission.')
output = FINAL_ROOT / 'IEEE_Access_final_revision_evidence.md'
output.write_text('\n\n'.join(report), encoding='utf-8')
print('Report:', output)
print('Ready for final-file submission:', bool(sf.is_file() and fp.is_file()))
