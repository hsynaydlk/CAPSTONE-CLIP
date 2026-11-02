import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
RUN_PROFILE = 'paper_full'

@dataclass
class CFG:
    run_tag: str = 'q1_capstone_clip_v10'
    protocol_version: str = 'capstone_clip_disjoint5fold_validation_free_v10'
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
EXTERNAL_BASELINES = {'AVION': {'paper': 'CVPR 2026', 'code_url': None, 'status': 'official-code hook; do not reverse-engineer'}, 'MaPLe': {'paper': 'CVPR 2023', 'code_url': 'https://github.com/muzairkhattak/multimodal-prompt-learning', 'status': 'external official run'}, 'CLIP-LoRA': {'paper': 'parameter-efficient CLIP adaptation', 'code_url': None, 'status': 'external official run required'}}
METHOD_AUDIT = []
for m in cfg.primary_methods:
    ext = m in ('Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP')
    METHOD_AUDIT.append({'method': m, 'included': True, 'uses_external_caption_data': ext, 'target_validation': False, 'notes': 'external budget exported' if ext else 'support-only/frozen baseline'})
for m, v in EXTERNAL_BASELINES.items():
    METHOD_AUDIT.append({'method': m, 'included': False, 'uses_external_caption_data': m == 'AVION', 'target_validation': 'official protocol', 'notes': v['status']})
print(pd.DataFrame(METHOD_AUDIT).to_string(index=False))

def run_classification_method(method, name, K, seed, test_idx, outer_id, caption_dataset=None):
    set_outer_split(outer_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    info = {'available': True}
    if method == 'Zero-shot':
        met = evaluate_image_encoder(name, test_idx, lambda x: F.normalize(x, dim=-1), frozen_class_features(name))
        model = None
    elif method == 'LinearProbe':
        model = fit_linear_probe(name, K, seed, outer_id)
        met = eval_linear_probe(model, name, test_idx)
        info = {'selection': 'fixed_C1'}
    elif method == 'CLIP-Adapter':
        model, info = train_clip_adapter(name, K, seed, outer_id)
        met = evaluate_image_encoder(name, test_idx, model, frozen_class_features(name))
    elif method == 'Tip-Adapter':
        keys, labels, hp = build_tip(name, K, seed, outer_id)
        model = None
        met = eval_tip(name, test_idx, keys, labels, hp)
    elif method == 'Tip-Adapter-F':
        keys, labels, info = train_tipf(name, K, seed, outer_id)
        model = None
        met = eval_tip(name, test_idx, keys, labels, cfg.presets['Tip-Adapter-F'])
    elif method == 'CoOp':
        model, info = train_coop(name, K, seed, outer_id)
        met = eval_coop(model, name, test_idx)
    elif method == 'TaskRes':
        model, info = train_taskres(name, K, seed, outer_id)
        met = evaluate_image_encoder(name, test_idx, lambda x: F.normalize(x, dim=-1), model.class_features())
    elif method == 'RLA':
        model, info = train_rla(name, K, seed, outer_id)
        met = evaluate_image_encoder(name, test_idx, model, frozen_class_features(name))
    elif method == 'ProLIP':
        model, info = train_prolip(name, K, seed, outer_id)
        if model is None:
            return (None, info)
        met = eval_prolip(model, name, test_idx)
    elif method == 'HOSO-Adapter':
        model, info = train_hoso(name, K, seed, outer_id)
        if model is None:
            return (None, info)
        met = evaluate_image_encoder(name, test_idx, model, frozen_class_features(name))
    elif method == 'Shared-Caption':
        model, info = train_shared_caption(name, K, seed, outer_id, caption_dataset)
        met = evaluate_shared(model, name, test_idx)
    elif method == 'Capacity-Matched-Shared':
        model, info = train_capacity_matched_shared(name, K, seed, outer_id, caption_dataset)
        met = evaluate_shared(model, name, test_idx)
    elif method == 'CAPSTONE-CLIP':
        model, info = train_capstone(name, K, seed, outer_id, caption_dataset, 'joint_full', 1.0)
        met = evaluate_capstone(model, name, test_idx)
    else:
        raise KeyError(method)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
    info = {**info, 'train_eval_s': elapsed, 'peak_gpu_delta_mb': peak, 'trainable_params': n_trainable(model) if isinstance(model, nn.Module) else None}
    free_gpu()
    return (met, info)
TEST_UNLOCKED = True
if cfg.run_primary:
    for name in cfg.datasets:
        for K in cfg.shots:
            for oid in range(cfg.n_outer_folds):
                ti = test_indices(name, oid)
                for seed in cfg.support_seeds:
                    for method in cfg.primary_methods:
                        key = f'primary/{ACTIVE_BACKBONE_KEY}/{method}/{name}/K{K}/f{oid}/s{seed}'
                        if is_done(key):
                            row = state_data(key)
                            if row:
                                RESULTS['classification_runs'].append(row)
                            continue
                        try:
                            met, info = run_classification_method(method, name, K, seed, ti, oid, cfg.caption_dataset)
                            row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'method': method, **(met or {'Accuracy': np.nan, 'MacroF1': np.nan, 'BalancedAccuracy': np.nan, 'NLL': np.nan, 'ECE': np.nan, 'N': 0}), **info}
                        except Exception as e:
                            row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'method': method, 'Accuracy': np.nan, 'MacroF1': np.nan, 'BalancedAccuracy': np.nan, 'NLL': np.nan, 'ECE': np.nan, 'N': 0, 'available': False, 'reason': f'{type(e).__name__}: {e}'}
                            print('[FAILED]', key, row['reason'])
                        RESULTS['classification_runs'].append(row)
                        mark_done(key, row)
print('Primary rows:', len(RESULTS['classification_runs']))
if cfg.run_caption_controls:
    corpora = [d for d in cfg.caption_datasets if (ACTIVE_BACKBONE_KEY, d) in CAPTION_BANKS]
    corpus_options = corpora + (['RSICD+RSITMD'] if len(corpora) >= 2 else [])
    for name in [d for d in cfg.caption_control_datasets if d in cfg.datasets]:
        for K in cfg.caption_control_shots:
            for oid in range(cfg.n_outer_folds):
                ti = test_indices(name, oid)
                for seed in cfg.support_seeds:
                    for control in cfg.caption_controls:
                        scales = cfg.caption_scales if control == 'joint_full' else [1.0]
                        for scale in scales:
                            for corpus in corpus_options if control == 'joint_full' and scale == 1.0 else [cfg.caption_dataset]:
                                key = f'caption_control/{name}/K{K}/f{oid}/s{seed}/{control}/{scale}/{corpus}'
                                if is_done(key):
                                    row = state_data(key)
                                    if row:
                                        RESULTS['caption_control_runs'].append(row)
                                    continue
                                original = None
                                try:
                                    if corpus == 'RSICD+RSITMD':
                                        original = CAPTION_BANKS[ACTIVE_BACKBONE_KEY, cfg.caption_dataset]['train']
                                        CAPTION_BANKS[ACTIVE_BACKBONE_KEY, '__MERGED__'] = {'train': merge_caption_banks([cap_bank('RSICD', 'train'), cap_bank('RSITMD', 'train')])}
                                        cap_name = '__MERGED__'
                                    else:
                                        cap_name = corpus
                                    m, info = train_capstone(name, K, seed, oid, cap_name, control, scale)
                                    met = evaluate_capstone(m, name, ti)
                                    row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'control': control, 'caption_scale': scale, 'caption_corpus': corpus, **met, **info}
                                except Exception as e:
                                    row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'control': control, 'caption_scale': scale, 'caption_corpus': corpus, 'Accuracy': np.nan, 'available': False, 'reason': f'{type(e).__name__}: {e}'}
                                RESULTS['caption_control_runs'].append(row)
                                mark_done(key, row)
                                free_gpu()
print('Caption-control rows:', len(RESULTS['caption_control_runs']))
ABLATION_VARIANTS = {'full': {}, 'no_decor': {'lambda_decor': 0.0}, 'no_consistency': {'lambda_cons': 0.0}, 'no_shared_aux_cls': {'lambda_shared_cls': 0.0}}
if cfg.run_architecture_ablations:
    for name in [d for d in cfg.ablation_datasets if d in cfg.datasets]:
        for K in [k for k in cfg.ablation_shots if k in cfg.shots]:
            for oid in range(cfg.n_outer_folds):
                ti = test_indices(name, oid)
                for seed in cfg.support_seeds:
                    for variant, overrides in ABLATION_VARIANTS.items():
                        key = f'ablation/{ACTIVE_BACKBONE_KEY}/{name}/K{K}/f{oid}/s{seed}/{variant}'
                        if is_done(key):
                            row = state_data(key)
                            if row:
                                RESULTS['ablation_runs'].append(row)
                            continue
                        try:
                            model, info = train_capstone(name, K, seed, oid, cfg.caption_dataset, 'joint_full', 1.0, hp_overrides=overrides)
                            met = evaluate_capstone(model, name, ti)
                            row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'variant': variant, **met, **info}
                        except Exception as e:
                            row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'variant': variant, 'Accuracy': np.nan, 'available': False, 'reason': f'{type(e).__name__}: {e}'}
                        RESULTS['ablation_runs'].append(row)
                        mark_done(key, row)
                        free_gpu()
print('Architecture-ablation rows:', len(RESULTS['ablation_runs']))

@torch.no_grad()
def retrieval_metrics(image_z, text_z, owner, n_img):
    sim = (F.normalize(image_z, dim=-1) @ F.normalize(text_z, dim=-1).t()).cpu()
    owner = owner.cpu()
    ranked_owner = owner[sim.argsort(1, descending=True)]
    gt = torch.arange(n_img).unsqueeze(1)
    first = (ranked_owner == gt).float().argmax(1)
    i2t = [float((first < k).float().mean()) for k in (1, 5, 10)]
    r2 = sim.t().argsort(1, descending=True)
    pos = (r2 == owner.unsqueeze(1)).float().argmax(1)
    t2i = [float((pos < k).float().mean()) for k in (1, 5, 10)]
    return {'i2t_R1': i2t[0], 'i2t_R5': i2t[1], 'i2t_R10': i2t[2], 't2i_R1': t2i[0], 't2i_R5': t2i[1], 't2i_R10': t2i[2], 'mR': float(np.mean(i2t + t2i)), 'N_images': n_img}

@torch.no_grad()
def evaluate_retrieval(model, bank, kind, max_images=None):
    n = bank['n_img'] if max_images is None else min(bank['n_img'], int(max_images))
    keep = int(bank['cap_start'][n - 1] + bank['cap_count'][n - 1])
    owner = bank['cap_owner'][:keep]
    g = bank['g'][:n].float().to(DEVICE)
    u = bank['u'][:keep].float().to(DEVICE)
    if kind == 'frozen':
        iz, uz = (F.normalize(g, dim=-1), F.normalize(u, dim=-1))
    elif kind == 'shared':
        iz, uz = (model.encode_image(g), model.encode_text(u))
    elif kind == 'capstone':
        iz, uz = (model.encode_ret_image(g), model.encode_ret_text(u))
    else:
        raise ValueError(kind)
    return retrieval_metrics(iz, uz, owner, n)
if cfg.run_retrieval:
    anchor = [d for d in cfg.datasets if d in ('NWPU_RESISC45', 'AID')][:1] or cfg.datasets[:1]
    name = anchor[0]
    K = 4 if 4 in cfg.shots else cfg.shots[-1]
    for ds in [d for d in cfg.caption_datasets if (ACTIVE_BACKBONE_KEY, d) in CAPTION_BANKS]:
        for oid in range(cfg.n_outer_folds):
            for seed in cfg.support_seeds:
                for method in ('Frozen-CLIP', 'Shared-Caption', 'CAPSTONE-CLIP'):
                    key = f'retrieval/{ds}/{method}/f{oid}/s{seed}'
                    if is_done(key):
                        row = state_data(key)
                        if row:
                            RESULTS['retrieval_runs'].append(row)
                        continue
                    if method == 'Frozen-CLIP':
                        model = None
                        rm = evaluate_retrieval(None, cap_bank(ds, 'test'), 'frozen', cfg.retrieval_max_images)
                        cm = evaluate_image_encoder(name, test_indices(name, oid), lambda x: F.normalize(x, dim=-1), frozen_class_features(name))
                    elif method == 'Shared-Caption':
                        model, _ = train_shared_caption(name, K, seed, oid, ds)
                        rm = evaluate_retrieval(model, cap_bank(ds, 'test'), 'shared', cfg.retrieval_max_images)
                        cm = evaluate_shared(model, name, test_indices(name, oid))
                    else:
                        model, _ = train_capstone(name, K, seed, oid, ds, 'joint_full', 1.0)
                        rm = evaluate_retrieval(model, cap_bank(ds, 'test'), 'capstone', cfg.retrieval_max_images)
                        cm = evaluate_capstone(model, name, test_indices(name, oid))
                    row = {'backbone': ACTIVE_BACKBONE_KEY, 'caption_dataset': ds, 'classification_dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'method': method, 'classification_Accuracy': cm['Accuracy'], **rm}
                    RESULTS['retrieval_runs'].append(row)
                    mark_done(key, row)
                    free_gpu()
print('Retrieval rows:', len(RESULTS['retrieval_runs']))

@torch.no_grad()
def capstone_logits(model, name, idx):
    fc = get_dataset_cache(name)
    ids = torch.as_tensor(idx, dtype=torch.long)
    cf = class_features_for(model, name, 'capstone')
    outs = []
    ys = []
    for s in range(0, len(ids), cfg.eval_chunk):
        ii = ids[s:s + cfg.eval_chunk]
        g = fc['g'][ii].float().to(DEVICE)
        outs.append(model.logit_scale.exp() * model.encode_class_image(g) @ cf.t())
        ys.append(fc['y'][ii])
    return (torch.cat(outs), torch.cat(ys))

def save_class_diagnostics(name, model, idx, K, fold=0, seed=0):
    logits, y = capstone_logits(model, name, idx)
    probs = logits.softmax(-1)
    conf, pred = probs.max(1)
    yc = y.cpu().numpy()
    pc = pred.cpu().numpy()
    names = get_dataset_cache(name)['class_names']
    cm = confusion_matrix(yc, pc, labels=np.arange(len(names)), normalize='true')
    per = pd.DataFrame({'class_id': np.arange(len(names)), 'class_name': names, 'recall': np.diag(cm)})
    per.to_csv(os.path.join(TAB_DIR, f'per_class_{name}_K{K}_f{fold}_s{seed}.csv'), index=False)
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.22), max(6, len(names) * 0.22)))
    im = ax.imshow(cm, aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'CAPSTONE-CLIP normalized confusion: {name}, K={K}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'confusion_{name}_K{K}.png'), dpi=220)
    plt.close(fig)
    edges = np.linspace(0, 1, cfg.ece_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf.cpu().numpy() > lo) & (conf.cpu().numpy() <= hi)
        if m.any():
            rows.append({'lo': lo, 'hi': hi, 'mean_confidence': float(conf.cpu().numpy()[m].mean()), 'accuracy': float((pc[m] == yc[m]).mean()), 'n': int(m.sum())})
    rel = pd.DataFrame(rows)
    rel.to_csv(os.path.join(TAB_DIR, f'reliability_{name}_K{K}.csv'), index=False)
    if len(rel):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.plot(rel.mean_confidence, rel.accuracy, marker='o')
        ax.set(xlabel='Confidence', ylabel='Accuracy', title=f'Reliability: {name}')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'reliability_{name}_K{K}.png'), dpi=220)
        plt.close(fig)
    err = np.where(pc != yc)[0]
    order = err[np.argsort(-conf.cpu().numpy()[err])[:100]] if len(err) else []
    paths = np.asarray(get_dataset_cache(name)['paths'])[np.asarray(idx)]
    errors = pd.DataFrame([{'path': paths[j], 'true_id': int(yc[j]), 'true_class': names[int(yc[j])], 'pred_id': int(pc[j]), 'pred_class': names[int(pc[j])], 'confidence': float(conf[j])} for j in order])
    errors.to_csv(os.path.join(TAB_DIR, f'high_confidence_errors_{name}_K{K}.csv'), index=False)
    return {'dataset': name, 'K': K, 'fold': fold, 'support_seed': seed, 'mean_per_class_recall': float(np.diag(cm).mean()), 'n_errors': int((pc != yc).sum()), 'n_test': int(len(yc))}
if cfg.run_class_diagnostics:
    K = cfg.diagnostic_shot if cfg.diagnostic_shot in cfg.shots else cfg.shots[-1]
    for name in cfg.datasets:
        key = f'class_diagnostics/{ACTIVE_BACKBONE_KEY}/{name}/K{K}/f0/s0'
        if is_done(key):
            row = state_data(key)
            if row:
                RESULTS['class_diagnostics'].append(row)
            continue
        m, _ = train_capstone(name, K, 0, 0, cfg.caption_dataset, 'joint_full', 1.0)
        row = save_class_diagnostics(name, m, test_indices(name, 0), K, 0, 0)
        RESULTS['class_diagnostics'].append(row)
        mark_done(key, row)
        free_gpu()
print('Class-diagnostic rows:', len(RESULTS['class_diagnostics']))

def base_new_classes(name):
    C = len(get_dataset_cache(name)['class_names'])
    order = np.arange(C)
    rng = np.random.RandomState(stable_int('base_new', name))
    rng.shuffle(order)
    cut = (C + 1) // 2
    return (sorted(order[:cut].tolist()), sorted(order[cut:].tolist()))

def subset_test_indices(name, oid, classes):
    idx = test_indices(name, oid)
    y = get_dataset_cache(name)['y'].numpy()
    return idx[np.isin(y[idx], classes)]
if cfg.run_base_to_new:
    for name in [d for d in cfg.base_to_new_datasets if d in cfg.datasets]:
        base, new = base_new_classes(name)
        for K in cfg.base_to_new_shots:
            for oid in range(cfg.n_outer_folds):
                for seed in cfg.support_seeds:
                    key = f'base_to_new/{name}/K{K}/f{oid}/s{seed}'
                    if is_done(key):
                        row = state_data(key)
                        if row:
                            RESULTS['base_to_new_runs'].append(row)
                        continue
                    m, info = train_capstone(name, K, seed, oid, cfg.caption_dataset, 'joint_full', 1.0, class_subset=base)
                    mb = evaluate_capstone(m, name, subset_test_indices(name, oid, base), base)
                    mn = evaluate_capstone(m, name, subset_test_indices(name, oid, new), new)
                    hm = 2 * mb['Accuracy'] * mn['Accuracy'] / max(1e-12, mb['Accuracy'] + mn['Accuracy'])
                    row = {'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'BaseAccuracy': mb['Accuracy'], 'NovelAccuracy': mn['Accuracy'], 'HarmonicMean': hm, 'n_base': len(base), 'n_novel': len(new), **info}
                    RESULTS['base_to_new_runs'].append(row)
                    mark_done(key, row)
                    free_gpu()
if cfg.run_cross_dataset and cfg.cross_dataset_source in cfg.datasets:
    src = cfg.cross_dataset_source
    K = cfg.cross_dataset_shot
    for oid in range(cfg.n_outer_folds):
        for seed in cfg.support_seeds:
            m, info = train_capstone(src, K, seed, oid, cfg.caption_dataset, 'joint_full', 1.0)
            for target in [d for d in cfg.datasets if d != src]:
                key = f'cross_dataset/{src}/{target}/f{oid}/s{seed}'
                if is_done(key):
                    row = state_data(key)
                    if row:
                        RESULTS['cross_dataset_runs'].append(row)
                    continue
                met = evaluate_capstone(m, target, test_indices(target, oid))
                row = {'backbone': ACTIVE_BACKBONE_KEY, 'source': src, 'target': target, 'K': K, 'fold': oid, 'support_seed': seed, **met, **info}
                RESULTS['cross_dataset_runs'].append(row)
                mark_done(key, row)
            free_gpu()
print('Base-to-new rows:', len(RESULTS['base_to_new_runs']), '| cross-dataset rows:', len(RESULTS['cross_dataset_runs']))

def reset_active_caches():
    global FEATS, CLASS_U, CAPTION_BANKS, SPLITS, AVAILABLE
    FEATS = {}
    CLASS_U = {}
    CAPTION_BANKS = {}
    free_gpu()
if cfg.run_cross_backbone:
    original = cfg.primary_backbone
    for bk in cfg.cross_backbone_keys:
        if bk == original:
            pass
        else:
            activate_backbone(bk)
            reset_active_caches()
            AVAILABLE = list(cfg.datasets)
            for n in AVAILABLE:
                get_dataset_cache(n)
            CLASS_U = {n: raw_prompt_embeddings(get_dataset_cache(n)['prompts']).to(DEVICE) for n in AVAILABLE}
            SPLITS = {n: build_disjoint_folds(get_dataset_cache(n)['y'].numpy()) for n in AVAILABLE}
            load_caption_banks()
        for name in cfg.datasets:
            for K in cfg.cross_backbone_shots:
                for oid in range(cfg.n_outer_folds):
                    for seed in cfg.support_seeds:
                        for method in cfg.cross_backbone_methods:
                            key = f'cross_backbone/{bk}/{method}/{name}/K{K}/f{oid}/s{seed}'
                            if is_done(key):
                                row = state_data(key)
                                if row:
                                    RESULTS['cross_backbone_runs'].append(row)
                                continue
                            try:
                                met, info = run_classification_method(method, name, K, seed, test_indices(name, oid), oid, cfg.caption_dataset)
                            except Exception as e:
                                met = None
                                info = {'available': False, 'reason': f'{type(e).__name__}: {e}'}
                            row = {'backbone': bk, 'dataset': name, 'K': K, 'fold': oid, 'support_seed': seed, 'method': method, **(met or {'Accuracy': np.nan, 'MacroF1': np.nan, 'BalancedAccuracy': np.nan, 'NLL': np.nan, 'ECE': np.nan}), **info, 'pretraining_overlap_risk': 'unknown_or_possible' if bk in ('remoteclip_b32', 'georsclip_b32') else 'foundation_model_unknown'}
                            RESULTS['cross_backbone_runs'].append(row)
                            mark_done(key, row)
    if ACTIVE_BACKBONE_KEY != original:
        activate_backbone(original)
        reset_active_caches()
        AVAILABLE = list(cfg.datasets)
        for n in AVAILABLE:
            get_dataset_cache(n)
        CLASS_U = {n: raw_prompt_embeddings(get_dataset_cache(n)['prompts']).to(DEVICE) for n in AVAILABLE}
        SPLITS = {n: build_disjoint_folds(get_dataset_cache(n)['y'].numpy()) for n in AVAILABLE}
        load_caption_banks()
print('Cross-backbone rows:', len(RESULTS['cross_backbone_runs']))

@torch.no_grad()
def near_duplicate_feature_audit(name, outer_id, thresholds=(0.99, 0.995, 0.999), max_test=500, train_block=4096, top_pairs=20):
    fc = get_dataset_cache(name)
    train_idx = train_pool_indices(name, outer_id)
    test_idx = test_indices(name, outer_id)
    rng = np.random.RandomState(stable_int('near_duplicate_audit', name, outer_id, ACTIVE_BACKBONE_KEY))
    sampled = np.asarray(test_idx, dtype=np.int64)
    if len(sampled) > max_test:
        sampled = np.sort(rng.choice(sampled, max_test, replace=False))
    q = F.normalize(fc['g'][torch.as_tensor(sampled)].float().to(DEVICE), dim=-1)
    best = torch.full((len(sampled),), -1.0, device=DEVICE)
    best_train = torch.full((len(sampled),), -1, dtype=torch.long, device=DEVICE)
    train_arr = np.asarray(train_idx, dtype=np.int64)
    for s in range(0, len(train_arr), train_block):
        ids = train_arr[s:s + train_block]
        k = F.normalize(fc['g'][torch.as_tensor(ids)].float().to(DEVICE), dim=-1)
        vals, pos = (q @ k.t()).max(1)
        improve = vals > best
        best[improve] = vals[improve]
        best_train[improve] = torch.as_tensor(ids, device=DEVICE)[pos[improve]]
    scores = best.cpu().numpy()
    partners = best_train.cpu().numpy()
    order = np.argsort(-scores)[:min(top_pairs, len(scores))]
    rows = []
    for th in thresholds:
        rows.append({'backbone': ACTIVE_BACKBONE_KEY, 'dataset': name, 'fold': outer_id, 'threshold': float(th), 'sampled_test': int(len(sampled)), 'total_test': int(len(test_idx)), 'train_pool': int(len(train_idx)), 'n_above': int((scores >= th).sum()), 'fraction_above': float((scores >= th).mean()), 'sample_based': bool(len(sampled) < len(test_idx))})
    pairs = [{'dataset': name, 'fold': outer_id, 'similarity': float(scores[j]), 'test_index': int(sampled[j]), 'train_index': int(partners[j]), 'test_path': fc['paths'][int(sampled[j])], 'train_path': fc['paths'][int(partners[j])]} for j in order]
    return (rows, pairs)
if cfg.run_near_duplicate_audit:
    pair_rows = []
    for name in cfg.datasets:
        for oid in range(cfg.n_outer_folds):
            key = f'near_duplicate/{ACTIVE_BACKBONE_KEY}/{name}/f{oid}'
            if is_done(key):
                payload = state_data(key) or {}
                RESULTS['near_duplicate_audit'].extend(payload.get('summary', []))
                pair_rows.extend(payload.get('pairs', []))
                continue
            rows, pairs = near_duplicate_feature_audit(name, oid)
            RESULTS['near_duplicate_audit'].extend(rows)
            pair_rows.extend(pairs)
            mark_done(key, {'summary': rows, 'pairs': pairs})
    pd.DataFrame(pair_rows).to_csv(os.path.join(TAB_DIR, 'near_duplicate_top_pairs.csv'), index=False)
print('Near-duplicate audit rows:', len(RESULTS['near_duplicate_audit']))
import statsmodels.formula.api as smf

def holm_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1, running)
    return out

def hierarchical_bootstrap_diff(df, candidate, reference, metric='Accuracy', B=None):
    B = B or cfg.bootstrap_repetitions
    rng = np.random.RandomState(cfg.seed)
    datasets = sorted(df.dataset.unique())
    vals = []
    for _ in range(B):
        ds_sample = rng.choice(datasets, len(datasets), replace=True)
        diffs = []
        for ds in ds_sample:
            sub = df[df.dataset == ds]
            folds = sorted(sub.fold.unique())
            fs = rng.choice(folds, len(folds), replace=True)
            for f in fs:
                a = sub[(sub.fold == f) & (sub.method == candidate)][metric]
                b = sub[(sub.fold == f) & (sub.method == reference)][metric]
                if len(a) and len(b):
                    diffs.append(a.mean() - b.mean())
        if diffs:
            vals.append(np.mean(diffs))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]

def mixed_effects_compare(df, candidate, reference, metric='Accuracy'):
    q = df[df.method.isin([candidate, reference])].copy()
    q = q.groupby(['dataset', 'K', 'fold', 'method'], as_index=False)[metric].mean()
    if q.method.nunique() < 2 or q.dataset.nunique() < 2:
        return {'mixedlm_available': False, 'mixedlm_reason': 'insufficient groups'}
    q['is_candidate'] = (q.method == candidate).astype(int)
    q['fold_id'] = q.dataset.astype(str) + '_' + q.fold.astype(str)
    try:
        fit = smf.mixedlm(f'{metric} ~ is_candidate + C(K)', q, groups=q['dataset'], vc_formula={'fold': '0+C(fold_id)'}).fit(reml=False, method='lbfgs', disp=False)
        return {'mixedlm_available': True, 'mixedlm_coef': float(fit.params.get('is_candidate', np.nan)), 'mixedlm_p': float(fit.pvalues.get('is_candidate', np.nan)), 'mixedlm_converged': bool(getattr(fit, 'converged', False))}
    except Exception as e:
        return {'mixedlm_available': False, 'mixedlm_reason': f'{type(e).__name__}: {e}'}

def compare_methods(df, candidate, reference, metric='Accuracy'):
    fold = df.groupby(['dataset', 'K', 'fold', 'method'], as_index=False)[metric].mean()
    a = fold[fold.method == candidate].rename(columns={metric: 'a'})
    b = fold[fold.method == reference].rename(columns={metric: 'b'})
    m = a.merge(b, on=['dataset', 'K', 'fold'], suffixes=('_a', '_b'))
    d = (m.a - m.b).to_numpy()
    if len(d) < 2:
        return None
    pt = scipy_stats.ttest_rel(m.a, m.b).pvalue
    try:
        pw = scipy_stats.wilcoxon(d, zero_method='wilcox').pvalue
    except Exception:
        pw = np.nan
    cl = float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
    return {'candidate': candidate, 'reference': reference, 'metric': metric, 'n_dataset_fold': len(d), 'mean_delta': float(d.mean()), 'median_delta': float(np.median(d)), 'paired_t_p': float(pt), 'wilcoxon_p': float(pw), 'common_language': cl, 'hier_boot_ci95': hierarchical_bootstrap_diff(fold, candidate, reference, metric), **mixed_effects_compare(df, candidate, reference, metric)}
stat_rows = []
if RESULTS['classification_runs']:
    rdf = pd.DataFrame(RESULTS['classification_runs'])
    rdf = rdf[np.isfinite(rdf.Accuracy)]
    comparisons = [('CAPSTONE-CLIP', 'ProLIP'), ('CAPSTONE-CLIP', 'RLA'), ('CAPSTONE-CLIP', 'Shared-Caption')]
    for c, r in comparisons:
        if c in set(rdf.method) and r in set(rdf.method):
            q = compare_methods(rdf, c, r, 'Accuracy')
            if q:
                stat_rows.append(q)
    if stat_rows:
        adj = holm_adjust([x['paired_t_p'] for x in stat_rows])
        for x, p in zip(stat_rows, adj):
            x['paired_t_p_holm'] = float(p)
RESULTS['statistics'] = stat_rows
print(pd.DataFrame(stat_rows).to_string(index=False) if stat_rows else 'No statistics yet')
import os
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore', category=RuntimeWarning)
STAT_METRIC = 'Accuracy'
CONFIRMATORY_COMPARISONS = [('CAPSTONE-CLIP', 'ProLIP'), ('CAPSTONE-CLIP', 'RLA'), ('CAPSTONE-CLIP', 'Shared-Caption')]
ROPE_ACCURACY_FRACTION = 0.0025
HIER_BOOTSTRAP_REPETITIONS = 5000
BAYESIAN_DRAWS = 100000
STAT_RANDOM_SEED = int(getattr(cfg, 'seed', 20260717)) if 'cfg' in globals() else 20260717
MIN_COMPLETE_BLOCKS_FOR_CD = 5
CD_PREFERRED_METHODS = ['Zero-shot', 'Linear-Probe', 'CoOp', 'CLIP-Adapter', 'Tip-Adapter', 'Tip-Adapter-F', 'TaskRes', 'HOSO-Adapter', 'RLA', 'ProLIP', 'Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP']
if 'TAB_DIR' not in globals():
    base_out = getattr(cfg, 'local_out', '/content/capstone_outputs') if 'cfg' in globals() else '/content/capstone_outputs'
    TAB_DIR = os.path.join(base_out, 'tables')
if 'FIG_DIR' not in globals():
    base_out = getattr(cfg, 'local_out', '/content/capstone_outputs') if 'cfg' in globals() else '/content/capstone_outputs'
    FIG_DIR = os.path.join(base_out, 'figures')
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def locate_primary_results_csv():
    candidates = [os.path.join(TAB_DIR, 'raw_primary_classification.csv'), os.path.join(getattr(cfg, 'local_out', ''), 'tables', 'raw_primary_classification.csv') if 'cfg' in globals() else '', '/content/raw_primary_classification.csv']
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    search_roots = [Path('/content'), Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing')]
    found = []
    for root in search_roots:
        if root.exists():
            try:
                found.extend(root.rglob('raw_primary_classification.csv'))
            except Exception:
                pass
    if not found:
        return None
    found = sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(found[0])

def load_primary_classification_results():
    if 'RESULTS' in globals() and isinstance(RESULTS, dict) and (len(RESULTS.get('classification_runs', [])) > 0):
        df = pd.DataFrame(RESULTS['classification_runs']).copy()
        source = 'RESULTS["classification_runs"]'
    else:
        csv_path = locate_primary_results_csv()
        if csv_path is None:
            raise FileNotFoundError('Primary classification sonuçları bulunamadı. Önce ana benchmark hücresini ve çıktı kaydetme hücresini çalıştırın.')
        df = pd.read_csv(csv_path)
        source = csv_path
    required = {'dataset', 'K', 'fold', 'support_seed', 'method', STAT_METRIC}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'İstatistik analizi için eksik kolonlar: {missing}')
    df[STAT_METRIC] = pd.to_numeric(df[STAT_METRIC], errors='coerce')
    df['K'] = pd.to_numeric(df['K'], errors='coerce')
    df['fold'] = pd.to_numeric(df['fold'], errors='coerce')
    df['support_seed'] = pd.to_numeric(df['support_seed'], errors='coerce')
    df = df.dropna(subset=['dataset', 'K', 'fold', 'support_seed', 'method', STAT_METRIC]).copy()
    if 'available' in df.columns:
        available = df['available'].astype(str).str.lower()
        df = df[~available.isin(['false', '0', 'no'])].copy()
    if df.empty:
        raise RuntimeError('Filtreleme sonrasında geçerli classification sonucu kalmadı.')
    print(f'[statistics] Kaynak: {source}')
    print(f'[statistics] Ham koşu sayısı: {len(df):,} | dataset={df.dataset.nunique()} | method={df.method.nunique()} | K={sorted(df.K.unique().tolist())}')
    return df
raw_stats_df = load_primary_classification_results()
metric_max = float(raw_stats_df[STAT_METRIC].abs().max())
if metric_max > 1.5:
    METRIC_SCALE = 'percentage'
    ROPE = 100.0 * ROPE_ACCURACY_FRACTION
else:
    METRIC_SCALE = 'fraction'
    ROPE = ROPE_ACCURACY_FRACTION
print(f'[statistics] Metric scale: {METRIC_SCALE}; practical-equivalence ROPE: ±{ROPE:.6f}')
fold_df = raw_stats_df.groupby(['dataset', 'K', 'fold', 'method'], as_index=False, dropna=False).agg(metric_mean=(STAT_METRIC, 'mean'), metric_sd_across_support_seeds=(STAT_METRIC, 'std'), n_support_seeds=('support_seed', 'nunique'))
fold_df.to_csv(os.path.join(TAB_DIR, 'statistics_fold_level_seed_averaged.csv'), index=False)

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid_idx = np.where(np.isfinite(p_values))[0]
    if len(valid_idx) == 0:
        return adjusted
    valid_p = p_values[valid_idx]
    order = np.argsort(valid_p)
    running_max = 0.0
    temp = np.empty_like(valid_p)
    m = len(valid_p)
    for rank, local_idx in enumerate(order):
        value = (m - rank) * valid_p[local_idx]
        running_max = max(running_max, value)
        temp[local_idx] = min(1.0, running_max)
    adjusted[valid_idx] = temp
    return adjusted

def paired_frame(fold_level_df, candidate, reference, k_value=None):
    q = fold_level_df.copy()
    if k_value is not None:
        q = q[q['K'] == k_value].copy()
    a = q[q.method == candidate][['dataset', 'K', 'fold', 'metric_mean']].rename(columns={'metric_mean': 'candidate_score'})
    b = q[q.method == reference][['dataset', 'K', 'fold', 'metric_mean']].rename(columns={'metric_mean': 'reference_score'})
    paired = a.merge(b, on=['dataset', 'K', 'fold'], how='inner', validate='one_to_one')
    paired['difference'] = paired['candidate_score'] - paired['reference_score']
    return paired

def hierarchical_bootstrap_paired(paired, repetitions=5000, random_seed=20260717):
    if paired.empty:
        return {'bootstrap_mean': np.nan, 'bootstrap_ci_low': np.nan, 'bootstrap_ci_high': np.nan, 'bootstrap_p_gt_0': np.nan}
    rng = np.random.default_rng(random_seed)
    datasets = paired['dataset'].drop_duplicates().to_numpy()
    boot_values = np.empty(repetitions, dtype=float)
    grouped = {ds: paired.loc[paired.dataset == ds, 'difference'].to_numpy(float) for ds in datasets}
    for b in range(repetitions):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        sampled_dataset_means = []
        for ds in sampled_datasets:
            values = grouped[ds]
            sampled_values = rng.choice(values, size=len(values), replace=True)
            sampled_dataset_means.append(sampled_values.mean())
        boot_values[b] = np.mean(sampled_dataset_means)
    return {'bootstrap_mean': float(np.mean(boot_values)), 'bootstrap_ci_low': float(np.quantile(boot_values, 0.025)), 'bootstrap_ci_high': float(np.quantile(boot_values, 0.975)), 'bootstrap_p_gt_0': float(np.mean(boot_values > 0))}

def bayesian_rope_probability(differences, rope, draws=100000, random_seed=20260717):
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {'bayes_p_candidate_worse': np.nan, 'bayes_p_practical_equivalence': np.nan, 'bayes_p_candidate_better': np.nan, 'bayes_decision': 'insufficient_data'}
    counts = np.array([np.sum(d < -rope), np.sum(np.abs(d) <= rope), np.sum(d > rope)], dtype=float)
    posterior_alpha = counts + 0.5
    rng = np.random.default_rng(random_seed)
    posterior_samples = rng.dirichlet(posterior_alpha, size=draws)
    posterior_mean = posterior_samples.mean(axis=0)
    labels = ['reference_better', 'practical_equivalence', 'candidate_better']
    return {'bayes_left_count': int(counts[0]), 'bayes_rope_count': int(counts[1]), 'bayes_right_count': int(counts[2]), 'bayes_p_candidate_worse': float(posterior_mean[0]), 'bayes_p_practical_equivalence': float(posterior_mean[1]), 'bayes_p_candidate_better': float(posterior_mean[2]), 'bayes_decision': labels[int(np.argmax(posterior_mean))]}

def mixed_effects_pair(raw_df, candidate, reference):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {'mixedlm_available': False, 'mixedlm_reason': 'statsmodels is not installed'}
    q = raw_df[raw_df.method.isin([candidate, reference])].copy()
    q = q.dropna(subset=[STAT_METRIC])
    if q.empty or q.method.nunique() < 2 or q.dataset.nunique() < 2:
        return {'mixedlm_available': False, 'mixedlm_reason': 'insufficient paired groups'}
    q['is_candidate'] = (q.method == candidate).astype(int)
    q['dataset_fold'] = q['dataset'].astype(str) + '_f' + q['fold'].astype(int).astype(str)
    try:
        model = smf.mixedlm(f'{STAT_METRIC} ~ is_candidate + C(K)', data=q, groups=q['dataset'], vc_formula={'dataset_fold': '0 + C(dataset_fold)'})
        fit = model.fit(reml=False, method='lbfgs', maxiter=1000, disp=False)
        ci = fit.conf_int()
        return {'mixedlm_available': True, 'mixedlm_coef': float(fit.params.get('is_candidate', np.nan)), 'mixedlm_se': float(fit.bse.get('is_candidate', np.nan)), 'mixedlm_p': float(fit.pvalues.get('is_candidate', np.nan)), 'mixedlm_ci_low': float(ci.loc['is_candidate', 0] if 'is_candidate' in ci.index else np.nan), 'mixedlm_ci_high': float(ci.loc['is_candidate', 1] if 'is_candidate' in ci.index else np.nan), 'mixedlm_converged': bool(getattr(fit, 'converged', False))}
    except Exception as exc:
        return {'mixedlm_available': False, 'mixedlm_reason': f'{type(exc).__name__}: {exc}'}

def paired_confirmatory_statistics(fold_level_df, raw_df, candidate, reference, k_value=None):
    paired = paired_frame(fold_level_df, candidate, reference, k_value=k_value)
    if len(paired) < 2:
        return None
    d = paired['difference'].to_numpy(float)
    t_result = scipy_stats.ttest_rel(paired['candidate_score'], paired['reference_score'], nan_policy='omit')
    try:
        if np.allclose(d, 0):
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        else:
            w_result = scipy_stats.wilcoxon(d, zero_method='pratt', alternative='two-sided', method='auto')
            wilcoxon_stat = float(w_result.statistic)
            wilcoxon_p = float(w_result.pvalue)
    except Exception:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan
    d_sd = np.std(d, ddof=1)
    cohens_dz = float(np.mean(d) / d_sd) if d_sd > 0 and len(d) > 1 else np.nan
    n_win = int(np.sum(d > 0))
    n_tie = int(np.sum(d == 0))
    n_loss = int(np.sum(d < 0))
    common_language = float((n_win + 0.5 * n_tie) / len(d))
    bootstrap = hierarchical_bootstrap_paired(paired, repetitions=HIER_BOOTSTRAP_REPETITIONS, random_seed=STAT_RANDOM_SEED)
    bayesian = bayesian_rope_probability(d, rope=ROPE, draws=BAYESIAN_DRAWS, random_seed=STAT_RANDOM_SEED)
    mixed = mixed_effects_pair(raw_df, candidate, reference)
    scope = 'all_K' if k_value is None else f'K{k_value}'
    return {'scope': scope, 'candidate': candidate, 'reference': reference, 'metric': STAT_METRIC, 'n_dataset_fold_pairs': int(len(paired)), 'n_datasets': int(paired.dataset.nunique()), 'n_shot_levels': int(paired.K.nunique()), 'mean_candidate': float(paired.candidate_score.mean()), 'mean_reference': float(paired.reference_score.mean()), 'mean_paired_difference': float(d.mean()), 'median_paired_difference': float(np.median(d)), 'sd_paired_difference': float(np.std(d, ddof=1)), 'cohens_dz': cohens_dz, 'common_language_probability': common_language, 'wins': n_win, 'ties': n_tie, 'losses': n_loss, 'paired_t_statistic': float(t_result.statistic), 'paired_t_p': float(t_result.pvalue), 'wilcoxon_statistic': wilcoxon_stat, 'wilcoxon_p': wilcoxon_p, 'rope': float(ROPE), **bootstrap, **bayesian, **mixed}
available_methods = set(fold_df['method'].unique())
confirmatory_rows = []
for candidate, reference in CONFIRMATORY_COMPARISONS:
    if candidate not in available_methods:
        print(f'[skip] Candidate bulunamadı: {candidate}')
        continue
    if reference not in available_methods:
        print(f'[skip] Reference bulunamadı: {reference}')
        continue
    result = paired_confirmatory_statistics(fold_df, raw_stats_df, candidate, reference, k_value=None)
    if result is not None:
        confirmatory_rows.append(result)
    for k_value in sorted(fold_df['K'].unique()):
        result_k = paired_confirmatory_statistics(fold_df, raw_stats_df, candidate, reference, k_value=k_value)
        if result_k is not None:
            confirmatory_rows.append(result_k)
confirmatory_df = pd.DataFrame(confirmatory_rows)
if not confirmatory_df.empty:
    main_mask = confirmatory_df['scope'].eq('all_K')
    confirmatory_df['paired_t_p_holm'] = np.nan
    confirmatory_df['wilcoxon_p_holm'] = np.nan
    confirmatory_df.loc[main_mask, 'paired_t_p_holm'] = holm_adjust(confirmatory_df.loc[main_mask, 'paired_t_p'].to_numpy())
    confirmatory_df.loc[main_mask, 'wilcoxon_p_holm'] = holm_adjust(confirmatory_df.loc[main_mask, 'wilcoxon_p'].to_numpy())
    confirmatory_df['holm_family'] = np.where(main_mask, 'predefined_primary_all_K', 'secondary_shot_specific_not_adjusted')
    confirmatory_path = os.path.join(TAB_DIR, 'table_advanced_confirmatory_statistics.csv')
    confirmatory_df.to_csv(confirmatory_path, index=False)
    print('\n=== Ana confirmatory karşılaştırmalar ===')
    display_columns = ['candidate', 'reference', 'n_dataset_fold_pairs', 'mean_paired_difference', 'median_paired_difference', 'bootstrap_ci_low', 'bootstrap_ci_high', 'paired_t_p_holm', 'wilcoxon_p_holm', 'cohens_dz', 'common_language_probability', 'bayes_p_candidate_better', 'bayes_p_practical_equivalence', 'bayes_p_candidate_worse', 'mixedlm_coef', 'mixedlm_p']
    display(confirmatory_df.loc[confirmatory_df.scope == 'all_K', [c for c in display_columns if c in confirmatory_df.columns]].round(6))
    print('Kaydedildi:', confirmatory_path)
else:
    print('Confirmatory paired comparison üretilemedi.')

def prepare_complete_method_blocks(fold_level_df, preferred_methods=None):
    q = fold_level_df.copy()
    methods = sorted(q['method'].unique())
    if preferred_methods:
        selected = [m for m in preferred_methods if m in methods]
    else:
        selected = methods
    if len(selected) < 3:
        return (None, [])
    wide = q[q.method.isin(selected)].pivot_table(index=['dataset', 'K', 'fold'], columns='method', values='metric_mean', aggfunc='mean')
    complete = wide.dropna(axis=0, how='any').copy()
    if len(complete) < MIN_COMPLETE_BLOCKS_FOR_CD:
        coverage = wide.notna().mean(axis=0).sort_values(ascending=False)
        fallback_methods = coverage[coverage >= 0.8].index.tolist()
        if 'CAPSTONE-CLIP' in wide.columns and 'CAPSTONE-CLIP' not in fallback_methods:
            fallback_methods.insert(0, 'CAPSTONE-CLIP')
        fallback_methods = list(dict.fromkeys(fallback_methods))
        if len(fallback_methods) >= 3:
            complete = wide[fallback_methods].dropna(axis=0, how='any')
            selected = fallback_methods
    return (complete, selected)

def nemenyi_critical_difference(n_methods, n_blocks, alpha=0.05):
    try:
        q_alpha = scipy_stats.studentized_range.ppf(1.0 - alpha, n_methods, np.inf) / math.sqrt(2.0)
    except Exception:
        q_alpha = 2.569
    cd = q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))
    return (float(cd), float(q_alpha))

def draw_cd_diagram(average_ranks, cd, output_path, title):
    ranks = average_ranks.sort_values()
    methods = ranks.index.tolist()
    values = ranks.to_numpy(float)
    n_methods = len(methods)
    fig_height = max(4.5, 0.42 * n_methods + 2.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    min_rank = 1.0
    max_rank = float(n_methods)
    ax.set_xlim(min_rank - 0.25, max_rank + 0.25)
    ax.set_ylim(-1.2, n_methods + 1.8)
    axis_y = n_methods + 0.7
    ax.hlines(axis_y, min_rank, max_rank, linewidth=1.5)
    for r in range(1, n_methods + 1):
        ax.vlines(r, axis_y - 0.12, axis_y + 0.12, linewidth=1)
        ax.text(r, axis_y + 0.22, str(r), ha='center', va='bottom', fontsize=10)
    for idx, (method, rank_value) in enumerate(zip(methods, values)):
        y = n_methods - idx - 0.1
        ax.plot([rank_value, rank_value], [axis_y - 0.12, y], linewidth=0.8)
        ax.scatter([rank_value], [y], s=28, zorder=3)
        ax.text(rank_value + 0.06, y, f'{method}  ({rank_value:.3f})', ha='left', va='center', fontsize=10)
    cd_y = n_methods + 1.45
    cd_start = min_rank
    cd_end = min(max_rank, min_rank + cd)
    ax.hlines(cd_y, cd_start, cd_end, linewidth=3)
    ax.vlines([cd_start, cd_end], cd_y - 0.12, cd_y + 0.12, linewidth=2)
    ax.text((cd_start + cd_end) / 2, cd_y + 0.16, f'Critical Difference = {cd:.3f}', ha='center', va='bottom', fontsize=10)
    candidate_groups = []
    for left in range(n_methods):
        right = left
        while right + 1 < n_methods and values[right + 1] - values[left] <= cd:
            right += 1
        if right > left:
            candidate_groups.append((left, right))
    maximal_groups = []
    for group in candidate_groups:
        contained = any((other != group and other[0] <= group[0] and (other[1] >= group[1]) for other in candidate_groups))
        if not contained:
            maximal_groups.append(group)
    group_base_y = axis_y - 0.45
    for group_idx, (left, right) in enumerate(maximal_groups):
        y = group_base_y - 0.18 * group_idx
        ax.hlines(y, values[left], values[right], linewidth=4)
    ax.set_title(title, pad=34)
    ax.set_xlabel('Average rank — lower is better')
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
complete_blocks, cd_methods = prepare_complete_method_blocks(fold_df, preferred_methods=CD_PREFERRED_METHODS)
nemenyi_summary = {'available': False, 'metric': STAT_METRIC}
if complete_blocks is not None and complete_blocks.shape[0] >= MIN_COMPLETE_BLOCKS_FOR_CD and (complete_blocks.shape[1] >= 3):
    rank_matrix = complete_blocks.rank(axis=1, method='average', ascending=False)
    average_ranks = rank_matrix.mean(axis=0).sort_values()
    friedman_result = scipy_stats.friedmanchisquare(*[complete_blocks[col].to_numpy(float) for col in complete_blocks.columns])
    n_blocks = int(complete_blocks.shape[0])
    n_methods = int(complete_blocks.shape[1])
    cd_005, q_alpha = nemenyi_critical_difference(n_methods=n_methods, n_blocks=n_blocks, alpha=0.05)
    rank_table = average_ranks.rename('average_rank').reset_index().rename(columns={'index': 'method'})
    rank_table['n_complete_blocks'] = n_blocks
    rank_table['critical_difference_alpha_0_05'] = cd_005
    rank_table['friedman_statistic'] = float(friedman_result.statistic)
    rank_table['friedman_p'] = float(friedman_result.pvalue)
    pairwise_rows = []
    for i, method_i in enumerate(average_ranks.index):
        for j, method_j in enumerate(average_ranks.index):
            if j <= i:
                continue
            rank_diff = abs(float(average_ranks[method_i]) - float(average_ranks[method_j]))
            pairwise_rows.append({'method_a': method_i, 'method_b': method_j, 'average_rank_a': float(average_ranks[method_i]), 'average_rank_b': float(average_ranks[method_j]), 'absolute_rank_difference': rank_diff, 'critical_difference': cd_005, 'nemenyi_significant_alpha_0_05': bool(rank_diff > cd_005)})
    pairwise_nemenyi_df = pd.DataFrame(pairwise_rows)
    rank_path = os.path.join(TAB_DIR, 'table_average_ranks_friedman_nemenyi.csv')
    pairwise_path = os.path.join(TAB_DIR, 'table_pairwise_nemenyi_rank_differences.csv')
    cd_figure_path = os.path.join(FIG_DIR, 'figure_critical_difference_nemenyi.png')
    rank_table.to_csv(rank_path, index=False)
    pairwise_nemenyi_df.to_csv(pairwise_path, index=False)
    draw_cd_diagram(average_ranks=average_ranks, cd=cd_005, output_path=cd_figure_path, title=f'Friedman–Nemenyi Critical Difference Diagram\n{n_methods} methods, {n_blocks} complete dataset × shot × fold blocks')
    nemenyi_summary = {'available': True, 'metric': STAT_METRIC, 'n_complete_blocks': n_blocks, 'n_methods': n_methods, 'methods': list(complete_blocks.columns), 'friedman_statistic': float(friedman_result.statistic), 'friedman_p': float(friedman_result.pvalue), 'nemenyi_cd_alpha_0_05': cd_005, 'q_alpha': q_alpha, 'rank_table_path': rank_path, 'pairwise_table_path': pairwise_path, 'cd_figure_path': cd_figure_path}
    print('\n=== Friedman–Nemenyi analizi ===')
    print(f'Complete blocks: {n_blocks} | Methods: {n_methods} | Friedman p={friedman_result.pvalue:.6g} | CD={cd_005:.4f}')
    display(rank_table.round(6))
else:
    reason = 'En az üç yöntem için yeterli sayıda ortak ve eksiksiz dataset × K × fold bloğu bulunamadı.'
    nemenyi_summary['reason'] = reason
    print('[CD skipped]', reason)
advanced_statistics_payload = {'analysis_version': 'capstone_advanced_statistics_v1', 'metric': STAT_METRIC, 'metric_scale': METRIC_SCALE, 'rope': float(ROPE), 'hierarchical_bootstrap_repetitions': HIER_BOOTSTRAP_REPETITIONS, 'bayesian_draws': BAYESIAN_DRAWS, 'random_seed': STAT_RANDOM_SEED, 'support_seed_handling': 'Support seeds were averaged within dataset × K × fold × method before confirmatory paired comparisons.', 'holm_family': [{'candidate': c, 'reference': r} for c, r in CONFIRMATORY_COMPARISONS], 'confirmatory_results': confirmatory_df.replace({np.nan: None}).to_dict('records') if not confirmatory_df.empty else [], 'nemenyi_summary': nemenyi_summary}
json_path = os.path.join(TAB_DIR, 'advanced_statistics_summary.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(advanced_statistics_payload, f, ensure_ascii=False, indent=2)
if 'RESULTS' in globals() and isinstance(RESULTS, dict):
    RESULTS['advanced_statistics'] = advanced_statistics_payload
print('\n=== Oluşturulan ana çıktılar ===')
if not confirmatory_df.empty:
    print(os.path.join(TAB_DIR, 'table_advanced_confirmatory_statistics.csv'))
print(os.path.join(TAB_DIR, 'statistics_fold_level_seed_averaged.csv'))
print(json_path)
if nemenyi_summary.get('available', False):
    print(nemenyi_summary['rank_table_path'])
    print(nemenyi_summary['pairwise_table_path'])
    print(nemenyi_summary['cd_figure_path'])
print("\nNot: Bayesian bölüm, küçük örneklem için Bayesian paired sign/ROPE analizidir; makalede yanlış biçimde 'Bayesian signed-rank' olarak adlandırılmamalıdır.")

def tensor_bytes(x):
    return 0 if x is None else x.nelement() * x.element_size()

def cache_size_report(name):
    fc = get_dataset_cache(name)
    return {'g_mb': tensor_bytes(fc['g']) / 1024 ** 2, 't_mb': tensor_bytes(fc['t']) / 1024 ** 2, 'h_mb': tensor_bytes(fc['h']) / 1024 ** 2}
if cfg.run_efficiency:
    models = {'RLA': RLAHead(EMBED_DIM), 'Shared-Caption': SharedCaptionAdapter(EMBED_DIM), 'Capacity-Matched-Shared': CapacityMatchedSharedCaptionAdapter(EMBED_DIM), 'CAPSTONE-CLIP': CAPSTONECLIP(EMBED_DIM)}
    if VISUAL_PROJ0 is not None:
        models['ProLIP'] = ProLIPProjector(VISUAL_PROJ0)
    total = sum((p.numel() for p in clip_model.parameters()))
    for name, m in models.items():
        tr = n_trainable(m)
        RESULTS['efficiency'].append({'method': name, 'trainable_params': tr, 'trainable_pct_of_backbone': 100 * tr / total})
    for name in cfg.datasets:
        RESULTS['efficiency'].append({'method': 'feature_cache', 'dataset': name, **cache_size_report(name)})
print(pd.DataFrame(RESULTS['efficiency']).to_string(index=False))
EXTERNAL_SCHEMA_COLUMNS = ['method', 'backbone', 'dataset', 'K', 'fold', 'support_seed', 'Accuracy', 'MacroF1', 'BalancedAccuracy', 'NLL', 'ECE', 'protocol_hash', 'official_code_commit']
external_template = pd.DataFrame(columns=EXTERNAL_SCHEMA_COLUMNS)
external_template.to_csv(os.path.join(TAB_DIR, 'external_baseline_import_template.csv'), index=False)

def import_external_baseline_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in EXTERNAL_SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'External baseline CSV missing {missing}')
    bad = df.protocol_hash.astype(str) != PROTOCOL_HASH
    if bad.any():
        raise ValueError('External baseline protocol hash mismatch')
    RESULTS['external_baselines'].extend(df.to_dict('records'))
    return df
print('External baseline template:', os.path.join(TAB_DIR, 'external_baseline_import_template.csv'))

def save_table(df, name):
    path = os.path.join(TAB_DIR, name + '.csv')
    df.to_csv(path, index=False)
    return path

def summary_table(raw, group, metrics):
    if raw.empty:
        return pd.DataFrame()
    agg = {m: ['mean', 'std', 'count'] for m in metrics if m in raw.columns}
    out = raw.groupby(group, dropna=False).agg(agg)
    out.columns = [f'{a}_{b}' for a, b in out.columns]
    return out.reset_index()
raw_primary = pd.DataFrame(RESULTS['classification_runs'])
raw_controls = pd.DataFrame(RESULTS['caption_control_runs'])
raw_ret = pd.DataFrame(RESULTS['retrieval_runs'])
raw_ab = pd.DataFrame(RESULTS['ablation_runs'])
raw_diag = pd.DataFrame(RESULTS['class_diagnostics'])
raw_b2n = pd.DataFrame(RESULTS['base_to_new_runs'])
raw_cross = pd.DataFrame(RESULTS['cross_dataset_runs'])
raw_cb = pd.DataFrame(RESULTS['cross_backbone_runs'])
for df, name in [(raw_primary, 'raw_primary_classification'), (raw_controls, 'raw_caption_controls'), (raw_ret, 'raw_retrieval'), (raw_ab, 'raw_architecture_ablations'), (raw_diag, 'raw_class_diagnostics'), (raw_b2n, 'raw_base_to_new'), (raw_cross, 'raw_cross_dataset'), (raw_cb, 'raw_cross_backbone')]:
    save_table(df, name)
primary_summary = summary_table(raw_primary, ['backbone', 'dataset', 'K', 'method'], ['Accuracy', 'MacroF1', 'BalancedAccuracy', 'NLL', 'ECE'])
control_summary = summary_table(raw_controls, ['dataset', 'K', 'control', 'caption_scale', 'caption_corpus'], ['Accuracy', 'MacroF1', 'ECE', 'external_to_support_ratio'])
ret_summary = summary_table(raw_ret, ['backbone', 'caption_dataset', 'method'], ['mR', 'classification_Accuracy', 'i2t_R1', 't2i_R1'])
ablation_summary = summary_table(raw_ab, ['backbone', 'dataset', 'K', 'variant'], ['Accuracy', 'MacroF1', 'ECE'])
save_table(primary_summary, 'table_primary_summary')
save_table(control_summary, 'table_caption_controls')
save_table(ret_summary, 'table_retrieval_pareto')
save_table(ablation_summary, 'table_architecture_ablations')
save_table(pd.DataFrame(RESULTS['statistics']), 'table_confirmatory_statistics')
save_table(pd.DataFrame(RESULTS['efficiency']), 'table_efficiency')
save_table(pd.DataFrame(RESULTS['near_duplicate_audit']), 'table_near_duplicate_audit')
save_table(pd.DataFrame(METHOD_AUDIT), 'table_method_audit')
if not primary_summary.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    focus = primary_summary[primary_summary.method.isin(['RLA', 'ProLIP', 'Shared-Caption', 'CAPSTONE-CLIP'])]
    avg = focus.groupby(['K', 'method'], as_index=False).Accuracy_mean.mean()
    for method, g in avg.groupby('method'):
        ax.plot(g.K, g.Accuracy_mean, marker='o', label=method)
    ax.set_xlabel('Shots per class')
    ax.set_ylabel('Mean accuracy')
    ax.set_title('CAPSTONE-CLIP: accuracy vs shot')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'figure_accuracy_vs_shot.png'), dpi=220)
    plt.close(fig)
if not control_summary.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    q = control_summary[control_summary.control == 'joint_full']
    if len(q):
        z = q.groupby('caption_scale', as_index=False).Accuracy_mean.mean()
        ax.plot(z.caption_scale, z.Accuracy_mean, marker='o')
    ax.set_xlabel('Caption corpus fraction')
    ax.set_ylabel('Mean accuracy')
    ax.set_title('Caption data scaling')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'figure_caption_scaling.png'), dpi=220)
    plt.close(fig)
claim_audit = [{'claim': 'Validation-free target protocol', 'status': 'PASS', 'evidence': 'No target val function; disjoint outer folds.'}, {'claim': 'Pure few-shot', 'status': 'DO NOT CLAIM', 'evidence': 'Main model uses external image-caption pairs.'}, {'claim': 'Caption grounding causes gains', 'status': 'CONDITIONAL', 'evidence': 'Requires matched-budget, shuffled-caption and image-only controls.'}, {'claim': 'Explainable attention', 'status': 'REMOVED', 'evidence': 'No explainability mechanism in main architecture.'}, {'claim': 'Retrieval SOTA', 'status': 'DO NOT CLAIM', 'evidence': 'Requires same-backbone official protocol comparison.'}, {'claim': 'Train-test near-duplicate leakage', 'status': 'AUDITED', 'evidence': 'Feature-space sample audit exported; inspect threshold table and top pairs.'}, {'claim': 'Inductive unseen benchmark', 'status': 'UNVERIFIED', 'evidence': 'RemoteCLIP/GeoRSCLIP pretraining overlap may exist.'}]
RESULTS['claim_audit'] = claim_audit
save_table(pd.DataFrame(claim_audit), 'table_claim_audit')
RESULTS['timing'] = {**TIMING, 'session_wall_clock_s': time.time() - T0_SESSION, 'hardware': HW_NAME}
json_path = os.path.join(cfg.local_out, 'results_capstone_q1_v10.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, indent=2, default=str)
config_path = os.path.join(cfg.local_out, 'config_capstone_q1_v10.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(asdict(cfg), f, indent=2, default=str)
claim_md = '# CAPSTONE-CLIP manuscript claim guide\n\n## Recommended positioning\nCaption-augmented, validation-free, label-efficient remote-sensing scene adaptation with a shared-private task-oriented residual architecture.\n\n## Required evidence before the strongest claim\n- CAPSTONE vs ProLIP/RLA/Shared-Caption and capacity-matched shared control on disjoint folds.\n- Matched-budget, shuffled-caption and image-only controls.\n- RSICD/RSITMD corpus transfer and caption scaling.\n- Base-to-new and cross-dataset generalization.\n- OpenCLIP, RemoteCLIP and GeoRSCLIP replication.\n\n## Forbidden or unsupported wording\n- Pure few-shot / equal total data budget.\n- Explainable or faithful attention.\n- Retrieval SOTA without same-protocol official comparison.\n- Pretraining-unseen data without documented audit.\n'
open(os.path.join(cfg.local_out, 'MANUSCRIPT_CLAIM_GUIDE.md'), 'w', encoding='utf-8').write(claim_md)
summary = f"# CAPSTONE-CLIP Q1 run summary\n\n- Profile: {RUN_PROFILE}\n- Protocol hash: `{PROTOCOL_HASH}`\n- Target validation: No\n- Disjoint folds: {cfg.n_outer_folds}\n- Support seeds: {len(cfg.support_seeds)}\n- Datasets: {', '.join(cfg.datasets)}\n- Main model: CAPSTONE-CLIP\n- Primary rows: {len(raw_primary)}\n- Caption-control rows: {len(raw_controls)}\n- Retrieval rows: {len(raw_ret)}\n- Ablation rows: {len(raw_ab)}\n\nThe strongest manuscript claims remain conditional on the claim-audit table.\n"
open(os.path.join(cfg.local_out, 'RUN_SUMMARY_CAPSTONE_Q1.md'), 'w', encoding='utf-8').write(summary)
if IN_COLAB and cfg.cache_to_drive:
    safe_makedirs(cfg.out_root, exist_ok=True, required=False)
    for p in glob.glob(os.path.join(cfg.local_out, '*')):
        if os.path.isfile(p):
            safe_copy2(p, os.path.join(cfg.out_root, os.path.basename(p)), required=False)
    for sub in ('tables', 'figures'):
        src = os.path.join(cfg.local_out, sub)
        dst = os.path.join(cfg.out_root, sub)
        if os.path.isdir(src):
            safe_copytree(src, dst, dirs_exist_ok=True, required=False)
zip_path = shutil.make_archive(cfg.local_out, 'zip', cfg.local_out)
print('Results JSON:', json_path)
print('Output ZIP:', zip_path)
