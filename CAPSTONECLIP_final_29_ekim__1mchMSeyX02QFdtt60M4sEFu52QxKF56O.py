from pathlib import Path
import json, hashlib, time, os, sys, subprocess, ast, platform
MODE = 'analysis'
assert MODE in {'dry_run', 'audit', 'rerun', 'analysis'}
DRIVE_ROOT = Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing')
OUT = DRIVE_ROOT / 'reviewer_revision_final_29_ekim' if MODE != 'dry_run' else Path('/content/capstone_final_dry_run')
if MODE != 'dry_run':
    from google.colab import drive
    drive.mount('/content/drive')
for sub in ['audit', 'runs', 'tables', 'state']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)

def status(stage, state, detail):
    atomic_json(OUT / 'status.json', dict(stage=stage, state=state, detail=str(detail), time=time.time()))
READY = MODE in {'audit', 'rerun'}
SESSION_START = time.monotonic()
SESSION_BUDGET_HOURS = 4.9
print('Mod:', MODE, 'Çıktı:', OUT)
if READY:
    try:
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
        cfg.run_tag = 'final_29_ekim'
        cfg.out_root = str(OUT / 'source_harness')
        cfg.local_out = '/content/capstone_final_29_ekim'
        cfg.drive_root = str(DRIVE_ROOT)
        cfg.data_root = str(DRIVE_ROOT / 'data')
        cfg.cache_to_drive = False
        cfg.backbone_specs[0]['pretrained_candidates'] = ['laion2b_s34b_b79k']
    except Exception as exc:
        READY = False
        status('source_cell_1', 'BLOCKED', repr(exc))
        print('Kaynak aşama 1 tamamlanamadı:', repr(exc))
if READY:
    try:
        import sys, subprocess

        def pip_install(pkgs):
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=False)
        pip_install(['open_clip_torch', 'ftfy', 'regex', 'scikit-learn', 'scipy', 'pandas', 'matplotlib', 'tqdm', 'huggingface_hub', 'statsmodels', 'psutil', 'ImageHash', 'scikit-image'])
        print('Dependencies installed.')
    except Exception as exc:
        READY = False
        status('source_cell_3', 'BLOCKED', repr(exc))
        print('Kaynak aşama 3 tamamlanamadı:', repr(exc))
if READY:
    try:
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

        def drive_mirror(*args, **kwargs):
            return None
    except Exception as exc:
        READY = False
        status('source_cell_4', 'BLOCKED', repr(exc))
        print('Kaynak aşama 4 tamamlanamadı:', repr(exc))
if READY:
    try:
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
    except Exception as exc:
        READY = False
        status('source_cell_6', 'BLOCKED', repr(exc))
        print('Kaynak aşama 6 tamamlanamadı:', repr(exc))
if READY:
    try:
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
    except Exception as exc:
        READY = False
        status('source_cell_7', 'BLOCKED', repr(exc))
        print('Kaynak aşama 7 tamamlanamadı:', repr(exc))
if READY:
    try:
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
    except Exception as exc:
        READY = False
        status('source_cell_9', 'BLOCKED', repr(exc))
        print('Kaynak aşama 9 tamamlanamadı:', repr(exc))
if READY:
    try:

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
    except Exception as exc:
        READY = False
        status('source_cell_11', 'BLOCKED', repr(exc))
        print('Kaynak aşama 11 tamamlanamadı:', repr(exc))
if READY:
    try:
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
    except Exception as exc:
        READY = False
        status('source_cell_13', 'BLOCKED', repr(exc))
        print('Kaynak aşama 13 tamamlanamadı:', repr(exc))
if READY:
    try:

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
    except Exception as exc:
        READY = False
        status('source_cell_15', 'BLOCKED', repr(exc))
        print('Kaynak aşama 15 tamamlanamadı:', repr(exc))
if READY:
    try:

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
    except Exception as exc:
        READY = False
        status('source_cell_17', 'BLOCKED', repr(exc))
        print('Kaynak aşama 17 tamamlanamadı:', repr(exc))
if READY:
    try:
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
    except Exception as exc:
        READY = False
        status('source_cell_19', 'BLOCKED', repr(exc))
        print('Kaynak aşama 19 tamamlanamadı:', repr(exc))
if READY:
    try:

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
    except Exception as exc:
        READY = False
        status('source_cell_21', 'BLOCKED', repr(exc))
        print('Kaynak aşama 21 tamamlanamadı:', repr(exc))
PHASH_MAX = 4
COS_MIN = 0.99
CHUNK = 64

def candidate_mask(qhash, thash, qfeat, tfeat):
    import numpy as np
    d = np.unpackbits(np.bitwise_xor(qhash[:, None, :], thash[None, :, :]), axis=2).sum(2)
    sim = np.einsum('ik,jk->ij', qfeat, tfeat, optimize=False)
    return (d, sim, (d <= PHASH_MAX) | (sim >= COS_MIN))

def audit_full():
    import numpy as np, pandas as pd, torch, imagehash, html
    from PIL import Image
    bank = cap_bank('RSITMD', 'train')
    target = get_dataset_cache('UCMerced')
    splits, annotation = build_caption_splits_at(DATA_AT['RSITMD'], 'RSITMD', [str(Path(cfg.data_root) / resolve_dataset_folder('RSITMD', cfg.data_root)), cfg.data_root])
    qp = [p for p, _ in splits['train']]
    tp = list(target['paths'])
    assert len(qp) == bank['n_img'] == 3858, 'RSITMD train sayısı kaynak protokolle farklı'
    assert len(tp) == 2100, 'UCMerced sayısı farklı'

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def ph(p):
        with Image.open(p) as im:
            return str(imagehash.phash(im.convert('RGB'), hash_size=8))
    qh = [ph(p) for p in qp]
    th = [ph(p) for p in tp]
    qs = [sha(p) for p in qp]
    ts = [sha(p) for p in tp]
    feature_digest = hashlib.sha256(bank['g'].cpu().numpy().tobytes() + target['g'].cpu().numpy().tobytes()).hexdigest()
    inventory = dict(feature_digest=feature_digest, qpaths=qp, tpaths=tp, qsha=qs, tsha=ts, annotation=annotation, backbone=ACTIVE_BACKBONE_KEY, thresholds=[PHASH_MAX, COS_MIN])
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    atomic_json(OUT / 'audit' / 'inventory.json', inventory)
    qg = F.normalize(bank['g'].float(), dim=-1).cpu().numpy()
    tg = F.normalize(target['g'].float(), dim=-1).cpu().numpy()
    fresh, _, _ = encode_paths(qp, desc='RSITMD cache alignment', keep_tokens=False, keep_preproj=False)
    assert torch.allclose(F.normalize(fresh.float(), dim=-1), torch.from_numpy(qg), atol=0.003, rtol=0.003), 'Cache/path sırası veya backbone farklı'
    fresh_t, _, _ = encode_paths(tp, desc='UCMerced cache alignment', keep_tokens=False, keep_preproj=False)
    assert torch.allclose(F.normalize(fresh_t.float(), dim=-1), torch.from_numpy(tg), atol=0.003, rtol=0.003), 'Target cache/path uyumsuzluğu'
    qbytes = np.array([list(bytes.fromhex(h)) for h in qh], dtype=np.uint8)
    tbytes = np.array([list(bytes.fromhex(h)) for h in th], dtype=np.uint8)
    rows = []
    for start in range(0, len(qp), CHUNK):
        part = OUT / 'audit' / f'block_{start:05d}_{fingerprint[:12]}.json'
        if part.exists():
            block = json.loads(part.read_text())
        else:
            d, s, m = candidate_mask(qbytes[start:start + CHUNK], tbytes, qg[start:start + CHUNK], tg)
            block = [dict(q=int(start + i), t=int(j), phash_distance=int(d[i, j]), cosine=float(s[i, j])) for i, j in zip(*np.where(m))]
            atomic_json(part, block)
        rows.extend(block)
    checked = []
    from skimage.metrics import structural_similarity
    pairdir = OUT / 'audit' / 'pairs'
    pairdir.mkdir(exist_ok=True)
    for r in rows:
        i, j = (r['q'], r['t'])
        a, b = (qp[i], tp[j])
        f, _, _ = encode_paths([a, b], desc='pair independent check', keep_tokens=False, keep_preproj=False)
        f = F.normalize(f.float(), dim=-1)
        cosine = float((f[0] * f[1]).sum())
        distance = int(imagehash.hex_to_hash(ph(a)) - imagehash.hex_to_hash(ph(b)))
        assert abs(cosine - r['cosine']) < 0.004 and distance == r['phash_distance'], 'Bağımsız kontrol uyuşmadı'
        with Image.open(a) as im:
            ia = im.convert('RGB').resize((256, 256))
        with Image.open(b) as im:
            ib = im.convert('RGB').resize((256, 256))
        ssim = float(structural_similarity(np.array(ia), np.array(ib), channel_axis=2, data_range=255))
        canvas = Image.new('RGB', (512, 256))
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (256, 0))
        pid = f'q{i}_t{j}'
        canvas.save(pairdir / f'{pid}.png')
        checked.append(dict(pair_id=pid, **r, qpath=a, tpath=b, sha_equal=qs[i] == ts[j], independent_cosine=cosine, independent_phash=distance, ssim=ssim))
    cols = ['pair_id', 'q', 't', 'phash_distance', 'cosine', 'qpath', 'tpath', 'sha_equal', 'independent_cosine', 'independent_phash', 'ssim']
    df = pd.DataFrame(checked, columns=cols)
    df.to_csv(OUT / 'audit' / 'candidates.csv', index=False)
    review = OUT / 'audit' / 'review.csv'
    oldmeta = OUT / 'audit' / 'review_fingerprint.txt'
    if not review.exists():
        rr = df[['pair_id']].copy()
        for c in ['reviewer1', 'reviewer2', 'decision1', 'decision2', 'adjudication', 'notes']:
            rr[c] = ''
        rr.to_csv(review, index=False)
        oldmeta.write_text(fingerprint)
    else:
        assert oldmeta.read_text() == fingerprint, 'İnceleme farklı veri sürümüne ait; yeni çıktı kökü kullanın'
    page = "<meta charset='utf-8'><h1>RSITMD train — UCMerced</h1>" + ''.join((f"<h3>{r['pair_id']}</h3><img src='pairs/{r['pair_id']}.png'><p>{html.escape(r['qpath'])}<br>{html.escape(r['tpath'])}</p>" for r in checked))
    (OUT / 'audit' / 'pairs.html').write_text(page)
    atomic_json(OUT / 'audit' / 'coverage.json', dict(complete=True, fingerprint=fingerprint, queries=len(qp), targets=len(tp), pairs=len(qp) * len(tp), candidates=len(df), top_k_truncation=False))
    return df
if READY and MODE == 'audit':
    try:
        audit_full()
        status('audit', 'REVIEW_REQUIRED', 'İki bağımsız inceleme sonrası rerun modunu çalıştırın')
    except Exception as exc:
        status('audit', 'BLOCKED', repr(exc))
        print(repr(exc))
if READY:

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
        completed = 0
        if RUN_CHECKPOINT.exists():
            ck = torch.load(RUN_CHECKPOINT, map_location=DEVICE, weights_only=False)
            net.load_state_dict(ck['model'])
            opt.load_state_dict(ck['optimizer'])
            completed = ck['epoch']
            random.setstate(ck['python_rng'])
            np.random.set_state(ck['numpy_rng'])
            torch.set_rng_state(ck['torch_rng'].cpu())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
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
                if ep < completed:
                    continue
                steps = max(1, cfg.caption_steps_per_epoch if use_caption else math.ceil(len(y_local) / cfg.batch_size))
                for _ in range(steps):
                    loss = classification_step()
                    if use_caption:
                        loss = loss + hp['lambda_cap'] * auxiliary_step()
                    loss = loss + hp['lambda_gate'] * net.gate_regularizer()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                ck = dict(model=net.state_dict(), optimizer=opt.state_dict(), epoch=ep + 1, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
                tmp = RUN_CHECKPOINT.with_suffix('.tmp')
                torch.save(ck, tmp)
                tmp.replace(RUN_CHECKPOINT)
        info = {'selection': 'prespecified_no_target_validation', 'control': control, 'caption_scale': caption_scale, 'support_images': len(support_idx), 'external_images': bank['n_img'] if use_caption else 0, 'external_to_support_ratio': bank['n_img'] / len(support_idx) if use_caption else 0.0, 'classes': classes, 'caption_dataset': caption_dataset or cfg.caption_dataset}
        return (net, info)
if READY:

    def drop_caption_images(bank, drop_ids):
        drop = set((int(x) for x in drop_ids))
        ids = np.asarray([i for i in range(bank['n_img']) if i not in drop], dtype=int)
        gs = bank['g'][ids]
        ts = bank['t'][ids]
        caps = []
        starts = []
        counts = []
        owners = []
        us = []
        for new_i, old_i in enumerate(ids.tolist()):
            st = int(bank['cap_start'][old_i])
            ct = int(bank['cap_count'][old_i])
            sel = torch.arange(st, st + ct)
            starts.append(len(caps))
            counts.append(ct)
            caps.extend([bank['captions'][j] for j in sel.tolist()])
            owners.extend([new_i] * ct)
            us.append(bank['u'][sel])
        return {'g': gs, 't': ts, 'u': torch.cat(us), 'cap_start': torch.tensor(starts), 'cap_count': torch.tensor(counts), 'cap_owner': torch.tensor(owners), 'n_img': len(ids), 'captions': caps, 'source_ids': torch.tensor(ids)}

def reviewed_drop():
    import pandas as pd
    coverage = json.loads((OUT / 'audit' / 'coverage.json').read_text())
    assert coverage['complete'] and coverage['pairs'] == 3858 * 2100
    assert (OUT / 'audit' / 'review_fingerprint.txt').read_text() == coverage['fingerprint']
    c = pd.read_csv(OUT / 'audit' / 'candidates.csv')
    r = pd.read_csv(OUT / 'audit' / 'review.csv').fillna('')
    assert not r.pair_id.duplicated().any() and set(r.pair_id) == set(c.pair_id)
    allowed = {'duplicate', 'distinct', 'uncertain'}
    for row in r.to_dict('records'):
        assert row['reviewer1'].strip() and row['reviewer2'].strip() and (row['reviewer1'].strip() != row['reviewer2'].strip()), 'İki farklı değerlendirici gerekli'
        assert row['decision1'] in allowed and row['decision2'] in allowed
        if row['decision1'] != row['decision2']:
            assert row['adjudication'] in allowed, 'Uyuşmazlığı çözün'
    r['resolved'] = r.apply(lambda x: x.adjudication if x.decision1 != x.decision2 else x.decision1, axis=1)
    atomic_json(OUT / 'audit' / 'review_summary.json', r.resolved.value_counts().to_dict())
    return (sorted(c.q.astype(int).unique()), coverage)

def rerun_matched():
    import numpy as np, pandas as pd, torch
    from contextlib import contextmanager
    global RUN_CHECKPOINT, TEST_UNLOCKED
    drops, coverage = reviewed_drop()
    inv = json.loads((OUT / 'audit' / 'inventory.json').read_text())
    assert all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in zip(inv['qpaths'] + inv['tpaths'], inv['qsha'] + inv['tsha']))), 'Audit sonrası veri değişti'
    original = CAPTION_BANKS[ACTIVE_BACKBONE_KEY, 'RSITMD']['train']
    target = get_dataset_cache('UCMerced')
    assert hashlib.sha256(original['g'].cpu().numpy().tobytes() + target['g'].cpu().numpy().tobytes()).hexdigest() == inv['feature_digest'], 'Audit sonrası cache değişti'
    clean = drop_caption_images(original, drops)
    assert clean['n_img'] == original['n_img'] - len(drops) and clean['n_img'] > 0
    atomic_json(OUT / 'state' / 'source_split_manifest.json', manifest)
    records = []
    for fold in range(5):
        sup = support_indices('UCMerced', 4, 0, fold)
        old = TEST_UNLOCKED
        TEST_UNLOCKED = True
        try:
            test = test_indices('UCMerced', fold)
        finally:
            TEST_UNLOCKED = old
        assert not set(map(int, sup)) & set(map(int, test))
        index_record = dict(support=list(map(int, sup)), test=list(map(int, test)))
        ih = hashlib.sha256(json.dumps(index_record, sort_keys=True).encode()).hexdigest()
        atomic_json(OUT / 'runs' / f'indices_fold{fold}.json', index_record)
        for condition, bank in [('original', original), ('decontaminated', clean)]:
            ident = dict(dataset='UCMerced', K=4, fold=fold, support_seed=0, condition=condition, audit=coverage['fingerprint'], index_hash=ih, source_protocol=PROTOCOL_HASH, epochs=cfg.caption_epochs, steps_per_epoch=cfg.caption_steps_per_epoch)
            rh = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
            folder = OUT / 'runs' / f'f{fold}_{condition}_{rh}'
            folder.mkdir(exist_ok=True)
            result = folder / 'result.json'
            RUN_CHECKPOINT = folder / 'epoch.pt'
            if result.exists():
                records.append(json.loads(result.read_text()))
                continue
            if time.monotonic() - SESSION_START > SESSION_BUDGET_HOURS * 3600:
                status('rerun', 'PARTIAL', 'Süre sınırı; aynı modda tekrar çalıştırın')
                return
            was_resumed = RUN_CHECKPOINT.exists()
            atomic_json(folder / 'started.json', dict(time=time.time(), resumed=was_resumed))
            CAPTION_BANKS[ACTIVE_BACKBONE_KEY, 'RSITMD']['train'] = bank
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                model, info = train_capstone('UCMerced', 4, 0, fold, 'RSITMD', 'joint_full', 1.0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_s = time.perf_counter() - start
                start = time.perf_counter()
                old = TEST_UNLOCKED
                TEST_UNLOCKED = True
                try:
                    metrics = evaluate_capstone(model, 'UCMerced', test)
                finally:
                    TEST_UNLOCKED = old
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                eval_s = time.perf_counter() - start
                row = {**ident, **metrics, **info, 'train_s_this_session': train_s, 'eval_s': eval_s, 'resumed': was_resumed, 'hardware': HW_NAME, 'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None, 'optimizer_steps': cfg.caption_epochs * cfg.caption_steps_per_epoch, 'caption_pairs_drawn': cfg.caption_epochs * cfg.caption_steps_per_epoch * min(cfg.caption_batch_size, bank['n_img']), 'caption_images_removed': drops, 'timing_scope': 'cached-feature adaptation; excludes staging/encoding/audit; checkpoint IO included', 'python': platform.python_version(), 'torch': torch.__version__}
                atomic_json(result, row)
                records.append(row)
                del model
            except Exception as exc:
                atomic_json(folder / 'error.json', dict(error=repr(exc)))
                status('rerun', 'BLOCKED', repr(exc))
                return
            finally:
                CAPTION_BANKS[ACTIVE_BACKBONE_KEY, 'RSITMD']['train'] = original
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'tables' / 'matched_k4_raw.csv', index=False)
    assert len(df) == 10
    for metric in ['Accuracy', 'MacroF1', 'BalancedAccuracy', 'ECE', 'NLL']:
        p = df.pivot(index=['fold', 'support_seed', 'index_hash'], columns='condition', values=metric)
        p['clean_minus_original'] = p.decontaminated - p.original
        p.to_csv(OUT / 'tables' / f'matched_k4_{metric}.csv')
    df[['fold', 'condition', 'hardware', 'epochs', 'steps_per_epoch', 'optimizer_steps', 'caption_pairs_drawn', 'train_s_this_session', 'eval_s', 'resumed', 'timing_scope']].to_csv(OUT / 'tables' / 'runtime_budget.csv', index=False)
    status('rerun', 'COMPLETE', '10 eşleşmiş koşu; beş fold bağımsız veri kümesi sayılmaz. Farklar betimseldir.')
if READY and MODE == 'rerun':
    try:
        rerun_matched()
    except Exception as exc:
        status('rerun', 'BLOCKED', repr(exc))
        print(repr(exc))
PRIMARY_CSV = DRIVE_ROOT / 'outputs_capstone_disjoint5fold_support5_v1' / 'tables' / 'raw_primary_classification.csv'
WIDTH_CSV = DRIVE_ROOT / 'reviewer_revision_final_clean_20261006' / 'tables' / 'adapter_width_sweep_raw.csv'
if MODE == 'analysis':
    print('Primary adayları:', list(DRIVE_ROOT.glob('outputs_*/tables/raw_primary_classification.csv')))
    print('Width adayları:', list((DRIVE_ROOT / 'reviewer_revision_final_clean_20261006').glob('tables/*width*raw*.csv')))
METHODS = ['Zero-shot', 'LinearProbe', 'CLIP-Adapter', 'Tip-Adapter', 'Tip-Adapter-F', 'CoOp', 'TaskRes', 'RLA', 'ProLIP', 'Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP']
DATASETS = ['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT']

def complete_grid(df, axes, keys):
    import itertools
    assert not df.duplicated(keys).any(), 'Tekrarlanan sonuç anahtarı'
    expected = set(itertools.product(*(axes[k] for k in keys)))
    actual = set(df[keys].itertuples(index=False, name=None))
    assert actual == expected, f'Eksik/fazla ızgara: missing={len(expected - actual)}, extra={len(actual - expected)}'
    assert df.Accuracy.notna().all() and df.Accuracy.between(0, 1).all(), 'Accuracy birimi kaynak gibi 0–1 olmalı'

def friedman_sensitivity(df):
    import pandas as pd, numpy as np
    from scipy.stats import friedmanchisquare, rankdata
    axes = dict(dataset=DATASETS, K=[1, 2, 4, 8, 16], fold=list(range(5)), support_seed=list(range(5)), method=METHODS)
    q = df[df.method.isin(METHODS)].copy()
    if 'backbone' in q:
        q = q[q.backbone == 'openclip_b32']
    complete_grid(q, axes, list(axes))
    cell = q.groupby(['dataset', 'K', 'fold', 'method']).Accuracy.mean().groupby(['dataset', 'K', 'method']).mean().unstack('method').reindex(columns=METHODS)
    blocks = cell.groupby('dataset').mean().reindex(DATASETS)
    blocks.to_csv(OUT / 'tables' / 'friedman_dataset_blocks.csv')
    ranks = pd.DataFrame([rankdata(-r, method='average') for r in blocks.to_numpy()], index=blocks.index, columns=METHODS)
    ranks.to_csv(OUT / 'tables' / 'friedman_dataset_ranks.csv')
    results = []
    for name, b in [('dataset_blocks_sensitivity', blocks), ('dependent_dataset_shot_exploratory', cell)]:
        stat, p = friedmanchisquare(*(b[m] for m in METHODS))
        results.append(dict(analysis=name, n_blocks=len(b), statistic=float(stat), p_asymptotic=float(p), interpretation='exploratory; no independent 30-block inference'))
    rng = np.random.default_rng(20261005)
    obs = ranks.mean().var(ddof=0)
    hits = 0
    B = 20000
    a = ranks.to_numpy()
    for _ in range(B):
        perm = np.array([rng.permutation(row) for row in a])
        hits += perm.mean(0).var() >= obs - 1e-14
    results[0]['permutation_p'] = (hits + 1) / (B + 1)
    results[0]['permutations'] = B
    atomic_json(OUT / 'tables' / 'friedman_sensitivity.json', results)

def width_inference(df):
    import pandas as pd, numpy as np, itertools
    axes = dict(dataset=DATASETS, K=[1, 4, 16], fold=list(range(5)), support_seed=[0, 1], width=[32, 64, 128, 256, 512])
    q = df.copy()
    if 'backbone' in q:
        q = q[q.backbone == 'openclip_b32']
    complete_grid(q, axes, list(axes))
    p = q.pivot(index=['dataset', 'K', 'fold', 'support_seed'], columns='width', values='Accuracy')
    d = ((p[512] - p[128]) * 100).groupby('dataset').mean().reindex(DATASETS)
    signs = np.array(list(itertools.product([-1, 1], repeat=6)))
    pv = float((abs((signs * d.to_numpy()).mean(1)) >= abs(d.mean()) - 1e-12).mean())
    rng = np.random.default_rng(20261005)
    boot = rng.choice(d.to_numpy(), size=(20000, 6), replace=True).mean(1)
    d.to_csv(OUT / 'tables' / 'width_dataset_effects_pp.csv')
    atomic_json(OUT / 'tables' / 'width_inference.json', dict(delta_pp=d.mean(), dataset_bootstrap_ci=np.quantile(boot, [0.025, 0.975]).tolist(), exact_sign_flip_p=pv, n_datasets=6, claim='Observed effects within this adapter/backbone/protocol only. No equivalence test or universal saturation claim. Bootstrap uncertainty with six datasets is limited.'))
if MODE == 'analysis':
    import pandas as pd
    if not PRIMARY_CSV.exists():
        candidates = list(DRIVE_ROOT.glob('outputs_*/tables/raw_primary_classification.csv'))
        if len(candidates) == 1:
            PRIMARY_CSV = candidates[0]
    for name, path, func in [('friedman', PRIMARY_CSV, friedman_sensitivity), ('width', WIDTH_CSV, width_inference)]:
        try:
            func(pd.read_csv(path))
            print(name, 'tamamlandı')
        except Exception as exc:
            atomic_json(OUT / 'tables' / f'{name}_BLOCKED.json', dict(path=str(path), reason=repr(exc)))
            print(name, repr(exc))
if MODE == 'dry_run':
    import numpy as np
    qh = np.array([[0] * 8, [255] * 8], dtype=np.uint8)
    th = np.array([[0] * 8, [170] * 8], dtype=np.uint8)
    qf = np.array([[1.0, 0.0], [0.0, 1.0]])
    tf = np.array([[0.0, 1.0], [1.0, 0.0]])
    d, s, m = candidate_mask(qh, th, qf, tf)
    assert d[0, 0] == 0 and d[1, 1] == 32
    assert m[0, 0] and m[0, 1] and m[1, 0] and (not m[1, 1])
    assert len([(fold, seed, condition) for fold in range(5) for seed in [0] for condition in ['original', 'decontaminated']]) == 10
    status('dry_run', 'PASS', 'Sentetik eşik/ızgara kontrolü geçti. Gerçek veri/GPU/eğitim doğrulanmadı.')
    print('Dry-run geçti. Şimdi Colab GPU ortamında audit modunu seçin.')
