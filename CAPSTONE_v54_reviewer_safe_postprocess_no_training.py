from __future__ import annotations
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

@dataclass
class Config:
    search_roots: Tuple[Path, ...] = (Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'), Path('/content'), Path('/mnt/data'), Path.cwd())
    output_dir: Path = Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/CAPSTONE_CLIP_v54_REVIEWER_SAFE')
    compile_pdf: bool = True
    strict: bool = True
    random_seed: int = 20260717
    factor_datasets: Tuple[str, ...] = ('AID', 'EuroSAT', 'NWPU_RESISC45')
    factor_k: int = 1
    caption_corpus: str = 'RSICD'
    caption_scale: float = 1.0
    capacity_small_method: str = 'Shared-Caption'
    capacity_large_method: str = 'Capacity-Matched-Shared'
    candidate_method: str = 'CAPSTONE-CLIP'
    cluster_references: Tuple[str, ...] = ('APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared')
CFG = Config()
ZIP_PATTERNS: Dict[str, str] = {'canonical': 'CAPSTONE_CLIP_v53_CANONICAL*.zip', 'q1': 'outputs_q1_capstone_clip_v11_full_external*.zip', 'post': 'outputs_capstone_paper_postprocess_v1*.zip'}
DISPLAY_NAMES: Dict[str, str] = {'CAPSTONE-CLIP': 'CAPSTONE-CLIP-Full', 'Shared-Caption': 'CAPSTONE-CLIP-Shared', 'Capacity-Matched-Shared': 'CAPSTONE-CLIP-CM', 'LinearProbe': 'Linear Probe', 'APPLeNet-PM': 'APPLeNet-PM', 'AVION-PM': 'AVION-PM'}
KEYS = ['backbone', 'dataset', 'K', 'fold']

def log(message: str) -> None:
    print(f'[CAPSTONE-v54] {message}', flush=True)

def mount_google_drive_if_needed() -> None:
    if not Path('/content').exists() or Path('/content/drive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        log('Mounting Google Drive...')
        drive.mount('/content/drive')
    except Exception as exc:
        log(f'Drive mount skipped: {exc}')

def resolve_output_dir(cfg: Config) -> Path:
    try:
        cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
        probe = cfg.output_dir.parent / '.capstone_v54_write_test'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        return cfg.output_dir
    except Exception:
        fallback = Path('/content/CAPSTONE_CLIP_v54_REVIEWER_SAFE') if Path('/content').exists() else Path.cwd() / 'CAPSTONE_CLIP_v54_REVIEWER_SAFE'
        fallback.parent.mkdir(parents=True, exist_ok=True)
        log(f'Preferred output path unavailable; using {fallback}')
        return fallback

def discover_latest(pattern: str, roots: Sequence[Path], exclude_dirs: Sequence[Path]=()) -> Path:
    candidates: List[Path] = []
    excluded = [p.resolve() for p in exclude_dirs if p.exists()]
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in list(root.glob(pattern)) + list(root.glob(f'**/{pattern}')):
                if not p.is_file():
                    continue
                rp = p.resolve()
                if any((str(rp).startswith(str(e)) for e in excluded)):
                    continue
                candidates.append(rp)
        except Exception:
            continue
    candidates = sorted(set(candidates), key=lambda p: (p.stat().st_mtime_ns, p.stat().st_size), reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No file matched {pattern!r}. Searched: ' + ', '.join((str(r) for r in roots)))
    return candidates[0]

def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        base = destination.resolve()
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        zf.extractall(destination)

def find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f'Could not find {filename} below {root}')
    return matches[0]

def find_dir_containing(root: Path, filename: str) -> Path:
    return find_file(root, filename).parent

def read_csv(path: Path, required: Optional[Iterable[str]]=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if required:
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f'{path.name}: missing columns {missing}')
    return df

def fmt_pp(value: float, signed: bool=True, digits: int=2) -> str:
    return f'{100.0 * value:+.{digits}f}' if signed else f'{100.0 * value:.{digits}f}'

def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return '---'
    if value < 0.001:
        exponent = int(math.floor(math.log10(value))) if value > 0 else -999
        coefficient = value / 10 ** exponent if value > 0 else 0.0
        return f'${coefficient:.2f}\\times10^{{{exponent}}}$'
    return f'{value:.3f}'

def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(source_dir.rglob('*')):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(source_dir))

def replace_environment_by_label(tex: str, label: str, new_block: str) -> str:
    marker = f'\\label{{{label}}}'
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f'LaTeX label not found: {label}')
    start_candidates = [tex.rfind('\\begin{table', 0, idx), tex.rfind('\\begin{figure', 0, idx)]
    start = max(start_candidates)
    if start < 0:
        raise ValueError(f'Could not locate environment start for {label}')
    begin_line_end = tex.find('}', start) + 1
    begin_text = tex[start:begin_line_end]
    if 'table*' in begin_text:
        end_marker = '\\end{table*}'
    elif 'table' in begin_text:
        end_marker = '\\end{table}'
    elif 'figure*' in begin_text:
        end_marker = '\\end{figure*}'
    else:
        end_marker = '\\end{figure}'
    end = tex.find(end_marker, idx)
    if end < 0:
        raise ValueError(f'Could not locate environment end for {label}')
    end += len(end_marker)
    return tex[:start] + new_block + tex[end:]

def insert_after_environment_label(tex: str, label: str, block: str) -> str:
    marker = f'\\label{{{label}}}'
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f'LaTeX label not found: {label}')
    starts = [tex.rfind('\\begin{table', 0, idx), tex.rfind('\\begin{figure', 0, idx)]
    start = max(starts)
    begin_line_end = tex.find('}', start) + 1
    begin_text = tex[start:begin_line_end]
    if 'table*' in begin_text:
        end_marker = '\\end{table*}'
    elif 'table' in begin_text:
        end_marker = '\\end{table}'
    elif 'figure*' in begin_text:
        end_marker = '\\end{figure*}'
    else:
        end_marker = '\\end{figure}'
    end = tex.find(end_marker, idx)
    if end < 0:
        raise ValueError(f'Could not locate environment end for {label}')
    end += len(end_marker)
    return tex[:end] + '\n\n' + block + tex[end:]

def replace_between_regex(tex: str, start_pattern: str, end_pattern: str, replacement: str) -> str:
    pattern = re.compile(start_pattern + '.*?' + end_pattern, flags=re.S)
    new_tex, count = pattern.subn(lambda _m: replacement, tex, count=1)
    if count != 1:
        raise ValueError(f'Expected one replacement between {start_pattern!r} and {end_pattern!r}; got {count}')
    return new_tex

def compute_factor_decomposition(raw_controls: pd.DataFrame, overall: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    required_controls = {'no_caption', 'image_only', 'matched_budget', 'joint_full', 'shuffled_caption'}
    subset = raw_controls[raw_controls['dataset'].isin(cfg.factor_datasets) & (raw_controls['K'] == cfg.factor_k) & (raw_controls['caption_corpus'] == cfg.caption_corpus) & np.isclose(raw_controls['caption_scale'].astype(float), cfg.caption_scale) & raw_controls['control'].isin(required_controls)].copy()
    present = set(subset['control'].unique())
    missing = sorted(required_controls - present)
    if missing:
        raise ValueError(f'Factor decomposition controls missing: {missing}')
    dataset_means = subset.groupby(['dataset', 'control'], as_index=False)['Accuracy'].mean().pivot(index='dataset', columns='control', values='Accuracy').reindex(cfg.factor_datasets)
    if dataset_means.isna().any().any():
        raise ValueError('Incomplete factor-control grid after filtering')
    means = dataset_means.mean(axis=0)
    no_caption = float(means['no_caption'])
    image_only = float(means['image_only'])
    matched_budget = float(means['matched_budget'])
    joint_full = float(means['joint_full'])
    shuffled = float(means['shuffled_caption'])
    exposure = image_only - no_caption
    pairing_fixed_budget = matched_budget - image_only
    matched_scale = joint_full - matched_budget
    matched_total = joint_full - image_only
    total_gain = joint_full - no_caption
    shuffled_penalty = shuffled - no_caption
    lookup = overall.set_index('method')['macro_average_Accuracy']
    capacity = float(lookup[cfg.capacity_large_method] - lookup[cfg.capacity_small_method])
    values = {'no_caption': no_caption, 'image_only': image_only, 'matched_budget': matched_budget, 'joint_full': joint_full, 'shuffled_caption': shuffled, 'image_exposure': exposure, 'correct_pairing_fixed_budget': pairing_fixed_budget, 'matched_corpus_scale': matched_scale, 'matched_semantics_total': matched_total, 'total_joint_gain': total_gain, 'shuffled_penalty_vs_no_caption': shuffled_penalty, 'capacity_increment': capacity}
    if not np.isclose(exposure + pairing_fixed_budget + matched_scale, total_gain, atol=1e-12):
        raise AssertionError('Additive factor terms do not sum to total gain')
    rows = [{'component': 'Image exposure', 'definition': 'image-only - no-caption', 'contribution': exposure, 'share_of_total': exposure / total_gain, 'analysis_axis': 'K=1 three-dataset factor control'}, {'component': 'Correct pairing at fixed budget', 'definition': 'matched-budget - image-only', 'contribution': pairing_fixed_budget, 'share_of_total': pairing_fixed_budget / total_gain, 'analysis_axis': 'K=1 three-dataset factor control'}, {'component': 'Scale of correctly matched pairs', 'definition': 'joint-full - matched-budget', 'contribution': matched_scale, 'share_of_total': matched_scale / total_gain, 'analysis_axis': 'K=1 three-dataset factor control'}, {'component': 'Matched semantics total', 'definition': 'joint-full - image-only', 'contribution': matched_total, 'share_of_total': matched_total / total_gain, 'analysis_axis': 'derived subtotal; not an additional additive term'}, {'component': 'Adapter capacity', 'definition': 'CAPSTONE-CLIP-CM - CAPSTONE-CLIP-Shared', 'contribution': capacity, 'share_of_total': capacity / total_gain, 'analysis_axis': 'six-dataset overall benchmark; separate axis'}]
    return (pd.DataFrame(rows), values)

def make_factor_figure(values: Mapping[str, float], output_png: Path, output_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    components = [values['image_exposure'], values['correct_pairing_fixed_budget'], values['matched_corpus_scale']]
    labels = ['Image exposure', 'Correct pairing\nat fixed budget', 'Scale of correctly\nmatched pairs']
    total = values['total_joint_gain']
    shares = [x / total for x in components]
    fig = plt.figure(figsize=(10.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.2], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(components))
    bars = ax.bar(x, np.array(components) * 100.0)
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Incremental accuracy contribution (percentage points)')
    ax.set_title(f'Additive decomposition of the K=1 gain (total = {100 * total:.2f} points)')
    upper = max(np.array(components) * 100.0) * 1.28
    ax.set_ylim(0, upper)
    for bar, value, share in zip(bars, components, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + upper * 0.025, f'+{100 * value:.2f} pts\n({100 * share:.1f}% of total)', ha='center', va='bottom', fontsize=9.5)
    ax.text(0.99, 0.98, f"Matched semantics together: +{100 * values['matched_semantics_total']:.2f} pts ({100 * values['matched_semantics_total'] / total:.1f}%)", transform=ax.transAxes, ha='right', va='top', fontsize=9.5, bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.85})
    ax2 = fig.add_subplot(gs[0, 1])
    cap_pp = 100.0 * values['capacity_increment']
    bar2 = ax2.bar([0], [cap_pp])
    ax2.axhline(0, linewidth=0.8)
    ax2.set_xticks([0], ['265K to 528K\nparameters'])
    ax2.set_ylabel('Accuracy difference (points)')
    ax2.set_title('Separate capacity contrast')
    span = max(0.12, abs(cap_pp) * 3.5)
    ax2.set_ylim(-span * 0.35, span)
    ax2.text(bar2[0].get_x() + bar2[0].get_width() / 2, cap_pp + span * 0.05, f'{cap_pp:+.2f} pts', ha='center', va='bottom', fontsize=10)
    ax2.text(0.5, 0.04, 'Capacity is shown separately because it is\ncomputed on the six-dataset overall benchmark.', transform=ax2.transAxes, ha='center', va='bottom', fontsize=8.7)
    fig.suptitle('Controlled contribution analysis without double counting', fontsize=13.5, fontweight='bold')
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def exact_sign_flip_p(dataset_effects: np.ndarray) -> float:
    values = np.asarray(dataset_effects, dtype=float)
    if len(values) == 0:
        return float('nan')
    observed = abs(float(values.mean()))
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        stats.append(abs(float(np.mean(values * np.asarray(signs)))))
    stats_arr = np.asarray(stats)
    return float(np.mean(stats_arr >= observed - 1e-15))

def exact_dataset_bootstrap_ci(dataset_effects: np.ndarray, alpha: float=0.05) -> Tuple[float, float]:
    values = np.asarray(dataset_effects, dtype=float)
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'))
    if n ** n <= 250000:
        means = np.fromiter((float(values[list(index_tuple)].mean()) for index_tuple in itertools.product(range(n), repeat=n)), dtype=float, count=n ** n)
    else:
        rng = np.random.default_rng(20260717)
        means = np.asarray([rng.choice(values, size=n, replace=True).mean() for _ in range(100000)])
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(low), float(high))

def compute_cluster_sensitivity(fold_table: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate = fold_table[fold_table['method'] == cfg.candidate_method][KEYS + ['Accuracy']].rename(columns={'Accuracy': 'candidate_accuracy'})
    rows: List[Dict[str, object]] = []
    lodo_rows: List[Dict[str, object]] = []
    for reference in cfg.cluster_references:
        ref = fold_table[fold_table['method'] == reference][KEYS + ['Accuracy']].rename(columns={'Accuracy': 'reference_accuracy'})
        paired = candidate.merge(ref, on=KEYS, how='inner', validate='one_to_one')
        paired['difference'] = paired['candidate_accuracy'] - paired['reference_accuracy']
        dataset_effects = paired.groupby('dataset', sort=True)['difference'].mean()
        values = dataset_effects.to_numpy(float)
        if len(values) < 2:
            raise ValueError(f'Insufficient dataset clusters for {reference}')
        mean_delta = float(values.mean())
        ci_low, ci_high = exact_dataset_bootstrap_ci(values)
        p_exact = exact_sign_flip_p(values)
        loo = []
        for dataset in dataset_effects.index:
            retained = dataset_effects.drop(dataset)
            loo_delta = float(retained.mean())
            loo.append(loo_delta)
            lodo_rows.append({'reference': reference, 'omitted_dataset': dataset, 'mean_delta_without_dataset': loo_delta, 'mean_delta_without_dataset_pp': 100.0 * loo_delta})
        rows.append({'candidate': cfg.candidate_method, 'reference': reference, 'display_candidate': DISPLAY_NAMES.get(cfg.candidate_method, cfg.candidate_method), 'display_reference': DISPLAY_NAMES.get(reference, reference), 'n_datasets': len(values), 'n_paired_blocks': len(paired), 'mean_delta': mean_delta, 'mean_delta_pp': 100.0 * mean_delta, 'dataset_cluster_ci95_low': ci_low, 'dataset_cluster_ci95_high': ci_high, 'dataset_cluster_ci95_low_pp': 100.0 * ci_low, 'dataset_cluster_ci95_high_pp': 100.0 * ci_high, 'exact_sign_flip_p': p_exact, 'lodo_min': float(np.min(loo)), 'lodo_max': float(np.max(loo)), 'lodo_min_pp': 100.0 * float(np.min(loo)), 'lodo_max_pp': 100.0 * float(np.max(loo)), 'all_dataset_effects_positive': bool(np.all(values > 0)), 'dataset_effects_json': json.dumps({str(k): float(v) for k, v in dataset_effects.items()}, sort_keys=True)})
    return (pd.DataFrame(rows), pd.DataFrame(lodo_rows))

def make_cluster_sensitivity_table(cluster: pd.DataFrame) -> str:
    order = ['APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared']
    df = cluster.set_index('reference').loc[order].reset_index()
    lines: List[str] = []
    for _, r in df.iterrows():
        ref = DISPLAY_NAMES.get(str(r['reference']), str(r['reference']))
        delta = float(r['mean_delta_pp'])
        lo = float(r['dataset_cluster_ci95_low_pp'])
        hi = float(r['dataset_cluster_ci95_high_pp'])
        p = fmt_p(float(r['exact_sign_flip_p']))
        lmin = float(r['lodo_min_pp'])
        lmax = float(r['lodo_max_pp'])
        if lo <= 0 <= hi:
            reading = 'includes zero'
        elif delta > 0:
            reading = 'positive across cluster interval'
        else:
            reading = 'reference favored'
        lines.append(f'{ref} & ${delta:+.2f}$ & $[{lo:+.2f},\\,{hi:+.2f}]$ & {p} & $[{lmin:+.2f},\\,{lmax:+.2f}]$ & {reading}\\\\')
    return textwrap.dedent(f'\n        \\begin{{table*}}[!t]\n        \\caption{{Dataset-cluster sensitivity analysis for accuracy. Each dataset contributes one equally weighted effect. The interval is the exact ordered dataset bootstrap over the six dataset-level effects; the $p$ value is an exact two-sided sign-flip test over datasets. LODO gives the leave-one-dataset-out range. This analysis is conservative and has coarse resolution with only six datasets.\\label{{tab:cluster_sensitivity}}}}\n        \\centering\n        \\scriptsize\n        \\setlength{{\\tabcolsep}}{{3.5pt}}\n        \\begin{{tabular}}{{lccccc}}\n        \\toprule\n        \\textbf{{Reference}} & \\textbf{{Mean $\\Delta$}} & \\textbf{{Dataset-cluster 95\\% CI}} & \\textbf{{Exact $p$}} & \\textbf{{LODO range}} & \\textbf{{Reading}}\\\\\n        \\midrule\n        {os.linesep.join(lines)}\n        \\bottomrule\n        \\end{{tabular}}\n        \\end{{table*}}\n        ').strip()

def make_tradeoff_figure(overall: pd.DataFrame, efficiency: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    df = overall.merge(efficiency[['method', 'train_eval_s']], on='method', how='left', validate='one_to_one')
    df['display'] = df['method'].map(DISPLAY_NAMES).fillna(df['method'])
    df['accuracy_pp'] = 100.0 * df['macro_average_Accuracy']
    df['ece_pp'] = 100.0 * df['macro_average_ECE']
    df = df[df['method'] != 'HOSO-Adapter'].copy()
    order = ['CAPSTONE-CLIP', 'Shared-Caption', 'Capacity-Matched-Shared', 'APPLeNet-PM', 'CLIP-Adapter', 'ProLIP', 'TaskRes', 'RLA', 'CoOp', 'AVION-PM', 'LinearProbe', 'Tip-Adapter-F', 'Tip-Adapter', 'Zero-shot']
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7))
    ax1, ax2 = axes
    marker_cycle = ['o', 's', 'D', '^', 'v', 'P', 'X', '<', '>', 'h', '*', 'd', 'p', '8']
    handles = []
    labels = []
    for marker, (_, r) in zip(marker_cycle, df.iterrows()):
        label = str(r['display'])
        handle = ax1.scatter(float(r['ece_pp']), float(r['accuracy_pp']), marker=marker, s=58)
        runtime = float(r['train_eval_s'])
        if np.isfinite(runtime) and runtime > 0:
            ax2.scatter(runtime, float(r['accuracy_pp']), marker=marker, s=58)
        handles.append(handle)
        labels.append(label)
    ax1.set_xlabel('ECE (%) - lower is better')
    ax1.set_ylabel('Accuracy (%) - higher is better')
    ax1.set_title('Accuracy-calibration trade-off')
    ax1.grid(True, alpha=0.25)
    ax2.set_xscale('log')
    ax2.set_xlabel('Mean train + evaluation time per run (s, log scale)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy-training-cost trade-off')
    ax2.grid(True, alpha=0.25)
    fig.suptitle('Accuracy, calibration, and run-level cost', fontsize=13.5, fontweight='bold')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=8.2, frameon=True, columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.19, 1, 0.94])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def make_hypothesis_figure(output_png: Path, output_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.8, 4.1))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis('off')
    cards = [(0.08, 'H1 - Semantic supervision', 'Matched image-caption supervision\ndrives few-shot gains; the gain\ngrows as K decreases', 'SUPPORTED'), (1.04, 'H2 - Image exposure', 'Additional image exposure alone\ndoes not yield gains comparable\nto matched semantic supervision', 'REJECTED'), (2.0, 'H3 - Adapter capacity', 'Increasing capacity from 265K to\n528K parameters yields no systematic\nclassification gain', 'NOT SUPPORTED')]
    for x, title, statement, verdict in cards:
        top = FancyBboxPatch((x, 0.68), 0.88, 0.2, boxstyle='round,pad=0.02', linewidth=1.2, facecolor='white')
        bottom = FancyBboxPatch((x + 0.14, 0.13), 0.6, 0.18, boxstyle='round,pad=0.02', linewidth=1.2, facecolor='white')
        ax.add_patch(top)
        ax.add_patch(bottom)
        ax.text(x + 0.44, 0.78, title, ha='center', va='center', fontsize=10.8, fontweight='bold')
        ax.text(x + 0.44, 0.5, statement, ha='center', va='center', fontsize=9.1)
        ax.annotate('', xy=(x + 0.44, 0.32), xytext=(x + 0.44, 0.4), arrowprops={'arrowstyle': '-|>', 'lw': 1.2})
        ax.text(x + 0.44, 0.22, verdict, ha='center', va='center', fontsize=11.0, fontweight='bold')
    fig.suptitle('Hypothesis summary under the leakage-controlled protocol', fontsize=13.2, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def tradeoff_latex() -> str:
    return textwrap.dedent('\n        \\begin{figure*}[!t]\n        \\centering\n        \\includegraphics[width=0.96\\textwidth]{figs/fig17_tradeoffs_v54.png}\n        \\caption{Accuracy--calibration and accuracy--training-cost trade-offs. CAPSTONE-CLIP-Full has the highest mean accuracy among the displayed methods, but APPLeNet-PM and CAPSTONE-CLIP-Shared are better calibrated. The CAPSTONE-CLIP shared-only configurations attain essentially the same classification accuracy as the full shared--private configuration at roughly half its run-level training and evaluation cost. Times are measured in the frozen-feature implementations used in this study and are not official-repository benchmarks.}\n        \\label{fig:tradeoffs}\n        \\end{figure*}\n        ').strip()

def standardize_names(tex: str) -> str:
    tex = tex.replace('Capacity-Matched-Shared', 'CAPSTONE-CLIP-CM')
    tex = tex.replace('Shared-Caption', 'CAPSTONE-CLIP-Shared')
    replacements = {"CAPSTONE-CLIP's": "CAPSTONE-CLIP-Full's", 'CAPSTONE-CLIP differs': 'CAPSTONE-CLIP-Full differs', 'CAPSTONE-CLIP exceeds': 'CAPSTONE-CLIP-Full exceeds', 'CAPSTONE-CLIP records': 'CAPSTONE-CLIP-Full records', 'CAPSTONE-CLIP improves': 'CAPSTONE-CLIP-Full improves', 'CAPSTONE-CLIP averages': 'CAPSTONE-CLIP-Full averages', 'CAPSTONE-CLIP remains': 'CAPSTONE-CLIP-Full remains', 'CAPSTONE-CLIP reaches': 'CAPSTONE-CLIP-Full reaches', 'CAPSTONE-CLIP and the': 'CAPSTONE-CLIP-Full and the'}
    for old, new in replacements.items():
        tex = tex.replace(old, new)
    tex = re.sub('(?m)^CAPSTONE-CLIP\\s*&', 'CAPSTONE-CLIP-Full &', tex)
    return tex

def patch_manuscript(tex: str, values: Mapping[str, float], cluster: pd.DataFrame, cluster_table: str) -> Tuple[str, List[str]]:
    changes: List[str] = []
    tex = standardize_names(tex)
    changes.append('Standardized visible CAPSTONE-CLIP family names')
    app = cluster.loc[cluster['reference'] == 'APPLeNet-PM'].iloc[0]
    prolip = cluster.loc[cluster['reference'] == 'ProLIP'].iloc[0]
    rla = cluster.loc[cluster['reference'] == 'RLA'].iloc[0]
    avion = cluster.loc[cluster['reference'] == 'AVION-PM'].iloc[0]
    app_clause = f"Against APPLeNet-PM, a protocol-matched reimplementation, the mean difference is {float(app['mean_delta_pp']):+.2f} points and dataset dependent; a conservative dataset-cluster sensitivity interval includes zero (95\\% CI [{float(app['dataset_cluster_ci95_low_pp']):+.2f}, {float(app['dataset_cluster_ci95_high_pp']):+.2f}], exact sign-flip $p={float(app['exact_sign_flip_p']):.3f}$)."
    tex, n = re.subn('Against APPLeNet-PM, a protocol-matched reimplementation,.*?dataset-dependent wins and losses;', lambda _m: app_clause + ' ', tex, count=1, flags=re.S)
    if n:
        changes.append('Replaced abstract APPLeNet claim with dataset-cluster sensitivity wording')
    h1 = f"\nCaption-supervised shared adaptation reaches 88.11\\% overall (Table~\\ref{{tab:overall}}), exceeding ProLIP by {float(prolip['mean_delta_pp']):.2f} points and RLA by {float(rla['mean_delta_pp']):.2f} points under the common leakage-controlled protocol. CAPSTONE-CLIP-Full exceeds AVION-PM by {float(avion['mean_delta_pp']):.2f} points. Against APPLeNet-PM, however, the mean difference is only {float(app['mean_delta_pp']):+.2f} points and is heterogeneous across datasets: gains are concentrated on AID and WHU-RS19, PatternNet favors APPLeNet-PM, and three datasets are close to zero (Table~\\ref{{tab:pm_dataset}}). The pre-specified block-level analysis yields the result reported in Table~\\ref{{tab:confirmatory}}, but a conservative sensitivity analysis that treats the six datasets as the inferential clusters gives a 95\\% interval of [{float(app['dataset_cluster_ci95_low_pp']):+.2f}, {float(app['dataset_cluster_ci95_high_pp']):+.2f}] points and an exact sign-flip $p={float(app['exact_sign_flip_p']):.3f}$ (Table~\\ref{{tab:cluster_sensitivity}}). Omitting AID reduces the mean difference to {float(app['lodo_min_pp']):+.2f} points, demonstrating that the small aggregate advantage is not robust evidence of architectural superiority. H1's distinguishing supervision prediction remains supported by the larger, monotone low-shot margins over class-name-only validation-free baselines; no universal advantage over APPLeNet is claimed. \\textbf{{Verdict: H1 supported for matched semantic supervision; architectural superiority over APPLeNet-PM is not established.}}\n".strip()
    try:
        tex = replace_between_regex(tex, 'Caption-supervised shared adaptation reaches', '\\\\textbf\\{Verdict: H1 supported[^}]*\\}', h1)
        changes.append('Replaced H1 results paragraph with reviewer-safe inference language')
    except ValueError:
        tex = replace_between_regex(tex, 'Caption-supervised shared adaptation reaches', '\\\\subsection\\{H2', h1 + '\n\n\\subsection{H2')
        changes.append('Replaced H1 results paragraph using subsection fallback')
    tex = tex.replace('Global paired accuracy comparisons. Differences are', 'Pre-specified block-level paired accuracy comparisons. Differences are', 1)
    tex = tex.replace('95\\% intervals use dataset--shot/fold cluster bootstrap, and Wilcoxon tests are Holm-adjusted', '95\\% intervals use the original dataset--shot/fold bootstrap, and Wilcoxon tests are Holm-adjusted', 1)
    tex = re.sub('(APPLeNet-PM\\$\\^\\{\\\\ddagger\\}\\$\\s*&\\s*\\$\\+0\\.70\\$.*?&\\s*)equivalent(\\\\\\\\)', '\\1small; cluster sensitivity includes zero\\2', tex, count=1)
    if '\\label{tab:cluster_sensitivity}' not in tex:
        tex = insert_after_environment_label(tex, 'tab:confirmatory', cluster_table)
        changes.append('Inserted dataset-cluster sensitivity table')
    stats_sentence = ' The block-level Wilcoxon family is retained as the pre-specified analysis. Because folds and shot levels are nested within only six datasets, we additionally report an exact dataset-level sign-flip test, an exact dataset bootstrap interval, and leave-one-dataset-out sensitivity for the key comparisons (Table~\\ref{tab:cluster_sensitivity}). The dataset-cluster analysis is conservative and has coarse resolution with six clusters; it is used to assess robustness rather than to replace the descriptive block-level effect estimates.'
    anchor = 'Exploratory dataset-level breakdowns are reported separately from the global confirmatory family.'
    if anchor in tex and 'exact dataset-level sign-flip' not in tex:
        tex = tex.replace(anchor, anchor + stats_sentence, 1)
        changes.append('Added dataset-cluster sensitivity to statistical validation')
    h3 = '\nH3 asks whether increasing the shared-adapter capacity from approximately 265K to 528K trainable parameters yields a systematic classification gain when semantic supervision, image budget, schedule, and optimization settings are held fixed. It does not: CAPSTONE-CLIP-CM differs from CAPSTONE-CLIP-Shared by only +0.03 points overall, and CAPSTONE-CLIP-Full is practically equivalent to both controls. This result supports a narrow conclusion about the evaluated two-point capacity contrast, not the universal claim that capacity can never matter. Small retrieval or dataset-specific differences remain exploratory and do not rescue the confirmatory capacity prediction. \\textbf{Verdict: H3 not supported for the evaluated 265K-to-528K capacity increase.}\n'.strip()
    try:
        tex = replace_between_regex(tex, 'H3 asks whether additional trainable capacity', '\\\\subsection\\{Cross-Modal Retrieval', h3 + '\n\n\\subsection{Cross-Modal Retrieval')
        changes.append('Narrowed H3 to the evaluated 265K-to-528K contrast')
    except ValueError:
        pass
    tex = re.sub('H3 --- Added capacity yields further gains under fixed supervision\\s*&\\s*Partially supported\\s*&.*?\\\\\\\\', lambda _m: 'H3 --- Increasing capacity from 265K to 528K yields further gains under fixed supervision & Not supported & CAPSTONE-CLIP-Shared, CAPSTONE-CLIP-CM, and CAPSTONE-CLIP-Full are practically equivalent for classification\\\\', tex, count=1, flags=re.S)
    tex = tex.replace('H3 partially supported', 'H3 not supported')
    tex = tex.replace('H3-partially-supported', 'H3-not-supported')
    tex = tex.replace('H3 partially-supported', 'H3 not supported')
    tex = tex.replace('partially supports H3', 'does not support H3')
    total_pp = 100.0 * values['total_joint_gain']
    exposure_pp = 100.0 * values['image_exposure']
    fixed_pair_pp = 100.0 * values['correct_pairing_fixed_budget']
    scale_pp = 100.0 * values['matched_corpus_scale']
    matched_pp = 100.0 * values['matched_semantics_total']
    capacity_pp = 100.0 * values['capacity_increment']
    factor_paragraph = f"\nFig.~\\ref{{fig:contrib}} reports an additive decomposition that avoids double counting. Of the {total_pp:.2f}-point $K{{=}}1$ gain of joint-full over no-caption (three-dataset mean), image exposure contributes {exposure_pp:+.2f} points ({100 * values['image_exposure'] / values['total_joint_gain']:.1f}\\%), correct image--caption pairing at a support-sized fixed budget contributes an additional {fixed_pair_pp:+.2f} points ({100 * values['correct_pairing_fixed_budget'] / values['total_joint_gain']:.1f}\\%), and scaling correctly matched pairs from the support-sized budget to the full corpus contributes {scale_pp:+.2f} points ({100 * values['matched_corpus_scale'] / values['total_joint_gain']:.1f}\\%). The two semantic terms together contribute {matched_pp:+.2f} points ({100 * values['matched_semantics_total'] / values['total_joint_gain']:.1f}\\%). The separate six-dataset capacity contrast is only {capacity_pp:+.2f} points. Fig.~\\ref{{fig:verdicts}} and Table~\\ref{{tab:hypotheses}} state the corresponding hypothesis verdicts.\n".strip()
    tex = re.sub('Fig\\.~\\\\ref\\{fig:contrib\\} condenses.*?Fig\\.~\\\\ref\\{fig:verdicts\\} and Table~\\\\ref\\{tab:hypotheses\\} state the corresponding hypothesis verdicts\\.', lambda _m: factor_paragraph, tex, count=1, flags=re.S)
    changes.append('Corrected additive Figure 18 interpretation')
    factor_env = textwrap.dedent('\n        \\begin{figure}[!t]\n        \\centering\n        \\includegraphics[width=0.99\\columnwidth]{figs/fig18_factor_decomposition_v54.png}\n        \\caption{Controlled contribution analysis without double counting. The three bars in the left panel are incremental terms and sum exactly to the joint-full minus no-caption gain at $K{=}1$ over AID, EuroSAT, and NWPU-RESISC45. Correct pairing at fixed budget and the scale of correctly matched pairs together constitute the matched-semantic contribution. The adapter-capacity contrast is shown on a separate axis because it is computed from the six-dataset overall benchmark. Percentages are protocol-specific, not universal constants.}\n        \\label{fig:contrib}\n        \\end{figure}\n        ').strip()
    tex = replace_environment_by_label(tex, 'fig:contrib', factor_env)
    changes.append('Replaced Figure 18 environment')
    if '\\label{fig:tradeoffs}' not in tex:
        marker = 'Fig.~\\ref{fig:contrib}'
        idx = tex.find(marker)
        if idx >= 0:
            tex = tex[:idx] + tradeoff_latex() + '\n\n' + tex[idx:]
            changes.append('Inserted non-overlapping trade-off figure')
    tex = tex.replace('Matched pairing accounts for $\\approx79\\%$ of the controlled $K{=}1$ gain', 'Matched semantics account for $\\approx88\\%$ of the controlled $K{=}1$ gain; the full-corpus scale term alone accounts for $\\approx79\\%$')
    tex = tex.replace('Matched pairing accounts for ${\\approx}79\\%$ of the controlled $K{=}1$ gain', 'Matched semantics account for ${\\approx}88\\%$ of the controlled $K{=}1$ gain; the full-corpus scale term alone accounts for ${\\approx}79\\%$')
    tex = tex.replace('Matched pairing accounts for ≈79% of the controlled K=1 gain', 'Matched semantics account for ≈88% of the controlled K=1 gain; the full-corpus scale term alone accounts for ≈79%')
    tex = re.sub('while the advantage over APPLeNet-PM is modest \\(0\\.70 points overall\\) and dataset dependent\\.', 'while the mean difference from APPLeNet-PM is modest (+0.70 points), dataset dependent, and includes zero in the conservative dataset-cluster sensitivity analysis.', tex, count=1)
    tex = tex.replace('CAPSTONE-CLIP-Full and the CAPSTONE-CLIP-Shared and CAPSTONE-CLIP-CM controls', 'The three CAPSTONE-CLIP configurations')
    tex = re.sub('\\n\\s*\\.\\s*\\n(\\\\appendix|\\\\section\\{APPENDIX)', '\\n\\1', tex)
    return (tex, changes)

def compile_latex(package_dir: Path, tex_name: str) -> Tuple[bool, str]:
    pdflatex = shutil.which('pdflatex')
    if not pdflatex:
        return (False, 'pdflatex not available; compilation skipped')
    logs: List[str] = []
    for _ in range(2):
        proc = subprocess.run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', tex_name], cwd=package_dir, text=True, capture_output=True)
        logs.append(proc.stdout + '\n' + proc.stderr)
        if proc.returncode != 0:
            (package_dir / 'compile_error_v54.log').write_text('\n\n'.join(logs), encoding='utf-8')
            return (False, 'LaTeX compilation failed; see compile_error_v54.log')
    pdf = package_dir / tex_name.replace('.tex', '.pdf')
    return (pdf.exists(), 'LaTeX compiled successfully')

def audit_outputs(tex: str, factor_table: pd.DataFrame, values: Mapping[str, float], cluster: pd.DataFrame, package_dir: Path) -> pd.DataFrame:
    checks: List[Tuple[str, bool, str]] = []
    additive_sum = float(values['image_exposure'] + values['correct_pairing_fixed_budget'] + values['matched_corpus_scale'])
    checks.append(('Factor components sum to total', bool(np.isclose(additive_sum, values['total_joint_gain'], atol=1e-12)), f"sum={100 * additive_sum:.6f} pp; total={100 * values['total_joint_gain']:.6f} pp"))
    checks.append(('No old Shared-Caption visible name', 'Shared-Caption' not in tex, f"count={tex.count('Shared-Caption')}"))
    checks.append(('No old Capacity-Matched-Shared visible name', 'Capacity-Matched-Shared' not in tex, f"count={tex.count('Capacity-Matched-Shared')}"))
    checks.append(('H3 no longer partially supported', 'partially supported' not in tex.lower() and 'partially-supported' not in tex.lower(), 'searched manuscript text'))
    checks.append(('Dataset-cluster table inserted', '\\label{tab:cluster_sensitivity}' in tex, 'tab:cluster_sensitivity'))
    checks.append(('Corrected factor figure inserted', 'fig18_factor_decomposition_v54.png' in tex and (package_dir / 'figs' / 'fig18_factor_decomposition_v54.png').exists(), 'fig18_factor_decomposition_v54.png'))
    checks.append(('Trade-off figure generated', (package_dir / 'figs' / 'fig17_tradeoffs_v54.png').exists(), 'fig17_tradeoffs_v54.png'))
    checks.append(('Hypothesis figure updated', (package_dir / 'figs' / 'fig5_hypotheses.png').exists(), 'fig5_hypotheses.png regenerated with H3 NOT SUPPORTED'))
    app = cluster.loc[cluster['reference'] == 'APPLeNet-PM'].iloc[0]
    checks.append(('APPLeNet cluster interval includes zero', float(app['dataset_cluster_ci95_low']) <= 0 <= float(app['dataset_cluster_ci95_high']), f"CI=[{app['dataset_cluster_ci95_low_pp']:.3f}, {app['dataset_cluster_ci95_high_pp']:.3f}] pp"))
    checks.append(('Factor table has three additive components', int((factor_table['analysis_axis'] == 'K=1 three-dataset factor control').sum()) == 3, f'rows={len(factor_table)}'))
    return pd.DataFrame(checks, columns=['check', 'pass', 'evidence'])

def main(cfg: Config=CFG) -> Path:
    mount_google_drive_if_needed()
    output_dir = resolve_output_dir(cfg)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / '_inputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    log('Discovering existing ZIP artifacts...')
    zip_paths = {name: discover_latest(pattern, cfg.search_roots, exclude_dirs=(output_dir,)) for name, pattern in ZIP_PATTERNS.items()}
    for name, path in zip_paths.items():
        log(f'{name}: {path}')
    extracted: Dict[str, Path] = {}
    for name, path in zip_paths.items():
        dest = input_dir / name
        safe_extract(path, dest)
        extracted[name] = dest
    canonical_package = find_dir_containing(extracted['canonical'], 'main_v53.tex')
    q1_root = find_dir_containing(extracted['q1'], 'raw_caption_controls.csv')
    post_root = find_dir_containing(extracted['post'], 'table_fold_level_seed_averaged.csv')
    package_dir = output_dir / 'CAPSTONE_CLIP_v54_LaTeX'
    shutil.copytree(canonical_package, package_dir)
    raw_controls = read_csv(q1_root / 'raw_caption_controls.csv', required=['dataset', 'K', 'control', 'caption_scale', 'caption_corpus', 'Accuracy'])
    overall = read_csv(post_root / 'paper_table_overall_macro_averages.csv', required=['method', 'macro_average_Accuracy', 'macro_average_ECE'])
    efficiency = read_csv(post_root / 'paper_table_efficiency_from_runs.csv', required=['method', 'train_eval_s', 'Accuracy'])
    fold_table = read_csv(post_root / 'table_fold_level_seed_averaged.csv', required=KEYS + ['method', 'Accuracy'])
    log('Computing corrected additive factor decomposition...')
    factor_table, factor_values = compute_factor_decomposition(raw_controls, overall, cfg)
    factor_table.to_csv(output_dir / 'V54_FACTOR_DECOMPOSITION.csv', index=False)
    pd.DataFrame([factor_values]).to_csv(output_dir / 'V54_FACTOR_VALUES.csv', index=False)
    log('Computing dataset-cluster sensitivity and leave-one-dataset-out results...')
    cluster, lodo = compute_cluster_sensitivity(fold_table, cfg)
    cluster.to_csv(output_dir / 'V54_DATASET_CLUSTER_SENSITIVITY.csv', index=False)
    lodo.to_csv(output_dir / 'V54_LEAVE_ONE_DATASET_OUT.csv', index=False)
    cluster_table = make_cluster_sensitivity_table(cluster)
    (output_dir / 'table_dataset_cluster_sensitivity_v54.tex').write_text(cluster_table, encoding='utf-8')
    figs = package_dir / 'figs'
    make_factor_figure(factor_values, figs / 'fig18_factor_decomposition_v54.png', figs / 'fig18_factor_decomposition_v54.pdf')
    make_tradeoff_figure(overall, efficiency, figs / 'fig17_tradeoffs_v54.png', figs / 'fig17_tradeoffs_v54.pdf')
    make_hypothesis_figure(figs / 'fig5_hypotheses.png', figs / 'fig5_hypotheses_v54.pdf')
    source_tex = package_dir / 'main_v53.tex'
    if not source_tex.exists():
        source_tex = package_dir / 'main.tex'
    tex = source_tex.read_text(encoding='utf-8')
    patched_tex, changes = patch_manuscript(tex, factor_values, cluster, cluster_table)
    target_tex = package_dir / 'main_v54.tex'
    target_tex.write_text(patched_tex, encoding='utf-8')
    (package_dir / 'main.tex').write_text(patched_tex, encoding='utf-8')
    audit = audit_outputs(patched_tex, factor_table, factor_values, cluster, package_dir)
    audit.to_csv(output_dir / 'V54_AUDIT.csv', index=False)
    print(audit.to_string(index=False))
    if cfg.strict and (not bool(audit['pass'].all())):
        failed = audit[~audit['pass']]
        raise RuntimeError('V54 audit failed:\n' + failed.to_string(index=False))
    compiled = False
    compile_message = 'Compilation disabled'
    if cfg.compile_pdf:
        compiled, compile_message = compile_latex(package_dir, 'main_v54.tex')
        log(compile_message)
    report_lines = ['# CAPSTONE-CLIP v54 reviewer-safe post-processing report', '', '## No-training guarantee', '', 'This pipeline reads existing CSV/ZIP artifacts only. It does not instantiate, train, or fine-tune a model.', '', '## Corrected factor decomposition', '', f"- Image exposure: {100 * factor_values['image_exposure']:+.2f} points ({100 * factor_values['image_exposure'] / factor_values['total_joint_gain']:.1f}% of total)", f"- Correct pairing at fixed budget: {100 * factor_values['correct_pairing_fixed_budget']:+.2f} points ({100 * factor_values['correct_pairing_fixed_budget'] / factor_values['total_joint_gain']:.1f}% of total)", f"- Scale of correctly matched pairs: {100 * factor_values['matched_corpus_scale']:+.2f} points ({100 * factor_values['matched_corpus_scale'] / factor_values['total_joint_gain']:.1f}% of total)", f"- Matched semantics total: {100 * factor_values['matched_semantics_total']:+.2f} points ({100 * factor_values['matched_semantics_total'] / factor_values['total_joint_gain']:.1f}% of total)", f"- Total joint gain: {100 * factor_values['total_joint_gain']:+.2f} points", f"- Separate capacity contrast: {100 * factor_values['capacity_increment']:+.2f} points", '', '## APPLeNet-PM dataset-cluster sensitivity', '']
    app = cluster.loc[cluster['reference'] == 'APPLeNet-PM'].iloc[0]
    report_lines.extend([f"- Mean difference: {float(app['mean_delta_pp']):+.2f} points", f"- Exact dataset-bootstrap 95% CI: [{float(app['dataset_cluster_ci95_low_pp']):+.2f}, {float(app['dataset_cluster_ci95_high_pp']):+.2f}] points", f"- Exact two-sided dataset sign-flip p: {float(app['exact_sign_flip_p']):.3f}", f"- Leave-one-dataset-out range: [{float(app['lodo_min_pp']):+.2f}, {float(app['lodo_max_pp']):+.2f}] points", '- Interpretation: the +0.70-point mean is small and dataset dependent; the conservative cluster sensitivity does not establish an overall architectural advantage.', '', '## Automatic manuscript changes', ''])
    report_lines.extend((f'- {c}' for c in changes))
    report_lines.extend(['', '## Compilation', '', f'- {compile_message}'])
    (output_dir / 'V54_CHANGELOG_AND_REVIEW.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    summary = {'inputs': {k: str(v) for k, v in zip_paths.items()}, 'output_dir': str(output_dir), 'package_dir': str(package_dir), 'compiled': compiled, 'compile_message': compile_message, 'factor_values': factor_values, 'applenet_cluster_sensitivity': app.to_dict(), 'changes': changes, 'audit_passed': bool(audit['pass'].all())}
    (output_dir / 'RUN_SUMMARY_v54.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    shutil.rmtree(input_dir, ignore_errors=True)
    zip_path = output_dir.parent / 'CAPSTONE_CLIP_v54_REVIEWER_SAFE.zip'
    zip_directory(output_dir, zip_path)
    log(f'Completed: {zip_path}')
    return zip_path
if __name__ == '__main__':
    try:
        result = main()
        print(f'\nFINAL PACKAGE: {result}')
    except Exception as exc:
        print(f'\n[CAPSTONE-v54][ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
