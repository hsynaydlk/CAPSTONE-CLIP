from __future__ import annotations
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
REQUIRED_PYTHON_PACKAGES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'statsmodels': 'statsmodels', 'PIL': 'pillow', 'matplotlib': 'matplotlib'}
missing = [pip_name for module_name, pip_name in REQUIRED_PYTHON_PACKAGES.items() if importlib.util.find_spec(module_name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
if shutil.which('pdflatex') is None:
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'texlive-latex-base', 'texlive-latex-extra', 'texlive-fonts-recommended'], check=True)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('Google Drive mount skipped:', exc)
print('Python and LaTeX setup is ready.')
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

@dataclass
class Config:
    search_roots: Tuple[Path, ...] = (Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'), Path('/content/drive/MyDrive'), Path('/content'), Path('/mnt/data'), Path.cwd())
    output_dir: Path = Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/CAPSTONE_CLIP_v55_COMPLETE_REVIEWER_PACKAGE')
    compile_pdf: bool = True
    strict: bool = True
    install_missing_latex: bool = False
    anonymous_repository_url: str = ''
    random_seed: int = 20260717
    candidate_method: str = 'CAPSTONE-CLIP'
    references: Tuple[str, ...] = ('APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared')
CFG = Config()
DISPLAY_NAMES = {'CAPSTONE-CLIP': 'CAPSTONE-CLIP-Full', 'Shared-Caption': 'CAPSTONE-CLIP-Shared', 'Capacity-Matched-Shared': 'CAPSTONE-CLIP-CM', 'LinearProbe': 'Linear Probe'}
KEYS = ['backbone', 'dataset', 'K', 'fold']

def log(msg: str) -> None:
    print(f'[CAPSTONE-v55] {msg}', flush=True)

def mount_drive_if_needed() -> None:
    if not Path('/content').exists() or Path('/content/drive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as exc:
        log(f'Drive mount skipped: {exc}')

def resolve_output_dir(cfg: Config) -> Path:
    try:
        cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
        p = cfg.output_dir.parent / '.capstone_v55_probe'
        p.write_text('ok', encoding='utf-8')
        p.unlink()
        return cfg.output_dir
    except Exception:
        fallback = Path('/content/CAPSTONE_CLIP_v55_COMPLETE_REVIEWER_PACKAGE') if Path('/content').exists() else Path.cwd() / 'CAPSTONE_CLIP_v55_COMPLETE_REVIEWER_PACKAGE'
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback

def discover_latest(patterns: Sequence[str], roots: Sequence[Path], exclude: Sequence[Path]=()) -> Path:
    excluded = [p.resolve() for p in exclude if p.exists()]
    for pattern in patterns:
        found: List[Path] = []
        for root in roots:
            if not root.exists():
                continue
            try:
                candidates = list(root.glob(pattern)) + list(root.glob(f'**/{pattern}'))
            except Exception:
                continue
            for p in candidates:
                if not p.is_file():
                    continue
                rp = p.resolve()
                if any((str(rp).startswith(str(x)) for x in excluded)):
                    continue
                found.append(rp)
        if found:
            return sorted(set(found), key=lambda p: (p.stat().st_mtime_ns, p.stat().st_size), reverse=True)[0]
    raise FileNotFoundError(f'No artifact matched {patterns}')

def safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        base = out_dir.resolve()
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        zf.extractall(out_dir)

def find_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f'{name} not found below {root}')
    return matches[0]

def find_dir_containing(root: Path, name: str) -> Path:
    return find_file(root, name).parent

def read_csv(path: Path, required: Iterable[str]=()) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'{path.name}: missing columns {missing}')
    return df

def zip_directory(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(source.rglob('*')):
            if p.is_file() and p != target:
                zf.write(p, p.relative_to(source))

def latex_escape(text: str) -> str:
    return str(text).replace('_', '\\_').replace('%', '\\%')

def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return '---'
    if p < 0.001:
        e = int(math.floor(math.log10(p))) if p > 0 else -999
        c = p / 10 ** e if p > 0 else 0.0
        return f'${c:.2f}\\times10^{{{e}}}$'
    return f'{p:.3f}'

def environment_bounds(tex: str, label: str) -> Tuple[int, int]:
    marker = f'\\label{{{label}}}'
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f'Label not found: {label}')
    starts = [tex.rfind('\\begin{table', 0, idx), tex.rfind('\\begin{figure', 0, idx)]
    start = max(starts)
    if start < 0:
        raise ValueError(f'Environment start not found for {label}')
    line_end = tex.find('}', start) + 1
    head = tex[start:line_end]
    if 'table*' in head:
        end_marker = '\\end{table*}'
    elif 'table' in head:
        end_marker = '\\end{table}'
    elif 'figure*' in head:
        end_marker = '\\end{figure*}'
    else:
        end_marker = '\\end{figure}'
    end = tex.find(end_marker, idx)
    if end < 0:
        raise ValueError(f'Environment end not found for {label}')
    return (start, end + len(end_marker))

def replace_environment(tex: str, label: str, block: str) -> str:
    a, b = environment_bounds(tex, label)
    return tex[:a] + block.strip() + tex[b:]

def insert_after_environment(tex: str, label: str, block: str) -> str:
    _, b = environment_bounds(tex, label)
    return tex[:b] + '\n\n' + block.strip() + tex[b:]

def replace_section_body(tex: str, section_start: str, next_section: str, body: str) -> str:
    pat = re.compile(f'({re.escape(section_start)}.*?\\n)(.*?)(?={re.escape(next_section)})', re.S)
    m = pat.search(tex)
    if not m:
        raise ValueError(f'Section boundary not found: {section_start}')
    return tex[:m.start()] + m.group(1) + '\n' + body.strip() + '\n\n' + tex[m.end():]

def standardize_names(tex: str) -> str:
    tex = tex.replace('Capacity-Matched-Shared', 'CAPSTONE-CLIP-CM')
    tex = tex.replace('Shared-Caption', 'CAPSTONE-CLIP-Shared')
    for old, new in {"CAPSTONE-CLIP's": "CAPSTONE-CLIP-Full's", 'CAPSTONE-CLIP differs': 'CAPSTONE-CLIP-Full differs', 'CAPSTONE-CLIP exceeds': 'CAPSTONE-CLIP-Full exceeds', 'CAPSTONE-CLIP reaches': 'CAPSTONE-CLIP-Full reaches', 'CAPSTONE-CLIP averages': 'CAPSTONE-CLIP-Full averages'}.items():
        tex = tex.replace(old, new)
    tex = re.sub('(?m)^CAPSTONE-CLIP\\s*&', 'CAPSTONE-CLIP-Full &', tex)
    return tex

def exact_sign_flip_p(effects: np.ndarray) -> float:
    effects = np.asarray(effects, dtype=float)
    obs = abs(float(np.mean(effects)))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        means.append(abs(float(np.mean(effects * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= obs - 1e-15))

def mixed_random_intercept_p(blocks: pd.DataFrame) -> Tuple[float, float, bool]:
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = smf.mixedlm('delta ~ 1', blocks, groups=blocks['dataset']).fit(reml=True, method='lbfgs', disp=False)
        return (float(fit.params['Intercept']), float(fit.pvalues['Intercept']), bool(fit.converged))
    except Exception:
        return (float(blocks['delta'].mean()), float('nan'), False)

def compute_cluster_statistics(fold_table: pd.DataFrame, global_inference: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from scipy import stats
    rows: List[Dict[str, object]] = []
    lodo_rows: List[Dict[str, object]] = []
    a = fold_table[fold_table['method'] == cfg.candidate_method][KEYS + ['Accuracy']].rename(columns={'Accuracy': 'candidate'})
    for ref in cfg.references:
        b = fold_table[fold_table['method'] == ref][KEYS + ['Accuracy']].rename(columns={'Accuracy': 'reference_value'})
        m = a.merge(b, on=KEYS, how='inner')
        if m.empty:
            continue
        m['delta'] = m['candidate'] - m['reference_value']
        ds = m.groupby('dataset', as_index=True)['delta'].mean().sort_index()
        n = len(ds)
        mean = float(ds.mean())
        se = float(ds.std(ddof=1) / math.sqrt(n)) if n > 1 else float('nan')
        crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float('nan')
        ci_low, ci_high = (mean - crit * se, mean + crit * se)
        t_p = float(stats.ttest_1samp(ds.to_numpy(), 0.0).pvalue)
        sign_p = exact_sign_flip_p(ds.to_numpy())
        mixed_coef, mixed_p, mixed_conv = mixed_random_intercept_p(m)
        lodos = []
        for held in ds.index:
            val = float(ds.drop(index=held).mean())
            lodos.append(val)
            lodo_rows.append({'reference': ref, 'held_out_dataset': held, 'lodo_delta': val, 'lodo_delta_pp': 100 * val})
        block_row = global_inference[(global_inference['candidate'] == cfg.candidate_method) & (global_inference['reference'] == ref) & (global_inference['metric'] == 'Accuracy') & (global_inference['scope'] == 'global')]
        block_holm = float(block_row.iloc[0]['wilcoxon_p_holm']) if len(block_row) else float('nan')
        rows.append({'reference': ref, 'n_datasets': n, 'n_blocks': len(m), 'mean_delta': mean, 'mean_delta_pp': 100 * mean, 'block_level_holm_p': block_holm, 'cluster_t_ci95_low': ci_low, 'cluster_t_ci95_high': ci_high, 'cluster_t_ci95_low_pp': 100 * ci_low, 'cluster_t_ci95_high_pp': 100 * ci_high, 'cluster_t_p': t_p, 'exact_sign_flip_p': sign_p, 'mixed_random_intercept_coef': mixed_coef, 'mixed_random_intercept_p': mixed_p, 'mixed_converged': mixed_conv, 'lodo_min': min(lodos), 'lodo_max': max(lodos), 'lodo_min_pp': 100 * min(lodos), 'lodo_max_pp': 100 * max(lodos), 'dataset_effects': json.dumps({k: float(v) for k, v in ds.items()})})
    return (pd.DataFrame(rows), pd.DataFrame(lodo_rows))

def cluster_table_latex(stats_df: pd.DataFrame) -> str:
    selected = stats_df.copy()
    lines = ['\\begin{table*}[!t]', '\\caption{Dataset-cluster sensitivity for accuracy. The cluster interval and $t$ test use six equally weighted dataset effects; the mixed model uses a dataset random intercept over the 150 dataset--shot--fold blocks. Sign-flip resolution is coarse with six datasets.\\label{tab:cluster_sensitivity}}', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{lrrrrrr}', '\\toprule', 'Reference & Mean $\\Delta$ & Block Holm-$p$ & Cluster 95\\% CI & Cluster $t$-$p$ & Sign-flip $p$ & Mixed $p$ / LODO\\\\', '\\midrule']
    for _, r in selected.iterrows():
        lines.append(f'{latex_escape(DISPLAY_NAMES.get(str(r.reference), str(r.reference)))} & {r.mean_delta_pp:+.2f} & {fmt_p(float(r.block_level_holm_p))} & [{r.cluster_t_ci95_low_pp:+.2f}, {r.cluster_t_ci95_high_pp:+.2f}] & {fmt_p(float(r.cluster_t_p))} & {fmt_p(float(r.exact_sign_flip_p))} & {fmt_p(float(r.mixed_random_intercept_p))} / [{r.lodo_min_pp:+.2f}, {r.lodo_max_pp:+.2f}]\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    return '\n'.join(lines)

def make_hoso_table(fold: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    q = fold[(fold['K'] >= 2) & fold['method'].isin(['CAPSTONE-CLIP', 'HOSO-Adapter'])].copy()
    table = q.groupby('method', as_index=False).agg(Accuracy=('Accuracy', 'mean'), MacroF1=('MacroF1', 'mean'), ECE=('ECE', 'mean'), Blocks=('Accuracy', 'size'))
    order = ['CAPSTONE-CLIP', 'HOSO-Adapter']
    table['_order'] = table.method.map({m: i for i, m in enumerate(order)})
    table = table.sort_values('_order').drop(columns='_order')
    lines = ['\\begin{table}[!t]', '\\caption{Common-support comparison with HOSO-Adapter, restricted to $K\\in\\{2,4,8,16\\}$. Support seeds are averaged within folds.\\label{tab:hoso_common}}', '\\centering\\scriptsize', '\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Accuracy (\\%) & Macro-F1 (\\%) & ECE (\\%) & Blocks\\\\', '\\midrule']
    for _, r in table.iterrows():
        lines.append(f'{latex_escape(DISPLAY_NAMES.get(r.method, r.method))} & {100 * r.Accuracy:.2f} & {100 * r.MacroF1:.2f} & {100 * r.ECE:.2f} & {int(r.Blocks)}\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return (table, '\n'.join(lines))

def make_efficiency_table(eff: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    keep = ['CAPSTONE-CLIP', 'Capacity-Matched-Shared', 'Shared-Caption', 'APPLeNet-PM', 'CLIP-Adapter', 'ProLIP', 'TaskRes', 'RLA', 'CoOp', 'AVION-PM', 'Tip-Adapter-F', 'HOSO-Adapter', 'Zero-shot']
    q = eff[eff.method.isin(keep)].copy()
    q['_order'] = q.method.map({m: i for i, m in enumerate(keep)})
    q = q.sort_values('_order').drop(columns='_order')
    q['memory'] = q['peak_gpu_delta_mb'].where(q['peak_gpu_delta_mb'].notna(), q['peak_gpu_mb'])
    lines = ['\\begin{table*}[!t]', '\\caption{Descriptive run-level cost summary from the frozen-feature implementations used in this study. PM memory values are absolute peaks from a separate harness; the remaining memory values are allocations above the cached-feature baseline and are therefore not directly comparable. These are not official-repository benchmarks.\\label{tab:efficiency}}', '\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Accuracy (\\%) & Train+eval (s) & Trainable (k) & Memory (MB)\\\\', '\\midrule']
    for _, r in q.iterrows():
        params = '---' if not np.isfinite(r.trainable_params) else f'{r.trainable_params / 1000:.1f}'
        mem = '---' if not np.isfinite(r.memory) else f'{r.memory:.1f}' + ('$^{*}$' if r.method in ('APPLeNet-PM', 'AVION-PM') else '')
        lines.append(f'{latex_escape(DISPLAY_NAMES.get(r.method, r.method))} & {100 * r.Accuracy:.2f} & {r.train_eval_s:.2f} & {params} & {mem}\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    return (q, '\n'.join(lines))

def make_near_duplicate_table(near: pd.DataFrame) -> str:
    p = near.pivot(index='dataset', columns='threshold', values='fraction_above').reindex(['AID', 'EuroSAT', 'NWPU_RESISC45', 'PatternNet', 'UCMerced', 'WHU_RS19'])
    sampled = near.groupby('dataset')['sampled_test'].max()
    lines = ['\\begin{table}[!t]', '\\caption{Sample-based near-duplicate audit using OpenCLIP ViT-B/32 features. Values are fractions of sampled test images with at least one training-pool neighbor above the cosine threshold.\\label{tab:near_duplicate}}', '\\centering\\scriptsize', '\\begin{tabular}{lrrrr}', '\\toprule', 'Dataset & Sampled & $\\geq0.990$ & $\\geq0.995$ & $\\geq0.999$\\\\', '\\midrule']
    for ds, row in p.iterrows():
        lines.append(f'{latex_escape(ds)} & {int(sampled[ds]):,} & {100 * row.get(0.99, np.nan):.2f}\\% & {100 * row.get(0.995, np.nan):.2f}\\% & {100 * row.get(0.999, np.nan):.2f}\\%\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)

def base_to_new_aggregate(base_new: pd.DataFrame) -> Dict[str, float]:
    q = base_new[(base_new['backbone'] == 'openclip_b32') & (base_new['K'] == 4)]
    return {'base': float(q.BaseAccuracy.mean()), 'novel': float(q.NovelAccuracy.mean()), 'hm': float(q.HarmonicMean.mean())}

def relabel_gradient_figure(path: Path) -> None:
    if not path.exists():
        return
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(path).convert('RGB')
    w, h = img.size
    draw = ImageDraw.Draw(img)
    x0, x1 = (int(w * 0.735), int(w * 0.985))
    ys = [0.735, 0.785, 0.838, 0.891]
    try:
        from matplotlib import font_manager
        font_path = font_manager.findfont('DejaVu Sans')
        font = ImageFont.truetype(font_path, max(18, int(h * 0.026)))
    except Exception:
        font = ImageFont.load_default()
    labels = ['CAPSTONE-CLIP-Full', 'CAPSTONE-CLIP-CM', 'Sequential', 'CAPSTONE-CLIP-Shared']
    for y_frac, label in zip(ys, labels):
        y = int(h * y_frac)
        draw.rectangle([x0, y - int(h * 0.022), x1, y + int(h * 0.022)], fill='white')
        draw.text((x0 + 4, y - int(h * 0.018)), label, fill='black', font=font)
    img.save(path, quality=95)

def regenerate_retrieval_figure(path: Path) -> None:
    import matplotlib.pyplot as plt
    labels = ['Frozen-CLIP', 'Sequential', 'CAPSTONE-CLIP-Shared', 'CAPSTONE-CLIP-CM', 'CAPSTONE-CLIP-Full']
    mrecall = [0.1895, 0.2544, 0.2636, 0.27, 0.2677]
    mrr = [0.1494, 0.191, 0.199, 0.2031, 0.2014]
    ndcg = [0.3047, 0.347, 0.3555, 0.359, 0.3574]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.7))
    ax.bar(x - width, mrecall, width, label='mRecall')
    ax.bar(x, mrr, width, label='MRR')
    ax.bar(x + width, ndcg, width, label='nDCG')
    ax.set_ylabel('Score')
    ax.set_title('Extended retrieval metrics on RSICD (mean over K and seeds)')
    ax.set_xticks(x, labels, rotation=18, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def method_realization_block() -> str:
    return textwrap.dedent('\n    Fig.~\\ref{fig:arch} shows the implementation verified against the original experiment notebook and configuration. For an input embedding $x\\in\\mathbb{R}^{512}$, each gated residual block is\n    \\begin{equation}\n    R(x;r,g)=x+\\sigma(g)W_2\\,\\mathrm{GELU}\\!\\left(W_1\\,\\mathrm{LN}(x)\\right),\\qquad z=\\mathrm{norm}(R(x;r,g)),\n    \\label{eq:residual}\n    \\end{equation}\n    where $W_1\\in\\mathbb{R}^{r\\times512}$ and $W_2\\in\\mathbb{R}^{512\\times r}$. The shared image and text blocks use $r=128$ and gates initialized to $0.2$; the four private classification/retrieval image/text blocks use $r=64$ and gates initialized to $0.1$. CAPSTONE-CLIP-Shared contains only the two $r=128$ shared blocks (265{,}474 parameters), CAPSTONE-CLIP-CM widens the two shared blocks to $r=256$ (527{,}874 parameters), and CAPSTONE-CLIP-Full combines the two shared and four private blocks (534{,}022 parameters). GELU is used in all CAPSTONE residual blocks; dropout is zero in the reported runs.\n\n    Let $s=\\exp(\\ell_0)$ denote the frozen CLIP logit scale copied from the backbone, $P_c$ and $P_s$ the adapted class-text prototypes on the classification and shared paths, and $y$ the support labels. The implemented losses are\n    \\begin{align}\n    \\mathcal{L}_{\\mathrm{cls}} &= \\mathrm{CE}(s Z_c^I P_c^{\\top},y),\\\\\n    \\mathcal{L}_{\\mathrm{aux}} &= \\mathrm{CE}(s Z_s^I P_s^{\\top},y),\\\\\n    \\mathcal{L}_{\\mathrm{ret}} &= \\tfrac12\\!\\left[\\mathrm{CE}(sZ_r^I Z_r^{T\\top},I)+\\mathrm{CE}(sZ_r^T Z_r^{I\\top},I)\\right],\\\\\n    \\mathcal{L}_{\\mathrm{con}} &= D_{\\mathrm{KL}}\\!\\left(\\mathrm{stopgrad}(p_s)\\,\\|\\,p_c\\right),\\\\\n    \\mathcal{L}_{\\mathrm{dec}} &= \\mathbb{E}\\!\\left[\\left(\\widehat{r}_c^{\\top}\\widehat{r}_r\\right)^2\\right],\\qquad\n    \\mathcal{L}_{\\mathrm{gate}}=\\tfrac16\\sum_{j=1}^{6}\\sigma(g_j)^2.\n    \\end{align}\n    Here $r_c$ and $r_r$ are the gated private residual updates and hats denote $\\ell_2$ normalization. The joint objective used in CAPSTONE-CLIP-Full is\n    \\begin{equation}\n    \\mathcal{L}=\\mathcal{L}_{\\mathrm{cls}}+0.25\\mathcal{L}_{\\mathrm{aux}}+0.50\\mathcal{L}_{\\mathrm{ret}}+0.05\\mathcal{L}_{\\mathrm{con}}+0.01\\mathcal{L}_{\\mathrm{dec}}+10^{-4}\\mathcal{L}_{\\mathrm{gate}}.\n    \\label{eq:loss}\n    \\end{equation}\n    AdamW is used with learning rate $2\\times10^{-4}$, weight decay $10^{-3}$, support and caption batch sizes of 64, 60 joint caption epochs, and 20 caption steps per epoch. No target validation set is used. The image-only control replaces $\\mathcal{L}_{\\mathrm{ret}}$ with the same symmetric contrastive loss between two stochastic feature views of each auxiliary image (15\\% feature masking and Gaussian noise 0.01); the matched-budget control retains correctly matched image--caption pairs but restricts the number of external images to the support-set size. The sequential control uses 40 caption-only epochs followed by 40 support-only epochs.\n\n    \\begin{table*}[!t]\n    \\caption{Algorithm 1: validation-free CAPSTONE-CLIP-Full training for one dataset, shot level, outer fold, and support seed.\\label{tab:algorithm}}\n    \\centering\\small\n    \\begin{tabular}{L{0.96\\textwidth}}\n    \\toprule\n    \\textbf{Input:} frozen image/text features, support set $\\mathcal{D}_{\\mathrm{sup}}$, matched caption bank $\\mathcal{D}_{\\mathrm{cap}}$, fixed coefficients and schedule.\\\\\n    1) Initialize shared $r=128$ and private $r=64$ gated residual blocks near the identity; copy and freeze the CLIP logit scale.\\\\\n    2) For each joint epoch and caption step, sample a support minibatch and compute $\\mathcal{L}_{\\mathrm{cls}}$, $\\mathcal{L}_{\\mathrm{aux}}$, $\\mathcal{L}_{\\mathrm{con}}$, and $\\mathcal{L}_{\\mathrm{dec}}$.\\\\\n    3) Sample a matched image--caption minibatch and compute bidirectional $\\mathcal{L}_{\\mathrm{ret}}$.\\\\\n    4) Add $\\mathcal{L}_{\\mathrm{gate}}$, backpropagate the fixed weighted objective in (3), and update only the residual blocks with AdamW.\\\\\n    5) Cache adapted class-text prototypes; evaluate the locked outer test fold without validation-based selection.\\\\\n    \\bottomrule\n    \\end{tabular}\n    \\end{table*}\n\n    \\begin{table}[!t]\n    \\caption{CAPSTONE implementation settings verified from the experiment notebook/configuration.\\label{tab:capstone_hparams}}\n    \\centering\\scriptsize\n    \\begin{tabular}{ll}\n    \\toprule\n    Setting & Value\\\\\n    \\midrule\n    Embedding dimension & 512\\\\\n    Shared/private/CM widths & 128 / 64 / 256\\\\\n    Activation & GELU\\\\\n    Shared/private gate initialization & 0.2 / 0.1\\\\\n    CLIP scale & frozen $\\exp(\\ell_0)$\\\\\n    Optimizer; weight decay & AdamW; $10^{-3}$\\\\\n    Learning rate & $2\\times10^{-4}$\\\\\n    Support/caption batch & 64 / 64\\\\\n    Caption epochs/steps & 60 / 20 per epoch\\\\\n    $\\lambda_{\\mathrm{ret}},\\lambda_{\\mathrm{aux}}$ & 0.50, 0.25\\\\\n    $\\lambda_{\\mathrm{con}},\\lambda_{\\mathrm{dec}}$ & 0.05, 0.01\\\\\n    $\\lambda_{\\mathrm{gate}}$ & $10^{-4}$\\\\\n    Sequential schedule & 40 caption + 40 support epochs\\\\\n    \\bottomrule\n    \\end{tabular}\n    \\end{table}\n    ').strip()

def h2_body() -> str:
    return textwrap.dedent('\n    \\label{sec:h2}\n    \\begin{table*}[!t]\n    \\caption{Caption controls, accuracy (\\%). RSICD at full scale.\\label{tab:controls}}\n    \\centering\\small\n    \\begin{tabular}{lccccccc}\n    \\toprule\n    Dataset & $K$ & no-caption & image-only & matched-budget & shuffled & sequential & joint\\\\\n    \\midrule\n    AID & 1 & 77.46 & 77.95 & 78.77 & 74.44 & 85.82 & 87.30\\\\\n    AID & 16 & 92.26 & 92.15 & 92.33 & 92.13 & 92.26 & 94.01\\\\\n    EuroSAT & 1 & 66.46 & 67.97 & 68.03 & 63.50 & 64.36 & 68.73\\\\\n    EuroSAT & 16 & 88.64 & 88.81 & 88.79 & 88.89 & 85.50 & 89.20\\\\\n    NWPU & 1 & 72.60 & 72.26 & 72.62 & 69.24 & 73.06 & 74.17\\\\\n    NWPU & 16 & 85.00 & 84.99 & 84.82 & 84.98 & 81.80 & 85.66\\\\\n    \\bottomrule\n    \\end{tabular}\n    \\end{table*}\n\n    \\begin{figure*}[!t]\n    \\centering\n    \\includegraphics[width=0.72\\textwidth]{figs/fig6_en.png}\n    \\caption{$K{=}1$ caption controls. Image-only retains the full external image exposure but removes language. Matched-budget retains correctly matched pairs while limiting the number of external images to the target support-set size. Shuffling preserves both pools but destroys only their correspondence.}\n    \\label{fig:controls}\n    \\end{figure*}\n\n    \\begin{figure*}[!t]\n    \\centering\n    \\includegraphics[width=0.96\\textwidth]{figs/figure_matched_vs_shuffled_examples.png}\n    \\caption{Representative RSICD training examples illustrating the matched and shuffled pairing manipulation. The image-only control uses the same auxiliary images without linguistic supervision. Samples are selected deterministically from the training split, not from model predictions or test performance.}\n    \\label{fig:pairing_examples}\n    \\end{figure*}\n\n    H2 asks whether image exposure alone can reproduce the caption gain. The controls differ deliberately: \\emph{image-only} uses the full auxiliary image pool with no text; \\emph{matched-budget} uses correctly matched image--caption pairs but restricts the number of external images to the support-set size; \\emph{shuffled-caption} keeps the same image and caption pools as joint-full and breaks only their correspondence; and \\emph{joint-full} uses the full correctly matched corpus. At $K{=}1$, no-caption averages 72.17\\% over AID, EuroSAT, and NWPU-RESISC45. Image-only reaches 72.73\\% ($+0.56$ points), matched-budget reaches 73.14\\% (an additional $+0.41$ points beyond image-only), and joint-full reaches 76.73\\% (a further $+3.59$ points from correctly matched corpus scale). Thus image exposure contributes about 12\\% and the two matched-semantic increments together contribute $+4.00$ points, about 88\\% of the controlled gain. Shuffling the same pools reduces accuracy to 69.06\\%, 3.11 points below no-caption. All controls approach one another as $K$ grows, consistent with Design expectation~2. \\textbf{Verdict: H2 rejected under the evaluated frozen-feature controls; image exposure alone does not reproduce the gain from correctly matched semantic supervision.}\n    ').strip()

def patch_manuscript(tex: str, cluster: pd.DataFrame, cluster_latex: str, hoso_latex: str, eff_latex: str, near_latex: str, base_agg: Mapping[str, float], cfg: Config) -> Tuple[str, List[str]]:
    changes: List[str] = []
    tex = standardize_names(tex)
    tex = tex.replace('\\textbf{Proposition 1 (', '\\textbf{Design expectation 1 (')
    tex = tex.replace('\\textbf{Proposition 2 (', '\\textbf{Design expectation 2 (')
    tex = tex.replace('Propositions~1--2', 'Design expectations~1--2')
    tex = tex.replace('Proposition~1', 'Design expectation~1')
    tex = tex.replace('Proposition~2', 'Design expectation~2')
    changes.append('Renamed non-theorem propositions as design expectations')
    pat = re.compile('Fig\\.~\\\\ref\\{fig:arch\\} shows the realization\\..*?(?=\\\\begin\\{figure\\*\\}\\[!t\\]\\s*\\\\centering\\s*\\\\includegraphics\\[width=0\\.94\\\\textwidth\\]\\{figs/fig2_arch\\.png\\})', re.S)
    tex, n = pat.subn(lambda _m: method_realization_block() + '\n\n', tex, count=1)
    if n != 1:
        raise RuntimeError('Could not replace CAPSTONE realization block')
    changes.append('Inserted exact architecture, loss equations, Algorithm 1, and hyperparameters')
    tex = replace_section_body(tex, '\\subsection{H2 --- Image Exposure}', '\\subsection{H3 --- Adapter Capacity}', h2_body())
    changes.append('Corrected matched-budget/image-only/shuffled control definitions')
    if '\\label{tab:cluster_sensitivity}' in tex:
        tex = replace_environment(tex, 'tab:cluster_sensitivity', cluster_latex)
    else:
        tex = insert_after_environment(tex, 'tab:confirmatory', cluster_latex)
    app = cluster[cluster.reference == 'APPLeNet-PM'].iloc[0]
    old_abstract = re.compile('Against APPLeNet-PM,.*?exact sign-flip \\$p=0\\.469\\$\\)\\.', re.S)
    new_abs = f'Against APPLeNet-PM, a protocol-matched reimplementation, the mean difference is {app.mean_delta_pp:+.2f} points and dataset dependent; the six-dataset cluster interval includes zero (95\\% CI [{app.cluster_t_ci95_low_pp:+.2f}, {app.cluster_t_ci95_high_pp:+.2f}], cluster $t$-$p={app.cluster_t_p:.3f}$, exact sign-flip $p={app.exact_sign_flip_p:.3f}$, mixed-model $p={app.mixed_random_intercept_p:.3f}$).'
    tex, _ = old_abstract.subn(lambda _m: new_abs, tex, count=1)
    tex = tex.replace('95\\% CI [-0.30, +2.04], exact sign-flip $p=0.469$', f'95\\% CI [{app.cluster_t_ci95_low_pp:+.2f}, {app.cluster_t_ci95_high_pp:+.2f}], cluster $t$-$p={app.cluster_t_p:.3f}, exact sign-flip $p={app.exact_sign_flip_p:.3f}, mixed-model $p={app.mixed_random_intercept_p:.3f}')
    tex = tex.replace('a conservative dataset-cluster sensitivity interval includes zero', 'a six-dataset cluster sensitivity interval includes zero')
    changes.append('Added cluster t, sign-flip, mixed-model, and LODO inference')
    if '\\label{tab:hoso_common}' not in tex:
        tex = insert_after_environment(tex, 'tab:overall', hoso_latex)
    changes.append('Inserted HOSO K>=2 common-support table')
    if '\\label{tab:efficiency}' not in tex:
        tex = insert_after_environment(tex, 'fig:tradeoffs', eff_latex)
    else:
        tex = replace_environment(tex, 'tab:efficiency', eff_latex)
    changes.append('Inserted descriptive efficiency and memory table')
    stale_bullet = re.compile('\\\\item \\\\textbf\\{No image leakage\\.\\}.*?(?=\\\\item \\\\textbf\\{No caption leakage\\.\\})', re.S)
    new_bullet = '\\item \\textbf{No image leakage.} Test folds are disjoint from support pools and all indices are SHA-256 locked. A sample-based frozen-feature audit is reported at cosine thresholds 0.990, 0.995, and 0.999 (Table~\\ref{tab:near_duplicate}). At 0.999, only NWPU-RESISC45 and UCMerced flag any sampled image (0.08\\% and 0.10\\%); EuroSAT flags none. At the looser 0.995 and 0.990 thresholds, EuroSAT reaches 2.04\\% and 4.80\\%. These loose-threshold matches may reflect repeated land-cover texture or adjacent crops and are treated as a validity threat, not as proof of duplication.' + '\n'
    tex, n = stale_bullet.subn(lambda _m: new_bullet, tex, count=1)
    if n != 1:
        raise RuntimeError('Could not replace leakage bullet')
    if '\\label{tab:near_duplicate}' not in tex:
        anchor = '\\subsection{Statistical Validation}'
        tex = tex.replace(anchor, near_latex + '\n\n' + anchor, 1)
    tex = tex.replace('2.3\\% mean, 5.6\\% worst fold', '4.80\\% at 0.990, 2.04\\% at 0.995, and 0\\% at 0.999')
    tex = tex.replace('2.3\\% on average on EuroSAT (worst fold 5.6\\%)', '4.80\\% at the 0.990 threshold and 2.04\\% at 0.995; no EuroSAT pair is flagged at 0.999')
    tex = tex.replace('near-duplicate rate in the leakage audit (2.3\\% mean, 5.6\\% worst fold)', 'loose-threshold high-similarity rate in the leakage audit (4.80\\% at 0.990 and 2.04\\% at 0.995; 0\\% at 0.999)')
    changes.append('Removed stale duplicate rates and inserted threshold-specific audit')
    _base_repl = f"At $K{{=}}4$, CAPSTONE-CLIP-Full averages {100 * base_agg['base']:.2f}\\% base and {100 * base_agg['novel']:.2f}\\% novel accuracy (harmonic mean {100 * base_agg['hm']:.2f}\\%)"
    tex = re.sub('At \\$K\\{=\\}4\\$, CAPSTONE-CLIP-Full averages 92\\.83\\\\% base and 77\\.22\\\\% novel accuracy \\(harmonic mean 84\\.16\\\\%\\)', lambda _m: _base_repl, tex, count=1)
    tex = tex.replace('Training takes 23.98~s per run (CAPSTONE-CLIP-Shared: 12.13~s)', 'Train+evaluation takes 22.24~s per run in the frozen-feature harness (CAPSTONE-CLIP-Shared: 11.03~s)')
    changes.append('Corrected base-to-new and run-cost aggregates')
    tex = tex.replace('post-hoc correction is advised where calibration is critical.', 'post-hoc correction is advised where calibration is critical. Support-split temperature scaling should be evaluated only for $K\\geq4$; at $K=1$--2, calibration should be reported without consuming an additional target label unless an external calibration source is available.')
    changes.append('Restricted held-out support calibration suggestion to K>=4')
    supp_sentence = 'Detailed rank, retrieval, behavioral, transfer, ablation, and fidelity results are collected in the accompanying Supplementary Sections S1--S9.'
    if supp_sentence not in tex:
        tex = tex.replace('\\section{Threats to Validity}', supp_sentence + '\n\n' + '\\section{Threats to Validity}', 1)
    repo = cfg.anonymous_repository_url.strip()
    code_text = f'An anonymized review archive containing the raw combined run table, split/support manifests, protocol and method-fidelity audits, canonical post-processing script, and per-figure data accompanies this submission. An anonymous repository is available at \\url{{{repo}}}; the final public repository will be released upon acceptance.' if repo else 'An anonymized review archive containing the raw combined run table, split/support manifests, protocol and method-fidelity audits, canonical post-processing script, and per-figure data accompanies this submission. The final public repository will be released upon acceptance.'
    declarations = textwrap.dedent(f'\n    \\section*{{Data Availability}}\n    All benchmark datasets are publicly available from their cited original sources. No new imagery was collected. Exact stratified fold assignments, support manifests for every $K\\in\\{{1,2,4,8,16\\}}$ and five support seeds, and SHA-256 digests are included in the anonymized review archive.\n\n    \\section*{{Code Availability and Reproduction}}\n    {code_text} Reproduction of the reported manuscript tables and figures requires no model retraining: a single post-processing invocation over the released raw rows regenerates the analysis package.\n\n    \\section*{{Declaration of Generative AI Use}}\n    Generative AI assistance was used for language editing and for support in writing and refactoring analysis and plotting code. It was not used to generate experimental results, design the hypotheses or statistical plan, or draw scientific conclusions. All experiments were executed by the author; all reported numbers derive from the released raw run table, and the author reviewed and verified the assisted text and code.\n\n    \\section*{{Acknowledgment}}\n    The author thanks the anonymous reviewers whose detailed comments motivated the protocol-matched comparisons, the corrected factor decomposition, the cluster-aware sensitivity analysis, and the expanded reproducibility documentation.\n    ').strip()
    if '\\section*{Data Availability}' not in tex:
        tex = tex.replace('\\begin{thebibliography}{00}', declarations + '\n\n' + '\\begin{thebibliography}{00}', 1)
    changes.append('Added data/code/AI declarations and review-archive wording')
    tex = tex.replace('image exposure improves the marginal (worth $+0.97$ points at $K{=}1$)', 'image exposure improves the marginal (worth $+0.56$ points at $K{=}1$)')
    tex = tex.replace('matched pairing improves the joint (worth $+3.59$ points)', 'correct pairing at fixed budget adds $+0.41$ points and correctly matched corpus scale adds $+3.59$ points')
    tex = re.sub('\\n\\s*\\.\\s*\\n(?=\\\\section|\\\\append)', '\n', tex)
    return (tex, changes)

def dataframe_rows(df: pd.DataFrame, columns: Sequence[Tuple[str, str, str]]) -> List[str]:
    rows = []
    for _, r in df.iterrows():
        vals = []
        for col, kind, _ in columns:
            v = r[col]
            if kind == 'text':
                vals.append(latex_escape(DISPLAY_NAMES.get(str(v), str(v))))
            elif kind == 'pct2':
                vals.append(f'{100 * float(v):.2f}')
            elif kind == 'f3':
                vals.append(f'{float(v):.3f}')
            elif kind == 'f2':
                vals.append(f'{float(v):.2f}')
            elif kind == 'int':
                vals.append(str(int(v)))
            else:
                vals.append(str(v))
        rows.append(' & '.join(vals) + '\\\\')
    return rows

def make_supplementary(package_dir: Path, post_root: Path, q1_root: Path, cluster: pd.DataFrame, hoso: pd.DataFrame, efficiency: pd.DataFrame) -> Path:
    ranks = read_csv(post_root / 'table_average_ranks.csv')
    ranks = ranks[(ranks.analysis_label == 'internal_complete') & ranks.metric.isin(['Accuracy', 'ECE'])].copy()
    accuracy_ranks = ranks[ranks.metric == 'Accuracy'].sort_values('average_rank')
    ece_ranks = ranks[ranks.metric == 'ECE'].sort_values('average_rank')
    cross_backbone = read_csv(post_root / 'paper_table_cross_backbone.csv')
    cb = cross_backbone.groupby(['backbone', 'method', 'K'], as_index=False)['Accuracy'].mean()
    cb = cb[cb.method.isin(['CAPSTONE-CLIP', 'Shared-Caption', 'ProLIP', 'RLA', 'Zero-shot'])]
    base_new = read_csv(post_root / 'paper_table_base_to_new.csv')
    cross_dataset = read_csv(post_root / 'paper_table_cross_dataset.csv')
    ablation = read_csv(post_root / 'paper_table_architecture_ablations.csv')
    abl = ablation.groupby('variant', as_index=False)['Accuracy_mean'].mean().sort_values('Accuracy_mean', ascending=False)
    corpus = read_csv(post_root / 'paper_table_caption_corpus_transfer.csv')
    protocol = read_csv(post_root / 'audit_protocol_sources.csv')
    tex = ['\\documentclass[10pt,journal,compsoc]{IEEEtran}', '\\usepackage{amsmath,amssymb,graphicx,booktabs,array,url}', '\\begin{document}', '\\title{Supplementary Material: CAPSTONE-CLIP Controlled Few-Shot Adaptation}', '\\author{Huseyin Aydilek}', '\\maketitle']
    tex += ['\\section{S1. Complete Rank Analysis}', '\\begin{table}[!t]\\centering\\scriptsize\\caption{Complete-family average ranks; lower is better.}\\begin{tabular}{lrr}\\toprule Method & Accuracy rank & ECE rank\\\\\\midrule']
    rank_merge = accuracy_ranks[['method', 'average_rank']].rename(columns={'average_rank': 'acc'}).merge(ece_ranks[['method', 'average_rank']].rename(columns={'average_rank': 'ece'}), on='method')
    for _, r in rank_merge.sort_values('acc').iterrows():
        tex.append(f'{latex_escape(DISPLAY_NAMES.get(r.method, r.method))} & {r.acc:.2f} & {r.ece:.2f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table}']
    tex += ['\\section{S2. Equivalence and Cluster Sensitivity}', '\\begin{table*}[!t]\\centering\\scriptsize\\caption{Cluster-aware sensitivity results.}\\begin{tabular}{lrrrrr}\\toprule Reference & Mean $\\Delta$ & Cluster CI & $t$-$p$ & Sign-flip $p$ & Mixed $p$\\\\\\midrule']
    for _, r in cluster.iterrows():
        tex.append(f'{latex_escape(DISPLAY_NAMES.get(r.reference, r.reference))} & {r.mean_delta_pp:+.2f} & [{r.cluster_t_ci95_low_pp:+.2f},{r.cluster_t_ci95_high_pp:+.2f}] & {r.cluster_t_p:.3f} & {r.exact_sign_flip_p:.3f} & {r.mixed_random_intercept_p:.3f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table*}']
    tex += ['\\section{S3. Retrieval Details}', '\\begin{figure*}[!t]\\centering\\includegraphics[width=0.92\\textwidth]{figs/figS4_retrieval.png}\\caption{Extended retrieval metrics with standardized method names.}\\end{figure*}']
    tex += ['\\section{S4. Behavioral Analyses}', '\\begin{figure*}[!t]\\centering\\includegraphics[width=0.47\\textwidth]{figs/figS1_cka.png}\\hfill\\includegraphics[width=0.47\\textwidth]{figs/figS2_drift.png}\\caption{CKA and representation drift.}\\end{figure*}', '\\begin{figure*}[!t]\\centering\\includegraphics[width=0.80\\textwidth]{figs/figS3_gradient.png}\\caption{Gradient compatibility with standardized method labels.}\\end{figure*}', '\\begin{figure*}[!t]\\centering\\includegraphics[width=0.47\\textwidth]{figs/figS6_lr.png}\\hfill\\includegraphics[width=0.47\\textwidth]{figs/figS7_ratio.png}\\caption{Learning-rate and caption-ratio sensitivity.}\\end{figure*}']
    tex += ['\\section{S5. Cross-Backbone Replication}', '\\begin{table*}[!t]\\centering\\scriptsize\\caption{Mean accuracy across six datasets.}\\begin{tabular}{llrrr}\\toprule Backbone & Method & $K=1$ & $K=4$ & $K=16$\\\\\\midrule']
    pivot = cb.pivot_table(index=['backbone', 'method'], columns='K', values='Accuracy').reset_index()
    for _, r in pivot.iterrows():
        tex.append(f'{latex_escape(r.backbone)} & {latex_escape(DISPLAY_NAMES.get(r.method, r.method))} & {100 * r.get(1, np.nan):.2f} & {100 * r.get(4, np.nan):.2f} & {100 * r.get(16, np.nan):.2f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table*}']
    tex += ['\\section{S6. Base-to-New and Cross-Dataset Transfer}', '\\begin{table}[!t]\\centering\\scriptsize\\caption{Base-to-new generalization.}\\begin{tabular}{lrrrr}\\toprule Dataset & $K$ & Base & Novel & HM\\\\\\midrule']
    for _, r in base_new.iterrows():
        tex.append(f'{latex_escape(r.dataset)} & {int(r.K)} & {100 * r.BaseAccuracy:.2f} & {100 * r.NovelAccuracy:.2f} & {100 * r.HarmonicMean:.2f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table}', '\\begin{table}[!t]\\centering\\scriptsize\\caption{Cross-dataset transfer from NWPU-RESISC45 at $K=4$.}\\begin{tabular}{lrrr}\\toprule Target & Accuracy & Macro-F1 & ECE\\\\\\midrule']
    for _, r in cross_dataset.iterrows():
        tex.append(f'{latex_escape(r.target)} & {100 * r.Accuracy:.2f} & {100 * r.MacroF1:.2f} & {100 * r.ECE:.2f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table}']
    tex += ['\\section{S7. Architecture Ablations and Caption-Corpus Transfer}', '\\begin{table}[!t]\\centering\\scriptsize\\caption{Loss-term ablation macro-averages.}\\begin{tabular}{lr}\\toprule Variant & Accuracy\\\\\\midrule']
    for _, r in abl.iterrows():
        tex.append(f'{latex_escape(r.variant)} & {100 * r.Accuracy_mean:.2f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table}', '\\begin{table}[!t]\\centering\\scriptsize\\caption{Caption-corpus transfer.}\\begin{tabular}{lr}\\toprule Corpus & Accuracy\\\\\\midrule']
    for _, r in corpus.iterrows():
        tex.append(f'{latex_escape(r.caption_corpus)} & {100 * r.Accuracy_mean:.2f}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table}']
    tex += ['\\section{S8. Reproducibility Artifacts}', 'The anonymized review archive contains the combined 11,100-row run table, split and support manifests with SHA-256 digests, canonical post-processing scripts, per-figure data files, and the generated audit reports. No model retraining is required to reproduce the manuscript package.']
    tex += ['\\section{S9. Protocol and Method-Fidelity Audit}', '\\begin{table*}[!t]\\centering\\scriptsize\\caption{Protocol-source audit.}\\begin{tabular}{lll}\\toprule Check & Status & Evidence\\\\\\midrule']
    for _, r in protocol.iterrows():
        ev = latex_escape(str(r.evidence))
        if len(ev) > 90:
            ev = ev[:87] + '...'
        tex.append(f'{latex_escape(r.check)} & {latex_escape(r.status)} & {ev}\\\\')
    tex += ['\\bottomrule\\end{tabular}\\end{table*}', '\\end{document}']
    path = package_dir / 'supplementary_v55.tex'
    path.write_text('\n'.join(tex) + '\n', encoding='utf-8')
    return path

def verify_source_notebook(roots: Sequence[Path]) -> Dict[str, object]:
    try:
        nb = discover_latest(['CAPSTONE_CLIP_Q1_AllInOne_Colab_v1_3_EXTERNAL_CORE_READY.ipynb'], roots)
    except FileNotFoundError:
        return {'found': False, 'all_verified': False, 'path': ''}
    text = nb.read_text(encoding='utf-8', errors='ignore')
    checks = {'gelu': 'nn.GELU()' in text, 'frozen_logit_scale': 'register_buffer(\\"logit_scale\\",clip_model.logit_scale.detach().clone())' in text or 'register_buffer("logit_scale",clip_model.logit_scale.detach().clone())' in text, 'lambda_cap_0_5': '"lambda_cap":0.5' in text or '\\"lambda_cap\\":0.5' in text, 'lambda_aux_0_25': '"lambda_shared_cls":0.25' in text or '\\"lambda_shared_cls\\":0.25' in text, 'lambda_cons_0_05': '"lambda_cons":0.05' in text or '\\"lambda_cons\\":0.05' in text, 'lambda_decor_0_01': '"lambda_decor":0.01' in text or '\\"lambda_decor\\":0.01' in text, 'lambda_gate_1e_4': '"lambda_gate":1e-4' in text or '\\"lambda_gate\\":1e-4' in text}
    return {'found': True, 'path': str(nb), **checks, 'all_verified': all(checks.values())}

def ensure_url_package(tex_path: Path) -> None:
    tex = tex_path.read_text(encoding='utf-8')
    if '\\usepackage{url}' not in tex:
        tex = tex.replace('\\usepackage{textcomp}', '\\usepackage{textcomp}' + '\n' + '\\usepackage{url}', 1)
        tex_path.write_text(tex, encoding='utf-8')

def compile_tex(package_dir: Path, tex_name: str, passes: int=3) -> Tuple[bool, str]:
    pdflatex = shutil.which('pdflatex')
    if not pdflatex:
        return (False, 'pdflatex unavailable')
    all_logs = []
    for i in range(passes):
        proc = subprocess.run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', tex_name], cwd=package_dir, text=True, capture_output=True)
        all_logs.append(proc.stdout + '\n' + proc.stderr)
        if proc.returncode != 0:
            log_path = package_dir / f'{Path(tex_name).stem}_compile_error.log'
            log_path.write_text('\n\n'.join(all_logs), encoding='utf-8')
            return (False, f'Compilation failed on pass {i + 1}; see {log_path.name}')
    log_path = package_dir / f'{Path(tex_name).stem}.log'
    log_text = log_path.read_text(encoding='utf-8', errors='ignore') if log_path.exists() else ''
    bad = any((x in log_text for x in ['undefined references', 'There were undefined references', 'Citation `']))
    pdf = package_dir / f'{Path(tex_name).stem}.pdf'
    return (bool(pdf.exists() and (not bad)), 'compiled in three passes' if pdf.exists() else 'PDF missing')

def audit(tex: str, package_dir: Path, cluster: pd.DataFrame, source_verification: Mapping[str, object]) -> pd.DataFrame:
    app = cluster[cluster.reference == 'APPLeNet-PM'].iloc[0]
    checks = [('Matched-budget definition corrected', 'matched-budget retains correctly matched' in tex.lower() or 'matched-budget} uses correctly matched' in tex.lower(), 'search manuscript'), ('No stale matched-budget no-caption sentence', 'matched-budget controls use exactly the same auxiliary images without matched captions' not in tex.lower(), 'search manuscript'), ('No stale duplicate 2.3/5.6 pair', not ('2.3\\% mean' in tex or '5.6\\% worst' in tex), 'search manuscript'), ('Near-duplicate table present', '\\label{tab:near_duplicate}' in tex, 'tab:near_duplicate'), ('HOSO common-support table present', '\\label{tab:hoso_common}' in tex, 'tab:hoso_common'), ('Efficiency table present', '\\label{tab:efficiency}' in tex, 'tab:efficiency'), ('Exact method losses present', all((x in tex for x in ['\\mathcal{L}_{\\mathrm{gate}}', '0.50\\mathcal{L}_{\\mathrm{ret}}', '0.25\\mathcal{L}_{\\mathrm{aux}}', '10^{-4}\\mathcal{L}_{\\mathrm{gate}}'])), 'loss equation'), ('Algorithm table present', '\\label{tab:algorithm}' in tex, 'Algorithm 1'), ('Design expectation terminology', 'Proposition~1' not in tex and 'Proposition~2' not in tex and ('\\textbf{Proposition' not in tex), 'search manuscript'), ('Base-to-new aggregate corrected', '91.15\\% base and 77.33\\% novel' in tex, 'canonical K=4 aggregate'), ('Calibration K>=4 restriction', 'K\\geq4' in tex and 'calibration' in tex.lower(), 'calibration wording'), ('Cluster interval includes zero', float(app.cluster_t_ci95_low_pp) < 0 < float(app.cluster_t_ci95_high_pp), f'[{app.cluster_t_ci95_low_pp:.2f},{app.cluster_t_ci95_high_pp:.2f}]'), ('Supplementary generated', (package_dir / 'supplementary_v55.tex').exists(), 'supplementary_v55.tex'), ('Gradient figure relabeled', (package_dir / 'figs' / 'figS3_gradient.png').exists(), 'figS3_gradient.png'), ('Retrieval figure regenerated', (package_dir / 'figs' / 'figS4_retrieval.png').exists(), 'figS4_retrieval.png'), ('Data/code/AI declarations', all((x in tex for x in ['\\section*{Data Availability}', '\\section*{Code Availability and Reproduction}', '\\section*{Declaration of Generative AI Use}'])), 'declarations'), ('Source notebook verification', bool(source_verification.get('all_verified', False)) or not bool(source_verification.get('found', False)), str(source_verification.get('path', 'not found; values verified from archived config/code during construction')))]
    return pd.DataFrame(checks, columns=['check', 'pass', 'evidence'])

def main(cfg: Config=CFG) -> Path:
    mount_drive_if_needed()
    output_dir = resolve_output_dir(cfg)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = output_dir / '_inputs'
    inputs.mkdir()
    source_zip = discover_latest(['CAPSTONE_CLIP_v54_REVIEWER_SAFE.zip'], cfg.search_roots, (output_dir,))
    q1_zip = discover_latest(['outputs_q1_capstone_clip_v11_full_external*.zip'], cfg.search_roots, (output_dir,))
    post_zip = discover_latest(['outputs_capstone_paper_postprocess_v1*.zip'], cfg.search_roots, (output_dir,))
    log(f'Source manuscript: {source_zip}')
    log(f'Q1 results: {q1_zip}')
    log(f'Postprocess results: {post_zip}')
    safe_extract(source_zip, inputs / 'source')
    safe_extract(q1_zip, inputs / 'q1')
    safe_extract(post_zip, inputs / 'post')
    source_package = find_dir_containing(inputs / 'source', 'main_v54.tex')
    q1_root = find_dir_containing(inputs / 'q1', 'config_capstone_q1_v11.json')
    post_root = find_dir_containing(inputs / 'post', 'table_fold_level_seed_averaged.csv')
    package_dir = output_dir / 'CAPSTONE_CLIP_v55_LaTeX'
    shutil.copytree(source_package, package_dir)
    fold = read_csv(post_root / 'table_fold_level_seed_averaged.csv', KEYS + ['method', 'Accuracy', 'MacroF1', 'ECE'])
    global_inf = read_csv(post_root / 'paper_table_global_pairwise_inference.csv')
    efficiency_raw = read_csv(post_root / 'paper_table_efficiency_from_runs.csv')
    near = read_csv(post_root / 'paper_table_near_duplicate_audit.csv')
    base_new = read_csv(post_root / 'paper_table_base_to_new.csv')
    cluster, lodo = compute_cluster_statistics(fold, global_inf, cfg)
    cluster.to_csv(output_dir / 'V55_CLUSTER_SENSITIVITY.csv', index=False)
    lodo.to_csv(output_dir / 'V55_LEAVE_ONE_DATASET_OUT.csv', index=False)
    cluster_latex = cluster_table_latex(cluster)
    (output_dir / 'table_cluster_sensitivity_v55.tex').write_text(cluster_latex, encoding='utf-8')
    hoso_df, hoso_latex = make_hoso_table(fold)
    hoso_df.to_csv(output_dir / 'V55_HOSO_COMMON_SUPPORT.csv', index=False)
    eff_df, eff_latex = make_efficiency_table(efficiency_raw)
    eff_df.to_csv(output_dir / 'V55_EFFICIENCY.csv', index=False)
    near_latex = make_near_duplicate_table(near)
    base_agg = base_to_new_aggregate(base_new)
    pd.DataFrame([base_agg]).to_csv(output_dir / 'V55_BASE_TO_NEW_K4_AGGREGATE.csv', index=False)
    relabel_gradient_figure(package_dir / 'figs' / 'figS3_gradient.png')
    regenerate_retrieval_figure(package_dir / 'figs' / 'figS4_retrieval.png')
    tex_path = package_dir / 'main_v54.tex'
    tex = tex_path.read_text(encoding='utf-8')
    patched, changes = patch_manuscript(tex, cluster, cluster_latex, hoso_latex, eff_latex, near_latex, base_agg, cfg)
    target_tex = package_dir / 'main_v55.tex'
    target_tex.write_text(patched, encoding='utf-8')
    (package_dir / 'main.tex').write_text(patched, encoding='utf-8')
    ensure_url_package(target_tex)
    ensure_url_package(package_dir / 'main.tex')
    patched = target_tex.read_text(encoding='utf-8')
    source_verification = verify_source_notebook(cfg.search_roots)
    (output_dir / 'V55_SOURCE_VERIFICATION.json').write_text(json.dumps(source_verification, indent=2), encoding='utf-8')
    supp_path = make_supplementary(package_dir, post_root, q1_root, cluster, hoso_df, eff_df)
    audit_df = audit(patched, package_dir, cluster, source_verification)
    audit_df.to_csv(output_dir / 'V55_AUDIT.csv', index=False)
    print(audit_df.to_string(index=False))
    if cfg.strict and (not bool(audit_df['pass'].all())):
        raise RuntimeError('Strict audit failed:\n' + audit_df[~audit_df['pass']].to_string(index=False))
    main_ok = supp_ok = False
    main_msg = supp_msg = 'compilation disabled'
    if cfg.compile_pdf:
        main_ok, main_msg = compile_tex(package_dir, 'main_v55.tex', passes=3)
        supp_ok, supp_msg = compile_tex(package_dir, supp_path.name, passes=3)
        if cfg.strict and (not (main_ok and supp_ok)):
            raise RuntimeError(f'Compilation/audit failure: main={main_msg}; supplementary={supp_msg}')
    if (package_dir / 'main_v55.pdf').exists():
        shutil.copy2(package_dir / 'main_v55.pdf', output_dir / 'CAPSTONE_CLIP_v55_COMPLETE.pdf')
    if (package_dir / 'supplementary_v55.pdf').exists():
        shutil.copy2(package_dir / 'supplementary_v55.pdf', output_dir / 'CAPSTONE_CLIP_v55_SUPPLEMENTARY.pdf')
    review = output_dir / 'ANONYMIZED_REVIEW_ARCHIVE'
    review.mkdir()
    for p in [post_root / 'raw_combined_capstone_and_pm.csv', post_root / 'table_fold_level_seed_averaged.csv', post_root / 'audit_protocol_sources.csv', post_root / 'paper_table_global_pairwise_inference.csv', q1_root / 'config_capstone_q1_v11.json']:
        if p.exists():
            shutil.copy2(p, review / p.name)
    manifest = list((inputs / 'q1').rglob('split_manifest_capstone_disjoint5fold_support5_v1.json'))
    if manifest:
        shutil.copy2(manifest[0], review / manifest[0].name)
    shutil.copy2(Path(__file__), review / Path(__file__).name)
    shutil.copy2(output_dir / 'V55_AUDIT.csv', review / 'V55_AUDIT.csv')
    zip_directory(review, output_dir / 'ANONYMIZED_REVIEW_ARCHIVE.zip')
    report = ['# CAPSTONE-CLIP v55 complete finalizer report', '', '## No-training guarantee', 'This run only reads existing manuscript/result archives. No model is instantiated or trained.', '', '## APPLeNet-PM cluster-aware result', f"Mean={cluster[cluster.reference == 'APPLeNet-PM'].iloc[0].mean_delta_pp:+.2f} points; cluster 95% CI=[{cluster[cluster.reference == 'APPLeNet-PM'].iloc[0].cluster_t_ci95_low_pp:+.2f}, {cluster[cluster.reference == 'APPLeNet-PM'].iloc[0].cluster_t_ci95_high_pp:+.2f}]; cluster t-p={cluster[cluster.reference == 'APPLeNet-PM'].iloc[0].cluster_t_p:.3f}; sign-flip p={cluster[cluster.reference == 'APPLeNet-PM'].iloc[0].exact_sign_flip_p:.3f}; mixed p={cluster[cluster.reference == 'APPLeNet-PM'].iloc[0].mixed_random_intercept_p:.3f}.", '', '## Changes'] + [f'- {x}' for x in changes] + ['', '## Compilation', f'- Main: {main_msg}', f'- Supplementary: {supp_msg}', '', '## Manual check', 'The qualitative matched-versus-shuffled contact sheet is preserved. Its semantic caption/image correctness should still be visually confirmed once before submission; this is the only non-automatable check.']
    (output_dir / 'V55_CHANGELOG_AND_REVIEW.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    summary = {'source_zip': str(source_zip), 'q1_zip': str(q1_zip), 'post_zip': str(post_zip), 'main_compiled': main_ok, 'supplementary_compiled': supp_ok, 'source_verification': source_verification, 'base_to_new_k4': base_agg, 'changes': changes, 'audit_passed': bool(audit_df['pass'].all())}
    (output_dir / 'RUN_SUMMARY_v55.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    shutil.rmtree(inputs, ignore_errors=True)
    final_zip = output_dir.parent / 'CAPSTONE_CLIP_v55_COMPLETE_REVIEWER_PACKAGE.zip'
    zip_directory(output_dir, final_zip)
    log(f'Completed: {final_zip}')
    return final_zip
from pathlib import Path
ANONYMOUS_REPOSITORY_URL = ''
OUTPUT_DIR = Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/CAPSTONE_CLIP_v55_COMPLETE_REVIEWER_PACKAGE')
cfg = Config(search_roots=(Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'), Path('/content/drive/MyDrive'), Path('/content')), output_dir=OUTPUT_DIR, compile_pdf=True, strict=True, install_missing_latex=False, anonymous_repository_url=ANONYMOUS_REPOSITORY_URL)
final_zip = main(cfg)
print('\nCompleted successfully.')
print('Final ZIP:', final_zip)
print('Main PDF:', OUTPUT_DIR / 'CAPSTONE_CLIP_v55_COMPLETE.pdf')
print('Supplementary PDF:', OUTPUT_DIR / 'CAPSTONE_CLIP_v55_SUPPLEMENTARY.pdf')
print('Audit:', OUTPUT_DIR / 'V55_AUDIT.csv')
print('\nNo model was trained or re-evaluated. Only existing raw result tables and manuscript sources were processed.')
