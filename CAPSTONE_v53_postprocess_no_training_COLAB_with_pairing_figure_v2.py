from __future__ import annotations
import importlib.util
import subprocess
import sys
from pathlib import Path
required = {'numpy': 'numpy', 'pandas': 'pandas', 'matplotlib': 'matplotlib', 'PIL': 'Pillow'}
missing = [pip_name for module_name, pip_name in required.items() if importlib.util.find_spec(module_name) is None]
if missing:
    print('Eksik paketler kuruluyor:', missing)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
else:
    print('Gerekli Python paketleri hazır.')
INPUT_MODE = 'Bilinen proje klasöründeki ZIPleri kullan'
PROJECT_ROOT = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'
DATA_ROOT = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/data'
SAVE_OUTPUT_TO_DRIVE = True
AUTO_DOWNLOAD_FINAL_ZIP = False
INSTALL_LATEX = False
USE_CAPSTONE_FAMILY_DISPLAY_NAMES = True
STRICT_AUDIT = True
INCLUDE_HOSO_IN_MAIN_TABLE = False
GENERATE_PAIRING_FIGURE = True
INSERT_PAIRING_FIGURE_IN_MANUSCRIPT = True
PAIRING_EXAMPLES = 3
PAIRING_SEED = 20260717
STRICT_PAIRING_FIGURE = True
from pathlib import Path
import os
import shutil
import subprocess
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB and (INPUT_MODE.startswith('Bilinen') or SAVE_OUTPUT_TO_DRIVE or GENERATE_PAIRING_FIGURE):
    drive.mount('/content/drive')
if IN_COLAB and INPUT_MODE.startswith('Dört ZIP'):
    print('Lütfen dört ZIP paketini birlikte seçin.')
    uploaded = files.upload()
    print('Yüklenen dosyalar:', list(uploaded))
if INSTALL_LATEX:
    print('LaTeX derleyicisi kuruluyor...')
    env = dict(os.environ)
    env['DEBIAN_FRONTEND'] = 'noninteractive'
    subprocess.run(['apt-get', '-qq', 'update'], check=True, env=env)
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'texlive-latex-base', 'texlive-latex-recommended', 'texlive-fonts-recommended', 'texlive-pictures'], check=True, env=env, stdout=subprocess.DEVNULL)
    print('pdflatex:', shutil.which('pdflatex'))
else:
    print('LaTeX kurulumu atlandı. pdflatex varsa PDF derlenir; yoksa LaTeX paketi oluşturulur.')
print('Bilinen proje klasörü:', PROJECT_ROOT)
print('Bilinen veri kökü:', DATA_ROOT)
print('RSICD beklenen klasör:', Path(DATA_ROOT) / 'RSICD')
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

@dataclass
class Config:
    search_roots: Tuple[Path, ...] = (Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'), Path('/content'), Path('/mnt/data'), Path.cwd())
    output_dir: Path = Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/CAPSTONE_CLIP_v53_CANONICAL')
    data_root: Path = Path('/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing/data')
    caption_dataset: str = 'RSICD'
    generate_pairing_figure: bool = True
    insert_pairing_figure_in_manuscript: bool = True
    pairing_figure_examples: int = 3
    pairing_figure_seed: int = 20260717
    strict_pairing_figure: bool = True
    equivalence_margin_pp: float = 1.0
    win_tie_margin_pp: float = 0.1
    include_hoso_in_main_overall_table: bool = False
    copy_optional_figures: bool = True
    compile_if_pdflatex_available: bool = True
    strict_audit: bool = True
CFG = Config()
ZIP_PATTERNS: Dict[str, str] = {'manuscript': 'CAPSTONE_CLIP_v52_LaTeX_final_20pp*.zip', 'q1': 'outputs_q1_capstone_clip_v11_full_external*.zip', 'pm': 'outputs_avion_applenet_capstone_protocol_matched_v1*.zip', 'post': 'outputs_capstone_paper_postprocess_v1*.zip'}
DISTINCTIVE_FILES: Dict[str, str] = {'manuscript': 'main.tex', 'q1': 'MANUSCRIPT_CLAIM_GUIDE.md', 'pm': 'protocol_audit.json', 'post': 'reports/PAPER_RESULTS_DIGEST.md'}
METHOD_DISPLAY = {'APPLeNet-PM': 'APPLeNet-PM$^{\\ddagger}$', 'AVION-PM': 'AVION-PM$^{\\ddagger}$', 'LinearProbe': 'Linear Probe', 'Shared-Caption': 'CAPSTONE-CLIP-Shared', 'Capacity-Matched-Shared': 'CAPSTONE-CLIP-CM', 'CAPSTONE-CLIP': 'CAPSTONE-CLIP-Full'}
CORE_CONFIRMATORY_REFERENCES = ['APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared']
OPTIONAL_FIGURES = ['figure_delta_heatmap_capstone_vs_applenet_pm.png', 'figure_delta_heatmap_capstone_vs_avion_pm.png', 'figure_accuracy_vs_runtime.png', 'figure_accuracy_ece_pareto.png', 'figure_rank_focus_methods_accuracy.png']

def log(message: str) -> None:
    print(f'[CAPSTONE-v53] {message}', flush=True)

def mount_google_drive_if_needed() -> None:
    if not Path('/content').exists() or Path('/content/drive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        log('Google Drive is being mounted...')
        drive.mount('/content/drive')
    except Exception as exc:
        log(f'Drive mount skipped: {exc}')

def resolve_output_dir(cfg: Config) -> Path:
    preferred = cfg.output_dir
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        probe = preferred.parent / '.capstone_write_test'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        return preferred
    except Exception:
        fallback = Path('/content/CAPSTONE_CLIP_v53_CANONICAL') if Path('/content').exists() else Path.cwd() / 'CAPSTONE_CLIP_v53_CANONICAL'
        fallback.parent.mkdir(parents=True, exist_ok=True)
        log(f'Preferred output path is unavailable; using {fallback}')
        return fallback

def discover_zip(pattern: str, roots: Sequence[Path]) -> Path:
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            candidates.extend(root.glob(pattern))
            candidates.extend(root.glob(f'**/{pattern}'))
        except Exception:
            continue
    candidates = sorted({p.resolve() for p in candidates if p.is_file()}, key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"ZIP not found for pattern: {pattern}\nSearched: {', '.join((str(r) for r in roots))}")
    return candidates[0]

def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        base = destination.resolve()
        for member in zf.infolist():
            member_path = (destination / member.filename).resolve()
            if not str(member_path).startswith(str(base)):
                raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        zf.extractall(destination)

def find_root_containing(extracted_dir: Path, distinctive_relative_path: str) -> Path:
    direct = extracted_dir / distinctive_relative_path
    if direct.exists():
        return extracted_dir
    matches = list(extracted_dir.rglob(Path(distinctive_relative_path).name))
    suffix = Path(distinctive_relative_path).parts
    for match in matches:
        parts = match.parts
        if len(parts) >= len(suffix) and tuple(parts[-len(suffix):]) == tuple(suffix):
            return match.parents[len(suffix) - 1]
    if matches and len(Path(distinctive_relative_path).parts) == 1:
        return matches[0].parent
    raise FileNotFoundError(f'Could not locate {distinctive_relative_path} below {extracted_dir}')

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'pass'})

def latex_escape(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '---'
    s = str(value)
    replacements = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}'}
    if '$' in s or '\\' in s:
        return s
    return ''.join((replacements.get(ch, ch) for ch in s))

def fmt_pp(x: float, signed: bool=False, digits: int=2) -> str:
    x_pp = 100.0 * float(x)
    return f'{x_pp:+.{digits}f}' if signed else f'{x_pp:.{digits}f}'

def fmt_p(p: float) -> str:
    if pd.isna(p):
        return '---'
    p = float(p)
    if p < 0.001:
        exp = int(math.floor(math.log10(p))) if p > 0 else -999
        coeff = p / 10 ** exp if p > 0 else 0.0
        return f'${coeff:.2f}\\times 10^{{{exp}}}$'
    return f'{p:.3f}'

def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)

def replace_environment_by_label(tex: str, label: str, new_block: str) -> str:
    marker = f'\\label{{{label}}}'
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f'Label not found in manuscript: {label}')
    starts = [(tex.rfind('\\begin{table*}', 0, idx), 'table*'), (tex.rfind('\\begin{table}', 0, idx), 'table')]
    start, env = max(starts, key=lambda item: item[0])
    if start < 0:
        raise ValueError(f'Table start not found for label: {label}')
    end_token = f'\\end{{{env}}}'
    end = tex.find(end_token, idx)
    if end < 0:
        raise ValueError(f'Table end not found for label: {label}')
    end += len(end_token)
    return tex[:start] + new_block.strip() + tex[end:]

def replace_between(tex: str, start_marker: str, end_marker: str, replacement: str, include_end: bool=False) -> str:
    start = tex.find(start_marker)
    if start < 0:
        raise ValueError(f'Start marker not found: {start_marker[:100]}')
    end = tex.find(end_marker, start)
    if end < 0:
        raise ValueError(f'End marker not found: {end_marker[:100]}')
    if include_end:
        end += len(end_marker)
    return tex[:start] + replacement.strip() + tex[end:]

def replace_abstract(tex: str, abstract_text: str) -> str:
    pattern = re.compile('\\\\begin\\{abstract\\}.*?\\\\end\\{abstract\\}', re.S)
    replacement = '\\begin{abstract}\n' + abstract_text.strip() + '\n\\end{abstract}'
    tex2, n = pattern.subn(lambda _: replacement, tex, count=1)
    if n != 1:
        raise ValueError('Abstract block could not be replaced')
    return tex2

def insert_after_environment_label(tex: str, label: str, block: str) -> str:
    marker = f'\\label{{{label}}}'
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f'Label not found: {label}')
    environments = ('table*', 'table', 'figure*', 'figure')
    starts = [(tex.rfind(f'\\begin{{{env}}}', 0, idx), env) for env in environments]
    start, env = max(starts, key=lambda item: item[0])
    if start < 0:
        raise ValueError(f'Label is not inside a supported table/figure environment: {label}')
    end_token = f'\\end{{{env}}}'
    end = tex.find(end_token, idx)
    if end < 0:
        raise ValueError(f'Environment end not found: {label}')
    end += len(end_token)
    return tex[:end] + '\n\n' + block.strip() + tex[end:]

def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

def _first_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_file():
            return path
    return None

def _rsicd_explicit_layout(data_root: Path) -> Tuple[Path, Path, List[Path]]:
    roots = [data_root / 'RSICD', data_root / 'rsicd']
    dataset_root = next((p for p in roots if p.is_dir()), roots[0])
    annotations = [dataset_root / 'dataset_rsicd.json', dataset_root / 'RSICD.json', dataset_root / 'rsicd.json', dataset_root / 'dataset.json', data_root / 'dataset_rsicd.json']
    annotation = _first_existing_path(annotations)
    if annotation is None:
        checked = '\n'.join((f'  - {p}' for p in annotations))
        raise FileNotFoundError('RSICD annotation was not found at the known explicit locations. No Drive-wide search was performed. Checked:\n' + checked)
    image_roots = [dataset_root / 'RSICD_images', dataset_root / 'images', dataset_root / 'Images', dataset_root / 'image', dataset_root]
    return (dataset_root, annotation, image_roots)

def _caption_texts(sentence_entries: object) -> List[str]:
    texts: List[str] = []
    if not isinstance(sentence_entries, list):
        return texts
    for entry in sentence_entries:
        if isinstance(entry, dict):
            text = entry.get('raw') or entry.get('sentence') or entry.get('caption') or ''
        else:
            text = str(entry)
        text = ' '.join(str(text).split()).strip()
        if text:
            texts.append(text)
    return texts

def _resolve_rsicd_image(filename: str, dataset_root: Path, image_roots: Sequence[Path]) -> Optional[Path]:
    rel = Path(str(filename).replace('\\', '/'))
    candidates = [dataset_root / rel]
    candidates.extend((root / rel.name for root in image_roots))
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None

def load_rsicd_training_pairs_explicit(data_root: Path) -> Tuple[List[Dict[str, str]], Path]:
    dataset_root, annotation_path, image_roots = _rsicd_explicit_layout(data_root)
    raw = json.loads(annotation_path.read_text(encoding='utf-8'))
    images = raw.get('images', []) if isinstance(raw, dict) else []
    records: List[Dict[str, str]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        split = str(item.get('split') or 'train').lower()
        if split not in {'train', 'training'}:
            continue
        filename = item.get('filename') or item.get('file_name') or item.get('imgid')
        captions = _caption_texts(item.get('sentences', []))
        if not filename or not captions:
            continue
        image_path = _resolve_rsicd_image(str(filename), dataset_root, image_roots)
        if image_path is None:
            continue
        records.append({'image_path': str(image_path), 'filename': image_path.name, 'matched_caption': captions[0], 'all_captions': ' || '.join(captions[:5])})
    if not records:
        raise RuntimeError(f'No usable RSICD training image-caption pairs were resolved from {annotation_path}. Expected images under RSICD/RSICD_images or another explicit RSICD image folder.')
    return (records, annotation_path)

def _rsicd_scene_category(filename: str) -> str:
    stem = Path(filename).stem.lower().strip()
    stem = re.sub('(?:[_-]?\\d+)$', '', stem)
    stem = re.sub('[_-]+', ' ', stem).strip()
    aliases = {'harbor': 'port', 'parking lot': 'parking', 'storage tank': 'storage tanks', 'railwaystation': 'railway station', 'sparseresidential': 'sparse residential'}
    return aliases.get(stem, stem)

def _best_caption_for_category(rec: Dict[str, str], category: str, keywords: Sequence[str]) -> str:
    captions = [c.strip() for c in str(rec.get('all_captions', '')).split('||') if c.strip()]
    if not captions:
        captions = [str(rec.get('matched_caption', '')).strip()]

    def score(caption: str) -> Tuple[int, int]:
        t = caption.lower()
        hits = sum((1 for w in keywords if w in t))
        return (hits, -len(caption))
    return max(captions, key=score)

def _select_representative_pairs(records: Sequence[Dict[str, str]], n: int, seed: int) -> List[Dict[str, str]]:
    preferred = [('airport', ('airport', 'runway', 'airplane', 'aircraft')), ('forest', ('forest', 'woodland', 'tree', 'trees', 'vegetation')), ('port', ('port', 'harbor', 'ship', 'ships', 'dock', 'container')), ('farmland', ('farmland', 'farm', 'field', 'agricultural', 'crop')), ('residential', ('residential', 'building', 'buildings', 'houses', 'urban')), ('river', ('river', 'water', 'stream'))]
    keyword_map = {cat: words for cat, words in preferred}
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for rec0 in records:
        rec = dict(rec0)
        category = _rsicd_scene_category(rec.get('filename', rec.get('image_path', '')))
        if not category:
            continue
        rec['scene_category'] = category
        by_category.setdefault(category, []).append(rec)
    rng = np.random.RandomState(seed)
    selected: List[Dict[str, str]] = []
    used_categories = set()
    for category, keywords in preferred:
        candidates = by_category.get(category, [])
        if not candidates:
            continue
        order = rng.permutation(len(candidates)).tolist()
        candidates = [candidates[i] for i in order]
        candidates.sort(key=lambda r: sum((w in str(r.get('all_captions', r.get('matched_caption', ''))).lower() for w in keywords)), reverse=True)
        rec = dict(candidates[0])
        rec['matched_caption'] = _best_caption_for_category(rec, category, keywords)
        selected.append(rec)
        used_categories.add(category)
        if len(selected) >= n:
            break
    if len(selected) < n:
        remaining_categories = [c for c in sorted(by_category) if c not in used_categories]
        remaining_categories = [remaining_categories[i] for i in rng.permutation(len(remaining_categories)).tolist()]
        for category in remaining_categories:
            candidates = by_category[category]
            rec = dict(candidates[int(rng.randint(len(candidates)))])
            rec['scene_category'] = category
            rec['matched_caption'] = _best_caption_for_category(rec, category, keyword_map.get(category, (category,)))
            selected.append(rec)
            used_categories.add(category)
            if len(selected) >= n:
                break
    if len(selected) < n:
        raise RuntimeError(f'Could select only {len(selected)} distinct RSICD scene categories; {n} were requested.')
    categories = [r['scene_category'] for r in selected]
    if len(set(categories)) != len(categories):
        raise RuntimeError(f'Pairing figure category diversity check failed: {categories}')
    return selected[:n]

def generate_pairing_figure(cfg: Config, output_png: Path, manifest_csv: Path) -> Dict[str, object]:
    import textwrap as _textwrap
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    records, annotation_path = load_rsicd_training_pairs_explicit(cfg.data_root)
    selected = _select_representative_pairs(records, cfg.pairing_figure_examples, cfg.pairing_figure_seed)
    if len(selected) < 3:
        raise RuntimeError('At least three category-diverse RSICD pairs are required for the illustration.')
    matched = [r['matched_caption'] for r in selected]
    categories = [r['scene_category'] for r in selected]
    shuffled = matched[1:] + matched[:1]
    shuffled_categories = categories[1:] + categories[:1]
    for rec, wrong, wrong_category in zip(selected, shuffled, shuffled_categories):
        if wrong_category == rec['scene_category']:
            raise RuntimeError('Shuffled caption came from the same scene category; derangement failed.')
        rec['shuffled_caption'] = wrong
        rec['shuffled_from_category'] = wrong_category
        rec['source_split'] = 'train'
        rec['annotation_file'] = str(annotation_path)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(manifest_csv, index=False)
    n = len(selected)
    fig = plt.figure(figsize=(13.2, 2.75 * n + 1.1))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[0.28] + [1] * n, width_ratios=[1.05, 1.25, 1.25])
    headers = ['RSICD training image', 'Matched caption (semantic alignment)', 'Cross-category shuffled caption']
    for col, title in enumerate(headers):
        ax = fig.add_subplot(gs[0, col])
        ax.axis('off')
        ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=11.5, fontweight='bold')
    for row, rec in enumerate(selected, start=1):
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.axis('off')
        with Image.open(rec['image_path']) as im:
            image = ImageOps.fit(im.convert('RGB'), (360, 260), method=Image.Resampling.LANCZOS)
        ax_img.imshow(image)
        ax_img.set_title(f"{rec['scene_category'].title()} — {rec['filename']}", fontsize=8)
        ax_match = fig.add_subplot(gs[row, 1])
        ax_match.axis('off')
        ax_match.text(0.02, 0.88, f"✓ Correct: {rec['scene_category']}", fontsize=11, fontweight='bold', va='top')
        ax_match.text(0.02, 0.7, _textwrap.fill(rec['matched_caption'], 46), fontsize=10.5, va='top', linespacing=1.35)
        ax_shuffle = fig.add_subplot(gs[row, 2])
        ax_shuffle.axis('off')
        ax_shuffle.text(0.02, 0.88, f"✗ Caption from: {rec['shuffled_from_category']}", fontsize=11, fontweight='bold', va='top')
        ax_shuffle.text(0.02, 0.7, _textwrap.fill(rec['shuffled_caption'], 46), fontsize=10.5, va='top', linespacing=1.35)
    fig.suptitle('Qualitative manipulation used by the caption-control experiments', fontsize=13.5, fontweight='bold', y=0.995)
    fig.text(0.5, 0.012, 'The selected rows come from distinct RSICD scene categories. The shuffled condition keeps exactly the same images and caption set but assigns every caption to a different scene category.', ha='center', va='bottom', fontsize=9.2)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return {'figure': str(output_png), 'manifest': str(manifest_csv), 'annotation': str(annotation_path), 'n_available_train_pairs': len(records), 'n_examples': len(selected), 'selected_categories': categories, 'category_diversity_pass': len(set(categories)) == len(categories), 'cross_category_shuffle_pass': all((a != b for a, b in zip(categories, shuffled_categories)))}

def pairing_figure_latex() -> str:
    return '\\begin{figure*}[!t]\n\\centering\n\\includegraphics[width=0.96\\textwidth]{figs/figure_matched_vs_shuffled_examples.png}\n\\caption{Representative RSICD training examples illustrating the caption-control manipulation. The matched condition preserves genuine image--caption correspondence, whereas the shuffled condition uses the same selected images and the same caption set but cyclically reassigns captions to different images. The image-only control retains external image exposure while removing linguistic supervision. The displayed samples are selected deterministically from the training split and are not selected using model predictions or test performance.}\n\\label{fig:pairing_examples}\n\\end{figure*}'

@dataclass
class AuditResult:
    checks: List[Tuple[str, bool, str]]

    @property
    def passed(self) -> bool:
        return all((ok for _, ok, _ in self.checks))

    def add(self, name: str, ok: bool, evidence: str) -> None:
        self.checks.append((name, bool(ok), evidence))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks, columns=['check', 'pass', 'evidence'])

def run_audit(roots: Dict[str, Path]) -> AuditResult:
    audit = AuditResult(checks=[])
    post_tables = roots['post'] / 'tables'
    final_checks = read_csv(post_tables / 'FINAL_CHECKS.csv')
    final_ok = bool(bool_series(final_checks['pass']).all())
    audit.add('Canonical postprocess FINAL_CHECKS', final_ok, f"{final_checks['pass'].sum()}/{len(final_checks)} reported true")
    protocol = json.loads((roots['pm'] / 'protocol_audit.json').read_text(encoding='utf-8'))
    manifest_ok = bool(protocol.get('protocol', {}).get('manifest_verified', False))
    all_match = all((bool(row.get('all_match', False)) for row in protocol.get('audit', [])))
    audit.add('PM manifest verified', manifest_ok, str(protocol.get('protocol', {}).get('manifest_path', '')))
    audit.add('PM fold/support protocol matches', all_match, f"{len(protocol.get('audit', []))} audited dataset-fold entries")
    target_validation = protocol.get('protocol', {}).get('target_validation', None)
    audit.add('No target validation for PM methods', target_validation is False, f'target_validation={target_validation}')
    fidelity = read_csv(roots['pm'] / 'tables' / 'method_fidelity_audit.csv')
    fidelity_ok = bool_series(fidelity['same_capstone_split']).all() and bool_series(fidelity['same_shots']).all() and bool_series(fidelity['same_support_seeds']).all() and bool_series(fidelity['same_backbone_key']).all() and (~bool_series(fidelity['official_repository_unchanged'])).all()
    audit.add('PM method fidelity labels', fidelity_ok, '; '.join(fidelity['allowed_table_label'].astype(str)))
    combined = read_csv(post_tables / 'raw_combined_capstone_and_pm.csv')
    main_rows = int((combined['source_notebook'] == 'CAPSTONE-main').sum()) if 'source_notebook' in combined else -1
    pm_rows = int(combined['method'].isin(['APPLeNet-PM', 'AVION-PM']).sum())
    audit.add('Combined raw row count', len(combined) == 11100, f'rows={len(combined)}')
    audit.add('Main completed rows', main_rows == 9600, f'rows={main_rows}')
    audit.add('PM completed rows', pm_rows == 1500, f'rows={pm_rows}')
    required_methods = {'CAPSTONE-CLIP', 'APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared'}
    missing_methods = sorted(required_methods - set(combined['method'].dropna().astype(str)))
    audit.add('Required comparison methods present', not missing_methods, f'missing={missing_methods}')
    global_inf = read_csv(post_tables / 'paper_table_global_pairwise_inference.csv')
    required_pairs = set(CORE_CONFIRMATORY_REFERENCES)
    found_pairs = set(global_inf.loc[(global_inf['scope'] == 'global') & (global_inf['metric'] == 'Accuracy'), 'reference'])
    audit.add('Required global pairwise comparisons present', required_pairs.issubset(found_pairs), f'found={sorted(found_pairs)}')
    hoso = global_inf[(global_inf['scope'] == 'global') & (global_inf['metric'] == 'Accuracy') & (global_inf['reference'] == 'HOSO-Adapter')]
    hoso_ok = len(hoso) == 1 and int(hoso.iloc[0]['n_paired_folds']) == 120
    audit.add('HOSO common-support scope', hoso_ok, 'Expected 120 paired folds over K>=2')
    return audit

def make_overall_table(overall: pd.DataFrame, include_hoso: bool) -> str:
    df = overall.copy()
    if not include_hoso:
        df = df[df['method'] != 'HOSO-Adapter'].copy()
    df = df.sort_values('macro_average_Accuracy', ascending=False).reset_index(drop=True)
    rows: List[str] = []
    best_acc = df['macro_average_Accuracy'].max()
    second_acc = df.loc[df['macro_average_Accuracy'] < best_acc, 'macro_average_Accuracy'].max()
    for _, r in df.iterrows():
        method = display_method(str(r['method']))
        acc = 100 * float(r['macro_average_Accuracy'])
        f1 = 100 * float(r['macro_average_MacroF1'])
        ece = 100 * float(r['macro_average_ECE'])
        nll = float(r['macro_average_NLL'])
        acc_text = f'{acc:.2f}'
        if np.isclose(float(r['macro_average_Accuracy']), best_acc):
            acc_text = f'\\textbf{{{acc_text}}}'
        elif np.isclose(float(r['macro_average_Accuracy']), second_acc):
            acc_text = f'\\underline{{{acc_text}}}'
        rows.append(f'{method} & {acc_text} & {f1:.2f} & {ece:.2f} & {nll:.3f}\\\\')
    note = '\\multicolumn{5}{p{0.96\\columnwidth}}{\\scriptsize $^{\\ddagger}$Protocol-matched reimplementation under the identical CAPSTONE split/support manifest; not an unchanged official-repository result. HOSO-Adapter is omitted because it is undefined at $K{=}1$ and is reported separately on the common $K\\geq2$ support.}' if not include_hoso else '\\multicolumn{5}{p{0.96\\columnwidth}}{\\scriptsize $^{\\ddagger}$Protocol-matched reimplementation; HOSO-Adapter is averaged only over $K\\geq2$.}'
    return textwrap.dedent(f'\n    \\begin{{table}}[!t]\n    \\caption{{Overall macro-average over six datasets and five shot levels.\\label{{tab:overall}}}}\n    \\centering\n    \\scriptsize\n    \\setlength{{\\tabcolsep}}{{2.8pt}}\n    \\begin{{tabular}}{{lcccc}}\n    \\toprule\n    \\textbf{{Method}} & \\textbf{{Acc. (\\%)}} & \\textbf{{F1 (\\%)}} & \\textbf{{ECE (\\%)}} & \\textbf{{NLL}}\\\\\n    \\midrule\n    {os.linesep.join(rows)}\n    \\bottomrule\n    {note}\n    \\end{{tabular}}\n    \\end{{table}}\n    ').strip()

def make_confirmatory_table(global_inf: pd.DataFrame, wtl: pd.DataFrame, cfg: Config) -> str:
    df = global_inf[(global_inf['scope'] == 'global') & (global_inf['metric'] == 'Accuracy') & (global_inf['candidate'] == 'CAPSTONE-CLIP') & global_inf['reference'].isin(CORE_CONFIRMATORY_REFERENCES)].copy()
    df['reference'] = pd.Categorical(df['reference'], categories=CORE_CONFIRMATORY_REFERENCES, ordered=True)
    df = df.sort_values('reference')
    joined = df.merge(wtl[['reference', 'wins', 'ties', 'losses', 'n_common_cells']], on='reference', how='left', validate='one_to_one')
    rows: List[str] = []
    for _, r in joined.iterrows():
        ref_raw = str(r['reference'])
        ref = display_method(ref_raw)
        delta = fmt_pp(float(r['benefit_delta_positive_is_better']), signed=True)
        lo = fmt_pp(float(r['hier_boot_ci95_low']), signed=True)
        hi = fmt_pp(float(r['hier_boot_ci95_high']), signed=True)
        wtl_text = f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}"
        dz = float(r['cohen_dz'])
        p = fmt_p(float(r['wilcoxon_p_holm']))
        equivalent = bool(r['equivalent_within_margin'])
        interpretation = 'equivalent' if equivalent else 'CAPSTONE higher' if float(r['benefit_delta_positive_is_better']) > 0 else 'reference higher'
        rows.append(f'{ref} & ${delta}$ & $[{lo},\\,{hi}]$ & {wtl_text} & {dz:.2f} & {p} & {interpretation}\\\\')
    return textwrap.dedent(f'\n    \\begin{{table*}}[!t]\n    \\caption{{Global paired accuracy comparisons. Differences are CAPSTONE-CLIP minus the reference method in percentage points. Support seeds are averaged within each external fold before inference; 95\\% intervals use dataset--shot/fold cluster bootstrap, and Wilcoxon tests are Holm-adjusted within the analysis family. W/T/L uses a descriptive $\\pm{cfg.win_tie_margin_pp:.1f}$-point cell margin; equivalence uses a pre-specified $\\pm{cfg.equivalence_margin_pp:.1f}$-point TOST margin.\\label{{tab:confirmatory}}}}\n    \\centering\n    \\scriptsize\n    \\setlength{{\\tabcolsep}}{{3.2pt}}\n    \\begin{{tabular}}{{lcccccc}}\n    \\toprule\n    \\textbf{{Reference}} & \\textbf{{Mean $\\Delta$}} & \\textbf{{Hier. 95\\% CI}} & \\textbf{{W/T/L}} & $\\boldsymbol{{d_z}}$ & \\textbf{{Holm-$p$}} & \\textbf{{Practical reading}}\\\\\n    \\midrule\n    {os.linesep.join(rows)}\n    \\bottomrule\n    \\multicolumn{{7}}{{p{{0.97\\textwidth}}}}{{\\scriptsize $^{{\\ddagger}}$APPLeNet-PM and AVION-PM are protocol-matched reimplementations using the identical CAPSTONE manifest, backbone key, shots, folds, and support seeds; they are not unchanged official-code results.}}\n    \\end{{tabular}}\n    \\end{{table*}}\n    ').strip()

def make_pm_dataset_table(per_dataset: pd.DataFrame) -> str:
    df = per_dataset[(per_dataset['scope'] == 'dataset') & (per_dataset['metric'] == 'Accuracy') & per_dataset['reference'].isin(['APPLeNet-PM', 'AVION-PM'])].copy()
    pivot: Dict[str, Dict[str, pd.Series]] = {}
    for _, row in df.iterrows():
        pivot.setdefault(str(row['dataset']), {})[str(row['reference'])] = row
    rows: List[str] = []
    dataset_order = ['AID', 'EuroSAT', 'NWPU_RESISC45', 'PatternNet', 'UCMerced', 'WHU_RS19']
    for ds in dataset_order:
        a = pivot[ds]['APPLeNet-PM']
        v = pivot[ds]['AVION-PM']
        a_delta = fmt_pp(a['benefit_delta_positive_is_better'], signed=True)
        a_lo = fmt_pp(a['hier_boot_ci95_low'], signed=True)
        a_hi = fmt_pp(a['hier_boot_ci95_high'], signed=True)
        v_delta = fmt_pp(v['benefit_delta_positive_is_better'], signed=True)
        v_lo = fmt_pp(v['hier_boot_ci95_low'], signed=True)
        v_hi = fmt_pp(v['hier_boot_ci95_high'], signed=True)
        ds_tex = ds.replace('_', '\\_')
        rows.append(f'{ds_tex} & ${a_delta}$ $[{a_lo},{a_hi}]$ & ${v_delta}$ $[{v_lo},{v_hi}]$\\\\')
    return textwrap.dedent(f'\n    \\begin{{table}}[!t]\n    \\caption{{Dataset-level accuracy differences (percentage points): CAPSTONE-CLIP minus each protocol-matched reimplementation. Intervals are hierarchical 95\\% bootstrap intervals.\\label{{tab:pm_dataset}}}}\n    \\centering\n    \\scriptsize\n    \\setlength{{\\tabcolsep}}{{3pt}}\n    \\begin{{tabular}}{{lcc}}\n    \\toprule\n    \\textbf{{Dataset}} & \\textbf{{vs. APPLeNet-PM$^{{\\ddagger}}$}} & \\textbf{{vs. AVION-PM$^{{\\ddagger}}$}}\\\\\n    \\midrule\n    {os.linesep.join(rows)}\n    \\bottomrule\n    \\end{{tabular}}\n    \\end{{table}}\n    ').strip()

def make_efficiency_table(efficiency: pd.DataFrame) -> str:
    selected = ['CAPSTONE-CLIP', 'APPLeNet-PM', 'AVION-PM', 'Capacity-Matched-Shared', 'Shared-Caption', 'ProLIP', 'RLA', 'CLIP-Adapter']
    df = efficiency[efficiency['method'].isin(selected)].copy()
    df['method'] = pd.Categorical(df['method'], categories=selected, ordered=True)
    df = df.sort_values('method')
    rows = []
    for _, r in df.iterrows():
        method = display_method(str(r['method']))
        sec = f"{float(r['train_eval_s']):.2f}" if pd.notna(r['train_eval_s']) else '---'
        params = f"{float(r['trainable_params']) / 1000.0:.1f}" if pd.notna(r['trainable_params']) else '---'
        gpu_val = r['peak_gpu_delta_mb'] if pd.notna(r['peak_gpu_delta_mb']) else r['peak_gpu_mb']
        gpu = f'{float(gpu_val):.1f}' if pd.notna(gpu_val) else '---'
        acc = f"{100 * float(r['Accuracy']):.2f}"
        rows.append(f'{method} & {acc} & {sec} & {params} & {gpu}\\\\')
    return textwrap.dedent(f'\n    \\begin{{table}}[!t]\n    \\caption{{Run-level efficiency summary. Timing is measured in the supplied frozen-feature implementations and should not be interpreted as official repository benchmarking.\\label{{tab:efficiency_v53}}}}\n    \\centering\n    \\scriptsize\n    \\setlength{{\\tabcolsep}}{{3pt}}\n    \\begin{{tabular}}{{lrrrr}}\n    \\toprule\n    \\textbf{{Method}} & \\textbf{{Acc. (\\%)}} & \\textbf{{Train/eval (s)}} & \\textbf{{Trainable k}} & \\textbf{{Peak/Delta MB}}\\\\\n    \\midrule\n    {os.linesep.join(rows)}\n    \\bottomrule\n    \\end{{tabular}}\n    \\end{{table}}\n    ').strip()

def make_hoso_common_support_table(fold_level: pd.DataFrame) -> str:
    df = fold_level[(fold_level['K'] >= 2) & fold_level['method'].isin(['CAPSTONE-CLIP', 'HOSO-Adapter'])].copy()
    summary = df.groupby('method', as_index=False).agg(accuracy=('Accuracy', 'mean'), macro_f1=('MacroF1', 'mean'), ece=('ECE', 'mean'), n=('Accuracy', 'count'))
    rows = []
    for method in ['CAPSTONE-CLIP', 'HOSO-Adapter']:
        r = summary[summary['method'] == method].iloc[0]
        rows.append(f"{method} & {100 * r['accuracy']:.2f} & {100 * r['macro_f1']:.2f} & {100 * r['ece']:.2f} & {int(r['n'])}\\\\")
    return textwrap.dedent(f'\n    \\begin{{table}}[!t]\n    \\caption{{Common-support comparison for HOSO-Adapter, restricted to $K\\in\\{{2,4,8,16\\}}$. Values average the 120 dataset--shot--fold blocks after support-seed averaging.\\label{{tab:hoso_common}}}}\n    \\centering\n    \\footnotesize\n    \\begin{{tabular}}{{lcccc}}\n    \\toprule\n    \\textbf{{Method}} & \\textbf{{Acc. (\\%)}} & \\textbf{{F1 (\\%)}} & \\textbf{{ECE (\\%)}} & \\textbf{{Blocks}}\\\\\n    \\midrule\n    {os.linesep.join(rows)}\n    \\bottomrule\n    \\end{{tabular}}\n    \\end{{table}}\n    ').strip()

def get_global_row(global_inf: pd.DataFrame, reference: str, metric: str='Accuracy') -> pd.Series:
    rows = global_inf[(global_inf['scope'] == 'global') & (global_inf['candidate'] == 'CAPSTONE-CLIP') & (global_inf['reference'] == reference) & (global_inf['metric'] == metric)]
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one global row for {reference}/{metric}; got {len(rows)}')
    return rows.iloc[0]

def manuscript_prose(global_inf: pd.DataFrame, overall: pd.DataFrame, per_dataset: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    app = get_global_row(global_inf, 'APPLeNet-PM')
    avion = get_global_row(global_inf, 'AVION-PM')
    prolip = get_global_row(global_inf, 'ProLIP')
    rla = get_global_row(global_inf, 'RLA')
    shared = get_global_row(global_inf, 'Shared-Caption')
    capmatch = get_global_row(global_inf, 'Capacity-Matched-Shared')
    cap_acc = float(overall.loc[overall['method'] == 'CAPSTONE-CLIP', 'macro_average_Accuracy'].iloc[0])
    app_acc = float(overall.loc[overall['method'] == 'APPLeNet-PM', 'macro_average_Accuracy'].iloc[0])
    app_ece = float(overall.loc[overall['method'] == 'APPLeNet-PM', 'macro_average_ECE'].iloc[0])
    cap_ece = float(overall.loc[overall['method'] == 'CAPSTONE-CLIP', 'macro_average_ECE'].iloc[0])
    abstract = f"\nVision--language foundation models provide strong zero-shot representations for remote sensing scene understanding, yet how these representations should be adapted when only a few labels per class are available remains open. Reported improvements often entangle three factors: semantic supervision from matched image--caption pairs, additional image exposure, and adapter capacity. We evaluate these factors with CAPSTONE-CLIP, a shared--private reference architecture, and matched controls under a leakage-controlled, validation-free protocol spanning six datasets, five shot levels, five outer folds, and five support seeds. Caption-supervised shared adaptation attains {100 * cap_acc:.2f}\\% mean accuracy. Relative to validation-free baselines, CAPSTONE-CLIP improves over ProLIP by {fmt_pp(prolip['benefit_delta_positive_is_better'])} points and RLA by {fmt_pp(rla['benefit_delta_positive_is_better'])} points. Against APPLeNet-PM, a protocol-matched reimplementation, the overall advantage is more modest at {fmt_pp(app['benefit_delta_positive_is_better'])} points (hierarchical 95\\% CI {fmt_pp(app['hier_boot_ci95_low'], signed=True)} to {fmt_pp(app['hier_boot_ci95_high'], signed=True)}; Holm-adjusted Wilcoxon $p={float(app['wilcoxon_p_holm']):.3f}$), with dataset-dependent wins and losses; AVION-PM is lower by {fmt_pp(avion['benefit_delta_positive_is_better'])} points. CAPSTONE-CLIP remains practically equivalent to Shared-Caption and Capacity-Matched-Shared, indicating that the principal classification gain is associated with matched semantic supervision rather than additional adapter capacity or image exposure alone. Behavioral analyses further show that adaptation largely preserves the pretrained representation geometry while supporting classification and retrieval. The contribution is methodological: a reusable controlled protocol for attributing few-shot adaptation gains to the evaluated design factors, rather than a claim of universal architectural superiority.\n".strip()
    h1 = f"\nCaption-supervised shared adaptation reaches {100 * cap_acc:.2f}\\% overall (Table~\\ref{{tab:overall}}), exceeding ProLIP by {fmt_pp(prolip['benefit_delta_positive_is_better'])} points and RLA by {fmt_pp(rla['benefit_delta_positive_is_better'])} points under leakage-controlled evaluation. The protocol-matched external comparisons refine this result: CAPSTONE-CLIP exceeds AVION-PM by {fmt_pp(avion['benefit_delta_positive_is_better'])} points, whereas the mean advantage over APPLeNet-PM is {fmt_pp(app['benefit_delta_positive_is_better'])} points with a hierarchical 95\\% CI of [{fmt_pp(app['hier_boot_ci95_low'], signed=True)}, {fmt_pp(app['hier_boot_ci95_high'], signed=True)}] and Holm-adjusted Wilcoxon $p={float(app['wilcoxon_p_holm']):.3f}$. This APPLeNet-PM difference is small and heterogeneous: CAPSTONE-CLIP records 14 wins, 3 practical ties, and 13 losses across the 30 dataset--shot cells, with gains concentrated on AID and WHU-RS19 and a consistent deficit on PatternNet (Table~\\ref{{tab:pm_dataset}}). H1's distinguishing prediction remains supported against the class-name-only validation-free baselines: the margin over ProLIP increases from 1.01 points at $K{{=}}16$ to 4.23 points at $K{{=}}1$, and the corresponding paired effect is large ($d_z={float(prolip['cohen_dz']):.2f}$). The external comparisons therefore support a narrower conclusion than universal method dominance: matched semantic supervision is beneficial under scarce labels, while the best architecture is dataset dependent. \\textbf{{Verdict: H1 supported for the supervision hypothesis; universal superiority is not claimed.}}\n".strip()
    h3 = f"\nH3 asks whether additional trainable capacity yields systematic gains once semantic supervision is fixed. The evidence does not support that prediction. CAPSTONE-CLIP differs from Shared-Caption by only {fmt_pp(shared['benefit_delta_positive_is_better'], signed=True)} points (hierarchical 95\\% CI [{fmt_pp(shared['hier_boot_ci95_low'], signed=True)}, {fmt_pp(shared['hier_boot_ci95_high'], signed=True)}]) and from Capacity-Matched-Shared by {fmt_pp(capmatch['benefit_delta_positive_is_better'], signed=True)} points (95\\% CI [{fmt_pp(capmatch['hier_boot_ci95_low'], signed=True)}, {fmt_pp(capmatch['hier_boot_ci95_high'], signed=True)}]); both comparisons are practically equivalent within the pre-specified $\\pm{cfg.equivalence_margin_pp:.1f}$-point margin. Small retrieval or dataset-specific differences are retained as exploratory observations, but they do not rescue the confirmatory capacity hypothesis. \\textbf{{Verdict: H3 not supported.}} Within the evaluated protocol, matched semantic supervision explains the classification improvement more consistently than parameter count, while CAPSTONE-CLIP should be interpreted as a measurement architecture rather than a universally superior adapter.\n".strip()
    methods = "\nUnder the same frozen-feature protocol we compare Zero-shot, Linear Probe, CLIP-Adapter \\cite{r8}, Tip-Adapter and Tip-Adapter-F \\cite{r9}, CoOp \\cite{r6}, TaskRes \\cite{r11}, RLA, ProLIP \\cite{r12}, HOSO-Adapter \\cite{r17}, and the hypothesis controls Shared-Caption and Capacity-Matched-Shared. We additionally report APPLeNet-PM and AVION-PM, which are protocol-matched reimplementations evaluated with the identical CAPSTONE split/support manifest, backbone key, shot levels, outer folds, and support seeds and without target validation. They are not unchanged official-repository implementations; the ``-PM'' suffix and table footnotes preserve this distinction. HOSO-Adapter is undefined at $K=1$, so it is excluded from the all-shot macro-average table and reported only on its common $K\\geq2$ support. Classification metrics are accuracy, macro-F1, balanced accuracy, NLL, and ECE; retrieval metrics are R@1, R@5, R@10, mean recall, MRR, nDCG, and mean rank. Hardware: NVIDIA L4.\n".strip()
    stats = f'\n\\textbf{{Fold and seed strategy.}} Five stratified outer folds are used per dataset, with five independent support-sampling seeds per fold. Support seeds are averaged within each external fold before confirmatory inference, producing 150 paired dataset--shot--fold blocks for methods defined at all shot levels. \\textbf{{Confidence intervals.}} Hierarchical 95\\% intervals are obtained with dataset--shot/fold cluster bootstrap. \\textbf{{Significance testing.}} Paired Wilcoxon signed-rank tests are Holm-adjusted within the global comparison family and are accompanied by paired effect sizes ($d_z$), mixed-effects estimates where available, and Friedman--Nemenyi rank analyses. \\textbf{{Practical interpretation.}} A pre-specified $\\pm{cfg.equivalence_margin_pp:.1f}$-percentage-point TOST margin is used for equivalence; the tighter $\\pm{cfg.win_tie_margin_pp:.1f}$-point threshold is used only to summarize descriptive win/tie/loss counts across dataset--shot cells. These thresholds serve different purposes and are not interchanged. Exploratory dataset-level breakdowns are reported separately from the global confirmatory family.\n'.strip()
    threats_external = '\n\\textbf{External validity.} The factor-isolation experiments are concentrated on optical RGB scene classification, and the behavioral suite is most complete for OpenCLIP ViT-B/32. APPLeNet-PM and AVION-PM extend the comparison under the identical CAPSTONE split and support protocol, but they are protocol-matched reimplementations rather than unchanged official-code reproductions. Consequently, the results support controlled within-protocol comparisons and do not claim exact reproduction of the original published systems. Applying the protocol to multispectral, SAR, hyperspectral, and multi-temporal settings remains future work.\n'.strip()
    threats_internal = '\n\\textbf{Internal validity.} There is no target-validation channel by construction; split and support manifests are SHA-256 locked, and the protocol audit passes for all dataset--fold entries. The sample-based near-duplicate scan finds no pairs above 0.999 cosine similarity for EuroSAT, but flags 4.8\\% and 2.0\\% of sampled images at the looser 0.990 and 0.995 thresholds, respectively. These high-similarity cases may include genuine near-duplicates or repeated land-cover patterns and therefore require cautious interpretation; the audit is reported as a limitation rather than treated as proof of leakage.\n'.strip()
    conclusion = f"\n\\textbf{{The primary contribution of this work is not another universally dominant CLIP adapter.}} It is a controlled evaluation of semantic supervision, image exposure, and adapter capacity under scarce target labels. Across six datasets, matched image--caption supervision improves substantially over ProLIP and RLA, while the advantage over APPLeNet-PM is modest ({fmt_pp(app['benefit_delta_positive_is_better'])} points overall) and dataset dependent. CAPSTONE-CLIP and the Shared-Caption and Capacity-Matched-Shared controls are practically equivalent for classification, so the capacity hypothesis is not supported. The same auxiliary image budget without correct captions does not reproduce the gain, and shuffled pairing is harmful.\n\nThe behavioral analyses indicate that successful adaptation is a low-dimensional, geometry-preserving deformation of the frozen representation. Within the evaluated protocol, the most defensible design guidance is therefore to prioritize correctly matched semantic supervision before increasing adapter size. CAPSTONE-CLIP serves as the reference instrument that makes this attribution measurable; the reusable contribution is the leakage-controlled protocol and its matched controls. Future work should reproduce the protocol with unchanged official implementations where feasible and extend it to non-RGB sensors, additional foundation-model families, and tasks beyond scene classification.\n".strip()
    return {'abstract': abstract, 'h1': h1, 'h3': h3, 'methods': methods, 'stats': stats, 'threats_external': threats_external, 'threats_internal': threats_internal, 'conclusion': conclusion, 'calibration_sentence': f'APPLeNet-PM has lower mean ECE ({100 * app_ece:.2f}%) than CAPSTONE-CLIP ({100 * cap_ece:.2f}%), while CAPSTONE-CLIP has {100 * (cap_acc - app_acc):.2f} points higher mean accuracy.'}

def patch_manuscript(main_tex: str, tables: Dict[str, str], prose: Dict[str, str]) -> Tuple[str, List[str]]:
    changes: List[str] = []
    tex = main_tex
    tex, n = re.subn('pre-registered', 'pre-specified', tex, flags=re.I)
    if n:
        changes.append(f'Replaced {n} occurrence(s) of pre-registered with pre-specified')
    tex, n2 = re.subn('pre-registration', 'pre-specification', tex, flags=re.I)
    if n2:
        changes.append(f'Replaced {n2} occurrence(s) of pre-registration with pre-specification')
    tex = replace_abstract(tex, prose['abstract'])
    changes.append('Replaced abstract with canonical v53 result language')
    tex = tex.replace('the controlled evidence supports H1, rejects H2, and partially supports H3.', 'the controlled evidence supports H1, rejects H2, and does not support H3.')
    tex = tex.replace('The contribution is therefore methodological: a reusable, leakage-controlled protocol for attributing adaptation gains to their actual causes.', 'The contribution is therefore methodological: a reusable, leakage-controlled protocol for attributing adaptation gains to the evaluated design factors.')
    changes.append('Updated introduction verdict and attribution wording')
    tex = replace_between(tex, 'Under the same frozen-feature protocol we compare Zero-shot, Linear Probe, CLIP-Adapter', 'Hardware: NVIDIA L4.', prose['methods'], include_end=True)
    changes.append('Added APPLeNet-PM/AVION-PM fidelity language and HOSO common-support rule')
    tex = replace_between(tex, '\\textbf{Fold strategy.}', 'central message.', prose['stats'], include_end=True)
    changes.append('Unified seed averaging, bootstrap, Holm, equivalence, and W/T/L language')
    tex = replace_environment_by_label(tex, 'tab:overall', tables['overall'])
    changes.append('Replaced overall results table')
    tex = replace_between(tex, 'Caption-supervised shared adaptation reaches', '\\textbf{Verdict: H1 supported.}', prose['h1'], include_end=True)
    changes.append('Replaced H1 results paragraph with protocol-matched external comparisons')
    tex = replace_environment_by_label(tex, 'tab:confirmatory', tables['confirmatory'])
    tex = insert_after_environment_label(tex, 'tab:confirmatory', tables['pm_dataset'])
    changes.append('Replaced confirmatory table and inserted dataset-level PM comparison table')
    tex = replace_between(tex, 'H3 asks whether more trainable parameters yield further gains once semantic supervision is fixed.', '\\subsection{Cross-Modal Retrieval with Extended Metrics}', prose['h3'] + '\n\n' + '\\subsection{Cross-Modal Retrieval with Extended Metrics}', include_end=True)
    changes.append('Changed H3 verdict from partially supported to not supported')
    tex = re.sub('H3 --- Increasing capacity under fixed semantics yields systematic gains & Partially supported & .*?\\\\\\\\', 'H3 --- Increasing capacity under fixed semantics yields systematic gains & Not supported & CAPSTONE-CLIP is practically equivalent to Shared-Caption and Capacity-Matched-Shared for classification\\\\\\\\', tex, flags=re.S)
    tex = tex.replace('H3 partially supported', 'H3 not supported')
    tex = tex.replace('partially supports H3', 'does not support H3')
    tex = replace_between(tex, '\\textbf{Internal validity.}', '\\textbf{Construct validity.}', prose['threats_internal'] + '\n\n' + '\\textbf{Construct validity.}', include_end=True)
    tex = replace_between(tex, '\\textbf{External validity.}', '\\textbf{Statistical inference.}', prose['threats_external'] + '\n\n' + '\\textbf{Statistical inference.}', include_end=True)
    changes.append('Updated internal and external validity language')
    tex = replace_between(tex, '\\section{Conclusion}', '\\begin{thebibliography}{00}', '\\section{Conclusion}' + '\n' + '\\label{sec:conclusion}' + '\n' + prose['conclusion'] + '\n\n' + '\\begin{thebibliography}{00}', include_end=True)
    tex = tex.replace('\\section{Conclusion}\\n\\label{sec:conclusion}\\n\\label{sec:conclusion}', '\\section{Conclusion}\\n\\label{sec:conclusion}')
    changes.append('Replaced conclusion with narrower, Q1-defensible claims')
    stale_replacements = {'systematically exceeds validation-free baselines': 'exceeds several validation-free baselines', 'actual causes': 'evaluated contributing factors', 'H3 partially supported': 'H3 not supported', 'Verdict: H3 partially supported': 'Verdict: H3 not supported', 'The five behavioral analyses form a causal chain rather than a checklist': 'The five behavioral analyses form a converging evidence chain rather than a checklist', 'causal-chain evidence': 'mechanistic evidence chain'}
    for old, new in stale_replacements.items():
        if old in tex:
            tex = tex.replace(old, new)
            changes.append(f'Replaced stale phrase: {old}')
    return (tex, changes)

def make_manual_review_report(tex: str, roots: Dict[str, Path], prose: Dict[str, str]) -> str:
    risky_patterns = [('pre-registered', 'Use pre-specified unless a public timestamped preregistration exists.'), ('actual cause', 'Prefer controlled attribution within the evaluated protocol.'), ('causal', 'Check that causal wording is framed as evidence or design rationale, not proof.'), ('systematically', 'Verify that the statement remains true after APPLeNet-PM results.'), ('official AVION', 'PM rows must not be described as official unchanged code.'), ('official APPLeNet', 'PM rows must not be described as official unchanged code.'), ('visual inspection of flagged pairs is ongoing', 'Do not leave an unfinished-action statement in the final paper.')]
    findings = []
    for pattern, note in risky_patterns:
        count = tex.lower().count(pattern.lower())
        findings.append((pattern, count, note))
    lines = ['# CAPSTONE-CLIP v53 manual review report', '', 'The package was generated without model training. Canonical numbers come from the supplied paper-postprocess outputs.', '', '## Required human checks before submission', '', '1. Confirm that every APPLeNet-PM and AVION-PM mention retains the protocol-matched reimplementation qualifier.', '2. Confirm that HOSO is not compared in an all-five-shot mean; use the generated common-support table for K>=2.', '3. Review page count after compiling; the new PM table may require moving one optional analysis to supplementary material.', '4. The qualitative matched-vs-shuffled RSICD figure is auto-inserted when enabled; decide separately whether to include the optional efficiency figures.', '5. Review the EuroSAT near-duplicate wording. The code reports thresholds accurately but does not perform new image inspection.', '6. Check journal-specific AI-assistance disclosure and data/code availability requirements.', '', '## Phrase audit in patched manuscript', '', '| Phrase | Count | Review note |', '|---|---:|---|']
    for pattern, count, note in findings:
        lines.append(f'| `{pattern}` | {count} | {note} |')
    lines.extend(['', '## Calibration note', '', prose['calibration_sentence'], '', '## Canonical source folders', ''])
    for name, root in roots.items():
        lines.append(f'- **{name}:** `{root}`')
    return '\n'.join(lines) + '\n'

def compile_latex(package_dir: Path) -> Tuple[bool, str]:
    pdflatex = shutil.which('pdflatex')
    if not pdflatex:
        return (False, 'pdflatex not available; compilation skipped')
    cmd = [pdflatex, '-interaction=nonstopmode', '-halt-on-error', 'main_v53.tex']
    logs = []
    for _ in range(2):
        proc = subprocess.run(cmd, cwd=package_dir, text=True, capture_output=True)
        logs.append(proc.stdout + '\n' + proc.stderr)
        if proc.returncode != 0:
            (package_dir / 'compile_error.log').write_text('\n\n'.join(logs), encoding='utf-8')
            return (False, 'LaTeX compilation failed; see compile_error.log')
    return ((package_dir / 'main_v53.pdf').exists(), 'LaTeX compiled successfully')

def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(source_dir.rglob('*')):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(source_dir))

def main(cfg: Config=CFG) -> Path:
    mount_google_drive_if_needed()
    output_dir = resolve_output_dir(cfg)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / '_extracted_inputs'
    work_dir.mkdir(parents=True, exist_ok=True)
    log('Discovering the four input ZIP packages...')
    try:
        zip_paths = {name: discover_zip(pattern, cfg.search_roots) for name, pattern in ZIP_PATTERNS.items()}
    except FileNotFoundError:
        try:
            from google.colab import files
            log('One or more ZIPs were not found. Select the four ZIP packages for upload.')
            files.upload()
            zip_paths = {name: discover_zip(pattern, cfg.search_roots) for name, pattern in ZIP_PATTERNS.items()}
        except ImportError:
            raise
    for name, path in zip_paths.items():
        log(f'{name}: {path}')
    roots: Dict[str, Path] = {}
    for name, zip_path in zip_paths.items():
        destination = work_dir / name
        log(f'Extracting {name}...')
        safe_extract(zip_path, destination)
        roots[name] = find_root_containing(destination, DISTINCTIVE_FILES[name])
        log(f'{name} root: {roots[name]}')
    log('Running canonical artifact and protocol audit...')
    audit = run_audit(roots)
    audit_df = audit.to_dataframe()
    audit_df.to_csv(output_dir / 'V53_AUDIT.csv', index=False)
    print(audit_df.to_string(index=False))
    if cfg.strict_audit and (not audit.passed):
        failed = audit_df[~audit_df['pass']]
        raise RuntimeError('Audit failed:\n' + failed.to_string(index=False))
    post_tables = roots['post'] / 'tables'
    overall = read_csv(post_tables / 'paper_table_overall_macro_averages.csv')
    global_inf = read_csv(post_tables / 'paper_table_global_pairwise_inference.csv')
    wtl = read_csv(post_tables / 'paper_table_win_tie_loss.csv')
    per_dataset = read_csv(post_tables / 'paper_table_capstone_pm_per_dataset_tests.csv')
    efficiency = read_csv(post_tables / 'paper_table_efficiency_from_runs.csv')
    fold_level = read_csv(post_tables / 'table_fold_level_seed_averaged.csv')
    tables = {'overall': make_overall_table(overall, cfg.include_hoso_in_main_overall_table), 'confirmatory': make_confirmatory_table(global_inf, wtl, cfg), 'pm_dataset': make_pm_dataset_table(per_dataset), 'efficiency': make_efficiency_table(efficiency), 'hoso': make_hoso_common_support_table(fold_level)}
    prose = manuscript_prose(global_inf, overall, per_dataset, cfg)
    latex_out = output_dir / 'generated_latex'
    latex_out.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        (latex_out / f'table_{name}_v53.tex').write_text(content + '\n', encoding='utf-8')
    for name, content in prose.items():
        (latex_out / f'prose_{name}_v53.tex').write_text(content + '\n', encoding='utf-8')
    package_dir = output_dir / 'CAPSTONE_CLIP_v53_LaTeX'
    copy_tree_contents(roots['manuscript'], package_dir)
    original_tex_path = package_dir / 'main.tex'
    original_tex = original_tex_path.read_text(encoding='utf-8')
    patched_tex, changes = patch_manuscript(original_tex, tables, prose)
    pairing_info: Optional[Dict[str, object]] = None
    if cfg.generate_pairing_figure:
        figure_path = package_dir / 'figs' / 'figure_matched_vs_shuffled_examples.png'
        manifest_path = output_dir / 'qualitative_pairing_examples.csv'
        try:
            pairing_info = generate_pairing_figure(cfg, figure_path, manifest_path)
            changes.append('Generated qualitative matched-vs-shuffled RSICD training-pair figure')
            if cfg.insert_pairing_figure_in_manuscript:
                patched_tex = insert_after_environment_label(patched_tex, 'fig:controls', pairing_figure_latex())
                patched_tex = patched_tex.replace('The pairing test is sharper:', 'The qualitative manipulation is illustrated in Fig.~\\ref{fig:pairing_examples}. The pairing test is sharper:', 1)
                changes.append('Inserted qualitative pairing figure after the quantitative caption-control figure')
        except Exception as exc:
            if cfg.strict_pairing_figure:
                raise
            log(f'Qualitative pairing figure skipped: {exc}')
            changes.append(f'Qualitative pairing figure skipped: {exc}')
    (package_dir / 'main_v53.tex').write_text(patched_tex, encoding='utf-8')
    optional_insert = '% Optional v53 inserts. Add only if page budget permits.\n\n' + tables['efficiency'] + '\n\n' + tables['hoso'] + '\n'
    (package_dir / 'OPTIONAL_INSERTS_v53.tex').write_text(optional_insert, encoding='utf-8')
    copied_figures: List[str] = []
    if cfg.copy_optional_figures:
        src_figs = roots['post'] / 'figures'
        dst_figs = package_dir / 'figs' / 'v53_optional'
        dst_figs.mkdir(parents=True, exist_ok=True)
        for filename in OPTIONAL_FIGURES:
            src = src_figs / filename
            if src.exists():
                shutil.copy2(src, dst_figs / filename)
                copied_figures.append(filename)
    canonical_dir = output_dir / 'canonical_tables'
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_files = ['paper_table_overall_macro_averages.csv', 'paper_table_global_pairwise_inference.csv', 'paper_table_win_tie_loss.csv', 'paper_table_capstone_pm_per_dataset_tests.csv', 'paper_table_efficiency_from_runs.csv', 'paper_table_near_duplicate_audit.csv', 'table_friedman_omnibus.csv', 'table_average_ranks.csv', 'table_best_method_cell_counts.csv', 'FINAL_CHECKS.csv']
    for filename in canonical_files:
        src = post_tables / filename
        if src.exists():
            shutil.copy2(src, canonical_dir / filename)
    shutil.copy2(roots['pm'] / 'protocol_audit.json', canonical_dir / 'protocol_audit.json')
    shutil.copy2(roots['pm'] / 'tables' / 'method_fidelity_audit.csv', canonical_dir / 'method_fidelity_audit.csv')
    report = make_manual_review_report(patched_tex, roots, prose)
    (output_dir / 'MANUAL_REVIEW_REQUIRED.md').write_text(report, encoding='utf-8')
    (output_dir / 'V53_CHANGELOG.md').write_text('# CAPSTONE-CLIP v53 automatic changelog\n\n' + '\n'.join((f'- {item}' for item in changes)) + '\n\n## Optional figures copied\n\n' + ('\n'.join((f'- {name}' for name in copied_figures)) if copied_figures else '- None') + '\n\n## Qualitative image-caption figure\n\n' + (json.dumps(pairing_info, indent=2, ensure_ascii=False) if pairing_info else 'Not generated') + '\n', encoding='utf-8')
    compile_status = 'Compilation disabled'
    compiled = False
    if cfg.compile_if_pdflatex_available:
        compiled, compile_status = compile_latex(package_dir)
    log(compile_status)
    shutil.rmtree(work_dir, ignore_errors=True)
    summary = {'training_rerun': False, 'canonical_postprocess_used': True, 'audit_passed': audit.passed, 'latex_compiled': compiled, 'output_dir': str(output_dir), 'manuscript': str(package_dir / 'main_v53.tex'), 'compiled_pdf': str(package_dir / 'main_v53.pdf') if compiled else None, 'equivalence_margin_pp': cfg.equivalence_margin_pp, 'win_tie_margin_pp': cfg.win_tie_margin_pp, 'hoso_in_main_overall': cfg.include_hoso_in_main_overall_table, 'qualitative_pairing_figure': pairing_info, 'training_data_root': str(cfg.data_root), 'drive_wide_training_data_search': False}
    (output_dir / 'RUN_SUMMARY_v53.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    final_zip = output_dir.parent / 'CAPSTONE_CLIP_v53_CANONICAL.zip'
    zip_directory(output_dir, final_zip)
    log(f'Completed. Output package: {final_zip}')
    return final_zip
if __name__ == '__main__':
    try:
        result = main()
        print(f'\nDONE: {result}')
    except Exception as exc:
        print(f'\nERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
from pathlib import Path
search_roots = []
if INPUT_MODE.startswith('Bilinen'):
    search_roots.append(Path(PROJECT_ROOT))
search_roots.extend([Path('/content'), Path('/mnt/data'), Path.cwd()])
_seen = set()
search_roots = tuple((p for p in search_roots if not (str(p) in _seen or _seen.add(str(p)))))
if SAVE_OUTPUT_TO_DRIVE and Path('/content/drive/MyDrive').exists():
    output_dir = Path(PROJECT_ROOT) / 'CAPSTONE_CLIP_v53_CANONICAL'
else:
    output_dir = Path('/content/CAPSTONE_CLIP_v53_CANONICAL')
CFG = Config(search_roots=search_roots, output_dir=output_dir, data_root=Path(DATA_ROOT), caption_dataset='RSICD', generate_pairing_figure=GENERATE_PAIRING_FIGURE, insert_pairing_figure_in_manuscript=INSERT_PAIRING_FIGURE_IN_MANUSCRIPT, pairing_figure_examples=max(2, int(PAIRING_EXAMPLES)), pairing_figure_seed=int(PAIRING_SEED), strict_pairing_figure=STRICT_PAIRING_FIGURE, equivalence_margin_pp=1.0, win_tie_margin_pp=0.1, include_hoso_in_main_overall_table=INCLUDE_HOSO_IN_MAIN_TABLE, copy_optional_figures=True, compile_if_pdflatex_available=True, strict_audit=STRICT_AUDIT)
if USE_CAPSTONE_FAMILY_DISPLAY_NAMES:
    METHOD_DISPLAY.update({'Shared-Caption': 'CAPSTONE-CLIP-Shared', 'Capacity-Matched-Shared': 'CAPSTONE-CLIP-CM', 'CAPSTONE-CLIP': 'CAPSTONE-CLIP-Full'})
print('ZIP arama kökleri — MyDrive geneli yok:')
for root in CFG.search_roots:
    print(' -', root)
print('Çıktı klasörü:', CFG.output_dir)
print('RSICD veri klasörü:', CFG.data_root / 'RSICD')
print('Nitel görsel:', CFG.generate_pairing_figure)
print('Makaleye otomatik ekleme:', CFG.insert_pairing_figure_in_manuscript)
print('Görünen yöntem adları:', {k: METHOD_DISPLAY.get(k, k) for k in ['Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP']})
FINAL_ZIP = main(CFG)
print('\nNihai paket:', FINAL_ZIP)
import json
from pathlib import Path
summary_path = CFG.output_dir / 'RUN_SUMMARY_v53.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
else:
    print('RUN_SUMMARY_v53.json bulunamadı.')
print('\nÜretilen temel dosyalar:')
for p in sorted(CFG.output_dir.rglob('*')):
    if p.is_file() and p.suffix.lower() in {'.tex', '.pdf', '.csv', '.md', '.json', '.png'}:
        print(' -', p.relative_to(CFG.output_dir))
figure_path = CFG.output_dir / 'CAPSTONE_CLIP_v53_LaTeX' / 'figs' / 'figure_matched_vs_shuffled_examples.png'
manifest_path = CFG.output_dir / 'qualitative_pairing_examples.csv'
print('\nNitel görsel:', figure_path)
print('Örnek manifesti:', manifest_path)
if figure_path.exists():
    try:
        from IPython.display import display, Image as IPyImage
        display(IPyImage(filename=str(figure_path)))
    except Exception as exc:
        print('Görsel ön izlemesi gösterilemedi:', exc)
if AUTO_DOWNLOAD_FINAL_ZIP:
    try:
        from google.colab import files
        files.download(str(FINAL_ZIP))
    except Exception as exc:
        print('Otomatik indirme başlatılamadı:', exc)
else:
    print('\nİndirmek için dosya panelinden şu paketi seçebilirsiniz:')
    print(FINAL_ZIP)
