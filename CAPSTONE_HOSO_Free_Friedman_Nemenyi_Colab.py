import io
import os
import re
import math
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from google.colab import files
DRIVE_PATH = ''
OUTPUT_DIR = Path('/content/hoso_free_friedman_nemenyi')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def norm_text(value):
    return re.sub('[^a-z0-9]+', '', str(value).lower())
METHOD_MAP = {'zeroshot': 'Zero-shot', 'linearprobe': 'Linear Probe', 'clipadapter': 'CLIP-Adapter', 'tipadapter': 'Tip-Adapter', 'tipadapterf': 'Tip-Adapter-F', 'coop': 'CoOp', 'taskres': 'TaskRes', 'rla': 'RLA', 'prolip': 'ProLIP', 'sharedcaption': 'CAPSTONE-CLIP-Shared', 'capstoneclipshared': 'CAPSTONE-CLIP-Shared', 'capacitymatchedshared': 'CAPSTONE-CLIP-CM', 'capstoneclipcm': 'CAPSTONE-CLIP-CM', 'capstoneclip': 'CAPSTONE-CLIP-Full', 'capstoneclipfull': 'CAPSTONE-CLIP-Full'}
EXPECTED_METHODS = ['Zero-shot', 'Linear Probe', 'CLIP-Adapter', 'Tip-Adapter', 'Tip-Adapter-F', 'CoOp', 'TaskRes', 'RLA', 'ProLIP', 'CAPSTONE-CLIP-Shared', 'CAPSTONE-CLIP-CM', 'CAPSTONE-CLIP-Full']

def find_column(df, candidates, required=True):
    normalized = {norm_text(c): c for c in df.columns}
    for candidate in candidates:
        key = norm_text(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f'Missing column. Expected one of {candidates}; available columns: {list(df.columns)}')
    return None

def collect_input_files():
    paths = []
    if DRIVE_PATH.strip():
        from google.colab import drive
        drive.mount('/content/drive')
        path = Path(DRIVE_PATH)
        if not path.exists():
            raise FileNotFoundError(f'Drive path not found: {path}')
        paths.append(path)
    else:
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError('No file was uploaded.')
        upload_dir = Path('/content/capstone_uploaded')
        upload_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in uploaded.items():
            path = upload_dir / Path(name).name
            path.write_bytes(payload)
            paths.append(path)
    expanded = []
    extract_root = Path('/content/capstone_extracted')
    extract_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.suffix.lower() == '.zip':
            target = extract_root / path.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as archive:
                archive.extractall(target)
            expanded.extend(target.rglob('*.csv'))
        elif path.suffix.lower() == '.csv':
            expanded.append(path)
        else:
            raise ValueError(f'Unsupported input: {path.name}. Upload a CSV or ZIP.')
    if not expanded:
        raise FileNotFoundError('No CSV file was found.')
    return list(dict.fromkeys(expanded))

def score_csv(path):
    try:
        sample = pd.read_csv(path, nrows=100)
    except Exception:
        return -1
    keys = {norm_text(c) for c in sample.columns}
    required_groups = [{'dataset', 'datasetname'}, {'k', 'shot', 'shots'}, {'method', 'model'}, {'accuracy', 'acc', 'testaccuracy'}]
    score = sum((any((norm_text(x) in keys for x in group)) for group in required_groups)) * 10
    name = path.name.lower()
    if name == 'raw_primary_classification.csv':
        score += 100
    elif 'combined_raw_capstone' in name:
        score += 70
    elif 'primary' in name and 'classification' in name:
        score += 50
    return score
csv_paths = collect_input_files()
scored = sorted(((score_csv(p), p) for p in csv_paths), reverse=True)
best_score, source_path = scored[0]
if best_score < 40:
    raise RuntimeError('A suitable raw classification table was not found.\n' + '\n'.join((f'{score:>3}: {path}' for score, path in scored[:20])))
print(f'Selected input: {source_path}')
raw = pd.read_csv(source_path)
dataset_col = find_column(raw, ['dataset', 'dataset_name', 'name'])
shot_col = find_column(raw, ['K', 'shot', 'shots', 'n_shot'])
method_col = find_column(raw, ['method', 'model', 'approach'])
accuracy_col = find_column(raw, ['Accuracy', 'accuracy', 'acc', 'test_accuracy'])
fold_col = find_column(raw, ['outer_id', 'outer_fold', 'fold', 'fold_id'], required=False)
status_col = find_column(raw, ['status', 'run_status'], required=False)
if fold_col is None:
    raise KeyError('The outer-fold column was not found. Expected outer_id, outer_fold, fold, or fold_id.')
df = raw.copy()
if status_col is not None:
    df = df[df[status_col].astype(str).str.lower().eq('completed')].copy()
df['_canonical_method'] = df[method_col].map(lambda x: METHOD_MAP.get(norm_text(x)))
unknown = sorted(df.loc[df['_canonical_method'].isna(), method_col].astype(str).unique())
if unknown:
    print('Ignored methods/conditions:', unknown)
df = df[df['_canonical_method'].isin(EXPECTED_METHODS)].copy()
df[shot_col] = pd.to_numeric(df[shot_col], errors='coerce')
df[accuracy_col] = pd.to_numeric(df[accuracy_col], errors='coerce')
df = df.dropna(subset=[dataset_col, shot_col, fold_col, accuracy_col])
available = sorted(df['_canonical_method'].unique())
missing_methods = [m for m in EXPECTED_METHODS if m not in available]
if missing_methods:
    raise RuntimeError('The following required HOSO-free internal methods are missing:\n- ' + '\n- '.join(missing_methods) + f'\n\nAvailable canonical methods: {available}')
fold_level = df.groupby([dataset_col, shot_col, fold_col, '_canonical_method'], as_index=False, dropna=False)[accuracy_col].mean().rename(columns={accuracy_col: 'Accuracy_fold_mean'})
cell_level = fold_level.groupby([dataset_col, shot_col, '_canonical_method'], as_index=False, dropna=False)['Accuracy_fold_mean'].mean().rename(columns={'Accuracy_fold_mean': 'Accuracy'})
matrix = cell_level.pivot(index=[dataset_col, shot_col], columns='_canonical_method', values='Accuracy')
matrix = matrix.reindex(columns=EXPECTED_METHODS).sort_index()
missing_cells = matrix.isna()
if missing_cells.any().any():
    details = []
    for index, row in missing_cells.iterrows():
        for method, is_missing in row.items():
            if is_missing:
                details.append(f'{index}: {method}')
    raise RuntimeError('The common dataset–shot support is incomplete:\n- ' + '\n- '.join(details[:100]))
n_blocks, n_methods = matrix.shape
if n_blocks != 30 or n_methods != 12:
    print(f'Warning: expected 30 dataset–shot blocks and 12 methods, but found {n_blocks} blocks and {n_methods} methods.')
friedman = friedmanchisquare(*[matrix[m].to_numpy() for m in EXPECTED_METHODS])
rank_rows = matrix.apply(lambda row: pd.Series(rankdata(-row.to_numpy(dtype=float), method='average'), index=matrix.columns), axis=1)
mean_ranks = rank_rows.mean(axis=0).sort_values()
mean_ranks_df = mean_ranks.rename('MeanRank').reset_index()
mean_ranks_df.columns = ['Method', 'MeanRank']
alpha = 0.05
q_alpha = studentized_range.ppf(1 - alpha, n_methods, np.inf) / math.sqrt(2)
critical_difference = q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))
rank_difference = pd.DataFrame(np.abs(mean_ranks.to_numpy()[:, None] - mean_ranks.to_numpy()[None, :]), index=mean_ranks.index, columns=mean_ranks.index)
nemenyi_significant = rank_difference > critical_difference
best_method = mean_ranks.index[0]
best_rank = float(mean_ranks.iloc[0])
significantly_worse_than_best = [method for method in mean_ranks.index[1:] if rank_difference.loc[best_method, method] > critical_difference]

def format_p(p):
    if p < 0.0001:
        return f'{p:.3e}'
    return f'{p:.6f}'
sentence = f'A Friedman analysis over the {n_blocks} dataset–shot blocks on which all {n_methods} HOSO-free internal methods were defined yielded χ²({n_methods - 1}) = {friedman.statistic:.3f}, p = {format_p(friedman.pvalue)}. {best_method} achieved the best mean rank ({best_rank:.3f}), and the Nemenyi critical difference at α = 0.05 was {critical_difference:.3f}.'
if significantly_worse_than_best:
    sentence += ' Its rank differed by more than the critical difference from ' + ', '.join(significantly_worse_than_best) + '.'
print('\n' + '=' * 80)
print('HOSO-FREE FRIEDMAN–NEMENYI RESULT')
print('=' * 80)
print(sentence)
print('\nMean ranks:')
print(mean_ranks_df.to_string(index=False))
print(f'\nSource: {source_path}')
print(f'Blocks: {n_blocks}; methods: {n_methods}; CD: {critical_difference:.6f}')
matrix.to_csv(OUTPUT_DIR / 'dataset_shot_accuracy_matrix.csv')
rank_rows.to_csv(OUTPUT_DIR / 'dataset_shot_rank_matrix.csv')
mean_ranks_df.to_csv(OUTPUT_DIR / 'mean_ranks.csv', index=False)
rank_difference.to_csv(OUTPUT_DIR / 'pairwise_rank_differences.csv')
nemenyi_significant.to_csv(OUTPUT_DIR / 'nemenyi_significance_matrix.csv')
fold_level.to_csv(OUTPUT_DIR / 'fold_level_averages.csv', index=False)
cell_level.to_csv(OUTPUT_DIR / 'dataset_shot_method_averages.csv', index=False)
report = f'HOSO-free Friedman–Nemenyi analysis\n\nInput file: {source_path}\nNumber of complete dataset–shot blocks: {n_blocks}\nNumber of internal methods: {n_methods}\n\nFriedman chi-square: {friedman.statistic:.12f}\nDegrees of freedom: {n_methods - 1}\nFriedman p-value: {friedman.pvalue:.12e}\nNemenyi alpha: {alpha}\nNemenyi q_alpha: {q_alpha:.12f}\nNemenyi critical difference: {critical_difference:.12f}\n\nManuscript-ready sentence:\n{sentence}\n\nMean ranks:\n{mean_ranks_df.to_string(index=False)}\n'
(OUTPUT_DIR / 'report.txt').write_text(report, encoding='utf-8')
zip_path = shutil.make_archive('/content/CAPSTONE_HOSO_Free_Friedman_Nemenyi', 'zip', OUTPUT_DIR)
print(f'\nSaved results: {zip_path}')
files.download(zip_path)
