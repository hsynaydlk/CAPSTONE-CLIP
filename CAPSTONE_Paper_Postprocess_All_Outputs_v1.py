import os, sys, subprocess, json, glob, hashlib, math, shutil, warnings, textwrap
from pathlib import Path
from itertools import combinations
from collections import defaultdict
os.environ.setdefault('MPLBACKEND', 'Agg')

def pip_install(packages):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + packages, check=False)
pip_install(['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'matplotlib', 'openpyxl'])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata
import statsmodels.formula.api as smf
from IPython.display import display, Markdown
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('Colab dışında çalışıyor; mevcut dosya sistemi kullanılacak.')
DRIVE_ROOT = '/content/drive/MyDrive/CAPSTONE_CLIP_RemoteSensing'
CAPSTONE_PREFERRED_ROOT = os.path.join(DRIVE_ROOT, 'outputs_q1_capstone_clip_v11_full_external')
PM_PREFERRED_ROOT = os.path.join(DRIVE_ROOT, 'outputs_avion_applenet_capstone_protocol_matched_v1')
POST_ROOT = os.path.join(DRIVE_ROOT, 'outputs_capstone_paper_postprocess_v1')
RUN_MODE = 'paper'
BOOTSTRAP_REPS = 10000 if RUN_MODE == 'paper' else 1000
RANDOM_SEED = 20260730
ALPHA = 0.05
NONINFERIORITY_MARGIN = 0.01
PRACTICAL_TIE_MARGIN = 0.001
STRICT_PROTOCOL_AUDIT = False
EXPECTED_DATASETS = ['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT']
EXPECTED_SHOTS = [1, 2, 4, 8, 16]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SUPPORT_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_PM_METHODS = ['AVION-PM', 'APPLeNet-PM']
FALLBACK_PRIMARY_METHODS = ['Zero-shot', 'LinearProbe', 'CLIP-Adapter', 'Tip-Adapter', 'Tip-Adapter-F', 'CoOp', 'TaskRes', 'RLA', 'ProLIP', 'HOSO-Adapter', 'Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP']
DISPLAY_LABEL = {'AVION-PM': 'AVION-PM (protocol-matched reimplementation)', 'APPLeNet-PM': 'APPLeNet-PM (protocol-matched reimplementation)'}
FOCUS_METHOD_ORDER = ['CAPSTONE-CLIP', 'APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared', 'TaskRes', 'CoOp', 'Tip-Adapter-F']
METRICS = ['Accuracy', 'MacroF1', 'BalancedAccuracy', 'NLL', 'ECE']
HIGHER_IS_BETTER = {'Accuracy': True, 'MacroF1': True, 'BalancedAccuracy': True, 'NLL': False, 'ECE': False}
TABLE_DIR = os.path.join(POST_ROOT, 'tables')
FIG_DIR = os.path.join(POST_ROOT, 'figures')
LATEX_DIR = os.path.join(POST_ROOT, 'latex')
REPORT_DIR = os.path.join(POST_ROOT, 'reports')
for p in [POST_ROOT, TABLE_DIR, FIG_DIR, LATEX_DIR, REPORT_DIR]:
    os.makedirs(p, exist_ok=True)
print('Post-process output:', POST_ROOT)
print('Bootstrap repetitions:', BOOTSTRAP_REPS)

def stable_int(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)

def find_output_root(preferred, patterns, required_rel):
    candidates = []
    if preferred:
        candidates.append(preferred)
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = list(dict.fromkeys(candidates))
    valid = []
    for root in candidates:
        req = os.path.join(root, required_rel)
        if os.path.isfile(req):
            valid.append(root)
    if not valid:
        return None
    valid.sort(key=lambda p: os.path.getmtime(os.path.join(p, required_rel)), reverse=True)
    return valid[0]
CAPSTONE_ROOT = find_output_root(CAPSTONE_PREFERRED_ROOT, [os.path.join(DRIVE_ROOT, 'outputs_q1_capstone*'), os.path.join(DRIVE_ROOT, 'outputs_*capstone_clip*v11*')], os.path.join('tables', 'raw_primary_classification.csv'))
PM_ROOT = find_output_root(PM_PREFERRED_ROOT, [os.path.join(DRIVE_ROOT, 'outputs_avion_applenet_capstone_protocol_matched*'), os.path.join(DRIVE_ROOT, 'outputs_*protocol_matched*')], os.path.join('tables', 'raw_protocol_matched_external.csv'))
if CAPSTONE_ROOT is None:
    raise FileNotFoundError('CAPSTONE raw_primary_classification.csv bulunamadı.')
if PM_ROOT is None:
    warnings.warn('PM çıktı kökü bulunamadı. Ana CAPSTONE analizleri yine üretilecek.')
print('CAPSTONE source:', CAPSTONE_ROOT)
print('PM source:', PM_ROOT)

def read_json(path, default=None):
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return default

def read_csv_rel(root, rel, required=False):
    path = os.path.join(root, rel) if root else None
    if path and os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            df.attrs['source_path'] = path
            return df
        except Exception as exc:
            if required:
                raise
            warnings.warn(f'Okunamadı: {path}: {exc}')
    if required:
        raise FileNotFoundError(path or rel)
    return pd.DataFrame()

def coerce_numeric(df, columns):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def save_csv(df, name):
    path = os.path.join(TABLE_DIR, name if name.endswith('.csv') else name + '.csv')
    df.to_csv(path, index=False)
    return path

def save_latex(text, name):
    path = os.path.join(LATEX_DIR, name if name.endswith('.tex') else name + '.tex')
    Path(path).write_text(text, encoding='utf-8')
    return path

def save_figure(fig, stem, dpi=300):
    png = os.path.join(FIG_DIR, stem + '.png')
    pdf = os.path.join(FIG_DIR, stem + '.pdf')
    fig.tight_layout()
    fig.savefig(png, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return (png, pdf)

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    idx = np.where(good)[0]
    pv = p[good]
    order = np.argsort(pv)
    running = 0.0
    adj = np.empty_like(pv)
    m = len(pv)
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * pv[j])
        adj[j] = min(1.0, running)
    out[idx] = adj
    return out

def t_ci(values, alpha=ALPHA):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    se = stats.sem(x)
    q = stats.t.ppf(1 - alpha / 2, len(x) - 1)
    return (float(x.mean() - q * se), float(x.mean() + q * se))

def two_level_bootstrap_ci(group, metric, B=BOOTSTRAP_REPS, seed=RANDOM_SEED):
    vals = []
    for _, fg in group.groupby('fold'):
        a = pd.to_numeric(fg[metric], errors='coerce').dropna().to_numpy(float)
        if len(a):
            vals.append(a)
    F = len(vals)
    if F == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    fold_boot = np.empty((F, B), float)
    for i, a in enumerate(vals):
        picks = rng.integers(0, len(a), size=(B, len(a)))
        fold_boot[i] = a[picks].mean(axis=1)
    fold_choice = rng.integers(0, F, size=(B, F))
    boot = fold_boot.T[np.arange(B)[:, None], fold_choice].mean(axis=1)
    return (float(np.quantile(boot, ALPHA / 2)), float(np.quantile(boot, 1 - ALPHA / 2)))

def display_name(method):
    return DISPLAY_LABEL.get(method, method)
MAIN_FILES = {'primary': 'tables/raw_primary_classification.csv', 'controls': 'tables/raw_caption_controls.csv', 'retrieval': 'tables/raw_retrieval.csv', 'ablations': 'tables/raw_architecture_ablations.csv', 'diagnostics': 'tables/raw_class_diagnostics.csv', 'base_to_new': 'tables/raw_base_to_new.csv', 'cross_dataset': 'tables/raw_cross_dataset.csv', 'cross_backbone': 'tables/raw_cross_backbone.csv', 'efficiency': 'tables/table_efficiency.csv', 'near_duplicate': 'tables/table_near_duplicate_audit.csv', 'near_duplicate_pairs': 'tables/near_duplicate_top_pairs.csv', 'method_audit': 'tables/table_method_audit.csv', 'claim_audit': 'tables/table_claim_audit.csv'}
MAIN = {k: read_csv_rel(CAPSTONE_ROOT, rel, required=k == 'primary') for k, rel in MAIN_FILES.items()}
PM = {'raw': read_csv_rel(PM_ROOT, 'tables/raw_protocol_matched_external.csv') if PM_ROOT else pd.DataFrame(), 'summary': read_csv_rel(PM_ROOT, 'tables/table_protocol_matched_summary.csv') if PM_ROOT else pd.DataFrame(), 'method_audit': read_csv_rel(PM_ROOT, 'tables/method_fidelity_audit.csv') if PM_ROOT else pd.DataFrame()}
main_config = read_json(os.path.join(CAPSTONE_ROOT, 'config_capstone_q1_v11.json'), {})
pm_protocol = read_json(os.path.join(PM_ROOT, 'protocol_audit.json'), {}) if PM_ROOT else {}
primary_methods = main_config.get('primary_methods', FALLBACK_PRIMARY_METHODS)
expected_datasets = main_config.get('datasets', EXPECTED_DATASETS)
expected_shots = [int(x) for x in main_config.get('shots', EXPECTED_SHOTS)]
expected_folds = list(range(int(main_config.get('n_outer_folds', len(EXPECTED_FOLDS)))))
expected_seeds = [int(x) for x in main_config.get('support_seeds', EXPECTED_SUPPORT_SEEDS)]
main_raw = coerce_numeric(MAIN['primary'], ['K', 'fold', 'support_seed'] + METRICS)
pm_raw = coerce_numeric(PM['raw'], ['K', 'fold', 'support_seed'] + METRICS)

def completed_rows(df, source):
    if df.empty:
        return df.copy()
    q = df.copy()
    q['source_notebook'] = source
    if 'status' in q.columns:
        q = q[q['status'].astype(str).str.lower().eq('completed')]
    if 'available' in q.columns:
        bad = q['available'].astype(str).str.lower().isin(['false', '0', 'no'])
        q = q[~bad]
    if 'Accuracy' in q.columns:
        q = q[np.isfinite(pd.to_numeric(q['Accuracy'], errors='coerce'))]
    return q
main_ok = completed_rows(main_raw, 'CAPSTONE-main')
pm_ok = completed_rows(pm_raw, 'PM-external')
KEYS = ['backbone', 'dataset', 'K', 'fold', 'support_seed', 'method']
for df in [main_ok, pm_ok]:
    for c in KEYS:
        if c not in df.columns:
            df[c] = np.nan
main_duplicates = main_ok[main_ok.duplicated(KEYS, keep=False)].sort_values(KEYS) if not main_ok.empty else pd.DataFrame()
pm_duplicates = pm_ok[pm_ok.duplicated(KEYS, keep=False)].sort_values(KEYS) if not pm_ok.empty else pd.DataFrame()
main_ok = main_ok.drop_duplicates(KEYS, keep='last')
pm_ok = pm_ok.drop_duplicates(KEYS, keep='last')
combined = pd.concat([main_ok, pm_ok], ignore_index=True, sort=False)
combined['method_display'] = combined['method'].map(display_name)
combined = coerce_numeric(combined, ['K', 'fold', 'support_seed'] + METRICS + ['train_eval_s', 'trainable_params', 'peak_gpu_mb', 'peak_gpu_delta_mb'])
save_csv(combined, 'raw_combined_capstone_and_pm')
save_csv(main_duplicates, 'audit_duplicate_main_runs')
save_csv(pm_duplicates, 'audit_duplicate_pm_runs')
print('Main completed rows:', len(main_ok))
print('PM completed rows:', len(pm_ok))
print('Combined methods:', sorted(combined.method.dropna().unique()))
display(combined.groupby(['source_notebook', 'method'], dropna=False).size().reset_index(name='rows'))

def expected_grid(methods):
    return pd.MultiIndex.from_product([expected_datasets, expected_shots, expected_folds, expected_seeds, methods], names=['dataset', 'K', 'fold', 'support_seed', 'method']).to_frame(index=False)

def coverage_audit(raw_all, completed, methods, source_label):
    exp = expected_grid(methods)
    found = completed[['dataset', 'K', 'fold', 'support_seed', 'method']].drop_duplicates()
    merged = exp.merge(found.assign(found=True), on=['dataset', 'K', 'fold', 'support_seed', 'method'], how='left')
    missing = merged[merged.found.isna()].drop(columns='found')
    rows = []
    for m in methods:
        e = exp[exp.method == m]
        c = found[found.method == m]
        raw_m = raw_all[raw_all.method == m] if 'method' in raw_all.columns else pd.DataFrame()
        failed = 0
        if not raw_m.empty:
            if 'status' in raw_m.columns:
                failed = int((raw_m.status.astype(str).str.lower() == 'failed').sum())
            elif 'available' in raw_m.columns:
                failed = int(raw_m.available.astype(str).str.lower().isin(['false', '0', 'no']).sum())
        rows.append({'source': source_label, 'method': m, 'expected_runs': len(e), 'completed_unique_runs': len(c), 'missing_runs': len(e) - len(c), 'failed_rows': failed, 'coverage_pct': 100 * len(c) / max(1, len(e))})
    return (pd.DataFrame(rows), missing)
cov_main, miss_main = coverage_audit(main_raw, main_ok, primary_methods, 'CAPSTONE-main')
cov_pm, miss_pm = coverage_audit(pm_raw, pm_ok, EXPECTED_PM_METHODS, 'PM-external')
coverage = pd.concat([cov_main, cov_pm], ignore_index=True)
protocol_rows = []
protocol_rows.append({'check': 'CAPSTONE raw exists', 'status': 'PASS' if len(main_raw) else 'FAIL', 'evidence': MAIN['primary'].attrs.get('source_path', '')})
manifest_verified = None
if pm_protocol:
    audit = pm_protocol.get('audit', [])
    manifest_verified = bool(audit) and all((bool(x.get('all_match')) for x in audit))
protocol_rows.append({'check': 'PM split/support manifest matches CAPSTONE', 'status': 'PASS' if manifest_verified else 'NOT AVAILABLE' if not pm_protocol else 'FAIL', 'evidence': os.path.join(PM_ROOT, 'protocol_audit.json') if PM_ROOT else ''})
if not pm_ok.empty and 'directly_comparable_protocol' in pm_ok.columns:
    comparable = pm_ok.directly_comparable_protocol.astype(str).str.lower().isin(['true', '1', 'yes']).all()
else:
    comparable = None
protocol_rows.append({'check': 'PM rows marked directly comparable protocol', 'status': 'PASS' if comparable else 'NOT AVAILABLE' if comparable is None else 'FAIL', 'evidence': 'raw_protocol_matched_external.csv'})
protocol_audit = pd.DataFrame(protocol_rows)
save_csv(coverage, 'audit_run_coverage')
save_csv(miss_main, 'audit_missing_main_runs')
save_csv(miss_pm, 'audit_missing_pm_runs')
save_csv(protocol_audit, 'audit_protocol_sources')
display(protocol_audit)
display(coverage)
if STRICT_PROTOCOL_AUDIT and (protocol_audit.status == 'FAIL').any():
    raise RuntimeError('STRICT_PROTOCOL_AUDIT=True ve protokol denetiminde FAIL var.')

def build_paper_summary(df, metrics=METRICS):
    rows = []
    group_cols = ['backbone', 'dataset', 'K', 'method']
    for keys, g in df.groupby(group_cols, dropna=False):
        backbone, dataset, K, method = keys
        for metric in metrics:
            if metric not in g.columns:
                continue
            z = g[np.isfinite(pd.to_numeric(g[metric], errors='coerce'))].copy()
            if z.empty:
                continue
            fold_means = z.groupby('fold')[metric].mean().to_numpy(float)
            lo_t, hi_t = t_ci(fold_means)
            lo_b, hi_b = two_level_bootstrap_ci(z, metric, B=BOOTSTRAP_REPS, seed=stable_int('cell-bootstrap', backbone, dataset, K, method, metric, RANDOM_SEED))
            seed_sd = z.groupby('fold')[metric].std(ddof=1).replace([np.inf, -np.inf], np.nan).mean()
            rows.append({'backbone': backbone, 'dataset': dataset, 'K': int(K), 'method': method, 'method_display': display_name(method), 'metric': metric, 'mean': float(np.mean(fold_means)), 'std_across_folds': float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0, 'mean_within_fold_seed_sd': float(seed_sd) if np.isfinite(seed_sd) else np.nan, 't_ci95_low': lo_t, 't_ci95_high': hi_t, 'hier_boot_ci95_low': lo_b, 'hier_boot_ci95_high': hi_b, 'n_folds': int(len(fold_means)), 'n_runs': int(len(z)), 'n_support_seeds': int(z.support_seed.nunique()) if 'support_seed' in z.columns else np.nan})
    return pd.DataFrame(rows)
paper_summary = build_paper_summary(combined)
fold_summary = combined.groupby(['backbone', 'dataset', 'K', 'fold', 'method'], dropna=False)[METRICS].mean(numeric_only=True).reset_index()
cell_summary = fold_summary.groupby(['backbone', 'dataset', 'K', 'method'], dropna=False)[METRICS].mean(numeric_only=True).reset_index()
save_csv(paper_summary, 'table_all_methods_paper_summary_long')
save_csv(fold_summary, 'table_fold_level_seed_averaged')
save_csv(cell_summary, 'table_dataset_shot_cell_means')
display(paper_summary[(paper_summary.metric == 'Accuracy') & paper_summary.method.isin(['CAPSTONE-CLIP', 'APPLeNet-PM', 'AVION-PM'])].head(30))

def metric_wide(summary_long, metric, methods=None):
    q = summary_long[summary_long.metric == metric].copy()
    if methods is not None:
        q = q[q.method.isin(methods)]
    q['formatted'] = q.apply(lambda r: f"{100 * r['mean']:.2f} ± {100 * r['std_across_folds']:.2f} [{100 * r['hier_boot_ci95_low']:.2f}, {100 * r['hier_boot_ci95_high']:.2f}]" if metric in ['Accuracy', 'MacroF1', 'BalancedAccuracy', 'ECE'] else f"{r['mean']:.4f} ± {r['std_across_folds']:.4f} [{r['hier_boot_ci95_low']:.4f}, {r['hier_boot_ci95_high']:.4f}]", axis=1)
    return q.pivot_table(index=['dataset', 'K'], columns='method', values='formatted', aggfunc='first').reset_index()
focus_three = [m for m in ['CAPSTONE-CLIP', 'APPLeNet-PM', 'AVION-PM'] if m in set(combined.method)]
for metric in METRICS:
    wide_all = metric_wide(paper_summary, metric)
    save_csv(wide_all, f'paper_table_{metric.lower()}_all_methods')
    if focus_three:
        wide_focus = metric_wide(paper_summary, metric, focus_three)
        save_csv(wide_focus, f'paper_table_{metric.lower()}_capstone_pm')
focus_numeric = paper_summary[paper_summary.method.isin(focus_three) & paper_summary.metric.isin(['Accuracy', 'MacroF1', 'ECE'])].copy()
focus_numeric['mean_pct'] = 100 * focus_numeric['mean']
focus_numeric['ci_low_pct'] = 100 * focus_numeric['hier_boot_ci95_low']
focus_numeric['ci_high_pct'] = 100 * focus_numeric['hier_boot_ci95_high']
save_csv(focus_numeric, 'paper_table_capstone_vs_pm_detailed')

def latex_accuracy_table(summary_long, methods):
    q = summary_long[(summary_long.metric == 'Accuracy') & summary_long.method.isin(methods)].copy()
    lines = ['\\begin{table*}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}']
    header = 'Dataset & Shot & ' + ' & '.join((display_name(m).replace('_', '\\_') for m in methods)) + ' \\\\'
    lines += ['\\begin{tabular}{ll' + 'c' * len(methods) + '}', '\\toprule', header, '\\midrule']
    for (ds, K), g in q.groupby(['dataset', 'K'], sort=True):
        vals = {r.method: r for _, r in g.iterrows()}
        available = [(m, vals[m]['mean']) for m in methods if m in vals]
        order = sorted(available, key=lambda x: x[1], reverse=True)
        best = order[0][0] if order else None
        second = order[1][0] if len(order) > 1 else None
        cells = []
        for m in methods:
            if m not in vals:
                cells.append('--')
                continue
            r = vals[m]
            text = f"{100 * r['mean']:.2f} $\\pm$ {100 * r['std_across_folds']:.2f}"
            if m == best:
                text = '\\textbf{' + text + '}'
            elif m == second:
                text = '\\underline{' + text + '}'
            cells.append(text)
        lines.append(f"{ds.replace('_', '\\_')} & {int(K)} & " + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\caption{Accuracy (\\%) under the identical CAPSTONE disjoint five-fold protocol. Values are mean $\\pm$ standard deviation across outer folds after averaging support seeds within each fold. AVION-PM and APPLeNet-PM are protocol-matched reimplementations.}', '\\label{tab:capstone_pm_accuracy}', '\\end{table*}']
    return '\n'.join(lines)
if focus_three:
    save_latex(latex_accuracy_table(paper_summary, focus_three), 'table_capstone_pm_accuracy')
print('Ana tablo yöntemleri:', focus_three)

def paired_fold_data(df, candidate, reference, metric, dataset=None, K=None):
    q = df.copy()
    if dataset is not None:
        q = q[q.dataset == dataset]
    if K is not None:
        q = q[q.K == K]
    q = q[q.method.isin([candidate, reference])]
    agg = q.groupby(['dataset', 'K', 'fold', 'method'], as_index=False)[metric].mean()
    a = agg[agg.method == candidate].rename(columns={metric: 'candidate_value'}).drop(columns='method')
    b = agg[agg.method == reference].rename(columns={metric: 'reference_value'}).drop(columns='method')
    return a.merge(b, on=['dataset', 'K', 'fold'], how='inner')

def rank_biserial_paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    return float((pos - neg) / (pos + neg))

def cluster_bootstrap_ci(pairs, B=BOOTSTRAP_REPS, seed=RANDOM_SEED):
    if pairs.empty:
        return (np.nan, np.nan)
    groups = []
    for _, g in pairs.groupby(['dataset', 'K'], dropna=False):
        a = g.benefit_delta.dropna().to_numpy(float)
        if len(a):
            groups.append(a)
    G = len(groups)
    if G == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    cell_boot = np.empty((G, B), float)
    for i, a in enumerate(groups):
        picks = rng.integers(0, len(a), size=(B, len(a)))
        cell_boot[i] = a[picks].mean(axis=1)
    group_choice = rng.integers(0, G, size=(B, G))
    boot = cell_boot.T[np.arange(B)[:, None], group_choice].mean(axis=1)
    return (float(np.quantile(boot, ALPHA / 2)), float(np.quantile(boot, 1 - ALPHA / 2)))

def mixedlm_global(df, candidate, reference, metric):
    q = df[df.method.isin([candidate, reference])].copy()
    q = q.groupby(['dataset', 'K', 'fold', 'method'], as_index=False)[metric].mean()
    if len(q) < 12 or q.dataset.nunique() < 2:
        return {'mixedlm_available': False, 'mixedlm_reason': 'insufficient groups'}
    sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
    q['oriented_value'] = sign * q[metric]
    q['is_candidate'] = (q.method == candidate).astype(int)
    q['fold_id'] = q.dataset.astype(str) + '_' + q.K.astype(str) + '_' + q.fold.astype(str)
    try:
        fit = smf.mixedlm('oriented_value ~ is_candidate + C(K)', q, groups=q['dataset'], vc_formula={'fold': '0+C(fold_id)'}).fit(reml=False, method='lbfgs', disp=False)
        return {'mixedlm_available': True, 'mixedlm_benefit_coef': float(fit.params.get('is_candidate', np.nan)), 'mixedlm_p': float(fit.pvalues.get('is_candidate', np.nan)), 'mixedlm_converged': bool(getattr(fit, 'converged', False))}
    except Exception as exc:
        return {'mixedlm_available': False, 'mixedlm_reason': f'{type(exc).__name__}: {exc}'}

def paired_stats(df, candidate, reference, metric, scope='global', dataset=None, K=None, bootstrap=False, run_mixedlm=False):
    m = paired_fold_data(df, candidate, reference, metric, dataset, K)
    if len(m) < 2:
        return None
    sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
    raw = (m.candidate_value - m.reference_value).to_numpy(float)
    benefit = sign * raw
    m = m.copy()
    m['benefit_delta'] = benefit
    try:
        t_p = float(stats.ttest_rel(m.candidate_value, m.reference_value).pvalue)
    except Exception:
        t_p = np.nan
    try:
        w_p = float(stats.wilcoxon(benefit, zero_method='wilcox', alternative='two-sided').pvalue)
    except Exception:
        w_p = np.nan
    sd = np.std(benefit, ddof=1)
    dz = float(np.mean(benefit) / sd) if sd > 0 else np.nan
    analytic_lo, analytic_hi = t_ci(benefit)
    if bootstrap:
        ci_lo, ci_hi = cluster_bootstrap_ci(m, B=BOOTSTRAP_REPS, seed=stable_int('diff', candidate, reference, metric, scope, dataset, K, RANDOM_SEED))
        ci_method = 'dataset-shot/fold cluster bootstrap'
    else:
        ci_lo, ci_hi = (analytic_lo, analytic_hi)
        ci_method = 'paired t confidence interval'
    row = {'scope': scope, 'dataset': dataset, 'K': K, 'candidate': candidate, 'reference': reference, 'metric': metric, 'n_paired_folds': len(m), 'raw_delta_candidate_minus_reference': float(np.mean(raw)), 'benefit_delta_positive_is_better': float(np.mean(benefit)), 'median_benefit_delta': float(np.median(benefit)), 'ci95_low': ci_lo, 'ci95_high': ci_hi, 'ci_method': ci_method, 'hier_boot_ci95_low': ci_lo, 'hier_boot_ci95_high': ci_hi, 'paired_t_p': t_p, 'wilcoxon_p': w_p, 'cohen_dz': dz, 'rank_biserial': rank_biserial_paired(benefit), 'common_language': float(np.mean(benefit > 0) + 0.5 * np.mean(benefit == 0))}
    if metric in ['Accuracy', 'MacroF1', 'BalancedAccuracy']:
        try:
            p_ni = float(stats.ttest_1samp(benefit, -NONINFERIORITY_MARGIN, alternative='greater').pvalue)
            p_lower = p_ni
            p_upper = float(stats.ttest_1samp(benefit, NONINFERIORITY_MARGIN, alternative='less').pvalue)
            row.update({'noninferiority_margin': NONINFERIORITY_MARGIN, 'noninferiority_p': p_ni, 'noninferior_at_alpha': bool(p_ni < ALPHA), 'equivalence_tost_p': max(p_lower, p_upper), 'equivalent_within_margin': bool(max(p_lower, p_upper) < ALPHA)})
        except Exception:
            pass
    if run_mixedlm:
        row.update(mixedlm_global(df, candidate, reference, metric))
    return row
methods_available = sorted(combined.method.dropna().unique())
comparators = [m for m in methods_available if m != 'CAPSTONE-CLIP']
key_refs = [m for m in ['APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared'] if m in comparators]
stat_rows = []
for metric in ['Accuracy', 'MacroF1', 'BalancedAccuracy', 'ECE', 'NLL']:
    for ref in comparators:
        r = paired_stats(combined, 'CAPSTONE-CLIP', ref, metric, 'global', bootstrap=True, run_mixedlm=metric == 'Accuracy' and ref in key_refs)
        if r:
            stat_rows.append(r)
for ref in key_refs:
    for ds in sorted(set(combined.dataset.dropna())):
        r = paired_stats(combined, 'CAPSTONE-CLIP', ref, 'Accuracy', 'dataset', dataset=ds, bootstrap=ref in ['APPLeNet-PM', 'AVION-PM'])
        if r:
            stat_rows.append(r)
    for shot in sorted(set(pd.to_numeric(combined.K, errors='coerce').dropna().astype(int))):
        r = paired_stats(combined, 'CAPSTONE-CLIP', ref, 'Accuracy', 'shot', K=shot, bootstrap=False)
        if r:
            stat_rows.append(r)
for ref in [m for m in ['APPLeNet-PM', 'AVION-PM'] if m in comparators]:
    for ds in sorted(set(combined.dataset.dropna())):
        for shot in sorted(set(pd.to_numeric(combined.K, errors='coerce').dropna().astype(int))):
            r = paired_stats(combined, 'CAPSTONE-CLIP', ref, 'Accuracy', 'dataset_shot', dataset=ds, K=shot, bootstrap=False)
            if r:
                stat_rows.append(r)
pairwise = pd.DataFrame(stat_rows)
if not pairwise.empty:
    pairwise['wilcoxon_p_holm'] = np.nan
    pairwise['paired_t_p_holm'] = np.nan
    for _, idx in pairwise.groupby(['scope', 'metric'], dropna=False).groups.items():
        idx = list(idx)
        pairwise.loc[idx, 'wilcoxon_p_holm'] = holm_adjust(pairwise.loc[idx, 'wilcoxon_p'].to_numpy())
        pairwise.loc[idx, 'paired_t_p_holm'] = holm_adjust(pairwise.loc[idx, 'paired_t_p'].to_numpy())
    pairwise['significant_wilcoxon_holm'] = pairwise.wilcoxon_p_holm < ALPHA
save_csv(pairwise, 'table_pairwise_inference_all_scopes')
save_csv(pairwise[pairwise.scope == 'global'], 'paper_table_global_pairwise_inference')
save_csv(pairwise[(pairwise.scope == 'dataset') & pairwise.reference.isin(['APPLeNet-PM', 'AVION-PM'])], 'paper_table_capstone_pm_per_dataset_tests')
display(pairwise[(pairwise.scope == 'global') & pairwise.reference.isin(['APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption'])].sort_values(['metric', 'reference']))

def ranking_analysis(cell_df, methods, metric, label):
    q = cell_df[cell_df.method.isin(methods)].copy()
    pivot = q.pivot_table(index=['dataset', 'K'], columns='method', values=metric, aggfunc='mean')
    pivot = pivot.dropna(subset=methods, how='any') if all((m in pivot.columns for m in methods)) else pd.DataFrame()
    if len(pivot) < 2 or len(methods) < 3:
        return (pd.DataFrame(), pd.DataFrame(), {'label': label, 'metric': metric, 'n_blocks': len(pivot), 'status': 'insufficient'})
    arr = [pivot[m].to_numpy(float) for m in methods]
    fried = stats.friedmanchisquare(*arr)
    higher = HIGHER_IS_BETTER[metric]
    ranks = np.vstack([rankdata(-row if higher else row, method='average') for row in pivot[methods].to_numpy(float)])
    rank_table = pd.DataFrame({'method': methods, 'average_rank': ranks.mean(axis=0), 'rank_std': ranks.std(axis=0, ddof=1)})
    rank_table['analysis_label'] = label
    rank_table['metric'] = metric
    rank_table['n_blocks'] = len(pivot)
    post = []
    for a, b in combinations(methods, 2):
        d = (pivot[a] - pivot[b]).to_numpy(float)
        if not higher:
            d = -d
        try:
            p = float(stats.wilcoxon(d, zero_method='wilcox').pvalue)
        except Exception:
            p = np.nan
        post.append({'analysis_label': label, 'metric': metric, 'method_a': a, 'method_b': b, 'mean_benefit_a_minus_b': float(np.mean(d)), 'wilcoxon_p': p})
    post = pd.DataFrame(post)
    if len(post):
        post['wilcoxon_p_holm'] = holm_adjust(post.wilcoxon_p.to_numpy())
    k = len(methods)
    N = len(pivot)
    try:
        q_alpha = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
        cd = float(q_alpha * np.sqrt(k * (k + 1) / (6 * N)))
    except Exception:
        cd = np.nan
    meta = {'label': label, 'metric': metric, 'n_blocks': N, 'n_methods': k, 'friedman_statistic': float(fried.statistic), 'friedman_p': float(fried.pvalue), 'nemenyi_cd': cd, 'status': 'ok'}
    return (rank_table, post, meta)
ranking_sets = {'internal_complete': [m for m in primary_methods if m in set(cell_summary.method)], 'capstone_pm': [m for m in ['CAPSTONE-CLIP', 'APPLeNet-PM', 'AVION-PM'] if m in set(cell_summary.method)], 'focus_methods': [m for m in FOCUS_METHOD_ORDER if m in set(cell_summary.method)]}
rank_tables = []
posthoc_tables = []
rank_meta = []
for label, methods in ranking_sets.items():
    for metric in ['Accuracy', 'MacroF1', 'ECE']:
        rt, ph, meta = ranking_analysis(cell_summary, methods, metric, label)
        if len(rt):
            rank_tables.append(rt)
        if len(ph):
            posthoc_tables.append(ph)
        rank_meta.append(meta)
rank_results = pd.concat(rank_tables, ignore_index=True) if rank_tables else pd.DataFrame()
posthoc_results = pd.concat(posthoc_tables, ignore_index=True) if posthoc_tables else pd.DataFrame()
rank_meta_df = pd.DataFrame(rank_meta)
save_csv(rank_results, 'table_average_ranks')
save_csv(posthoc_results, 'table_friedman_posthoc_wilcoxon_holm')
save_csv(rank_meta_df, 'table_friedman_omnibus')

def plot_rank_diagram(rank_table, meta, stem):
    if rank_table.empty:
        return
    q = rank_table.sort_values('average_rank')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(q) + 1)))
    y = np.arange(len(q))
    ax.scatter(q.average_rank, y, s=60)
    for yi, (_, r) in zip(y, q.iterrows()):
        ax.text(r.average_rank + 0.04, yi, display_name(r.method), va='center', fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel('Average rank (1 = best)')
    ax.set_xlim(0.8, max(len(q) + 0.5, q.average_rank.max() + 0.8))
    ax.grid(axis='x', alpha=0.3)
    cd = meta.get('nemenyi_cd', np.nan)
    if np.isfinite(cd):
        ycd = -0.6
        start = 1.0
        ax.plot([start, start + cd], [ycd, ycd], linewidth=3)
        ax.text(start + cd / 2, ycd - 0.15, f'CD={cd:.2f}', ha='center', va='top')
        ax.set_ylim(len(q) - 0.5, -1.2)
    ax.set_title(f"{meta['label']} — {meta['metric']} average ranks; Friedman p={meta.get('friedman_p', np.nan):.3g}")
    save_figure(fig, stem)
for _, meta in rank_meta_df[rank_meta_df.status == 'ok'].iterrows():
    rt = rank_results[(rank_results.analysis_label == meta.label) & (rank_results.metric == meta.metric)]
    plot_rank_diagram(rt, meta.to_dict(), f'figure_rank_{meta.label}_{meta.metric.lower()}')
display(rank_meta_df)
cell_acc = cell_summary[['dataset', 'K', 'method', 'Accuracy']].copy()
wtl_rows = []
for ref in [m for m in methods_available if m != 'CAPSTONE-CLIP']:
    a = cell_acc[cell_acc.method == 'CAPSTONE-CLIP'].rename(columns={'Accuracy': 'cap'}).drop(columns='method')
    b = cell_acc[cell_acc.method == ref].rename(columns={'Accuracy': 'ref'}).drop(columns='method')
    m = a.merge(b, on=['dataset', 'K'])
    d = m.cap - m.ref
    wtl_rows.append({'candidate': 'CAPSTONE-CLIP', 'reference': ref, 'n_common_cells': len(m), 'wins': int((d > PRACTICAL_TIE_MARGIN).sum()), 'ties': int((d.abs() <= PRACTICAL_TIE_MARGIN).sum()), 'losses': int((d < -PRACTICAL_TIE_MARGIN).sum()), 'mean_delta': float(d.mean()) if len(d) else np.nan, 'median_delta': float(d.median()) if len(d) else np.nan, 'practical_tie_margin': PRACTICAL_TIE_MARGIN})
wtl = pd.DataFrame(wtl_rows).sort_values(['wins', 'mean_delta'], ascending=False)
best_rows = []
for (ds, K), g in cell_acc.groupby(['dataset', 'K']):
    g = g.dropna(subset=['Accuracy'])
    if g.empty:
        continue
    mx = g.Accuracy.max()
    for m in g[np.isclose(g.Accuracy, mx, atol=1e-12)].method:
        best_rows.append({'dataset': ds, 'K': K, 'method': m, 'best': 1})
best_counts = pd.DataFrame(best_rows).groupby('method', as_index=False).best.sum().sort_values('best', ascending=False) if best_rows else pd.DataFrame()
overall = cell_summary.groupby('method', as_index=False)[METRICS].mean(numeric_only=True).rename(columns={m: f'macro_average_{m}' for m in METRICS})
overall['method_display'] = overall.method.map(display_name)
save_csv(wtl, 'paper_table_win_tie_loss')
save_csv(best_counts, 'table_best_method_cell_counts')
save_csv(overall, 'paper_table_overall_macro_averages')
if not wtl.empty:
    q = wtl.head(10).sort_values('wins')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(q) + 1)))
    y = np.arange(len(q))
    ax.barh(y, q.wins, label='Wins')
    ax.barh(y, q.ties, left=q.wins, label='Ties')
    ax.barh(y, q.losses, left=q.wins + q.ties, label='Losses')
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(x) for x in q.reference])
    ax.set_xlabel('Dataset-shot cells')
    ax.set_title('CAPSTONE-CLIP win/tie/loss by common dataset-shot cell')
    ax.legend()
    save_figure(fig, 'figure_win_tie_loss')
display(overall.sort_values('macro_average_Accuracy', ascending=False))
display(wtl.head(10))
focus_methods = [m for m in FOCUS_METHOD_ORDER if m in set(combined.method)]

def curve_table(metric):
    q = fold_summary[fold_summary.method.isin(focus_methods)].copy()
    rows = []
    for (K, m), g in q.groupby(['K', 'method']):
        vals = g[metric].dropna().to_numpy(float)
        lo, hi = t_ci(vals)
        rows.append({'K': int(K), 'method': m, 'mean': float(vals.mean()), 'ci_low': lo, 'ci_high': hi, 'n': len(vals)})
    return pd.DataFrame(rows)
for metric in ['Accuracy', 'MacroF1', 'ECE']:
    ct = curve_table(metric)
    save_csv(ct, f'figure_data_{metric.lower()}_vs_shot')
    if ct.empty:
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, g in ct.groupby('method'):
        g = g.sort_values('K')
        ax.plot(g.K, g['mean'], marker='o', label=display_name(method))
        ax.fill_between(g.K, g.ci_low, g.ci_high, alpha=0.12)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(ct.K.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Shots per class')
    ax.set_ylabel(metric if metric == 'NLL' else metric + ' (fraction)')
    ax.set_title(f'{metric} versus shot count, averaged across available datasets and folds')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, f'figure_{metric.lower()}_vs_shot_focus')
for ref in ['APPLeNet-PM', 'AVION-PM']:
    a = cell_acc[cell_acc.method == 'CAPSTONE-CLIP'].rename(columns={'Accuracy': 'cap'}).drop(columns='method')
    b = cell_acc[cell_acc.method == ref].rename(columns={'Accuracy': 'ref'}).drop(columns='method')
    m = a.merge(b, on=['dataset', 'K'])
    m['delta_pp'] = 100 * (m.cap - m.ref)
    save_csv(m, f"figure_data_delta_capstone_vs_{ref.lower().replace('-', '_')}")
    if m.empty:
        continue
    p = m.pivot(index='dataset', columns='K', values='delta_pp').reindex(index=expected_datasets, columns=expected_shots)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(p.to_numpy(float), aspect='auto')
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels(p.columns)
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(p.index)
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            v = p.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=9)
    ax.set_xlabel('Shots per class')
    ax.set_title(f'Accuracy delta: CAPSTONE-CLIP minus {display_name(ref)} (percentage points)')
    fig.colorbar(im, ax=ax, label='Accuracy delta (pp)')
    save_figure(fig, f"figure_delta_heatmap_capstone_vs_{ref.lower().replace('-', '_')}")
pareto = cell_summary.groupby('method', as_index=False).agg(Accuracy=('Accuracy', 'mean'), ECE=('ECE', 'mean'))
pareto = pareto[pareto.method.isin(focus_methods)]
save_csv(pareto, 'figure_data_accuracy_ece_pareto')
if len(pareto):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(100 * pareto.ECE, 100 * pareto.Accuracy, s=70)
    for _, r in pareto.iterrows():
        ax.annotate(display_name(r.method), (100 * r.ECE, 100 * r.Accuracy), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set_xlabel('ECE (%) — lower is better')
    ax.set_ylabel('Accuracy (%) — higher is better')
    ax.set_title('Performance–calibration trade-off')
    ax.grid(alpha=0.25)
    save_figure(fig, 'figure_accuracy_ece_pareto')
forest = pairwise[(pairwise.scope == 'global') & (pairwise.metric == 'Accuracy')].copy()
forest = forest[forest.reference.isin(focus_methods)].sort_values('benefit_delta_positive_is_better')
save_csv(forest, 'figure_data_global_accuracy_forest')
if len(forest):
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(forest) + 1)))
    y = np.arange(len(forest))
    x = 100 * forest.benefit_delta_positive_is_better
    lo = 100 * forest.hier_boot_ci95_low
    hi = 100 * forest.hier_boot_ci95_high
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt='o', capsize=3)
    ax.axvline(0, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(x) for x in forest.reference])
    ax.set_xlabel('CAPSTONE-CLIP accuracy benefit (percentage points)')
    ax.set_title('Hierarchical bootstrap forest plot')
    ax.grid(axis='x', alpha=0.25)
    save_figure(fig, 'figure_global_accuracy_forest')
controls = coerce_numeric(MAIN['controls'], ['K', 'fold', 'support_seed', 'caption_scale', 'Accuracy', 'MacroF1', 'ECE', 'external_to_support_ratio'])
controls = completed_rows(controls, 'CAPSTONE-main') if not controls.empty else controls
control_summary = pd.DataFrame()
if not controls.empty:
    control_summary = controls.groupby(['dataset', 'K', 'control', 'caption_scale', 'caption_corpus'], dropna=False).agg(Accuracy_mean=('Accuracy', 'mean'), Accuracy_std=('Accuracy', 'std'), MacroF1_mean=('MacroF1', 'mean'), ECE_mean=('ECE', 'mean'), n_runs=('Accuracy', 'count')).reset_index()
    save_csv(control_summary, 'paper_table_caption_controls')
    q = control_summary.groupby(['K', 'control'], as_index=False).Accuracy_mean.mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    for control, g in q.groupby('control'):
        g = g.sort_values('K')
        ax.plot(g.K, g.Accuracy_mean, marker='o', label=control)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(q.K.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Shots per class')
    ax.set_ylabel('Mean accuracy')
    ax.set_title('Caption causal-control comparison')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, 'figure_caption_control_accuracy')
    s = control_summary[control_summary.control == 'joint_full'].copy()
    s = s.groupby('caption_scale', as_index=False).Accuracy_mean.mean().sort_values('caption_scale')
    if len(s):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(s.caption_scale, s.Accuracy_mean, marker='o')
        ax.set_xlabel('Caption corpus fraction')
        ax.set_ylabel('Mean accuracy')
        ax.set_title('Caption-data scaling')
        ax.grid(alpha=0.25)
        save_figure(fig, 'figure_caption_scaling')
    corpus = control_summary[(control_summary.control == 'joint_full') & (control_summary.caption_scale == 1.0)].groupby('caption_corpus', as_index=False).Accuracy_mean.mean().sort_values('Accuracy_mean', ascending=False)
    save_csv(corpus, 'paper_table_caption_corpus_transfer')
    if len(corpus):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(corpus.caption_corpus, corpus.Accuracy_mean)
        ax.set_ylabel('Mean accuracy')
        ax.set_title('Caption-corpus transfer')
        save_figure(fig, 'figure_caption_corpus_transfer')
else:
    print('Caption-control raw çıktısı yok; bu bölüm atlandı.')
retrieval = coerce_numeric(MAIN['retrieval'], ['K', 'fold', 'support_seed', 'classification_Accuracy', 'mR', 'i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10'])
if not retrieval.empty:
    ret_summary = retrieval.groupby(['backbone', 'caption_dataset', 'classification_dataset', 'method'], dropna=False).agg(classification_Accuracy=('classification_Accuracy', 'mean'), mR=('mR', 'mean'), i2t_R1=('i2t_R1', 'mean'), i2t_R5=('i2t_R5', 'mean'), i2t_R10=('i2t_R10', 'mean'), t2i_R1=('t2i_R1', 'mean'), t2i_R5=('t2i_R5', 'mean'), t2i_R10=('t2i_R10', 'mean'), n_runs=('mR', 'count')).reset_index()
    save_csv(ret_summary, 'paper_table_retrieval')
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in ret_summary.iterrows():
        ax.scatter(100 * r.classification_Accuracy, 100 * r.mR, s=65)
        ax.annotate(f'{r.method}\n{r.caption_dataset}', (100 * r.classification_Accuracy, 100 * r.mR), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Classification accuracy (%)')
    ax.set_ylabel('Mean Recall (%)')
    ax.set_title('Classification–retrieval Pareto analysis')
    ax.grid(alpha=0.25)
    save_figure(fig, 'figure_retrieval_pareto')
else:
    ret_summary = pd.DataFrame()
    print('Retrieval raw çıktısı yok; bu bölüm atlandı.')
abl = coerce_numeric(MAIN['ablations'], ['K', 'fold', 'support_seed', 'Accuracy', 'MacroF1', 'ECE'])
if not abl.empty:
    abl_summary = abl.groupby(['dataset', 'K', 'variant'], dropna=False).agg(Accuracy_mean=('Accuracy', 'mean'), MacroF1_mean=('MacroF1', 'mean'), ECE_mean=('ECE', 'mean'), n_runs=('Accuracy', 'count')).reset_index()
    full = abl_summary[abl_summary.variant == 'full'][['dataset', 'K', 'Accuracy_mean']].rename(columns={'Accuracy_mean': 'full_acc'})
    abl_delta = abl_summary.merge(full, on=['dataset', 'K'], how='left')
    abl_delta['delta_vs_full_pp'] = 100 * (abl_delta.Accuracy_mean - abl_delta.full_acc)
    save_csv(abl_summary, 'paper_table_architecture_ablations')
    save_csv(abl_delta, 'paper_table_architecture_ablation_deltas')
    p = abl_delta[abl_delta.variant != 'full'].pivot_table(index='variant', columns=['dataset', 'K'], values='delta_vs_full_pp')
    if not p.empty:
        fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(p.columns)), 4.5))
        im = ax.imshow(p.to_numpy(float), aspect='auto')
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels(p.index)
        ax.set_xticks(range(len(p.columns)))
        ax.set_xticklabels([f'{a}\nK={b}' for a, b in p.columns], rotation=45, ha='right')
        for i in range(len(p.index)):
            for j in range(len(p.columns)):
                v = p.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=8)
        ax.set_title('Architecture ablation accuracy delta versus full CAPSTONE (pp)')
        fig.colorbar(im, ax=ax, label='Delta (pp)')
        save_figure(fig, 'figure_architecture_ablation_heatmap')
else:
    abl_summary = pd.DataFrame()
    print('Ablation raw çıktısı yok; bu bölüm atlandı.')
b2n = coerce_numeric(MAIN['base_to_new'], ['K', 'fold', 'support_seed', 'BaseAccuracy', 'NovelAccuracy', 'HarmonicMean'])
if not b2n.empty:
    b2n_summary = b2n.groupby(['backbone', 'dataset', 'K'], dropna=False).agg(BaseAccuracy=('BaseAccuracy', 'mean'), NovelAccuracy=('NovelAccuracy', 'mean'), HarmonicMean=('HarmonicMean', 'mean'), n_runs=('HarmonicMean', 'count')).reset_index()
    save_csv(b2n_summary, 'paper_table_base_to_new')
    avg = b2n_summary.groupby('K', as_index=False)[['BaseAccuracy', 'NovelAccuracy', 'HarmonicMean']].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in ['BaseAccuracy', 'NovelAccuracy', 'HarmonicMean']:
        ax.plot(avg.K, avg[c], marker='o', label=c)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(avg.K.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Shots per base class')
    ax.set_ylabel('Accuracy / harmonic mean')
    ax.set_title('Base-to-new generalization')
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, 'figure_base_to_new')
else:
    b2n_summary = pd.DataFrame()
    print('Base-to-new çıktısı yok.')
cross = coerce_numeric(MAIN['cross_dataset'], ['K', 'fold', 'support_seed'] + METRICS)
if not cross.empty:
    cross_summary = cross.groupby(['backbone', 'source', 'target', 'K'], dropna=False).agg(Accuracy=('Accuracy', 'mean'), MacroF1=('MacroF1', 'mean'), ECE=('ECE', 'mean'), n_runs=('Accuracy', 'count')).reset_index()
    save_csv(cross_summary, 'paper_table_cross_dataset')
    p = cross_summary.pivot_table(index='source', columns='target', values='Accuracy')
    if not p.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(100 * p.to_numpy(float), aspect='auto')
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels(p.index)
        ax.set_xticks(range(len(p.columns)))
        ax.set_xticklabels(p.columns, rotation=45, ha='right')
        for i in range(len(p.index)):
            for j in range(len(p.columns)):
                v = p.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{100 * v:.1f}', ha='center', va='center')
        ax.set_title('Cross-dataset accuracy (%)')
        fig.colorbar(im, ax=ax, label='Accuracy (%)')
        save_figure(fig, 'figure_cross_dataset_heatmap')
else:
    cross_summary = pd.DataFrame()
    print('Cross-dataset çıktısı yok.')
cb = coerce_numeric(MAIN['cross_backbone'], ['K', 'fold', 'support_seed'] + METRICS)
if not cb.empty:
    cb_summary = cb.groupby(['backbone', 'dataset', 'K', 'method'], dropna=False).agg(Accuracy=('Accuracy', 'mean'), MacroF1=('MacroF1', 'mean'), ECE=('ECE', 'mean'), n_runs=('Accuracy', 'count')).reset_index()
    save_csv(cb_summary, 'paper_table_cross_backbone')
    avg = cb_summary.groupby(['backbone', 'method'], as_index=False).Accuracy.mean()
    p = avg.pivot(index='method', columns='backbone', values='Accuracy')
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(p) + 1)))
    im = ax.imshow(100 * p.to_numpy(float), aspect='auto')
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(p.index)
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels(p.columns, rotation=30, ha='right')
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            v = p.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{100 * v:.1f}', ha='center', va='center', fontsize=8)
    ax.set_title('Cross-backbone mean accuracy (%)')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')
    save_figure(fig, 'figure_cross_backbone_heatmap')
else:
    cb_summary = pd.DataFrame()
    print('Cross-backbone çıktısı yok.')
eff_cols = [c for c in ['method', 'train_eval_s', 'trainable_params', 'peak_gpu_mb', 'peak_gpu_delta_mb'] if c in combined.columns]
run_eff = pd.DataFrame()
if len(eff_cols) > 1:
    agg = {c: 'mean' for c in eff_cols if c != 'method'}
    run_eff = combined.groupby('method', as_index=False).agg(agg)
    perf = cell_summary.groupby('method', as_index=False).Accuracy.mean()
    run_eff = run_eff.merge(perf, on='method', how='left')
    save_csv(run_eff, 'paper_table_efficiency_from_runs')
    if 'trainable_params' in run_eff.columns and run_eff.trainable_params.notna().any():
        q = run_eff.dropna(subset=['trainable_params', 'Accuracy'])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(q.trainable_params, 100 * q.Accuracy, s=65)
        for _, r in q.iterrows():
            ax.annotate(display_name(r.method), (r.trainable_params, 100 * r.Accuracy), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.set_xscale('log')
        ax.set_xlabel('Trainable parameters (log scale)')
        ax.set_ylabel('Mean accuracy (%)')
        ax.set_title('Accuracy–parameter efficiency')
        ax.grid(alpha=0.25)
        save_figure(fig, 'figure_accuracy_vs_trainable_parameters')
    if 'train_eval_s' in run_eff.columns and run_eff.train_eval_s.notna().any():
        q = run_eff.dropna(subset=['train_eval_s', 'Accuracy'])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(q.train_eval_s, 100 * q.Accuracy, s=65)
        for _, r in q.iterrows():
            ax.annotate(display_name(r.method), (r.train_eval_s, 100 * r.Accuracy), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.set_xlabel('Mean train + evaluation time per run (s)')
        ax.set_ylabel('Mean accuracy (%)')
        ax.set_title('Accuracy–runtime trade-off')
        ax.grid(alpha=0.25)
        save_figure(fig, 'figure_accuracy_vs_runtime')
stability = combined.groupby(['dataset', 'K', 'method'], dropna=False).agg(mean_accuracy=('Accuracy', 'mean'), run_sd=('Accuracy', 'std'), fold_sd=('Accuracy', lambda x: np.nan), n_runs=('Accuracy', 'count')).reset_index()
fold_sd = combined.groupby(['dataset', 'K', 'method', 'fold'], dropna=False).Accuracy.mean().reset_index().groupby(['dataset', 'K', 'method'], dropna=False).Accuracy.std().rename('fold_sd_real').reset_index()
stability = stability.drop(columns='fold_sd').merge(fold_sd, on=['dataset', 'K', 'method'], how='left')
save_csv(stability, 'paper_table_stability')
near = coerce_numeric(MAIN['near_duplicate'], ['fold', 'threshold', 'fraction_above', 'n_above', 'sampled_test'])
if not near.empty:
    near_summary = near.groupby(['backbone', 'dataset', 'threshold'], dropna=False).agg(fraction_above=('fraction_above', 'mean'), n_above=('n_above', 'sum'), sampled_test=('sampled_test', 'sum')).reset_index()
    save_csv(near_summary, 'paper_table_near_duplicate_audit')
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds, g in near_summary.groupby('dataset'):
        g = g.sort_values('threshold')
        ax.plot(g.threshold, 100 * g.fraction_above, marker='o', label=ds)
    ax.set_xlabel('Cosine-similarity threshold')
    ax.set_ylabel('Sampled test images above threshold (%)')
    ax.set_title('Train–test near-duplicate feature audit')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, 'figure_near_duplicate_audit')
else:
    near_summary = pd.DataFrame()
    print('Near-duplicate audit çıktısı yok.')
per_class_files = glob.glob(os.path.join(CAPSTONE_ROOT, 'tables', 'per_class_*.csv'))
reliability_files = glob.glob(os.path.join(CAPSTONE_ROOT, 'tables', 'reliability_*.csv'))
high_error_files = glob.glob(os.path.join(CAPSTONE_ROOT, 'tables', 'high_confidence_errors_*.csv'))
class_index = pd.DataFrame({'file': per_class_files, 'type': 'per_class'}) if per_class_files else pd.DataFrame(columns=['file', 'type'])
class_index = pd.concat([class_index, pd.DataFrame({'file': reliability_files, 'type': 'reliability'}), pd.DataFrame({'file': high_error_files, 'type': 'high_confidence_error'})], ignore_index=True)
save_csv(class_index, 'artifact_index_class_diagnostics')
worst = []
for p in per_class_files:
    try:
        d = pd.read_csv(p)
        recall_col = next((c for c in d.columns if c.lower() in ['recall', 'class_recall', 'accuracy']), None)
        class_col = next((c for c in d.columns if 'class' in c.lower() or 'name' in c.lower()), None)
        if recall_col and class_col:
            z = d.nsmallest(min(10, len(d)), recall_col).copy()
            z['source_file'] = os.path.basename(p)
            worst.append(z)
    except Exception:
        pass
worst_classes = pd.concat(worst, ignore_index=True) if worst else pd.DataFrame()
if not worst_classes.empty:
    save_csv(worst_classes, 'paper_table_worst_classes')
source_figures = []
for source, root in [('CAPSTONE-main', CAPSTONE_ROOT), ('PM-external', PM_ROOT)]:
    if not root:
        continue
    for p in glob.glob(os.path.join(root, 'figures', '*')):
        if os.path.isfile(p):
            source_figures.append({'source': source, 'file': p, 'basename': os.path.basename(p), 'size_bytes': os.path.getsize(p)})
source_figure_index = pd.DataFrame(source_figures)
save_csv(source_figure_index, 'artifact_index_source_figures')
excel_path = os.path.join(POST_ROOT, 'CAPSTONE_PAPER_TABLES_ALL.xlsx')
excel_tables = {'coverage': coverage, 'protocol_audit': protocol_audit, 'paper_summary': paper_summary, 'overall': overall, 'pairwise_global': pairwise[pairwise.scope == 'global'] if not pairwise.empty else pd.DataFrame(), 'pairwise_dataset': pairwise[pairwise.scope == 'dataset'] if not pairwise.empty else pd.DataFrame(), 'friedman': rank_meta_df, 'average_ranks': rank_results, 'win_tie_loss': wtl, 'caption_controls': control_summary, 'retrieval': ret_summary, 'ablations': abl_summary, 'base_to_new': b2n_summary, 'cross_dataset': cross_summary, 'cross_backbone': cb_summary, 'efficiency': run_eff, 'near_duplicate': near_summary}
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for name, df in excel_tables.items():
        if isinstance(df, pd.DataFrame) and (not df.empty):
            df.to_excel(writer, sheet_name=name[:31], index=False)
print('Excel:', excel_path)
if not overall.empty:
    tex = overall.sort_values('macro_average_Accuracy', ascending=False).to_latex(index=False, float_format='%.4f', escape=True)
    save_latex(tex, 'table_overall_macro_averages')
if not wtl.empty:
    save_latex(wtl.to_latex(index=False, float_format='%.4f', escape=True), 'table_win_tie_loss')
if not pairwise.empty:
    cols = [c for c in ['candidate', 'reference', 'metric', 'n_paired_folds', 'benefit_delta_positive_is_better', 'hier_boot_ci95_low', 'hier_boot_ci95_high', 'wilcoxon_p_holm', 'cohen_dz', 'noninferior_at_alpha', 'equivalent_within_margin'] if c in pairwise.columns]
    save_latex(pairwise[pairwise.scope == 'global'][cols].to_latex(index=False, float_format='%.4g', escape=True), 'table_global_inference')

def pct(x):
    return 'NA' if not np.isfinite(x) else f'{100 * x:.2f}%'

def global_result(ref, metric='Accuracy'):
    q = pairwise[(pairwise.scope == 'global') & (pairwise.metric == metric) & (pairwise.reference == ref)]
    return None if q.empty else q.iloc[0]
lines = []
lines.append('# CAPSTONE-CLIP combined paper-results digest')
lines.append('')
lines.append('## Source audit')
lines.append(f'- CAPSTONE source: `{CAPSTONE_ROOT}`')
lines.append(f'- Protocol-matched source: `{PM_ROOT}`')
lines.append(f'- Main completed rows: {len(main_ok)}')
lines.append(f'- PM completed rows: {len(pm_ok)}')
lines.append(f'- PM manifest verification: {manifest_verified}')
lines.append('')
lines.append('## Completion warnings')
for _, r in coverage[coverage.missing_runs > 0].iterrows():
    lines.append(f"- {r['method']}: {int(r['missing_runs'])} missing of {int(r['expected_runs'])} expected runs ({r['coverage_pct']:.1f}% complete).")
if not (coverage.missing_runs > 0).any():
    lines.append('- All expected grids are complete.')
lines.append('')
lines.append('## Main paired findings')
for ref in ['APPLeNet-PM', 'AVION-PM', 'ProLIP', 'RLA', 'Shared-Caption', 'Capacity-Matched-Shared']:
    r = global_result(ref)
    if r is None:
        continue
    lines.append(f"- CAPSTONE-CLIP vs {display_name(ref)}: mean accuracy benefit {100 * r['benefit_delta_positive_is_better']:+.2f} pp, hierarchical 95% CI [{100 * r['hier_boot_ci95_low']:+.2f}, {100 * r['hier_boot_ci95_high']:+.2f}] pp, Holm-adjusted Wilcoxon p={r.get('wilcoxon_p_holm', np.nan):.4g}, Cohen dz={r.get('cohen_dz', np.nan):.3f}.")
lines.append('')
lines.append('## Manuscript naming and claim rules')
lines.append('- Use `AVION-PM (protocol-matched reimplementation)` and `APPLeNet-PM (protocol-matched reimplementation)`.')
lines.append('- Do not describe PM rows as unchanged official-repository results.')
lines.append('- Do not claim pure few-shot or equal total data budget for caption-augmented methods.')
lines.append('- Caption-grounding claims require completed matched-budget, shuffled-caption and image-only controls.')
lines.append('- Retrieval SOTA claims require official same-protocol comparisons.')
lines.append('')
lines.append('## Statistical reporting rule')
lines.append('Support seeds are averaged inside each external fold before confirmatory inference. Dataset-shot-fold blocks are paired across methods; Holm correction is applied within each analysis family.')
digest = '\n'.join(lines) + '\n'
digest_path = os.path.join(REPORT_DIR, 'PAPER_RESULTS_DIGEST.md')
Path(digest_path).write_text(digest, encoding='utf-8')
artifacts = []
for root, _, files in os.walk(POST_ROOT):
    for fn in files:
        p = os.path.join(root, fn)
        if p.endswith('.zip'):
            continue
        artifacts.append({'relative_path': os.path.relpath(p, POST_ROOT), 'size_bytes': os.path.getsize(p)})
artifact_index = pd.DataFrame(artifacts).sort_values('relative_path')
save_csv(artifact_index, 'ARTIFACT_INDEX')
manifest = {'created_by': 'CAPSTONE_Paper_Postprocess_All_Outputs_v1', 'capstone_source': CAPSTONE_ROOT, 'pm_source': PM_ROOT, 'post_root': POST_ROOT, 'bootstrap_repetitions': BOOTSTRAP_REPS, 'alpha': ALPHA, 'noninferiority_margin': NONINFERIORITY_MARGIN, 'practical_tie_margin': PRACTICAL_TIE_MARGIN, 'datasets': expected_datasets, 'shots': expected_shots, 'folds': expected_folds, 'support_seeds': expected_seeds, 'method_labels': DISPLAY_LABEL, 'artifact_count': len(artifact_index)}
Path(os.path.join(POST_ROOT, 'OUTPUT_MANIFEST.json')).write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
print(digest)
print('Digest:', digest_path)
artifacts = []
for root, _, files in os.walk(POST_ROOT):
    for fn in files:
        p = os.path.join(root, fn)
        if p.endswith('.zip'):
            continue
        artifacts.append({'relative_path': os.path.relpath(p, POST_ROOT), 'size_bytes': os.path.getsize(p)})
artifact_index = pd.DataFrame(artifacts).sort_values('relative_path')
save_csv(artifact_index, 'ARTIFACT_INDEX')
zip_base = '/content/CAPSTONE_PAPER_POSTPROCESS_ALL_OUTPUTS_v1'
zip_path = shutil.make_archive(zip_base, 'zip', POST_ROOT)
zip_drive = os.path.join(POST_ROOT, os.path.basename(zip_path))
shutil.copy2(zip_path, zip_drive)
final_checks = pd.DataFrame([{'check': 'CAPSTONE raw loaded', 'pass': len(main_raw) > 0}, {'check': 'PM raw loaded', 'pass': len(pm_raw) > 0}, {'check': 'Combined raw exported', 'pass': os.path.isfile(os.path.join(TABLE_DIR, 'raw_combined_capstone_and_pm.csv'))}, {'check': 'Coverage audit exported', 'pass': os.path.isfile(os.path.join(TABLE_DIR, 'audit_run_coverage.csv'))}, {'check': 'Pairwise inference exported', 'pass': os.path.isfile(os.path.join(TABLE_DIR, 'table_pairwise_inference_all_scopes.csv'))}, {'check': 'Paper summary exported', 'pass': os.path.isfile(os.path.join(TABLE_DIR, 'table_all_methods_paper_summary_long.csv'))}, {'check': 'Excel workbook exported', 'pass': os.path.isfile(os.path.join(POST_ROOT, 'CAPSTONE_PAPER_TABLES_ALL.xlsx'))}, {'check': 'Digest exported', 'pass': os.path.isfile(os.path.join(REPORT_DIR, 'PAPER_RESULTS_DIGEST.md'))}, {'check': 'ZIP copied to Drive', 'pass': os.path.isfile(zip_drive)}])
save_csv(final_checks, 'FINAL_CHECKS')
display(final_checks)
print('Output root:', POST_ROOT)
print('Drive ZIP:', zip_drive)
print('Local ZIP:', zip_path)
