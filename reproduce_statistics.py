from pathlib import Path
import json, itertools, hashlib
import pandas as pd, numpy as np
from scipy.stats import friedmanchisquare, rankdata
O = Path(__file__).resolve().parent
O.mkdir(exist_ok=True)
p = pd.read_csv(O / 'source_primary.csv')
w = pd.read_csv(O / 'source_width.csv')
DS = ['NWPU_RESISC45', 'PatternNet', 'AID', 'UCMerced', 'WHU_RS19', 'EuroSAT']
M = ['Zero-shot', 'LinearProbe', 'CLIP-Adapter', 'Tip-Adapter', 'Tip-Adapter-F', 'CoOp', 'TaskRes', 'RLA', 'ProLIP', 'Shared-Caption', 'Capacity-Matched-Shared', 'CAPSTONE-CLIP']

def grid(df, axes):
    keys = list(axes)
    assert not df.duplicated(keys).any()
    assert set(df[keys].itertuples(index=False, name=None)) == set(itertools.product(*axes.values()))
    assert df.Accuracy.between(0, 1).all()
q = p[(p.backbone == 'openclip_b32') & p.method.isin(M)].copy()
grid(q, dict(dataset=DS, K=[1, 2, 4, 8, 16], fold=range(5), support_seed=range(5), method=M))
grid(w, dict(dataset=DS, K=[1, 4, 16], fold=range(5), support_seed=[0, 1], width=[32, 64, 128, 256, 512]))
for width, method in [(128, 'Shared-Caption'), (256, 'Capacity-Matched-Shared')]:
    a = w[w.width == width]
    b = q[(q.method == method) & q.K.isin([1, 4, 16]) & q.support_seed.isin([0, 1])]
    z = a.merge(b, on=['dataset', 'K', 'fold', 'support_seed'], suffixes=('_w', '_p'), validate='one_to_one')
    assert len(z) == 180
    assert np.allclose(z.Accuracy_w, z.Accuracy_p, atol=1e-14)
cell = q.groupby(['dataset', 'K', 'method']).Accuracy.mean().unstack('method').reindex(columns=M)
blocks = cell.groupby('dataset').mean().reindex(DS)
blocks.to_csv(O / 'friedman_six_dataset_blocks.csv')
cell.to_csv(O / 'friedman_30_dependent_cells.csv')
ranks = pd.DataFrame(rankdata(-blocks.to_numpy(), axis=1), index=DS, columns=M)
ranks.to_csv(O / 'friedman_six_dataset_ranks.csv')
rank30 = pd.DataFrame(rankdata(-cell.to_numpy(), axis=1), index=cell.index, columns=M)
rank_summary = pd.DataFrame({'six_dataset_rank': ranks.mean(), 'dependent_30_cell_rank': rank30.mean(), 'mean_accuracy_pct': blocks.mean() * 100}).sort_values('six_dataset_rank')
rank_summary.to_csv(O / 'rank_comparison.csv')
chi, p_asym = friedmanchisquare(*(blocks[m] for m in M))
oldchi, oldp = friedmanchisquare(*(cell[m] for m in M))
a = ranks.to_numpy()
n, k = a.shape
sums = a.sum(axis=0)
manual = 12 / (n * k * (k + 1)) * np.sum(sums * sums) - 3 * n * (k + 1)
ties = sum((sum((v ** 3 - v for v in np.unique(row, return_counts=True)[1])) for row in blocks.to_numpy()))
manual /= 1 - ties / (n * (k ** 3 - k))
assert np.isclose(manual, chi)
rng = np.random.default_rng(20261029)
B = 200000
hits = 0
obs = np.sum((a.mean(0) - (k + 1) / 2) ** 2)
for start in range(0, B, 2000):
    ix = rng.random((min(2000, B - start), n, k)).argsort(axis=2)
    perm = np.take_along_axis(np.broadcast_to(a, (len(ix), n, k)), ix, axis=2)
    score = np.sum((perm.mean(1) - (k + 1) / 2) ** 2, axis=1)
    hits += np.count_nonzero(score >= obs - 1e-12)
lodo = []
for omit in DS:
    b = blocks.drop(omit)
    rr = rankdata(-b.to_numpy(), axis=1).mean(0)
    st, pv = friedmanchisquare(*(b[m] for m in M))
    lodo.append(dict(omitted=omit, statistic=st, p_asymptotic=pv, best_rank_methods=';'.join(np.array(M)[np.isclose(rr, rr.min())])))
pd.DataFrame(lodo).to_csv(O / 'friedman_leave_one_dataset_out.csv', index=False)
signs = np.array(list(itertools.product([-1, 1], repeat=6)))
post = []
for method in M:
    if method == 'CAPSTONE-CLIP':
        continue
    d = (blocks['CAPSTONE-CLIP'] - blocks[method]).to_numpy() * 100
    pv = float((abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
    post.append(dict(comparator=method, full_minus_comparator_pp=d.mean(), p_exact=pv))
post = pd.DataFrame(post).sort_values('p_exact').reset_index(drop=True)
post['p_holm'] = np.minimum(1, np.maximum.accumulate(post.p_exact.to_numpy() * (len(post) - np.arange(len(post)))))
post.to_csv(O / 'secondary_pairwise_exact_holm.csv', index=False)
fried = dict(n_runs=len(q), n_methods=12, n_dataset_blocks=6, statistic=float(chi), df=11, p_asymptotic=float(p_asym), permutation_draws=B, permutation_hits=int(hits), permutation_p_plus_one=(hits + 1) / (B + 1), permutation_seed=20261029, old_30_cell_statistic=float(oldchi), old_30_cell_p=float(oldp), label='exploratory sensitivity; permutation assumes within-dataset method exchangeability; six datasets limit generalization')
piv = w.pivot(index=['dataset', 'K', 'fold', 'support_seed'], columns='width', values='Accuracy')
d = ((piv[512] - piv[128]) * 100).groupby('dataset').mean().reindex(DS)
d.to_csv(O / 'width512_minus128_dataset_effects_pp.csv')
rng = np.random.default_rng(20261005)
boot = rng.choice(d.sort_index().to_numpy(), size=(20000, 6), replace=True).mean(1)
lo, hi = np.quantile(boot, [0.025, 0.975])
pv = float((abs((signs * d.to_numpy()).mean(1)) >= abs(d.mean()) - 1e-12).mean())
wsummary = w.groupby('width').agg(n=('Accuracy', 'size'), accuracy_pct=('Accuracy', lambda x: x.mean() * 100), ece_pct=('ECE', lambda x: x.mean() * 100))
wsummary['params'] = [68674, 134274, 265474, 527874, 1052674]
wsummary.to_csv(O / 'width_summary.csv')
wl = []
for ds in DS:
    wl.append(dict(omitted=ds, remaining_mean_delta_pp=d.drop(ds).mean()))
pd.DataFrame(wl).to_csv(O / 'width_leave_one_dataset_out.csv', index=False)
width = dict(n_runs=len(w), mean512_minus128_pp=float(d.mean()), dataset_bootstrap_95_low_pp=float(lo), dataset_bootstrap_95_high_pp=float(hi), exact_sign_flip_p=pv, n_datasets=6, seed=20261005, bootstrap_B=20000, equivalence_established=False, initialization_note='support seed labels matched, but initialization RNG namespace differs by width; 128/256 reused from primary, 32/64/512 revision runs')
result = dict(friedman=fried, width=width, source_hashes={p: hashlib.sha256((O / ('source_' + p + '.csv')).read_bytes()).hexdigest() for p in ['primary', 'width']})
(O / 'statistics.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print(rank_summary.to_string())
print(post.to_string(index=False))
print('dataset width effects', d.to_dict())
