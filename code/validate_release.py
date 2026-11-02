import ast
import csv
import hashlib
import io
import json
import struct
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest_indices(values):
    return hashlib.sha256(b''.join(struct.pack('<q', int(x)) for x in values)).hexdigest()

def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def main():
    artifacts = json.loads((ROOT / 'provenance' / 'artifact_manifest.json').read_text())
    for item in artifacts:
        path = ROOT / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
    for path in (ROOT / 'code').glob('*.py'):
        source = path.read_text()
        tree = ast.parse(source)
        compile(tree, str(path), 'exec')
        assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)), path
        assert not ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node) is None, (path, node.name)
    lock = json.loads((ROOT / 'manifests' / 'split_manifest_capstone_disjoint5fold_support5_v1.json').read_text())
    integrity = json.loads((ROOT / 'provenance' / 'original_data_integrity.json').read_text())
    datasets = []
    for dataset, folds in lock['datasets'].items():
        directory = ROOT / 'manifests' / dataset
        images = read_rows(directory / 'image_index.csv')
        assignments = read_rows(directory / 'fold_assignments.csv')
        supports = read_rows(directory / 'support_assignments.csv')
        assert [int(x['image_index']) for x in images] == list(range(len(images)))
        assert len(assignments) == len(images)
        by_index = {int(x['image_index']): x['image_path'] for x in images}
        assert all(by_index[int(x['image_index'])] == x['image_path'] for x in assignments + supports)
        paths = [integrity[dataset]['root'] + '/' + x['image_path'].split('/', 1)[1] for x in images]
        assert hashlib.sha256('\n'.join(paths).encode()).hexdigest() == integrity[dataset]['used_path_digest']
        for fold in range(5):
            test = [int(x['image_index']) for x in assignments if int(x['test_fold']) == fold]
            pool = sorted(set(range(len(images))) - set(test))
            expected = folds[str(fold)]
            assert digest_indices(test) == expected['test_sha256']
            assert digest_indices(pool) == expected['train_pool_sha256']
            for shot in lock['shots']:
                for seed in lock['support_seeds']:
                    rows = [x for x in supports if int(x['fold']) == fold and int(x['K']) == shot and int(x['support_seed']) == seed]
                    values = [int(x['image_index']) for x in sorted(rows, key=lambda x: int(x['support_order']))]
                    assert digest_indices(values) == expected['supports'][f'K{shot}_s{seed}']
                    assert not set(values) & set(test)
        datasets.append({'dataset': dataset, 'records': len(images), 'support_memberships': len(supports), 'hashes_match': True})
    tables = []
    required = ['Accuracy', 'MacroF1', 'ECE', 'NLL']
    for relative in ['results/architecture_ablations/raw_architecture_ablations.csv', 'results/all_six_exposure/all6_exposure_controls_raw.csv', 'results/five_width_sweep/adapter_width_sweep_raw.csv', 'results/second_caption_corpus/caption_corpus_robustness_raw.csv', 'results/aid_decontamination/aid_decontaminated_rsicd_raw.csv']:
        rows = read_rows(ROOT / relative)
        assert rows
        assert all(all(row.get(metric, '') != '' for metric in required) for row in rows), relative
        for row in rows:
            assert 0 <= float(row['Accuracy']) <= 1
            assert 0 <= float(row['MacroF1']) <= 1
            assert 0 <= float(row['ECE']) <= 1
            assert float(row['NLL']) >= 0
        tables.append({'path': relative, 'rows': len(rows), 'required_metrics_complete': True})
    print(json.dumps({'status': 'PASS', 'artifact_hashes_checked': len(artifacts), 'datasets': datasets, 'tables': tables, 'training_rerun': False, 'known_duplicate_content_risk': 'docs/data_integrity.md'}, indent=2))

if __name__ == '__main__':
    main()
