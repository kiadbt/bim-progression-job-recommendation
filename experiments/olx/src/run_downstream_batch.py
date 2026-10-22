"""Check completed ranking outputs before statistical analysis."""
import numpy as np
import pandas as pd

def verify_output(directory, task, expected_sources):
    from artifact_hash import sha256
    from ranking_driver_core import read_json, INPUTS
    record = read_json(directory / 'readiness.json')
    if record['status'] != 'completed_comparisons' or record['tasks'] != [task]:
        raise AssertionError('Comparison completion record is missing or incomplete')
    if record['source_code'] != expected_sources:
        raise AssertionError('Executed driver sources differ from the passed integration test')
    for name, digest in record['artifact_hashes'].items():
        if sha256(directory / name) != digest:
            raise AssertionError(f'Output hash mismatch: {name}')
    import numpy as np
    import pandas as pd
    manifest = read_json(INPUTS / 'manifest.json')
    if task == 'sampled':
        if record['checks']['sampled']['cases'] != sum(manifest['sampled']['cases_by_split'].values()):
            raise AssertionError('Sampled run did not cover every approved case')
        execution = read_json(directory / 'sampled/execution.json')
        if execution['miniature_training_graph']:
            raise AssertionError('Full sampled comparison used a smoke graph')
        expected = pd.read_parquet(INPUTS / 'sampled_cases.parquet').set_index('attempt_id')
        for path in (directory / 'sampled').rglob('*paired_ranks.parquet'):
            actual = pd.read_parquet(path)
            np.testing.assert_array_equal(actual.candidate_row, expected.loc[actual.attempt_id].candidate_row)
            rank_columns = [c for c in actual if c.endswith('_rank')]
            if not actual[rank_columns].apply(lambda c: c.between(1, 1000).all()).all():
                raise AssertionError('Invalid sampled ranks')
    else:
        for phase in ['development', 'later_retrospective']:
            base = directory / 'full_catalog' / phase
            execution = read_json(base / 'execution.json')
            expected_n = (manifest['development_selection']['users'] if phase == 'development'
                          else next(r['users'] for r in manifest['full_catalog'] if r['name'] == phase))
            if execution['miniature_training_graph'] or execution['evaluated_users'] != expected_n:
                raise AssertionError('Full-catalog graph or evaluation size is incomplete')
            users = np.load(base / 'user_ids.npy')
            for path in base.rglob('*_per_user.parquet'):
                actual = pd.read_parquet(path)
                np.testing.assert_array_equal(actual.user_idx, users)
                values = actual[['precision', 'recall', 'ndcg', 'mAP', 'MRR', 'HR']].to_numpy()
                if not np.isfinite(values).all() or np.any((values < 0) | (values > 1 + 1e-12)):
                    raise AssertionError('Invalid full-catalog metric values')
    return {'status': 'passed', 'readiness_sha256': sha256(directory / 'readiness.json'),
            'duration_seconds': record['duration_seconds'], 'artifact_count': len(record['artifact_hashes'])}
