"""Read-only input gate for subsequent retrospective comparison drivers."""
import json
from pathlib import Path

from artifact_hash import sha256
from protocol_primitives import require_component_available

REV=Path(__file__).resolve().parents[1]


def load_approved_inputs(task,component_selection_end=None,is_learned_bim=False,
                         claim_fresh_holdout=False,claim_verified_active_catalog=False):
    if claim_fresh_holdout or claim_verified_active_catalog:
        raise ValueError('Neither a fresh holdout nor verified active listings is established')
    checks=REV/'protocol_checks'
    verified=json.loads((checks/'verification.json').read_text(encoding='utf-8'))
    tests=json.loads((checks/'unit_test_results.json').read_text(encoding='utf-8'))
    assert verified['status']=='passed_for_retrospective_reanalysis_only'
    assert tests['successful']
    assert tests['protocol_primitives_sha256']==sha256(REV/'src/protocol_primitives.py')
    manifest_path=checks/'evaluation_inputs/manifest.json'
    assert sha256(manifest_path)==verified['manifest_sha256']
    assert sha256(checks/'comparison_contract.json')==verified['comparison_contract_sha256']
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for name,record in manifest['files'].items():
        assert sha256(manifest_path.parent/name)==record['sha256'],name
    if task=='sampled_test':
        start=manifest['sampled']['validation_end_exclusive']
    elif task=='full_catalog_later':
        start=next(r['training_end_exclusive'] for r in manifest['full_catalog']
                   if r['name']=='later_retrospective')
    else:
        raise ValueError('Only documented later evaluation tasks are approved')
    if is_learned_bim and component_selection_end is None:
        raise ValueError('A learned BIM comparison must declare its selection boundary')
    if component_selection_end is not None:
        require_component_available(component_selection_end,start)
    return manifest


if __name__=='__main__':
    load_approved_inputs('sampled_test',1582493743,is_learned_bim=True)
    load_approved_inputs('full_catalog_later')
    try:
        load_approved_inputs('full_catalog_later',1582493743,is_learned_bim=True)
    except ValueError:
        print('Current BIM checkpoint correctly rejected for earlier full-catalog start')
    else:
        raise AssertionError('Component time gate failed')
    print('Read-only evaluation input gates passed; no comparisons executed')
