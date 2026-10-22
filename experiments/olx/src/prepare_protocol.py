"""Prepare local output folders and source identity for input validation."""
import json
import shutil
from rebuild_features import ROOT, REV, SOURCE
from artifact_hash import sha256

def main():
    for name in ['protocol_checks/evaluation_inputs', 'contact_analysis', 'driver_integration',
                 'normalization_experiment', 'downstream_comparisons']:
        (REV / name).mkdir(parents=True, exist_ok=True)
    for source, target in [
        (ROOT / 'configs/comparison_contract.json', REV / 'protocol_checks/comparison_contract.json'),
        (ROOT / 'configs/normalization.json', REV / 'normalization_experiment/analysis_specification.json')]:
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError(f'Existing configuration differs: {target.name}')
        else:
            shutil.copyfile(source, target)
    identity = REV / 'protocol_checks/legacy_protocol_audit.json'
    record = {'source_sha256': sha256(SOURCE)}
    if identity.exists() and json.loads(identity.read_text()) != record:
        raise ValueError('Existing source identity differs')
    identity.write_text(json.dumps(record, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
