"""Verify recorded training-budget diagnostics without fitting models."""
from hashlib import sha256
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    protocol_path = ROOT / 'configs/cosmetics_training_budget.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    result = json.loads((ROOT / 'results/cosmetics/training_budget_diagnostic.json').read_text(encoding='utf-8'))
    assert sha256(protocol_path.read_bytes()).hexdigest() == result['protocol_sha256']
    assert result['training_users'] == 50000
    assert result['december_accessed'] is False and result['ranking_rerun'] is False
    assert len(protocol['inputs']) == 4
    for model in result['models'].values():
        assert len(model['loss_curve']) == 32
        assert model['loss_curve'][-1] < model['loss_curve'][7]
        assert model['no_improvement_count'] < model['patience']
        assert -model['final_loss_change'] > model['tolerance']
        for phase, count in [('calibration', 1739), ('development', 2000)]:
            assert model['assessment'][phase]['users'] == count
    a = result['models']['actual']['assessment']['development']
    assert a['extended_32_epochs']['auc'] < a['original_eight_epochs']['auc']
    print('Verified 8-to-32 epoch diagnostic; December ranking records remain separate.')

if __name__ == '__main__':
    main()
