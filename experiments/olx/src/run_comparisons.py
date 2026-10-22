"""Run both ranking tasks after a matching shared-protocol smoke test."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from ranking_driver_core import REV
from run_repaired_ranking import source_hashes
from run_downstream_batch import verify_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readiness-record', type=Path, required=True)
    args = parser.parse_args()
    output = REV / 'downstream_comparisons/run_2026-10-20'
    output.mkdir(parents=True, exist_ok=False)
    status = {'stage': 'running', 'completed_tasks': []}
    path = output / 'batch_status.json'
    path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    for task, folder in [('sampled', 'sampled_run'), ('full', 'full_catalog_run')]:
        target = output / folder
        subprocess.run([sys.executable, str(REV / 'src/run_repaired_ranking.py'),
                        '--mode', 'compare', '--task', task, '--output', str(target),
                        '--batch-size', '128', '--block-size', '200',
                        '--readiness-record', str(args.readiness_record.resolve())], check=True)
        status['completed_tasks'].append({'task': task, **verify_output(target, task, source_hashes())})
        path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    status['stage'] = 'completed'
    path.write_text(json.dumps(status, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
