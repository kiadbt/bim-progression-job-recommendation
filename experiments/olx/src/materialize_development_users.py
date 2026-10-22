"""Fix the common development-user subset without scoring any model."""
import json
from pathlib import Path

import duckdb
import numpy as np

from artifact_hash import sha256

REV=Path(__file__).resolve().parents[1]


def main():
    inputs=REV/'protocol_checks/evaluation_inputs'
    path=inputs/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    output=inputs/'full_catalog_development_tuning_users.npy'
    if output.exists() or 'development_selection' in manifest:
        raise FileExistsError('Development subset already recorded')
    con=duckdb.connect()
    users=con.execute(f"SELECT DISTINCT user_idx FROM read_parquet('{(inputs/'full_catalog_development_relevance.parquet').as_posix()}') ORDER BY user_idx").fetchnumpy()['user_idx']
    con.close()
    selected=np.sort(np.random.default_rng(10).choice(users,min(30000,len(users)),replace=False))
    np.save(output,selected)
    manifest['development_selection']={'users':len(selected),'seed':10,'file':output.name,
        'scope':'Common tuning cohort for all full-catalog comparators; remaining development users are not an independent holdout'}
    manifest['files'][output.name]={'bytes':output.stat().st_size,'sha256':sha256(output)}
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['development_selection'],indent=2))


if __name__=='__main__':
    main()
