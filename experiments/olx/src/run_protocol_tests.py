"""Save a machine-readable execution record for the protocol regression suite."""
import io
import json
import unittest
from datetime import datetime,timezone
from pathlib import Path

from artifact_hash import sha256

REV=Path(__file__).resolve().parents[1]


def main():
    tests=unittest.defaultTestLoader.discover(str(REV/'tests'))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    record={'successful':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_primitives_sha256':sha256(REV/'src/protocol_primitives.py'),
        'test_file_sha256':{p.name:sha256(p) for p in (REV/'tests').glob('test_*.py')},
        'output':stream.getvalue()}
    (REV/'protocol_checks/unit_test_results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=='__main__':
    main()
