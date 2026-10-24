"""Verify the public allowlist and artifact checksums without private data."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'reproducibility/artifact_manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        path = ROOT / name
        assert path.suffix.lower() in {'.py', '.json', '.md', '.txt', ''}, name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=name)
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    private_extensions = {'.docx', '.doc', '.pdf', '.zip', '.parquet', '.npy', '.npz', '.pt', '.pth', '.pkl', '.pickle'}
    assert not [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts
                and p.suffix.lower() in private_extensions]
    print(f"Verified {len(manifest['files'])} public files; no document or row-level artifacts found.")

if __name__ == '__main__':
    main()
