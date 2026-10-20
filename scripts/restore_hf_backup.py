"""Verify and restore a downloaded deduplicated backup, without overwriting files."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_path(root, name):
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Unsafe archive path: ' + name)
    return p

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backup', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True,
                    help='New, nonexistent directory; no existing project is overwritten')
    args = ap.parse_args()
    backup, dest = args.backup.resolve(), args.destination.resolve()
    if dest.exists():
        raise SystemExit('Destination must not exist; restore into a new directory')
    plan = json.loads((backup / 'adapter_plan.json').read_text())
    evidence = json.loads((backup / 'evidence_manifest.json').read_text())
    expected = {x['path']: x for x in evidence}
    for x in plan['unique_adapter_files']:
        p = backup / 'adapter_objects' / x['sha256'] / Path(x['path']).name
        if p.stat().st_size != x['bytes'] or checksum(p) != x['sha256']:
            raise SystemExit('Adapter object verification failed: ' + x['path'])
    dest.mkdir(parents=True)
    with tarfile.open(backup / 'essential_evidence.tar.gz') as archive:
        seen = set()
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected archive entry: ' + member.name)
            p = safe_path(dest, member.name)
            p.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, p.open('xb') as output:
                shutil.copyfileobj(source, output)
            x = expected[member.name]
            if p.stat().st_size != x['bytes'] or checksum(p) != x['sha256']:
                raise ValueError('Evidence verification failed: ' + member.name)
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError('Missing evidence entries')
    canonical = {}
    for x in plan['adapter_files']:
        p = safe_path(dest, x['path'])
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise ValueError('Unexpected existing adapter: ' + x['path'])
        if x['sha256'] in canonical:
            p.hardlink_to(canonical[x['sha256']])
        else:
            source = backup / 'adapter_objects' / x['sha256'] / Path(x['path']).name
            shutil.copyfile(source, p)
            canonical[x['sha256']] = p
    print('Restored and checksum-verified', len(evidence), 'evidence files and',
          len(plan['adapter_files']), 'adapter paths to', dest)
    print('Duplicate adapters are hard-linked: treat restored weight files as immutable.')

if __name__ == '__main__':
    main()
