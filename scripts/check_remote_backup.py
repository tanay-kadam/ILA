"""Read-only remote comparison; never deletes or overwrites research artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / '.transfer' / 'ssh_audit_20261020'
out.mkdir(parents=True, exist_ok=True)
tree = ast.parse((root / 'scripts/sync_remote_artifacts.py').read_text())
remote = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in n.targets))
ssh = [r'C:\Windows\System32\OpenSSH\ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'seeklab', 'python3 -']
items = json.loads(subprocess.run(ssh, input=remote, text=True, capture_output=True, check=True).stdout)
(out / 'remote_manifest.json').write_text(json.dumps(items, indent=2))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()
counts = {'matching_local': 0, 'matching_staged_remote': 0}
unbacked = []
for item in items:
    candidates = [('matching_local', root / item['path']), ('matching_staged_remote', root / '.transfer/ssh_sync_20261010/incoming' / item['path'])]
    for label, path in candidates:
        try:
            valid = path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        except OSError:
            valid = False
        if valid:
            counts[label] += 1
            break
    else:
        unbacked.append(item)
report = {'remote_files': len(items), 'remote_bytes': sum(x['bytes'] for x in items), **counts, 'unbacked': unbacked, 'scope': 'Nonweight selected project files only; weights, environments and other excluded files are not verified backups.'}
(out / 'comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps({**report, 'unbacked': len(unbacked)}), flush=True)
