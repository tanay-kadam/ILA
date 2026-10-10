"""Fetch missing/different nonweight SSH artifacts without overwriting local work."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.transfer' / 'ssh_sync_20261010'
OUT.mkdir(parents=True, exist_ok=True)
SSH = [r'C:\Windows\System32\OpenSSH\ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'seeklab']
REMOTE = '''import pathlib,hashlib,json
r=pathlib.Path('/home/tkadam/ILA')
items=[]
for top in ['docs','scripts','src','configs','tests','results','data','logs','README.md','requirements.txt','fix_cuda.sh','.gitignore']:
 p=r/top
 for f in ([p] if p.is_file() else p.rglob('*')):
  if not f.is_file() or f.is_symlink(): continue
  rel=f.relative_to(r)
  if any(x in ['__pycache__','.venv','.git','.pytest_cache'] for x in rel.parts): continue
  if f.suffix.lower() in ['.safetensors','.bin','.pt','.pth','.ckpt','.pyc','.pkl','.pickle']: continue
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(1048576),b''): h.update(b)
  items.append({'path':rel.as_posix(),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(items))
'''
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

manifest_path = OUT / 'remote_manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else json.loads(subprocess.run(SSH + ['python3 -'], input=REMOTE, text=True, capture_output=True, check=True).stdout)
(OUT / 'remote_manifest.json').write_text(json.dumps(manifest, indent=2))
report_path = OUT / 'sync_report.json'
missing, different = [], []
if report_path.exists():
    old = json.loads(report_path.read_text())
    missing, different = old['missing'], old['different_preserved']
else:
    for item in manifest:
        p = ROOT / item['path']
        if not p.exists():
            missing.append(item)
        elif p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            different.append(item)
report = {'remote_files': len(manifest), 'missing': missing, 'different_preserved': different}
(OUT / 'sync_report.json').write_text(json.dumps(report, indent=2))
selected = missing + different
nonportable = [x for x in selected if any(ord(c) < 32 for c in x['path']) or any(c in x['path'] for c in '<>:"|?*')]
(OUT / 'nonportable_paths_skipped.json').write_text(json.dumps(nonportable, indent=2))
selected = [x for x in selected if x not in nonportable]
print(json.dumps({'remote_files':len(manifest),'missing':len(missing),'different':len(different),'transfer_bytes':sum(x['bytes'] for x in selected)}), flush=True)
if selected:
    stage = OUT / 'incoming'
    stage.mkdir(exist_ok=True)
    pending = []
    for item in selected:
        candidate = ROOT / item['path'] if item in missing else stage / item['path']
        if not candidate.exists() or digest(candidate) != item['sha256']:
            pending.append(item)
    pending.sort(key=lambda x: (x['path'].split('/')[0] not in ['scripts','src','configs','tests','docs'], x['path']))
    batches, batch, size = [], [], 0
    for item in pending:
        if batch and size + item['bytes'] > 8_000_000:
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += item['bytes']
    if batch:
        batches.append(batch)
    for number, batch in enumerate(batches):
        code = "import tarfile,pathlib; r=pathlib.Path('/home/tkadam/ILA'); paths=" + repr([x['path'] for x in batch]) + "; t=tarfile.open(fileobj=__import__('sys').stdout.buffer,mode='w|gz'); [(t.add(r/p,arcname=p,recursive=False)) for p in paths]; t.close()"
        archive = OUT / 'batch.tar.gz'
        for attempt in range(3):
            with archive.open('wb') as stream:
                result = subprocess.run(SSH + ['python3 -'], input=code.encode(), stdout=stream)
            if result.returncode == 0:
                break
            if attempt == 2:
                raise RuntimeError('SSH transfer failed after three attempts')
            time.sleep(2)
        with tarfile.open(archive) as t:
            t.extractall(stage, filter='data')
        for item in batch:
            p = stage / item['path']
            if digest(p) != item['sha256']:
                raise RuntimeError('Hash mismatch: ' + item['path'])
            if item in missing:
                dest = ROOT / item['path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise RuntimeError('Local file appeared during transfer: ' + str(dest))
                p.replace(dest)
        print('Verified batch', number + 1, '/', len(batches), flush=True)
print('Verified transfer complete; differing local files preserved; remote versions staged.', flush=True)
(OUT / 'COMPLETE.json').write_text(json.dumps({'verified_files':len(selected),'added_files':len(missing),'staged_differences':len(different)},indent=2))
