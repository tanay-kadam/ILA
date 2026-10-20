"""Server-side deduplicated private archive. NEVER deletes source files."""
import hashlib
import json
import re
import tarfile
import time
from pathlib import Path
from huggingface_hub import HfApi

ROOT=Path('/home/tkadam/ILA')
OUT=Path('/home/tkadam/ila_backup_20261020')
REPO='tk-8503/ila-private-backup'
assert str(ROOT.resolve()) == '/home/tkadam/ILA'
for _ in range(240):
    if (OUT/'adapter_plan.json').exists():
        break
    time.sleep(5)
else:
    raise RuntimeError('Checksum inventory not ready; no upload started')
plan=json.loads((OUT/'adapter_plan.json').read_text())
api=HfApi()
assert api.whoami()['name']=='tk-8503'
selected=[]
excluded=[]
weight_ext={'.safetensors','.bin','.pt','.pth','.ckpt','.gguf','.onnx','.pkl','.pickle','.pyc'}
for top in ['docs','scripts','src','configs','tests','data','results','README.md','requirements.txt','fix_cuda.sh']:
    p=ROOT/top
    for f in ([p] if p.is_file() else p.rglob('*')):
        if not f.is_file() or f.is_symlink():
            continue
        rel=f.relative_to(ROOT)
        if any(x in ['__pycache__','.git','.venv','.venv-vllm','.cache'] for x in rel.parts) or f.suffix.lower() in weight_ext or f.name.endswith(('.tar.gz','.zip')):
            excluded.append(str(rel))
            continue
        if f.name.startswith('.env') or f.suffix in ['.pem','.key'] or 'token' == f.name.lower():
            raise RuntimeError('Potential credential file; manual review required')
        selected.append(f)
total=plan['unique_adapter_bytes']+sum(p.stat().st_size for p in selected)
(OUT/'selection.json').write_text(json.dumps({'conservative_upload_bytes':total,'nonweight_files':len(selected),'excluded_paths':excluded,'scope':'All unique adapter_model files plus selected source/data/results. Other weights, environments, caches, logs, and transfer archives are not backed up.'},indent=2))
if total>95_000_000_000:
    raise RuntimeError('Backup exceeds conservative free private quota; no uploads or deletions performed')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

archive=OUT/'essential_evidence.tar.gz'
evidence=[]
secret=re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}')
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
    for p in selected:
        if p.stat().st_size<4_000_000 and secret.search(p.read_bytes()):
            raise RuntimeError('Credential pattern found in '+str(p.relative_to(ROOT)))
        evidence.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)})
        tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
(OUT/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
print('Archive prepared; planned bytes',total,flush=True)
api.create_repo(repo_id=REPO,repo_type='dataset',private=True,exist_ok=True)
assert api.repo_info(REPO,repo_type='dataset').private is True, 'Repository is not private; refusing upload'
uploads=[{'local':str(ROOT/x['path']),'path':'adapter_objects/'+x['sha256']+'/'+Path(x['path']).name,'bytes':x['bytes'],'sha256':x['sha256']} for x in plan['unique_adapter_files']]
uploads.append({'local':str(archive),'path':'essential_evidence.tar.gz','bytes':archive.stat().st_size,'sha256':digest(archive)})
for i,item in enumerate(uploads):
    assert api.repo_info(REPO,repo_type='dataset').private is True
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=item['local'],path_in_repo=item['path'],repo_id=REPO,repo_type='dataset',commit_message='Archive verified research object')
            break
        except Exception:
            if attempt==2:
                raise
            time.sleep(10)
    print('Uploaded',i+1,'/',len(uploads),flush=True)
for name in ['adapter_plan.json','evidence_manifest.json','selection.json']:
    api.upload_file(path_or_fileobj=str(OUT/name),path_in_repo=name,repo_id=REPO,repo_type='dataset')
info=api.repo_info(REPO,repo_type='dataset')
assert info.private is True
tree={f.path:f for f in api.list_repo_tree(REPO,repo_type='dataset',recursive=True,revision=info.sha) if hasattr(f,'size')}
for item in uploads:
    f=tree[item['path']]
    assert f.size==item['bytes']
    lfs=f.lfs
    sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
    assert sha==item['sha256'], 'Remote SHA256 mismatch or unavailable: '+item['path']
report={'status':'UPLOAD_OBJECT_HASHES_VERIFIED','repo':REPO,'private':True,'commit':info.sha,'verified_objects':len(uploads),'unique_adapter_bytes':plan['unique_adapter_bytes'],'adapter_paths_recoverable':len(plan['adapter_files']),'deletion_performed':False,'remaining_deletion_gate':'Review excluded files and merged-weight recoverability; no automatic deletion.'}
(OUT/'UPLOAD_VERIFIED.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
