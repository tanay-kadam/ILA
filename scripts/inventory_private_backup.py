"""Read-only remote inventory and credential-status check; never prints tokens."""
import json
from pathlib import Path
import subprocess

remote = r'''
import json, pathlib, collections, os
root=pathlib.Path('/home/tkadam/ILA')
assert str(root.resolve()) == '/home/tkadam/ILA'
sizes=collections.Counter()
adapters=[]
other_weights=[]
excluded=collections.Counter()
for p in root.rglob('*'):
 if not p.is_file() or p.is_symlink(): continue
 rel=p.relative_to(root)
 if any(x in ['.venv','.venv-vllm','__pycache__','.pytest_cache','.git'] for x in rel.parts):
  excluded['environments_caches_git']+=p.stat().st_size
  continue
 n=p.stat().st_size
 if p.name.startswith('adapter_model.'):
  adapters.append({'path':str(rel),'bytes':n})
 elif p.suffix in ['.safetensors','.bin','.pt','.pth','.ckpt','.gguf','.onnx']:
  other_weights.append({'path':str(rel),'bytes':n})
 else: sizes[rel.parts[0]]+=n
auth={'status':'not_checked'}
try:
 from huggingface_hub import HfApi, get_token
 token=get_token()
 if token:
  who=HfApi(token=token).whoami()
  auth={'status':'authenticated','username':who.get('name')}
 else: auth={'status':'login_required'}
except ImportError: auth={'status':'client_missing'}
except Exception as e: auth={'status':'check_failed','error_type':type(e).__name__}
print(json.dumps({'remote_root':str(root),'adapter_files':adapters,'adapter_bytes':sum(x['bytes'] for x in adapters),'other_weight_files':other_weights,'other_weight_bytes':sum(x['bytes'] for x in other_weights),'nonweight_bytes_by_root':dict(sizes),'excluded_bytes':dict(excluded),'huggingface_auth':auth}))
'''
ssh=[r'C:\Windows\System32\OpenSSH\ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=12','seeklab','python3 -']
result=subprocess.run(ssh,input=remote,text=True,capture_output=True,check=True)
data=json.loads(result.stdout)
out=Path(__file__).resolve().parents[1]/'.transfer/private_backup_20261020'
out.mkdir(parents=True,exist_ok=True)
(out/'inventory.json').write_text(json.dumps(data,indent=2))
print(json.dumps({k:v for k,v in data.items() if k not in ['adapter_files','other_weight_files']},indent=2))
print('Adapter file count:',len(data['adapter_files']))
print('Other weight file count:',len(data['other_weight_files']))
