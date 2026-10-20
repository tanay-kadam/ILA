"""Run on the server: create a checksum plan; do not upload or delete anything."""
import collections
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path('/home/tkadam/ILA')
OUT = Path('/home/tkadam/ila_backup_20261020')
assert str(ROOT.resolve()) == '/home/tkadam/ILA'
OUT.mkdir(exist_ok=True)
api = HfApi()
assert api.whoami()['name'] == 'tk-8503', 'Wrong Hugging Face account'
items, groups = [], {}
paths = sorted(ROOT.rglob('adapter_model.*'))
for i, path in enumerate(paths):
    if not path.is_file() or path.is_symlink() or '.venv' in path.parts:
        continue
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    item = {'path':str(path.relative_to(ROOT)), 'bytes':after.st_size, 'sha256':h.hexdigest()}
    items.append(item)
    groups.setdefault(h.hexdigest(), []).append(item)
    print('Hashed adapter', i + 1, '/', len(paths), flush=True)
unique = [rows[0] for rows in groups.values()]
plan = {'account':'tk-8503','proposed_repo':'tk-8503/ila-private-backup',
        'adapter_files':items,'unique_adapter_files':unique,
        'adapter_bytes':sum(x['bytes'] for x in items),
        'unique_adapter_bytes':sum(x['bytes'] for x in unique),
        'duplicate_adapter_files':len(items)-len(unique),
        'full_weights_not_selected':True,'deletion_authorized_by_this_script':False}
(OUT/'adapter_plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({k:v for k,v in plan.items() if not isinstance(v,list)}),flush=True)
