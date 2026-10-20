import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def test_restore_checks_hashes_and_reconstructs_duplicates(tmp_path):
    backup = tmp_path / 'backup'
    backup.mkdir()
    payload = b'synthetic adapter bytes'
    sha = hashlib.sha256(payload).hexdigest()
    obj = backup / 'adapter_objects' / sha / 'adapter_model.safetensors'
    obj.parent.mkdir(parents=True)
    obj.write_bytes(payload)
    adapter = {'path':'results/a/adapter_model.safetensors','bytes':len(payload),'sha256':sha}
    duplicate = {**adapter,'path':'results/b/adapter_model.safetensors'}
    (backup/'adapter_plan.json').write_text(json.dumps({'unique_adapter_files':[adapter],'adapter_files':[adapter,duplicate]}))
    evidence = tmp_path / 'config.json'
    evidence.write_bytes(b'{"seed":0}')
    item = {'path':'configs/config.json','bytes':evidence.stat().st_size,'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
    (backup/'evidence_manifest.json').write_text(json.dumps([item]))
    with tarfile.open(backup/'essential_evidence.tar.gz','w:gz') as tar:
        tar.add(evidence,arcname=item['path'])
    script = Path(__file__).resolve().parents[1]/'scripts/restore_hf_backup.py'
    dest = tmp_path/'restored'
    command = [sys.executable,str(script),'--backup',str(backup),'--destination',str(dest)]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert (dest/adapter['path']).read_bytes() == payload
    assert (dest/duplicate['path']).read_bytes() == payload
    assert (dest/item['path']).read_bytes() == evidence.read_bytes()
    assert subprocess.run(command,capture_output=True).returncode != 0
