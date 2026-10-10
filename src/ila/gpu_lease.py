"""Cross-branch physical-GPU leases; never preempt foreign workloads."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

ALLOWED_GPUS={0,1,2,3,4,7}


@contextmanager
def gpu_lease(root, gpu):
    if gpu not in ALLOWED_GPUS:
        raise ValueError("GPUs 5 and 6 are prohibited")
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/f"gpu{gpu}"
    try:target.mkdir()
    except FileExistsError:
        yield False;return
    owner=target/"owner.json"
    owner.write_text(json.dumps({"pid":os.getpid(),"physical_gpu":gpu}))
    try:yield True
    finally:
        if owner.exists() and json.loads(owner.read_text())["pid"]==os.getpid():
            owner.unlink();target.rmdir()
