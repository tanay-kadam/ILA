"""Checksum-based artifact transfer without environments, caches, or local overwrites."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SKIP_DIRS = {".git", ".venv", ".venv-vllm", "venv", "env", "__pycache__", ".pytest_cache", ".cache", "node_modules", ".transfer", ".transfer_export"}
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"}


def portable_name(name):
    if os.name != "nt":
        return name
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)}
    parts = []
    for part in PurePosixPath(name).parts:
        encoded = "".join(f"%{ord(c):02X}" if ord(c) < 32 or c in '<>:"\\|?*%' else c for c in part)
        while encoded.endswith((".", " ")):
            encoded = encoded[:-1] + ("%2E" if encoded[-1] == "." else "%20")
        if part.split(".")[0].upper() in reserved:
            encoded = f"%{ord(encoded[0]):02X}" + encoded[1:]
        parts.append(encoded)
    return "/".join(parts)


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entries(root, include_weights=False):
    out = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith((".venv", ".tmp_pytest")))
        for name in sorted(files):
            p = Path(directory) / name
            if p.is_symlink():
                continue
            is_weight = p.suffix.lower() in WEIGHTS
            out[p.relative_to(root).as_posix()] = {"size": p.stat().st_size, "weight": is_weight,
                "sha256": checksum(p) if include_weights or not is_weight else None}
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("inventory", "prepare", "apply"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--local-manifest", type=Path)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--include-weights", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.mode == "inventory":
        data = entries(root, args.include_weights)
        write_json(args.manifest, {"root": str(root), "files": data})
        print(json.dumps({"files": len(data), "bytes": sum(v["size"] for v in data.values())})); return
    if args.mode == "prepare":
        local = json.loads(args.local_manifest.read_text())["files"]
        source = entries(root, args.include_weights)
        selected, weights, conflicts = {}, {}, []
        for name, info in source.items():
            if info["weight"] and not args.include_weights:
                weights[name] = info; continue
            if local.get(name, {}).get("sha256") != info["sha256"]:
                selected[name] = info
                if name in local:
                    conflicts.append(name)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        if args.archive.exists():
            raise SystemExit("archive already exists; preserve it and choose a new name")
        with tarfile.open(args.archive, "w:gz", compresslevel=3) as archive:
            for name in selected:
                archive.add(root / name, arcname=name, recursive=False)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
                  "selected": selected, "conflicts": conflicts, "excluded_weights": weights,
                  "excluded_directories": sorted(SKIP_DIRS), "archive_sha256": checksum(args.archive),
                  "archive_bytes": args.archive.stat().st_size,
                  "selected_bytes": sum(v["size"] for v in selected.values()),
                  "excluded_weight_bytes": sum(v["size"] for v in weights.values())}
        write_json(args.manifest, report)
        print(json.dumps({k:v for k,v in report.items() if k not in ("selected", "excluded_weights", "conflicts")}) )
        return
    report = json.loads(args.manifest.read_text())
    if checksum(args.archive) != report["archive_sha256"]:
        raise SystemExit("archive checksum mismatch; nothing extracted")
    needed = report["selected_bytes"]
    if shutil.disk_usage(root).free < needed + 5 * 1024**3:
        raise SystemExit("insufficient free space with 5 GiB reserve; nothing extracted")
    copied, preserved, verified, renamed = [], [], [], {}
    with tarfile.open(args.archive, "r:gz") as archive:
        seen = set()
        for member in archive:
            name = member.name
            posix = PurePosixPath(name)
            if posix.is_absolute() or ".." in posix.parts or not member.isfile() or name not in report["selected"] or name in seen:
                raise SystemExit(f"unsafe/unexpected archive member: {name}")
            seen.add(name)
            expected = report["selected"][name]
            local_name = portable_name(name)
            if local_name != name:
                renamed[name] = local_name
            dst = (root / local_name).resolve()
            if not dst.is_relative_to(root):
                raise SystemExit("archive target escapes repository")
            if dst.exists() and checksum(dst) == expected["sha256"]:
                verified.append(name); continue
            if dst.exists():
                dst = root / ".transfer" / "remote_conflicts" / local_name
                preserved.append(name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            temp = dst.with_name(dst.name + ".transfer-part")
            h = hashlib.sha256()
            with archive.extractfile(member) as stream, open(temp, "wb") as target:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block); target.write(block)
            if h.hexdigest() != expected["sha256"] or temp.stat().st_size != expected["size"]:
                raise SystemExit(f"member checksum mismatch: {name}; partial file retained")
            temp.replace(dst)
            if name not in preserved:
                copied.append(name)
        if seen != set(report["selected"]):
            raise SystemExit("archive missing selected entries")
    summary = {"copied": copied, "already_verified": verified, "local_conflicts_preserved": preserved, "renamed_for_windows": renamed,
               "excluded_weights": report["excluded_weights"], "status": "NON_WEIGHT_TRANSFER_VERIFIED" if report["excluded_weights"] else "TRANSFER_VERIFIED"}
    write_json(root / ".transfer" / "transfer_report.json", summary)
    print(json.dumps({"status": summary["status"], "copied": len(copied), "already_verified": len(verified), "local_conflicts": preserved,
                      "weight_files_remaining": len(report["excluded_weights"]), "windows_filename_mappings": renamed}))


if __name__ == "__main__":
    main()
