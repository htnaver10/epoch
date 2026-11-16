"""Restore exact .pt checkpoints that are stored as lossless Git LFS parts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path outside repository: {relative}")
    return path


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Missing or wrong-sized file: {path}. Run git lfs pull first.")
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != entry["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")


def restore(root, entry, verify_only=False):
    destination = contained(root, entry["path"])
    parts = entry.get("parts", [])
    if not parts:
        verify(destination, entry)
        return "verified"
    for part in parts:
        verify(contained(root, part["path"]), part)
    if sum(p["size_bytes"] for p in parts) != entry["size_bytes"]:
        raise ValueError(f"Incorrect combined part size for {destination}")
    if destination.exists():
        verify(destination, entry)
        return "verified existing"
    if verify_only:
        # Hash the concatenated content without writing a multi-GB temporary file.
        checksum = hashlib.sha256()
        for part in parts:
            with contained(root, part["path"]).open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    checksum.update(block)
        if checksum.hexdigest() != entry["sha256"]:
            raise ValueError(f"Combined checksum mismatch for {destination}")
        return "verified parts"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".partial", delete=False) as target:
            temporary = Path(target.name)
            checksum = hashlib.sha256()
            for part in parts:
                with contained(root, part["path"]).open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        target.write(block)
                        checksum.update(block)
        if temporary.stat().st_size != entry["size_bytes"] or checksum.hexdigest() != entry["sha256"]:
            raise ValueError(f"Reconstructed checksum mismatch for {destination}")
        # Rename refuses to overwrite existing files on Windows. Check explicitly
        # on other platforms as well so an existing checkpoint stays intact.
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        temporary = None
        return "restored"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only", action="store_true", help="Verify all direct files and combined parts without restoring files")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "model-checkpoints.json").read_text(encoding="utf-8"))
    if manifest["schema"] not in {"epoch_model_checkpoints_v1", "epoch_model_checkpoints_v2"}:
        raise ValueError("Unsupported checkpoint inventory")
    for entry in manifest["files"]:
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid checksum in checkpoint inventory")
        result = restore(root, entry, args.verify_only)
        print(f"{result}: {entry['path']}", flush=True)
    print(f"Verified {len(manifest['files'])} checkpoints.")


if __name__ == "__main__":
    main()
