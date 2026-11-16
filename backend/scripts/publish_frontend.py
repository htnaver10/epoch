"""Copy a verified frontend build into the local runnable site without deleting files."""
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / "frontend/dist").resolve()
destination = (ROOT / "ready-to-run").resolve()
assert source.is_relative_to(ROOT) and destination.is_relative_to(ROOT)
if not (source / "index.html").is_file():
    raise SystemExit("Run npm.cmd run build in frontend first.")
# Vite can preserve CRLF from the input HTML while injecting LF asset tags.
# Normalize that generated file so Windows publication does not create a mixed-EOL diff.
index = source / "index.html"
index.write_bytes(index.read_bytes().replace(b"\r", b""))
files = [p for p in source.rglob("*") if p.is_file()]
changed = 0
for path in files:
    copied = destination / path.relative_to(source)
    with path.open("rb") as stream:
        expected = hashlib.file_digest(stream, "sha256").digest()
    actual = None
    if copied.is_file():
        with copied.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").digest()
    if actual != expected:
        copied.parent.mkdir(parents=True, exist_ok=True)
        temporary = copied.with_name(copied.name + ".publish-partial")
        shutil.copy2(path, temporary)
        with temporary.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").digest() == expected
        try:
            temporary.replace(copied)
        except PermissionError:
            # Windows readers can prevent replacing a currently served file.
            # Preserve the prior publisher's overwrite behavior in that case.
            shutil.copy2(temporary, copied)
            with copied.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").digest() == expected
            temporary.unlink()
        changed += 1
print(f"Published {changed} changed files and verified {len(files)} files in {destination}")
