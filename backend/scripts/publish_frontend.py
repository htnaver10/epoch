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
shutil.copytree(source, destination, dirs_exist_ok=True)
files = [p for p in source.rglob("*") if p.is_file()]
for path in files:
    copied = destination / path.relative_to(source)
    with path.open("rb") as a, copied.open("rb") as b:
        assert hashlib.file_digest(a, "sha256").digest() == hashlib.file_digest(b, "sha256").digest()
print(f"Published and verified {len(files)} files in {destination}")
