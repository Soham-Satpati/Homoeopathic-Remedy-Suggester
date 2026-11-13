"""Re-hash the llama.cpp server files in HoRUS.app's model manifest after code signing.

    python refresh_manifest.py dist/HoRUS.app

codesign writes a signature into every Mach-O file, so the llama.cpp server and its libraries no
longer match the SHA-256 values fetch_models.py recorded. sign_app.sh runs this after signing
the binaries and before sealing the bundle, so the app's own integrity check (``--selftest``,
"models_intact") checks the signed files. Only entries under llama-server/ are updated: a
mismatch anywhere else is real corruption and is left for the check to catch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(app: Path) -> int:
    found = [p for p in (app / "Contents" / "Resources").rglob("manifest.json")
             if p.parent.name == "models" and not p.is_symlink()]
    if not found:
        print("No model manifest in the bundle; nothing to refresh")
        return 0
    mf = found[0]
    m = json.loads(mf.read_text())
    changed = 0
    for rel, digest in m["files"].items():
        if not rel.startswith("llama-server/"):
            continue
        new = sha256(mf.parent / rel)
        if new != digest:
            m["files"][rel] = new
            changed += 1
    if changed:
        mf.write_text(json.dumps(m, indent=2))
    print(f"Manifest: re-hashed {changed} signed llama-server files")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
