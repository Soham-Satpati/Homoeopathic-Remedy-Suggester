"""Pack HoRUS.app into a disk image (drag HoRUS to Applications).

    python offline/packaging/build_dmg.py --app dist/HoRUS.app --out installer [--version 1.5.0]
                                          [--arch arm64] [--sign IDENTITY]

Writes ``<out>/HoRUS-<version>-macOS-<arch>.dmg``. The bundled AI models make the image larger than
GitHub's 2 GB per-file limit, so when it is bigger it is split into ``.dmg.part01``, ``.part02``…
(joined with ``cat``; see the user README) and the whole image is removed. ``SHA256SUMS.txt``
lists the joined image and every part.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

OFFLINE = Path(__file__).resolve().parents[1]
REPO = OFFLINE.parent
PART_SIZE = 1_900_000_000          # under GitHub's 2 GiB release-asset limit


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_image(app: Path, dmg: Path, version: str) -> None:
    stage = REPO / "build" / "dmg"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    # ditto keeps symlinks, permissions and code signatures intact
    subprocess.run(["ditto", str(app), str(stage / app.name)], check=True)
    os.symlink("/Applications", stage / "Applications")
    shutil.copy2(OFFLINE / "packaging" / "README.txt", stage / "Read me.txt")
    dmg.unlink(missing_ok=True)
    subprocess.run(["hdiutil", "create", "-volname", f"HoRUS {version}", "-srcfolder", str(stage),
                    "-fs", "HFS+", "-format", "ULFO", "-ov", str(dmg)], check=True)
    shutil.rmtree(stage)


def split(dmg: Path) -> list[Path]:
    parts = []
    with open(dmg, "rb") as f:
        i = 1
        while True:
            chunk = f.read(PART_SIZE)
            if not chunk:
                break
            part = dmg.with_name(f"{dmg.name}.part{i:02d}")
            part.write_bytes(chunk)
            parts.append(part)
            i += 1
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=REPO / "dist" / "HoRUS.app")
    ap.add_argument("--out", type=Path, default=REPO / "installer")
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--arch", default=os.uname().machine)
    ap.add_argument("--sign", default=os.environ.get("HORUS_SIGN_IDENTITY", "-"))
    ap.add_argument("--no-split", action="store_true", help="keep one image (for local use)")
    args = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("Build the disk image on a Mac.")

    args.out.mkdir(parents=True, exist_ok=True)
    dmg = args.out / f"HoRUS-{args.version}-macOS-{args.arch}.dmg"
    build_image(args.app, dmg, args.version)
    if args.sign != "-":
        subprocess.run(["codesign", "--force", "--timestamp", "--sign", args.sign, str(dmg)], check=True)

    sums = {dmg.name: sha256(dmg)}
    size = dmg.stat().st_size
    if size > PART_SIZE and not args.no_split:
        parts = split(dmg)
        sums.update({p.name: sha256(p) for p in parts})
        h = hashlib.sha256()                                      # the parts must join back exactly
        for p in parts:
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
        if h.hexdigest() != sums[dmg.name]:
            sys.exit("Split parts do not join back into the image")
        dmg.unlink()
        print(f"Disk image {size / 2**30:.2f} GB split into {len(parts)} parts")
    (args.out / "SHA256SUMS.txt").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    for p in sorted(args.out.iterdir()):
        print(f"{p.name}  {p.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
