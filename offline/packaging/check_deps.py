"""Fail if any binary in HoRUS.app links a library that is neither inside the bundle nor part of macOS.

    python check_deps.py dist/HoRUS.app [--arch arm64]

The build machines have Homebrew and Xcode installed, so a library picked up from /opt/homebrew or
/usr/local would not show up in tests there but would stop the app (or the AI engine) on a clinic
Mac. This reads each Mach-O file's load commands with ``otool`` and checks every linked library is
either a system library (/usr/lib, /System/Library) or resolves to a file inside the bundle
(@rpath, @loader_path, @executable_path). It also checks every binary contains the target
architecture, since one Intel-only library is enough to stop the app on Apple silicon.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/")
MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}
SKIP_SUFFIXES = {".gguf", ".bin", ".json", ".npz", ".csv", ".gz", ".pkl", ".onnx", ".txt", ".py", ".pyc", ".png"}


def is_macho(p: Path) -> bool:
    if p.suffix.lower() in SKIP_SUFFIXES or p.is_symlink() or not p.is_file():
        return False
    try:
        with open(p, "rb") as f:
            return f.read(4) in MACHO_MAGIC
    except OSError:
        return False


def otool(flag: str, p: Path) -> list[str]:
    return subprocess.run(["otool", flag, str(p)], capture_output=True, text=True).stdout.splitlines()


def linked(p: Path) -> list[str]:
    lines = otool("-L", p)
    # universal files print one "<file> (architecture x):" header per slice; skip every header line
    libs = [ln.strip().split(" (compatibility")[0] for ln in lines if ln.strip() and not ln.rstrip().endswith(":")]
    return sorted({lib for lib in libs if not lib.endswith("/" + p.name)})   # a dylib lists its own install name


def rpaths(p: Path) -> list[str]:
    out, lines = [], otool("-l", p)
    for i, ln in enumerate(lines):
        if ln.strip() == "cmd LC_RPATH":
            for nxt in lines[i + 1:i + 4]:
                nxt = nxt.strip()
                if nxt.startswith("path "):
                    out.append(nxt[5:].split(" (offset")[0])
    return out


def resolves(lib: str, binary: Path, main_exe_dir: Path) -> bool:
    def sub(s: str) -> str:
        return s.replace("@loader_path", str(binary.parent)).replace("@executable_path", str(main_exe_dir))
    if lib.startswith("@rpath/"):
        rest = lib[len("@rpath/"):]
        # the llama.cpp server runs as its own executable: its @executable_path is its own folder
        for exe_dir in {main_exe_dir, binary.parent}:
            for rp in rpaths(binary):
                cand = rp.replace("@loader_path", str(binary.parent)).replace("@executable_path", str(exe_dir))
                if Path(cand, rest).exists():
                    return True
        return False
    if lib.startswith(("@loader_path", "@executable_path")):
        return Path(os.path.normpath(sub(lib))).exists() or \
            Path(os.path.normpath(lib.replace("@executable_path", str(binary.parent)))).exists()
    return False


def archs(p: Path) -> set[str]:
    r = subprocess.run(["lipo", "-archs", str(p)], capture_output=True, text=True)
    return set(r.stdout.split())


def check(app: Path, arch: str) -> list[str]:
    main_exe_dir = app / "Contents" / "MacOS"
    problems = []
    for p in app.rglob("*"):
        if not is_macho(p):
            continue
        rel = p.relative_to(app.parent)
        a = archs(p)
        if a and arch not in a:
            problems.append(f"{rel} is built for {', '.join(sorted(a))}, not {arch}")
        for lib in linked(p):
            if lib.startswith(SYSTEM_PREFIXES):
                continue
            if lib.startswith("@") and resolves(lib, p, main_exe_dir):
                continue
            problems.append(f"{rel} needs {lib}")
    return sorted(set(problems))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("app", type=Path)
    ap.add_argument("--arch", default=os.uname().machine)
    args = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("Run on macOS (needs otool and lipo).")
    issues = check(args.app, args.arch)
    for i in issues:
        print(f"::error title=missing library::{i}")
    n = sum(1 for p in args.app.rglob("*") if is_macho(p))
    print(f"::notice title=library check::{len(issues)} problems across {n} binaries ({args.arch})")
    sys.exit(1 if issues else 0)
