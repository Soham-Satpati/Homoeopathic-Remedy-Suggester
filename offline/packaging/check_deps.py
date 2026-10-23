"""Fail if any shipped .exe/.dll imports a DLL that is neither shipped nor part of Windows itself.

    python check_deps.py <folder> [<folder> …]

The build machines have Visual Studio installed, so a missing runtime DLL would not show up in
tests there but would stop the app (or the AI engine) on a clean clinic PC. This reads each PE
file's import table and checks every imported DLL is either next to it, somewhere in the shipped
folders, or a DLL that ships with every Windows 10/11 installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pefile

# DLLs present on every Windows 10 (1809+) / 11 installation.
WINDOWS = {
    "kernel32.dll", "user32.dll", "gdi32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll",
    "ws2_32.dll", "bcrypt.dll", "ntdll.dll", "crypt32.dll", "secur32.dll", "shlwapi.dll", "comdlg32.dll",
    "comctl32.dll", "version.dll", "winmm.dll", "imm32.dll", "dwmapi.dll", "uxtheme.dll", "setupapi.dll",
    "cfgmgr32.dll", "iphlpapi.dll", "netapi32.dll", "userenv.dll", "wtsapi32.dll", "mpr.dll", "dbghelp.dll",
    "msvcrt.dll", "ucrtbase.dll", "psapi.dll", "powrprof.dll", "rpcrt4.dll", "mswsock.dll", "dnsapi.dll",
    "winhttp.dll", "wininet.dll", "d3d11.dll", "d3d12.dll", "dxgi.dll", "d3d9.dll", "dxva2.dll", "mfplat.dll",
    "mf.dll", "mfreadwrite.dll", "mfuuid.dll", "evr.dll", "opengl32.dll", "glu32.dll", "winspool.drv",
    "authz.dll", "dwrite.dll", "d2d1.dll", "propsys.dll", "shcore.dll", "wintrust.dll", "ncrypt.dll",
    "normaliz.dll", "hid.dll", "avrt.dll", "mmdevapi.dll", "ksuser.dll", "wldap32.dll", "dbgeng.dll",
    "esent.dll", "mscms.dll", "credui.dll", "netutils.dll", "srvcli.dll", "wkscli.dll", "pdh.dll",
    "dwmapi.dll", "wevtapi.dll", "msimg32.dll", "usp10.dll", "virtdisk.dll", "fltlib.dll", "msacm32.dll",
    "dsound.dll", "comsvcs.dll", "mfcore.dll", "mfsensorgroup.dll", "d3dcompiler_47.dll", "api-ms-win-core-synch-l1-2-0.dll",
    "nvcuda.dll", "opencl.dll",          # optional GPU paths, loaded only if present
}


def check(folders: list[Path]) -> list[str]:
    shipped = {p.name.lower() for f in folders for p in f.rglob("*") if p.suffix.lower() in (".dll", ".pyd", ".exe")}
    problems = []
    for f in folders:
        for pe_path in f.rglob("*"):
            if pe_path.suffix.lower() not in (".exe", ".dll", ".pyd"):
                continue
            try:
                pe = pefile.PE(str(pe_path), fast_load=True)
                pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                                                       pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
            except pefile.PEFormatError:
                continue
            delayed = {e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])}
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                dll = entry.dll.decode().lower()
                if dll in shipped or dll in WINDOWS or dll.startswith(("api-ms-win-", "ext-ms-")):
                    continue
                if dll in delayed:
                    continue
                problems.append(f"{pe_path.relative_to(f.parent)} needs {dll}")
    return sorted(set(problems))


if __name__ == "__main__":
    folders = [Path(a) for a in sys.argv[1:]]
    issues = check(folders)
    for i in issues:
        print(f"::error title=missing DLL::{i}")
    print(f"::notice title=DLL check::{len(issues)} unresolved imports across "
          f"{sum(1 for f in folders for _ in f.rglob('*.dll'))} DLLs")
    sys.exit(1 if issues else 0)
