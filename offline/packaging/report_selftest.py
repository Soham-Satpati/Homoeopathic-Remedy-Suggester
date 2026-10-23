"""Turn a selftest.json into GitHub annotations (one line per check)."""

import json
import sys
from pathlib import Path

path, label = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else ""
if not path.exists():
    print(f"::error title=selftest {label}::no selftest.json written")
    sys.exit(0)
d = json.loads(path.read_text(encoding="utf-8"))
for name, c in d.get("checks", {}).items():
    if c.get("ok"):
        print(f"::notice title={label} {name}::OK in {c.get('seconds')}s — {json.dumps(c.get('info'))[:400]}")
    else:
        tb = (c.get("traceback") or "").strip().splitlines()[-3:]
        print(f"::error title={label} {name}::{c.get('error')} | {' / '.join(tb)[:600]}")
if d.get("error"):
    print(f"::error title={label} selftest::{d['error']}")
print(f"::notice title={label} result::ok={d.get('ok')} failed={d.get('failed')} ai={d.get('ai')}")
