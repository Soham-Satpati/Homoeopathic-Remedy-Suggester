"""Clean Hering's Guiding Symptoms volumes and install them as shipped HoRUS books.

    python offline/scripts/build_guiding_books.py RAW_FOLDER [REPO_ROOT]

1. ``clean_guiding_symptoms.py`` cleans every volume in RAW_FOLDER (report written next to the books)
2. each cleaned volume is written to REPO_ROOT, compact
3. its symptom co-occurrence table is built (``*_conditional_probabilities.csv.gz``)
4. books.json lists the volumes (other shipped books are kept as they are)
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from clean_guiding_symptoms import FILE, VOLUMES, main as clean  # noqa: E402
from horus.clusters import build_cooccurrence  # noqa: E402

ROMAN = {v: i + 1 for i, v in enumerate(VOLUMES)}


def book_entry(vol: str, data: dict) -> dict:
    names = sorted(data)
    n = sum(len(i) for chs in data.values() for i in chs.values())
    first, last = (x.title() for x in (names[0], names[-1]))
    stem = FILE.format(vol)[:-5]
    return {
        "id": f"guiding_symptoms_{ROMAN[vol]}",
        "title": f"Hering — Guiding Symptoms, vol. {vol}",
        "json": FILE.format(vol),
        "cooccurrence": f"{stem}_conditional_probabilities.csv.gz",
        "exclude_chapters": ["relationship", "relations"],
        # clinic-added copies of the uncleaned scan that this cleaned volume replaces
        "replaces": [stem, stem.replace("_", " ")],
        "description": f"Hering's Guiding Symptoms of our Materia Medica, volume {vol}: {len(names)} remedies "
                       f"({first} to {last}), {n:,} symptoms. Cleaned from the scan — see "
                       f"guiding_symptoms_cleaning_report.md.",
    }


def run(raw: Path, root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean(raw, tmp)
        for name in ("guiding_symptoms_cleaning_report.md", "guiding_symptoms_cleaning_report.json"):
            (root / name).write_bytes((tmp / name).read_bytes())
        entries = []
        for vol in VOLUMES:
            src = tmp / FILE.format(vol)
            if not src.exists():
                continue
            data = json.loads(src.read_text(encoding="utf-8"))
            (root / FILE.format(vol)).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                                                 encoding="utf-8")
            e = book_entry(vol, data)
            t = time.time()
            cp = build_cooccurrence(data, exclude_chapters=e["exclude_chapters"])
            cp.to_csv(root / e["cooccurrence"], index=False, compression="gzip")
            print(f"vol. {vol}: {len(cp):,} co-occurrence rows in {time.time() - t:.0f}s")
            entries.append(e)
    manifest = root / "books.json"
    books = json.loads(manifest.read_text(encoding="utf-8"))["books"]
    books = [b for b in books if not b["id"].startswith("guiding_symptoms_")] + entries
    manifest.write_text(json.dumps({"books": books}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1])
