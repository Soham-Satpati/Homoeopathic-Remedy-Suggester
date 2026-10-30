"""Clean the scanned Hering "Guiding Symptoms of our Materia Medica" books for HoRUS.

    python offline/scripts/clean_guiding_symptoms.py RAW_FOLDER OUT_FOLDER

RAW_FOLDER holds the books as extracted from the scans
(Guiding_Symptoms_of_our_Materia_Medica-I.json … -X.json). OUT_FOLDER receives the cleaned
books (same file names) plus guiding_symptoms_cleaning_report.json / .md listing every rule's
effect, with examples, so each change can be checked against the printed book.

What is corrected (each rule is counted in the report):
  * section headings and case headers saved as symptoms ("passive motion", "woman, æt.")
  * remedy-relationship chapters and proving-history lines (not symptoms)
  * location lines cut from their heading ("pain: in forehead" / "in temples" → "pain in temples")
  * scanning debris (^ ■ \\ | …), stray bullet letters, the remedy name run into a symptom
  * "1." read for "l.", and "l." / "r." written out as left / right
  * "<" and ">" written out as worse and better
  * remedy names: truncated, misread, merged or misfiled names
  * remedy-wide modality labels copied onto every symptom of a remedy (cleared)
  * duplicate entries within a remedy
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

VOLUMES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
FILE = "Guiding_Symptoms_of_our_Materia_Medica-{}.json"

# ── remedy names ──────────────────────────────────────────────────────────────
# (volume, name in the scan) → correct name; None drops a block that can't be identified.
# Every entry was checked against the block's own opening lines (substance, provers, chapters).
RENAME = {
    ("I", "carbonicum"): "ammonium carbonicum",        # "sesquicarbonas… introduced by Hahnemann in 1828";
                                                       # the mind…ears chapters missing from "ammonium carbonicum"
    ("I", "muriaticum"): "ammonium muriaticum",        # "ammonium chloride, NH4Cl"; first half of that remedy
    ("I", "castus"): "agnus castus",                   # "Vitex agnus castus, a shrub…"
    ("I", "lacticum acidum"): "aceticum acidum",       # block opens "acetic acid… vinegar from the shops"
    ("I", "antiiracinum"): "anthracinum",              # misread; the rest of Anthracinum's chapters
    ("I", "alumina or argilla"): "alumina",
    ("II", "asaf"): "asafoetida",
    ("III", "cadmium sulfuratum cainca"): "cadmium sulfuratum",   # "sulphide of cadmium, CdS"
    ("III", "cantharides capsicum"): "cantharides",    # "Spanish flies"; Capsicum's text is not in the scan
    ("III", "cepa"): "allium cepa",
    ("III", "palladium"): "caladium seguinum",         # block opens "Caladium seguinum, proved in Surinam…";
                                                       # the real Palladium is in vol. VIII
    ("V", "douchos pruriens"): "dolichos pruriens",
    ("VI", "hippoznin"): "hippozaeninum",
    ("VI", "hydrocyanic acid"): "hydrocyanicum acidum",
    ("VIII", "nitrig acid"): "nitricum acidum",
    ("VIII", "picric acid"): "picricum acidum",
    ("VIII", "qisgrsgs"): None,                        # unreadable name; limbs…sensations of a remedy after
                                                       # Pulsatilla that the scan doesn't identify
    ("II", "benzoic acid"): "benzoicum acidum",
    ("X", "viola odorata xanthoxylum"): "viola odorata",
}

# ── entry rules ───────────────────────────────────────────────────────────────
DROP_CHAPTERS = {"relationship", "relations"}
LOC = (r"in|on|over|across|under|through|from|to|into|about|around|behind|below|above|between|along|at|"
       r"near|down|up|within|upon|beneath|towards?|extending|extends")
LOC_LINE = re.compile(rf"^({LOC})\b")
HEADING = re.compile(rf"^([a-z][a-z ,'\-]{{1,40}}?)\s*:\s*({LOC})\b")
CASE_HEADER = re.compile(r"^[a-z .,'\-]{0,40},\s*(set|aet|get|act|æt|oet|cet)\.?(\s*\d+)?\.?$")
HISTORY = re.compile(r"\b(proved (by|in|on)|provings|introduced (by|in|as|into)|published|edition|"
                     r"discovered in|archives|journ?\.|journal|old school|chemists?|materia medica pura|"
                     r"chronic diseases|hahnemann|allg\.|hom\. ?(jour|vier|klin|times|rec))")
EXTRA_HEADINGS = {"passive motion", "desires, aversions", "trachea and bronchia", "the element",
                  "abdomen, loins and groins", "appetite and thirst", "stages of life", "constitution"}
JUNK_CHARS = re.compile(r"[\^■□•|\\{}~_«»¬§¤†‡®©*]")
BULLET = re.compile(r"(^|[.;,:]\s+|\s)[bdq]\s+(?=[a-z]{3})|(^|[.;:]\s+)[il1]\s+(?=[a-z]{3})")   # ■/□ markers read as b, d, q, i, l
ONE_FOR_L = re.compile(r"(?<![\d.])\b1\.(?=\s*[a-z])")
SIDE = re.compile(r"\b([lr])\.\s*(?=[a-z])")


def letters_ratio(t: str) -> float:
    """Share of ordinary characters (letters, digits, spaces, usual punctuation)."""
    return sum(c.isalnum() or c.isspace() or c in ".,;:()'\"-!?/&" for c in t) / max(1, len(t))


def tidy(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([;:,.!?)])", r"\1", t)
    t = re.sub(r"([(])\s+", r"\1", t)
    t = re.sub(r"([;:,])(?=[^\s\d])", r"\1 ", t)
    t = re.sub(r"(;\s*){2,}", "; ", t)
    return t.strip(" ;,:-").strip()


class Cleaner:
    def __init__(self, remedy_names: set, chapter_names: set):
        self.remedy_names = remedy_names
        self.headings = chapter_names | EXTRA_HEADINGS
        self.count = defaultdict(Counter)            # volume → rule → n
        self.examples = defaultdict(list)            # rule → [(before, after)]

    def note(self, vol, rule, before, after=None):
        self.count[vol][rule] += 1
        if len(self.examples[rule]) < 25:
            self.examples[rule].append({"volume": vol, "before": before, "after": after})

    def text(self, vol: str, remedy: str, raw: str) -> str:
        """Character-level repairs of one entry."""
        t = re.sub(r"\s+", " ", raw).strip()
        before = t
        if JUNK_CHARS.search(t) or BULLET.search(t):
            t = JUNK_CHARS.sub(" ", t)
            t = BULLET.sub(lambda m: (lambda g: g if g.strip() else " ")(m.group(1) or m.group(2) or ""), t)
            if t != before:
                self.note(vol, "scanning debris removed", before, tidy(t))
        if "<" in t or ">" in t:
            t2 = re.sub(r"\s*<+c?(?![a-z])\s*", " worse ", t)
            t2 = re.sub(r"\s*>+\s*", " better ", t2)
            t2 = re.sub(r"\b(worse|better)(\s+\1\b)+", r"\1", t2)
            self.note(vol, '"<" / ">" written as worse / better', t, tidy(t2))
            t = t2
        if ONE_FOR_L.search(t):
            t2 = ONE_FOR_L.sub("l.", t)
            self.note(vol, '"1." read as "l."', t, t2)
            t = t2
        if SIDE.search(t):
            t = SIDE.sub(lambda m: "left " if m.group(1) == "l" else "right ", t)
        first = remedy.split()[0]
        words = t.lower().split()
        if (len(words) >= 4 and words[-1].strip(".,;") == first
                and words[-2] not in ("of", "from", "like", "the", "a", "with", "than", "or", "and", "by", "after")):
            self.note(vol, "remedy name run into the symptom", t, tidy(" ".join(t.split()[:-1])))
            t = " ".join(t.split()[:-1])
        # case reference numbers left at the end ("…great anxiety, 65")
        t = re.sub(r"(,|\s)\s*\d{1,3}\.?$", "", tidy(t)) if re.search(r"[a-z]{3}.*(,|\s)\s*\d{1,3}\.?$", t) \
            and not re.search(r"\b(\d{1,2})\s*(a|p)\.?m\.?$|\bto \d{1,3}$", t) else t
        return tidy(t)

    def drop_reason(self, t: str, chapter: str) -> str | None:
        low = t.lower()
        if len(re.findall(r"[a-z]{2,}", low)) == 0 or len(low) < 4:
            return "empty or too short"
        if letters_ratio(t) < 0.85:
            return "unreadable scan text"
        if CASE_HEADER.match(low):
            return "case header (e.g. 'woman, æt.')"
        if low in self.headings:
            return "section heading"
        if low.rstrip(".") in self.remedy_names or (len(low) <= 16 and
                                                    re.fullmatch(r"([a-z]{2,6}\.\s?){1,2}[a-z]{0,7}\.?", low)):
            return "remedy name or abbreviation"
        if HISTORY.search(low):
            return "proving history / publication note"
        return None

    def chapter(self, vol: str, remedy: str, chapter: str, items: list) -> list:
        out, head, misses = [], None, 0
        for it in items:
            if not isinstance(it, dict):
                continue
            raw = str(it.get("symptom") or "")
            t = self.text(vol, remedy, raw)
            low = t.lower()
            m = HEADING.match(low)
            if m:
                head, misses = m.group(1).strip(" ,"), 0
            elif LOC_LINE.match(low) and ":" not in low and len(low.split()) <= 12:
                if head:
                    joined = f"{head} {t}"
                    self.note(vol, "location line rejoined to its heading", t, joined)
                    t = joined
                elif len(low.split()) <= 3 and "," not in low:
                    self.note(vol, "location line without heading (dropped)", t)
                    continue
            else:
                misses += 1
                if misses > 3:
                    head = None
            # "heaviness: of head" → "heaviness of head" (one location after a heading)
            if ";" not in t:
                t = re.sub(rf"^([a-z][a-z ,'\-]{{1,40}}?)\s*:\s*(?=({LOC}|of)\b)", r"\1 ", t, flags=re.I)
            why = self.drop_reason(t, chapter)
            if why:
                self.note(vol, why, raw)
                continue
            out.append({**it, "symptom": t})
        return out


def clear_copied_labels(vol: str, remedy: str, chapters: dict, cleaner: Cleaner) -> None:
    """A label value found on ≥30% (and ≥10) of a remedy's entries was copied from the remedy's
    summary onto every symptom — it says nothing about the individual symptom."""
    entries = [it for items in chapters.values() for it in items]
    if not entries:
        return
    for key in ("Modalities", "Area", "Intensity/Severity", "Extension"):
        vals = Counter(str((it.get("labels") or {}).get(key)) for it in entries
                       if (it.get("labels") or {}).get(key) not in (None, "", "null"))
        for val, n in vals.items():
            if n >= 10 and n >= 0.3 * len(entries):
                for it in entries:
                    lab = it.get("labels") or {}
                    if str(lab.get(key)) == val:
                        lab[key] = None
                cleaner.count[vol][f"copied '{key}' label cleared"] += n
                if len(cleaner.examples[f"copied '{key}' label cleared"]) < 10:
                    cleaner.examples[f"copied '{key}' label cleared"].append(
                        {"volume": vol, "before": f"{remedy}: '{val}' on {n} of {len(entries)} entries", "after": None})


def main(raw: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    books = {v: json.loads((raw / FILE.format(v)).read_text(encoding="utf-8"))
             for v in VOLUMES if (raw / FILE.format(v)).exists()}
    names = {(RENAME.get((v, r), r) or "") for v, d in books.items() for r in d} - {""}
    short = {n.split()[0] for n in names if len(n.split()[0]) > 4}
    chapters = {c.lower() for d in books.values() for chs in d.values() for c in chs}
    cleaner = Cleaner(names | short, chapters)
    report = {"volumes": {}, "renamed": [], "dropped_blocks": [], "merged": []}

    for v, data in books.items():
        cleaned: dict = {}
        before_n = sum(len(i) for chs in data.values() for i in chs.values())
        for remedy, chs in data.items():
            target = RENAME.get((v, remedy), remedy)
            if target is None:
                report["dropped_blocks"].append({"volume": v, "name": remedy,
                                                 "entries": sum(len(i) for i in chs.values())})
                continue
            if target != remedy:
                report["renamed"].append({"volume": v, "from": remedy, "to": target})
            dest = cleaned.setdefault(target, {})
            if dest and target != remedy:
                report["merged"].append({"volume": v, "from": remedy, "into": target})
            for chap, items in chs.items():
                if chap.lower() in DROP_CHAPTERS:
                    cleaner.count[v]["relationship chapter (not symptoms)"] += len(items)
                    continue
                dest.setdefault(chap, []).extend(cleaner.chapter(v, target, chap, items))
        for remedy, chs in cleaned.items():
            seen = set()
            for chap in list(chs):
                keep = []
                for it in chs[chap]:
                    k = it["symptom"].lower()
                    if k in seen:
                        cleaner.count[v]["duplicate within remedy"] += 1
                        continue
                    seen.add(k)
                    keep.append(it)
                if keep:
                    chs[chap] = keep
                else:
                    del chs[chap]
            clear_copied_labels(v, remedy, chs, cleaner)
        cleaned = {r: c for r, c in cleaned.items() if c}
        after_n = sum(len(i) for chs in cleaned.values() for i in chs.values())
        (out / FILE.format(v)).write_text(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        report["volumes"][v] = {"remedies_before": len(data), "remedies_after": len(cleaned),
                                "entries_before": before_n, "entries_after": after_n,
                                "changes": dict(cleaner.count[v].most_common())}
    report["examples"] = cleaner.examples
    (out / "guiding_symptoms_cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "guiding_symptoms_cleaning_report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(rep: dict) -> str:
    L = ["# Cleaning report — Hering, Guiding Symptoms of our Materia Medica", "",
         "Produced by `offline/scripts/clean_guiding_symptoms.py`. Every rule's count per volume, "
         "then examples (before → after) to check against the printed book.", "",
         "| Volume | Remedies | Entries before | Entries after |", "|---|---|---|---|"]
    for v, s in rep["volumes"].items():
        L.append(f"| {v} | {s['remedies_before']} → {s['remedies_after']} | {s['entries_before']:,} | "
                 f"{s['entries_after']:,} |")
    rules = sorted({r for s in rep["volumes"].values() for r in s["changes"]})
    L += ["", "## Changes per volume", "", "| Rule | " + " | ".join(rep["volumes"]) + " |",
          "|---|" + "---|" * len(rep["volumes"])]
    for r in rules:
        L.append(f"| {r} | " + " | ".join(f"{s['changes'].get(r, 0):,}" for s in rep["volumes"].values()) + " |")
    L += ["", "## Remedy names", ""]
    L += [f"- vol. {x['volume']}: “{x['from']}” → **{x['to']}**" for x in rep["renamed"]]
    L += [f"- vol. {x['volume']}: “{x['from']}” merged into **{x['into']}** (the two blocks were halves of one remedy)"
          for x in rep["merged"]]
    L += [f"- vol. {x['volume']}: block “{x['name']}” ({x['entries']} entries) removed — the scan doesn't "
          f"identify the remedy" for x in rep["dropped_blocks"]]
    L += ["", "## Examples", ""]
    for rule, exs in rep["examples"].items():
        L.append(f"### {rule}")
        for e in exs[:8]:
            L.append(f"- ({e['volume']}) “{e['before'][:140]}”" + (f" → “{e['after'][:140]}”" if e["after"] else ""))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]))
    for v, s in r["volumes"].items():
        print(v, s["entries_before"], "→", s["entries_after"], dict(list(s["changes"].items())[:6]))
