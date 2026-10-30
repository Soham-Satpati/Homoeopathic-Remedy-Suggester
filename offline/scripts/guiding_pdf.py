"""Re-check the Guiding Symptoms JSON volumes against the scanned books (Internet Archive PDFs).

The JSON volumes were extracted from the same OCR text the PDFs carry, but the extraction lost track
of where one remedy ends and the next begins:

* several consecutive remedies are often filed under one name ("arsenicum hydrogenisatum" holds all
  of Arsenicum album; "aurum metallicum" holds Ars. met., Ars. sulph. rubrum, Aurum and Aurum mur.);
* a remedy whose title was missed is appended to the one before it (Gambogia inside Formica rufa);
* whole remedies, mostly the first ones of each volume, are missing (Bryonia, Natrum sulph., …).

This module locates every remedy in the book (title lines + running page headers), re-assigns each
JSON entry to the remedy whose pages it was printed on, and parses the pages the JSON doesn't cover.

    python offline/scripts/guiding_pdf.py RAW_JSON_FOLDER PDF_FOLDER OUT_FOLDER

writes corrected (still uncleaned) volumes plus ``guiding_symptoms_pdf_check.json`` to OUT_FOLDER;
``clean_guiding_symptoms.py`` then cleans them as before. Needs ``pdftotext`` (poppler).
"""

from __future__ import annotations

import bisect
import collections
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

VOLUMES = ["I", "II", "III", "V", "VI", "VII", "VIII", "IX", "X"]
FILE = "Guiding_Symptoms_of_our_Materia_Medica-{}.json"
PDF = "Guiding_Symptoms_of_our_Materia_Medica-{}.pdf"

# OCR misreadings of remedy titles → the book's spelling
TITLE_FIX = {
    "thusa cynapium": "aethusa cynapium", "aesculcs glabra": "aesculus glabra",
    "antimonium arsenituffl": "antimonium arsenicosum", "asafcetida": "asafoetida",
    "curara": "curare", "cyclamen europium": "cyclamen europaeum",
    "eupatoreum purpureum": "eupatorium purpureum", "hippoz nin": "hippozaeninum",
    "lodoformum": "iodoformum", "iodium": "iodum", "lodium": "iodum", "lodum": "iodum",
    "ferrum lodatum": "ferrum iodatum", "kali lodatum": "kali iodatum",
    "mercurius lodatus flavus": "mercurius iodatus flavus", "mercurius lodatos ruber": "mercurius iodatus ruber",
    "mercurius lodatus ruber": "mercurius iodatus ruber", "laci iesis": "lachesis",
    "cenanthe crocata": "oenanthe crocata", "p onia": "paeonia", "t ictodes fcetida": "ictodes foetida",
    "ictodes fcetida": "ictodes foetida", "natrum sulfuricum": "natrum sulphuricum",
    "ailantus glandulosa": "ailanthus glandulosa", "aloe socotrina": "aloes socotrina",
    "hepar sulph calc": "hepar sulphuris calcareum", "silica": "silicea",
    "alumina or argilla": "alumina", "vaccininum and variolinum": "vaccininum et variolinum",
    "sinapis nigra et alba": "sinapis nigra et alba", "mercurius": "mercurius solubilis",
    "ferrum": "ferrum metallicum", "arsenicum album": "arsenicum album", "arsenicum": "arsenicum album",
    "cepa": "allium cepa", "caladium": "caladium seguinum", "carbolic acid": "carbolicum acidum",
    "benzoic acid": "benzoicum acidum", "hydrocyanic acid": "hydrocyanicum acidum",
    "nitric acid": "nitricum acidum", "picric acid": "picricum acidum", "lactic acid": "lacticum acidum",
    "calcarea arsenica": "calcarea arsenicosa", "carboneum sulfuratum": "carboneum sulphuratum",
    "hyosciamus niger": "hyoscyamus niger", "asaf etida": "asafoetida", "arsenicum iddatum": "arsenicum iodatum",
    "aurum ffluriatlcum": "aurum muriaticum", "castoreuffl": "castoreum", "cepa allium": "allium cepa",
    "euphorbiuiw": "euphorbium", "ictodes f etida": "ictodes foetida", "ipomcea": "ipomoea",
    "natrum phosphorlcuffl": "natrum phosphoricum", "piconia": "paeonia", "trombidiuffl": "trombidium",
    "erigerok": "erigeron", "ferrum a": "ferrum metallicum", "mi helleborus": "helleborus",
    "alburnum opulus": "viburnum opulus", "lyssik": "lyssin", "douchos pruriens": "dolichos pruriens",
    "arnica montana radix": "arnica montana", "mercurius sulphuricus turpethum": "mercurius sulphuricus",
    "arsenicum sulphuratum flayum": "arsenicum sulphuratum flavum", "aceticum acidum": "aceticum acidum",
    "arsenicum metallicum": "arsenicum metallicum", "morphinum sulphuricum": "morphinum sulphuricum",
    "anacardium occidentale": "anacardium occidentale", "atropinum sulphuricum": "atropinum sulphuricum",
    "cadmium sulfuratum": "cadmium sulphuratum", "cuprum aceticum": "cuprum aceticum",
    "cuprum arsenicosum": "cuprum arsenicosum", "ferrum sulphuricum": "ferrum sulphuricum",
    "kali ferrocyanatum": "kali ferrocyanatum",
}
# the names HoRUS's other books use, so evidence for one remedy adds up across books
SAME_AS = {
    "hepar sulphuris calcareum": "hepar sulphuris", "mercurius solubilis": "mercurius", "silicea": "silica",
    "iodum": "iodium", "calcarea ostrearum": "calcarea carbonica", "lac vaccinum defloratum": "lac defloratum",
}
# lines in the body that look like titles but aren't remedies
NOT_TITLES = re.compile(r"^(volume|the|of|and|preface|index|contents|guiding symptoms|materia medica|"
                        r"relationship|relations|clinical authorities|errata|addenda|vol|qisgrsgs)\b")
END_MATTER = re.compile(r"^(stockholders|first annual report|second annual report|annual report|officers|"
                        r"charter|announcement|by laws|the american homoeopathic publishing|countway library|rare books|"
                        r"boston medical library|digitized by)")
_NORM = re.compile(r"[^a-z]+")


def norm(s: str) -> str:
    return " ".join(_NORM.sub(" ", s.lower()).split())


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def pdf_pages(pdf: Path) -> list[list[str]]:
    """Each page's lines (layout preserved); the first non-empty line is the running header."""
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True,
                         check=True).stdout
    return [[l.rstrip() for l in p.splitlines() if l.strip()] for p in txt.split("\f")]


def page_header(lines: list[str]) -> str | None:
    """Remedy name from the running header ("23        ACONITUM   NAPELLUS."), None if not one."""
    if not lines:
        return None
    h = caps_ocr(re.sub(r"^\W*[\dIl|!]*\s{2,}|\s{2,}[\dIl|!]+\W*$", "", lines[0].strip()))
    letters = [c for c in h if c.isalpha()]
    if len(letters) < 3 or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return None
    n = norm(h)
    return n if n and len(n) < 45 else None


def caps_ocr(s: str) -> str:
    """Common misreadings inside capitals: M → "ffl"/"WI"/"IW", I → "l"."""
    s = s.replace("ffl", "M").replace("UWI", "UM").replace("UIW", "UM").replace("Uiw", "UM")
    return re.sub(r"(?<=[A-Z])l(?=[A-Z])", "I", s)


def is_title(line: str) -> bool:
    s = caps_ocr(line.strip())
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 4 or len(s.split()) > 6 or len(line) - len(line.lstrip()) < 8:
        return False
    return (sum(c.isupper() for c in letters) / len(letters) >= 0.85 and bool(re.search(r"[.,]\s*$", s))
            and re.match(r"^[\^'\"\-®°]*\s*[A-Z]", s) is not None)


SNAPS: dict = {}                    # damaged title → vocabulary name (for the check report)


def vocabulary(raw_dir: Path) -> set:
    """Remedy names to repair OCR-damaged titles with: the JSON volumes' names, the names in HoRUS's
    other books, and the corrections above."""
    names = set(TITLE_FIX.values()) | set(SAME_AS)
    for vol in VOLUMES:
        f = raw_dir / FILE.format(vol)
        if f.exists():
            names |= set(json.loads(f.read_text(encoding="utf-8")))
    repo = Path(__file__).resolve().parents[2]
    try:
        for b in json.loads((repo / "books.json").read_text(encoding="utf-8"))["books"]:
            if not b["id"].startswith("guiding_symptoms_"):
                names |= {k.lower() for k in json.loads((repo / b["json"]).read_text(encoding="utf-8"))}
    except (OSError, KeyError, ValueError):
        pass
    bad = {"qisgrsgs", "castus", "carbonicum", "muriaticum", "asaf", "cepa", "antiiracinum", "hippoznin",
           "nitrig acid", "douchos pruriens", "aleteris farinosa", "hyosciamus niger",
           "cadmium sulfuratum cainca", "cantharides capsicum", "viola odorata xanthoxylum", "cuprum arsenicum",
           "cadmium sulfuratum"}
    return {n for n in names if n not in bad and len(n) > 3}


class Volume:
    """One scanned volume: its text as one searchable string, and where each remedy begins."""

    def __init__(self, pages: list[list[str]], known: set):
        """``known``: trusted remedy names (spelling reference for damaged titles)."""
        self.pages = pages
        self.known = known
        self.text, self.marks = self._index()
        self.segments = self._segments()            # [(offset, name, page, line)]
        self.seg_offsets = [s[0] for s in self.segments]

    # ── text index ──
    def _index(self):
        parts, marks, pos, prev_hyphen = [], [], 0, False
        for p, lines in enumerate(self.pages):
            for j, line in enumerate(lines):
                if j == 0:                                  # running header
                    continue
                n = norm(line)
                if not n:
                    continue
                sep = "" if prev_hyphen else " "
                marks.append((pos + len(sep), p, j))
                parts.append(sep + n)
                pos += len(sep) + len(n)
                prev_hyphen = line.rstrip().endswith("-")
        return "".join(parts) + " ", marks

    def offset(self, page: int, line: int) -> int:
        i = bisect.bisect_left(self.marks, (page, line), key=lambda m: (m[1], m[2]))
        return self.marks[min(i, len(self.marks) - 1)][0]

    def where(self, offset: int) -> tuple[int, int]:
        i = max(0, bisect.bisect_right(self.marks, offset, key=lambda m: m[0]) - 1)
        return self.marks[i][1], self.marks[i][2]

    # ── remedies ──
    def trusted(self, name: str) -> bool:
        return name in TITLE_FIX or TITLE_FIX.get(name, name) in self.known

    def canonical(self, name: str) -> str:
        """The book's title is the authority (the JSON's names are often wrong); only OCR damage is
        repaired — a name already in the vocabulary is kept, a damaged one snaps to the closest
        vocabulary name — and a few names follow HoRUS's other books."""
        name = TITLE_FIX.get(name, name)
        if name not in self.known:
            best = max(self.known, key=lambda k: ratio(name, k), default=None)
            if best and ratio(name, best) >= 0.82:
                SNAPS[name] = best
                name = best
        return SAME_AS.get(name, name)

    def _header_runs(self):
        runs = []                                           # [name, first, last]
        for p, lines in enumerate(self.pages):
            h = page_header(lines)
            if not h:
                continue
            if runs and (h == runs[-1][0] or ratio(h, runs[-1][0]) >= 0.85):
                runs[-1][2] = p
            else:
                runs.append([h, p, p])
        # 1-page runs are pages shared by two remedies
        return [r for r in runs if r[2] > r[1] and not NOT_TITLES.match(r[0]) and not END_MATTER.match(r[0])]

    def _segments(self):
        found = []                                          # (page, line, raw name)
        for p, lines in enumerate(self.pages):
            for j, line in enumerate(lines[1:-1], 1):
                if is_title(line):
                    found.append((p, j, norm(caps_ocr(line))))
        runs = self._header_runs()
        names_in_runs = [r[0] for r in runs]
        # the body runs from the first remedy title that is also a running header to the end matter
        def remedyish(n):
            return any(ratio(n, r) >= 0.75 or n.startswith(r) or r.startswith(n) for r in names_in_runs) \
                or any(ratio(TITLE_FIX.get(n, n), k) >= 0.85 for k in self.known)
        start = next((i for i, f in enumerate(found) if remedyish(f[2])), 0)
        body = []
        for f in found[start:]:
            if END_MATTER.match(f[2]):
                break
            if not NOT_TITLES.match(f[2]):
                body.append(f)
        end_page = len(self.pages)
        last = body[-1][0] if body else 0
        for p in range(last, len(self.pages)):
            h = page_header(self.pages[p]) or ""
            if END_MATTER.match(h) or h.startswith("index") or any(
                    f[0] == p and END_MATTER.match(f[2]) for f in found):
                end_page = p
                break
        # every running-header run needs a title near its first page
        for name, a, b in runs:
            if a > end_page:
                continue
            cname = self.canonical(name)
            near = [f for f in body if a - 1 <= f[0] <= b and self.canonical(f[2]) == cname]
            if near:
                continue
            hit = None
            for p in (a - 1, a):
                for j, line in enumerate(self.pages[p][1:] if 0 <= p < len(self.pages) else [], 1):
                    n = norm(line)
                    if n and len(n.split()) <= 6 and self.canonical(norm(caps_ocr(line))) == cname:
                        hit = (p, j, n)
            body.append(hit or (a, 1, name))
        body.sort()
        segs = []
        for p, j, n in body:
            name = self.canonical(n)
            # a broken running header ("s kali bromatum", "kali carbonic qm") is the same remedy
            if segs and (segs[-1][1] == name or not self.trusted(n) and (
                    ratio(name, segs[-1][1]) >= 0.8 or len(n) > 6 and (n in segs[-1][1] or segs[-1][1] in n))):
                continue
            segs.append((self.offset(p, j), name, p, j))
        self.end_offset = self.offset(end_page, 1) if end_page < len(self.pages) else len(self.text)
        return segs

    def segment_lines(self, i: int):
        """[(page, line_no, text)] of segment ``i`` (running headers left out)."""
        _, _, p0, j0 = self.segments[i]
        if i + 1 < len(self.segments):
            _, _, p1, j1 = self.segments[i + 1]
        else:
            p1, j1 = self.where(self.end_offset) if self.end_offset < len(self.text) else (len(self.pages), 0)
        out = []
        for p in range(p0, min(p1, len(self.pages) - 1) + 1):
            for j, line in enumerate(self.pages[p]):
                if j == 0 or (p == p0 and j < j0) or (p == p1 and j >= j1):
                    continue
                out.append((p, j, line))
        return out

    def parse_segment(self, i: int) -> list[tuple[int, str, str]]:
        """[(page, chapter, symptom text)] for segment ``i``, parsed page by page."""
        chapter, out = None, []
        by_page = collections.defaultdict(list)
        for p, j, line in self.segment_lines(i):
            by_page[p].append(line)
        for p in sorted(by_page):
            entries, chapter = parse_lines(by_page[p], chapter)
            for ch, text in entries:
                out += [(p, ch, t) for t in split_sentences(text)]
        return out

    def remedy_at(self, offset: int) -> str | None:
        if offset >= self.end_offset:
            return None
        i = bisect.bisect_right(self.seg_offsets, offset) - 1
        return self.segments[i][1] if i >= 0 else None

    def find_all(self, s: str, cap: int = 60) -> list[int]:
        out, i, needle = [], 0, f" {s} "
        while len(out) < cap:
            i = self.text.find(needle, i)
            if i < 0:
                break
            out.append(i + 1)
            i += 1
        return out


# ── parsing pages the JSON doesn't cover ─────────────────────────────────────
CHAPTERS = [
    "general characteristics", "mind", "sensorium", "inner head", "outer head", "sight and eyes",
    "hearing and ears", "smell and nose", "upper face", "lower face", "teeth and gums", "taste and tongue",
    "taste, speech, tongue", "inner mouth", "palate and throat", "throat", "appetite, thirst",
    "appetite and thirst", "desires, aversions", "eating and drinking", "hiccough, belching, nausea and vomiting",
    "scrobiculum and stomach", "hypochondria", "abdomen", "abdomen and loins", "stool and rectum",
    "urinary organs", "male sexual organs", "female sexual organs", "pregnancy, parturition, lactation",
    "pregnancy", "voice and larynx", "trachea and bronchia", "respiration", "cough", "inner chest and lungs",
    "heart, pulse and circulation", "outer chest", "neck and back", "upper limbs", "lower limbs",
    "limbs in general", "rest, position, motion", "rest", "position", "motion", "nerves", "sleep", "time",
    "temperature and weather", "fever", "attacks, periodicity", "locality and direction", "sensations",
    "tissues", "touch, passive motion, injuries", "touch", "skin", "stages of life, constitution",
    "relationship", "relations",
]
_CH_WORDS = [(c, [w for w in re.split(r"[^a-z]+", c) if w and w != "and"]) for c in CHAPTERS]
_CH_WORDS.sort(key=lambda t: -len(t[1]))
_CH_RES = [(c, re.compile(r"^" + r"[\s.,;:]+(?:and\s+)?".join(ws) + r"\s*\.", re.I)) for c, ws in _CH_WORDS]
# Hering prints the chapters in this order; a heading out of order is a word in the text ("Fever, …")
_CH_ORDER = {c: i for i, c in enumerate(CHAPTERS)}
_CH_ORDER.update({"relations": _CH_ORDER["relationship"], "pregnancy": _CH_ORDER["pregnancy, parturition, lactation"],
                  "rest": _CH_ORDER["rest, position, motion"], "position": _CH_ORDER["rest, position, motion"],
                  "motion": _CH_ORDER["rest, position, motion"], "touch": _CH_ORDER["touch, passive motion, injuries"],
                  "throat": _CH_ORDER["palate and throat"], "appetite and thirst": _CH_ORDER["appetite, thirst"],
                  "taste, speech, tongue": _CH_ORDER["taste and tongue"],
                  "abdomen and loins": _CH_ORDER["abdomen"]})
_FIRST_CHAPTER_MAX = _CH_ORDER["abdomen"]
_SIGNATURE = re.compile(r"^\s*(VOL|Vol)\.?\s+[IVXL]+\.?\s*$")
_LEAD = re.compile(r"^[^A-Za-z]*")
_MARK = re.compile(r"^(?:[■□•|!*]+\s*|(?:[Il1i]{1,2}|\|)\s+(?=[A-Z(\"'])|[Il1i](?=[A-Z][a-z]))+")
# θ (the disease a symptom was cured in) is OCR'd as "^", "/^", "θ", or after a gap as "d", "6", "0", "@", "e"
_CLINICAL = re.compile(r"(?:\s{2,}[d6@0e]\s?|\s*/?\^\s?|\s*θ\s?)(?=[A-Z][a-z])")
_REFERENCE = re.compile(r"\bvol\.\s*\d|\bp\.\s*\d|\bpp\.\s*\d")


def chapter_heading(line: str):
    """(chapter, rest of line) when ``line`` opens a chapter ("^^ Upper Limbs. Tearing…")."""
    s = _LEAD.sub("", line.strip())
    for name, rx in _CH_RES:
        m = rx.match(s)
        if m:
            rest = s[m.end():]
            # "Touch. Passive Motion. Injuries." — swallow the remaining parts of a combined heading
            while True:
                m2 = re.match(r"\s*(Passive Motion|Injuries|Position|Motion|Parturition|Lactation|Thirst|"
                              r"Periodicity|Constitution|Belching|Nausea and Vomiting)\s*[.,]", rest)
                if not m2:
                    break
                rest = rest[m2.end():]
            return name, rest.strip()
    return None


def modalities(text: str) -> str | None:
    parts = [f"aggravates with {m.strip(' ,;.')}" for m in re.findall(r"<\s*([^<>;]+)", text)]
    parts += [f"ameliorates with {m.strip(' ,;.')}" for m in re.findall(r">\s*([^<>;]+)", text)]
    parts = [p for p in parts if len(p) > 18]
    return ", ".join(parts) or None


def parse_lines(lines: list[str], chapter: str | None):
    """Entries from a run of book lines: [(chapter, text)], plus the chapter open at the end.

    Lines before the remedy's first chapter (description, history, clinical authorities) are
    skipped. An entry starts at the left margin — usually with a grade mark (■, I, 1 1) — and runs
    on over more-indented lines; a right-hand clinical note ("θ Prolapsus uteri") ends it."""
    out = []
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    base = min(indents) if indents else 0
    cur: list[str] = []
    closed = False

    def flush():
        nonlocal cur, closed
        if cur and chapter:
            text = ""
            for part in cur:
                text = text[:-1] + part if text.endswith("-") else (text + " " + part if text else part)
            text = re.sub(r"\s+", " ", text).strip(" .;,")
            if len(text) >= 3:
                out.append((chapter, text))
        cur, closed = [], False

    for line in lines:
        if not line.strip() or _SIGNATURE.match(line):
            continue
        h = chapter_heading(line)
        marked = bool(re.match(r"^\s*[\^'\"*®°\-~`]", line))      # the book's chapter sign, as OCR'd
        if h and (_REFERENCE.search(line) or (chapter is None and _CH_ORDER[h[0]] > _FIRST_CHAPTER_MAX)
                  or (not marked and chapter is not None and _CH_ORDER[h[0]] < _CH_ORDER[chapter])):
            h = None
        if h:
            flush()
            chapter, rest = h
            line = rest
            if not rest:
                continue
            start = True
        else:
            ind = len(line) - len(line.lstrip())
            stripped = line.strip()
            start = (ind <= base + 1 and (bool(_MARK.match(stripped)) or stripped[:1].isupper())) or \
                bool(re.match(r"^[■□]", stripped))
        if _REFERENCE.search(line) and len(_REFERENCE.findall(line)) >= 2:
            flush()
            continue
        if start:
            flush()
        if closed:
            continue
        text = line.strip()
        m = _CLINICAL.search(" " * 3 + text) if False else _CLINICAL.search(text)
        if m:
            text = text[:m.start()]
            closed = True
        text = _MARK.sub("", text) if start else text
        text = re.sub(r"\s{3,}", " ", text)
        if text.strip():
            cur.append(text.strip())
    flush()
    return out, chapter


def split_sentences(text: str) -> list[str]:
    """One printed paragraph can hold several symptoms ("Inability to think. ■ Very low-spirited.")."""
    parts = re.split(r"(?<=[a-z)\]])\.\s+(?:[■□|!]\s*|[Il1i]\s+)?(?=[A-Z])|\s*[■□]\s*", text)
    parts = [re.sub(r"^(?:[Il1i|!]{1,2}\s+)+(?=[A-Z(])", "", p.strip()) for p in parts if p]
    return [p.strip(" .;,") for p in parts if len(p.strip(" .;,")) >= 3]


def realign(raw: dict, vol: Volume) -> tuple[dict, collections.Counter, list]:
    """Re-file every JSON entry under the remedy whose pages it was printed on.

    An entry's text is looked up in the book; when it is printed in several places (common phrases),
    the occurrence nearest its neighbours in the same JSON block is taken. Entries that can't be
    found (OCR differences) follow the previous entry of the same chapter list.
    Returns (volume, Counter[(json name, book remedy)] of entries, [(remedy, chapter, offset, item)])."""
    placed = []                                             # (remedy, chapter, offset, item)
    moves: collections.Counter = collections.Counter()
    margin = 40_000                                         # ~ 15 pages of text
    for key, chapters in raw.items():
        if not isinstance(chapters, dict):
            continue
        lists = [(ch, items if isinstance(items, list) else []) for ch, items in chapters.items()]
        occ = {}
        for ch, items in lists:
            for i, it in enumerate(items):
                n = norm(str(it.get("symptom", ""))) if isinstance(it, dict) else ""
                occ[(ch, i)] = vol.find_all(n) if len(n.split()) >= 3 else []
        unique = sorted(o[0] for o in occ.values() if len(o) == 1)
        if unique:
            lo, hi = unique[len(unique) // 50], unique[-1 - len(unique) // 50]
        else:
            lo, hi = 0, len(vol.text)
        mid = unique[len(unique) // 2] if unique else None
        for ch, items in lists:
            pos: list = []
            prev = None
            for i, it in enumerate(items):
                cands = [o for o in occ[(ch, i)] if lo - margin <= o <= hi + margin]
                if cands:
                    ref = prev if prev is not None else (mid if mid is not None else cands[0])
                    after = [o for o in cands if o >= ref]
                    prev = min(after or cands, key=lambda o: abs(o - ref))
                    pos.append(prev)
                else:
                    pos.append(None)
            # unresolved: previous resolved in this list, else the next one, else the block's middle
            nxt = None
            for i in range(len(pos) - 1, -1, -1):
                if pos[i] is None:
                    pos[i] = nxt
                else:
                    nxt = pos[i]
            last = None
            for i, it in enumerate(items):
                off = pos[i] if pos[i] is not None else (last if last is not None else mid)
                if pos[i] is not None:
                    last = pos[i]
                rem = vol.remedy_at(off) if off is not None else None
                rem = rem or key
                moves[(key, rem)] += 1
                placed.append((rem, ch, off if off is not None else -1, it))
    return placed, moves


def entry(text: str, page: int) -> dict:
    return {"symptom": text.lower(), "labels": {"Area": None, "Intensity/Severity": None,
                                                 "Modalities": modalities(text), "Extension": None},
            "dose": "null", "source": f"scan p. {page + 1}"}


def correct(raw: dict, vol: Volume) -> tuple[dict, dict]:
    """(corrected volume, report): entries re-filed by page, and the pages the JSON doesn't cover parsed
    from the scan. Each remedy's chapters and entries come out in the book's order."""
    placed, moves = realign(raw, vol)
    json_on_page = collections.Counter(vol.where(off)[0] for _, _, off, _ in placed if off >= 0)
    have = collections.defaultdict(set)
    for rem, _, _, it in placed:
        have[rem].add(norm(str(it.get("symptom", ""))))
    added = collections.defaultdict(list)                  # remedy → [(offset, chapter, item)]
    pages_added = collections.defaultdict(list)
    for i, (off0, name, _, _) in enumerate(vol.segments):
        if off0 >= vol.end_offset:
            continue
        parsed = vol.parse_segment(i)
        per_page = collections.defaultdict(list)
        for p, ch, text in parsed:
            per_page[p].append((ch, text))
        for p, items in sorted(per_page.items()):
            known = sum(norm(t) in have[name] for _, t in items)
            # a page the JSON covers has most of its entries there already (or JSON entries located on it)
            if len(items) < 4 or known >= 0.2 * len(items) or json_on_page[p] >= 0.25 * len(items):
                continue
            pages_added[name].append(p + 1)
            base = vol.offset(p, 1)
            for k, (ch, text) in enumerate(items):
                n = norm(text)
                if n and n not in have[name]:
                    have[name].add(n)
                    added[name].append((base + k, ch, entry(text, p)))
    rows = [(off, rem, ch, it) for rem, ch, off, it in placed]
    rows += [(off, rem, ch, it) for rem, lst in added.items() for off, ch, it in lst]
    order = {name: i for i, (_, name, _, _) in enumerate(vol.segments)}
    rows.sort(key=lambda r: (order.get(r[1], len(order)), r[0] if r[0] >= 0 else float("inf")))
    out: dict = {}
    for _, rem, ch, it in rows:
        out.setdefault(rem, {}).setdefault(ch, []).append(it)
    report = {
        "remedies_in_book": [name for _, name, _, _ in vol.segments],
        "renamed_or_split": [{"json_name": a, "book_remedy": b, "entries": n}
                             for (a, b), n in sorted(moves.items(), key=lambda t: -t[1]) if a != b],
        "added_from_scan": {name: {"pages": pages_added[name], "entries": len(added[name])}
                            for name in added if added[name]},
        "json_names_not_in_book": sorted({a for (a, b) in moves} - {b for (a, b) in moves}),
        "title_repairs": dict(SNAPS),
    }
    SNAPS.clear()
    return out, report


def run(raw_dir: Path, pdf_dir: Path, out_dir: Path, volumes=VOLUMES) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = vocabulary(raw_dir)
    reports = {}
    for v in volumes:
        src, pdf = raw_dir / FILE.format(v), pdf_dir / PDF.format(v)
        if not src.exists():
            continue
        raw = json.loads(src.read_text(encoding="utf-8"))
        if not pdf.exists():
            print(f"vol. {v}: no scan, copied unchanged")
            (out_dir / FILE.format(v)).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            continue
        fixed, rep = correct(raw, Volume(pdf_pages(pdf), vocab))
        (out_dir / FILE.format(v)).write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
        n_before = sum(len(i) for chs in raw.values() for i in chs.values() if isinstance(i, list))
        n_after = sum(len(i) for chs in fixed.values() for i in chs.values())
        rep.update(entries_before=n_before, entries_after=n_after, remedies_before=len(raw),
                   remedies_after=len(fixed))
        reports[v] = rep
        print(f"vol. {v}: {len(raw)} → {len(fixed)} remedies, {n_before:,} → {n_after:,} entries; "
              f"{sum(x['entries'] for x in rep['renamed_or_split']):,} entries re-filed, "
              f"{sum(x['entries'] for x in rep['added_from_scan'].values()):,} added from the scan")
    (out_dir / "guiding_symptoms_pdf_check.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    return reports


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:] or VOLUMES)
