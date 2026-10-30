## HoRUS Offline for Windows

### New in 1.5

- **All of Hering's Guiding Symptoms (vols. I–III and V–X) are now built in, checked against the printed books and cleaned.** 360 remedies and about 142,000 symptoms. Every change is listed, with examples and book pages, in `guiding_symptoms_cleaning_report.md`:
  - every symptom was looked up in the scanned book and filed under the remedy it is printed under — about 24,700 had been filed under a neighbouring remedy (e.g. all of Arsenicum album under "Arsenicum hydrogenisatum", Mercurius solubilis under Merc. corrosivus and dulcis, Capsicum under Causticum, Iodium under Inula);
  - about 5,000 symptoms the extraction had skipped were read from the scanned pages — Bryonia, Hepar sulph., Natrum phos. and sulph., both Ranunculus, Lachnanthes, Lactic acid, the rest of Arnica and Staphisagria, and about 25 smaller remedies;
  - section headings and case headers that had been saved as symptoms ("passive motion", "woman, æt.") removed, along with remedy-relationship chapters and proving-history notes;
  - location lines rejoined to their heading ("pain: in forehead" / "in temples" → "pain in temples");
  - scanning debris removed; "1." read as "l." fixed; "l." / "r." written as left / right; "<" / ">" written as worse / better;
  - broken remedy names fixed — e.g. "castus" → Agnus castus, vol. III's "palladium" → Caladium seguinum (its real content), "carbonicum" / "muriaticum" rejoined to Ammonium carbonicum / muriaticum; the unreadable block "qisgrsgs" is part of Plumbum;
  - remedy-wide modality labels that had been copied onto every symptom cleared.
- **Books you added yourself from the raw scans are replaced automatically** by the cleaned built-in volumes (your settings move to them). *Settings → Books & weighting* lists the old copies with a button to delete them.
- **Symptom suggestions are grouped under each of the patient's symptoms.** Each symptom is a drop-down showing its own details (category, worse, better); open it to see the symptoms that often occur with it, how strongly, and which of the case's leading remedies each points to. Tick, add modalities, compare — as before.
- **The patient questionnaire shows three questions per screen**, on the clinic PC and on tablets, each with its own answer box and microphone.
- **Suggestions weigh the evidence across the book.** A symptom that goes with several entries close to the patient's (e.g. with five headache entries in different remedies) now ranks above a line printed beside a single entry, and every symptom of the case gets its share of the case-wide list.
- The engine's automated tests now run on Windows only, the platform HoRUS ships for.
- Lower memory use for the suggestion tables (about 600 MB less with all books loaded).

### New in 1.4

- **Symptom suggestions work with your own wording.** Before, suggestions only appeared when a symptom was typed exactly as in the book, so most cases showed *"No suggestions"*. Each symptom is now matched to the closest symptoms in the selected books ("headache in the forehead" ≈ "pain in forehead"), and the table shows what it was matched to.
- **More useful suggestions first.** Symptoms listed under the case's leading remedies come first, and each one shows the remedy it points to, so you know what to ask about. Weak links (from symptoms found under only one remedy) count less, and scanning debris from the books is hidden.
- **Fixed: the same remedy could appear twice in the top 10** (e.g. "Aesculus Hippocastanum" and "Aesculus hippocastanum") when books spelled it differently, which also split its score. Remedy names are now merged across books and saved cases.
- **Fixed: time-of-day modalities matched the wrong time** (e.g. "worse afternoon" ≈ "worse mornings"). Times must now agree.
- **Tip:** symptoms with their location and sensation ("pressing pain in forehead", "heaviness of head") match the books far better than bare words ("heaviness"). Edit them on the review screen before analysing.

### New in 1.3

- **Fixed: HoRUS froze while adding a large book.** Building the symptom clusters for a book of ~18,000 symptoms kept the app busy for about a minute. Books are now added in a separate background process, so the window stays responsive (measured: the app never pauses for more than a fraction of a second while an 18,635-symptom book is added).
- **Add several books at once.** *Settings → Books & weighting → Add books…* accepts multiple files. Name each book (or untick ones you don't want) and they are added one after another with an overall progress bar. A book that can't be read is reported and the others are still added.
- **Cancel** while books are being added.

### New in 1.2

- **Fixed: Settings → Books & weighting stayed on "The books are still loading" forever** if Settings was built before the books finished loading. The tab now fills itself in as soon as they're ready.
- **Progress bars** for loading and indexing the books (in the Books tab and the status bar) and for adding a book, with a message for each step.
- **Much faster first start.** The symptom index for the shipped books is now built into the installer, so HoRUS no longer spends several minutes indexing ~21,000 symptoms the first time it opens.
- **Delete button on every book.** Books added in the clinic are deleted from the computer; books that came with HoRUS are removed from use and can be brought back with *Restore removed books*. Only administrators can delete books, and every change is recorded in the audit log.

### New in 1.1

- **Books and decision bias.** Choose which books rank remedies, how strongly each counts, how much this clinic's own saved cases count, and the classical weights for mind, generals, particulars, modalities and causation. Set a clinic default in *Settings → Books & weighting* and change it for a single case from *New case*. The books and weights used are recorded on every case and printed on its PDF.
- **Hering's Guiding Symptoms, vol. I** added as a third book (45 remedies, about 18,000 symptoms), with its own symptom clusters for suggestions.
- **Add a book…** imports any book in the HoRUS JSON format and builds its clusters on this computer.
- **Live clusters.** Symptom suggestions and remedy weights update when a case is saved, and roll back exactly when a case or patient is deleted. Cases can now be deleted individually. Re-analysing a case updates it instead of saving a duplicate.
- **Suggestions now use every selected book's clusters.** 1.0 (like the online demo) only read the rheumatic book's table.
- **Delete my account** for administrators. If another administrator remains, only the account is removed; if you are the only user, the clinic is deleted and HoRUS returns to first-time setup.
- **Fixes:** the tick boxes on the symptom-suggestions and review screens are visible again; a rare crash when closing HoRUS while it was still preparing data.

Clinical homeopathic intelligence that runs entirely on your computer — no internet, no cloud. Patient records are encrypted on the PC.

### Download and install

The installer is split into parts because the offline AI models are large. **Download every `HoRUS-Setup` file below into the same folder**, then run `HoRUS-Setup.exe`.

- `HoRUS-Setup.exe`
- `HoRUS-Setup-1.bin`, `HoRUS-Setup-2.bin`, … (all of them)

Windows may show *"Windows protected your PC"* because the installer is not code-signed yet: click **More info → Run anyway**. No administrator rights are needed.

Requirements: Windows 10 (1809+) or 11, 64-bit · 8 GB RAM recommended · about 8 GB free disk space.

### What's inside

- **Three editions in one install** — Solo, Clinic (several logins with roles), Clinic LAN (patients fill in the questionnaire on a tablet over the clinic Wi-Fi). Chosen at first start, changeable later.
- **Case analysis** — type, paste or dictate the patient's words; the offline AI reads them into symptoms, you review, HoRUS ranks remedies from the dataset with the evidence behind each, and writes the treatment-plan PDF. Structured entry works without the AI.
- **Same decisions as the online demo** — remedy ranking, scores, miasm, red-flag warnings and the PDF are proven identical to the demo by automated tests. The AI only reads text and writes explanations; it never chooses remedies.
- **Symptom suggestions and comparative report**, **patient questionnaire with voice** (English, Bengali, Hindi → English, offline), **patient history**, **import of the demo's `patients.json`**.
- **Security** — AES-256 encrypted records, per-user passwords with roles, recovery key, lockout, auto-lock, tamper-evident audit log, sealed backups (automatic on exit).

### First start

Choose an edition, create the administrator account, and **save the recovery key** — it is the only way to unlock the records if the administrator password is lost.

Built and verified by an automated pipeline on Windows: engine parity tests, a full self-test of the installed app with the offline AI, a window test, and install/uninstall (patient data is kept on uninstall). See `SHA256SUMS.txt` to verify the downloads.
