## HoRUS Offline for Windows

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
