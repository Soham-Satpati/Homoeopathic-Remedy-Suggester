## HoRUS Offline for Windows

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
