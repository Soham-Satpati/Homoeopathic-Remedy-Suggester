HoRUS Offline for Windows
=========================

Clinical homeopathic intelligence that runs entirely on this computer.
Nothing is sent over the internet.


FIRST START
  1. Open HoRUS from the Start menu (or the desktop icon).
  2. Choose how HoRUS will be used:
       Solo        one doctor, one computer
       Clinic      several people on one computer (doctors and staff, each with a login)
       Clinic LAN  as Clinic, plus patients can fill in the questionnaire on a tablet
                   over the clinic Wi-Fi
  3. Create the administrator account.
  4. WRITE DOWN OR SAVE THE RECOVERY KEY. If the administrator password is lost, it is the
     only way to unlock the patient records.
  5. Optional: import patients.json from the online HoRUS demo.


DAILY USE
  New case          Type, paste or dictate the patient's words; the offline AI reads them
                    into symptoms (or use Structured entry). Review, then Analyse.
                    The case is saved automatically. Save PDF for the treatment plan.
                    "Suggest more symptoms" finds symptoms that usually occur with this
                    case and compares the two analyses.
  Patients          Search, open a record, see every case and intake, export PDFs.
  New intakes       Questionnaires patients filled in. "Start case from this intake".
  Patient
  questionnaire     Hands the computer to the patient: one question at a time, by voice
                    (English, Bengali or Hindi) or typing. A staff password is needed to
                    leave this screen.
  Settings          Users and roles, backups, audit log, clinic and AI settings, tablets.


THE OFFLINE AI
  The AI only reads free text into symptoms, writes explanations, transcribes speech and
  suggests follow-up questions. Remedies, their order, scores, the miasm and red-flag
  warnings are always decided by the dataset and fixed rules, never by the AI.
  It needs about 8 GB of RAM. The first use after sign-in takes up to a minute while it
  starts; it then runs in the background. It can be switched off in Settings.


YOUR DATA
  Patient records are encrypted (AES-256) in:
      %LOCALAPPDATA%\HoRUS\data\horus.db
  Automatic backups are written when HoRUS closes, to:
      %LOCALAPPDATA%\HoRUS\backups
  Copy backups to a USB drive or another computer regularly. A backup can only be
  opened with a HoRUS password or the recovery key.
  Uninstalling HoRUS does NOT delete patient records or backups.
  Each Windows user account has its own HoRUS data. Use one Windows account on a shared
  clinic PC and give each person their own HoRUS login.


TABLETS (Clinic LAN edition)
  Settings -> Tablets (LAN) -> Start tablet intake. Open the address shown on the tablet
  (same Wi-Fi), accept the certificate warning once after checking the fingerprint, and
  enter the pairing code. If Windows Firewall asks, allow HoRUS on Private networks.


MOVING TO A NEW COMPUTER
  Install HoRUS on the new computer, choose "Restore from a backup" on first start,
  and select your latest .horusbak file.


TROUBLESHOOTING
  Logs:  %LOCALAPPDATA%\HoRUS\logs\horus.log  (no patient details are logged)
  "Windows protected your PC" at install: the installer is not code-signed yet.
  Click "More info" -> "Run anyway".


For clinical reference only. Prescribing decisions rest with the practitioner.
