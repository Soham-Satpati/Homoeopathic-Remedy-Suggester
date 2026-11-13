HoRUS Offline macOS Edition
===========================

Clinical homeopathic intelligence that runs entirely on this Mac.
Nothing is sent over the internet.


INSTALLING
  1. If you downloaded several parts (HoRUS-...-macOS-arm64.dmg.part01, .part02, ...),
     put them all in your Downloads folder, open Terminal and run:
         cd ~/Downloads && cat HoRUS-*-macOS-arm64.dmg.part* > HoRUS-macOS.dmg && open HoRUS-macOS.dmg
  2. Drag HoRUS onto the Applications folder in the window that opens.
  3. Eject the HoRUS disk (Finder sidebar) and delete the downloaded files if you like.
  If macOS says HoRUS "cannot be opened" or is "from an unidentified developer":
  Control-click (or right-click) HoRUS in Applications and choose Open, then Open again.
  On macOS 15 or later, if there is no Open button: try once, then go to
  System Settings -> Privacy & Security and click "Open Anyway". Only needed the first time.


FIRST START
  1. Open HoRUS from Applications or Launchpad (drag it to the Dock to keep it there).
  2. Choose how HoRUS will be used:
       Solo        one doctor, one Mac
       Clinic      several people on one Mac (doctors and staff, each with a login)
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
  questionnaire     Hands the Mac to the patient: one question at a time, by voice
                    (English, Bengali or Hindi) or typing. A staff password is needed to
                    leave this screen.
  Settings          Users and roles, backups, audit log, clinic and AI settings, tablets.


THE OFFLINE AI
  The AI only reads free text into symptoms, writes explanations, transcribes speech and
  suggests follow-up questions. Remedies, their order, scores, the miasm and red-flag
  warnings are always decided by the dataset and fixed rules, never by the AI.
  It needs about 8 GB of memory. On Apple silicon it runs on the Mac's GPU. The first use
  after sign-in takes up to a minute while it starts; it then runs in the background.
  It can be switched off in Settings.


MICROPHONE
  The first time you dictate, macOS asks whether HoRUS may use the microphone: click
  Allow. To change this later: System Settings -> Privacy & Security -> Microphone.
  Audio is transcribed on this Mac and never leaves it.


YOUR DATA
  Patient records are encrypted (AES-256) in:
      ~/Library/Application Support/HoRUS/data/horus.db
  Automatic backups are written when HoRUS closes, to:
      ~/Library/Application Support/HoRUS/backups
  (In Finder: Go -> Go to Folder..., then paste the path.)
  Copy backups to a USB drive or another computer regularly. A backup can only be
  opened with a HoRUS password or the recovery key.
  Moving HoRUS to the Bin does NOT delete patient records or backups.
  Each macOS user account has its own HoRUS data. Use one macOS account on a shared
  clinic Mac and give each person their own HoRUS login.
  Time Machine backs up these folders too; the records stay encrypted in it.


TABLETS (Clinic LAN edition)
  Settings -> Tablets (LAN) -> Start tablet intake. Open the address shown on the tablet
  (same Wi-Fi), accept the certificate warning once after checking the fingerprint, and
  enter the pairing code. If macOS asks whether HoRUS may accept incoming network
  connections, or may find devices on the local network, click Allow.


MOVING TO A NEW COMPUTER
  Install HoRUS on the new Mac, choose "Restore from a backup" on first start,
  and select your latest .horusbak file. Backups from the Windows Edition work too.


REMOVING HORUS
  Drag HoRUS from Applications to the Bin. To also delete every patient record, delete
  ~/Library/Application Support/HoRUS (only after copying a backup somewhere safe).


TROUBLESHOOTING
  Logs:  ~/Library/Application Support/HoRUS/logs/horus.log  (no patient details are logged)


For clinical reference only. Prescribing decisions rest with the practitioner.
