HoRUS Offline macOS Edition - Engine Test Console
=================================================

This is a TEST BUILD of the offline engine (step 1 of 5). It is not the clinic app.

HOW TO RUN
  Unzip, then drag HoRUS-Offline-Test.app to Applications (or leave it in this folder).
  The first time, Control-click (or right-click) the app and choose Open, then Open again.
  On macOS 15 or later, if there is no Open button: try to open it once, then go to
  System Settings -> Privacy & Security and click "Open Anyway".
  The build is not notarised yet; that comes with the real app.

  The first start takes 10-30 seconds while the dataset and the matching model load.
  The first analysis takes a little longer while it indexes the dataset; after that it
  is fast (the index is cached).

WHAT TO TRY
  1. Pick a case from "Load a sample..." and press Analyse (or Command-Return).
  2. Read the Report tab: red flags, miasm, ranked remedies, and for each remedy the
     dataset rows it matched (patient symptom = dataset symptom, with similarity).
  3. Open "Symptom suggestions" for symptoms that often occur with this case.
  4. Save PDF... writes the same treatment-plan PDF as the demo.
  5. Enter your own cases. One symptom per line:
         burning pain in stomach | worse: night, cold drinks | better: warm drinks
     If you also paste the patient's own words at the top, symptoms that are not in
     those words are marked "unverified" and left out of the ranking (as in the demo).

WHAT IT DOES NOT DO YET
  - No patient database, logins or saved history   (step 2)
  - No AI: no free-text parsing, written explanations, voice or follow-up
    questions; reports use dataset evidence only    (step 3)
  - This window is a test harness; the clinic app replaces it (step 4)

PRIVACY
  Everything runs on this Mac. Nothing is sent over the internet.
  A log file is kept at ~/Library/Application Support/HoRUS/logs/test-console.log
  The symptom-index cache is at ~/Library/Application Support/HoRUS/cache

TO REMOVE
  Move HoRUS-Offline-Test.app to the Bin, and delete
  ~/Library/Application Support/HoRUS (Finder: Go -> Go to Folder...)
