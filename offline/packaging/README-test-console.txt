HoRUS Offline - Engine Test Console
===================================

This is a TEST BUILD of the offline engine (step 1 of 5). It is not the clinic app.

HOW TO RUN
  Double-click HoRUS-Offline-Test.exe (keep it inside this folder - it needs the
  "_internal" folder next to it).
  If Windows shows "Windows protected your PC": click "More info", then "Run anyway".
  The build is not code-signed yet; that comes with the real installer.

  The first start takes 10-30 seconds while the dataset and the matching model load.
  The first analysis takes a little longer while it indexes the dataset; after that it
  is fast (the index is cached).

WHAT TO TRY
  1. Pick a case from "Load a sample..." and press Analyse (or Ctrl+Enter).
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
  Everything runs on this computer. Nothing is sent over the internet.
  A log file is kept at %LOCALAPPDATA%\HoRUS\logs\test-console.log
  The symptom-index cache is at %LOCALAPPDATA%\HoRUS\cache

TO REMOVE
  Delete this folder and %LOCALAPPDATA%\HoRUS
