; Inno Setup script for HoRUS Offline (Windows x64).
;   iscc horus.iss /DAppVersion=1.0.0 /DSourceDir=..\..\dist\HoRUS /DOutputDir=..\..\installer
; The bundled AI models make the package larger than GitHub's 2 GB per-file limit, so the
; installer is split: HoRUS-Setup.exe plus HoRUS-Setup-1.bin, -2.bin... kept in the same folder.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\HoRUS"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\installer"
#endif

[Setup]
AppId={{7E3C4B9A-6F2D-4C1E-9A8B-5D0F2E7C1A43}
AppName=HoRUS
AppVersion={#AppVersion}
AppVerName=HoRUS {#AppVersion}
AppPublisher=Soham Satpati
AppComments=Clinical homeopathic intelligence that runs entirely offline.
DefaultDirName={autopf}\HoRUS
DefaultGroupName=HoRUS
DisableProgramGroupPage=yes
; Installs for the current user without administrator rights by default; an administrator can
; choose "all users" in the first dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
WizardStyle=modern
SetupIconFile=horus.ico
UninstallDisplayIcon={app}\HoRUS.exe
UninstallDisplayName=HoRUS {#AppVersion}
OutputDir={#OutputDir}
OutputBaseFilename=HoRUS-Setup
Compression=lzma2/normal
SolidCompression=no
LZMANumBlockThreads=4
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
InfoBeforeFile=install-info.txt
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "firewall"; Description: "Allow tablets on the clinic network to reach HoRUS (Clinic LAN edition)"; GroupDescription: "Network:"; Flags: unchecked; Check: IsAdminInstallMode

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HoRUS"; Filename: "{app}\HoRUS.exe"
Name: "{group}\HoRUS - read me"; Filename: "{app}\README.txt"
Name: "{group}\Uninstall HoRUS"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HoRUS"; Filename: "{app}\HoRUS.exe"; Tasks: desktopicon

[Run]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""HoRUS tablet intake"" dir=in action=allow program=""{app}\HoRUS.exe"" enable=yes profile=private"; Flags: runhidden; Tasks: firewall
Filename: "{app}\HoRUS.exe"; Description: "Start HoRUS"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""HoRUS tablet intake"""; Flags: runhidden; RunOnceId: "fwrule"; Check: IsAdminInstallMode

[Messages]
UninstalledAll=HoRUS has been removed.%n%nPatient records and backups were NOT deleted. They are in your Windows profile under AppData\Local\HoRUS.
