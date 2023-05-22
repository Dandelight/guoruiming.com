```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl install -d Ubuntu-18.04
wsl install -d Ubuntu-20.04
wsl --shutdown
wsl --set-version Ubuntu-18.04 2
wsl --set-version Ubuntu-20.04 1
```
