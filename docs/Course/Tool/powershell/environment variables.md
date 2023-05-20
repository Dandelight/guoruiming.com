# 在 PowerShell 中设置系统环境变量

```powershell
[Environment]::SetEnvironmentVariable("PATH", $Env:PATH + ";C:\Program Files\Scripts", [EnvironmentVariableTarget]::Machine)
```
