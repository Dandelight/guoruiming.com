# Windows 自带 OpenSSH 服务器的安装与使用

1. 在 Windows 的 设置 > 应用 > 应用和功能 > 管理可选功能 > 添加功能，然后找到 **OpenSSH 服务器** 安装。
2. 开启 `sshd` 服务 `Start-Service sshd`
3. 设置服务为开机自启动 `Set-Service -Name sshd -StartupType 'Automatic'`
4. 检查对应防火墙是否打开 `Get-NetFirewallRule -Name ssh`
5. 如果是放开的，那么结果会提示 `OpenSSH-Server-In-TCP`  状态为 `enabled`。
6. 用本机的 IP 被连接，用户名密码是 Windows 的用户名和密码。
