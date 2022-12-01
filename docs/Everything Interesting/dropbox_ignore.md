# The missing `.dropboxignore`

<https://github.com/sp1thas/dropboxignore> 项目目前（2022/11/24）只支持 `*nix`，所以 `Windows` 下还是需要根据官方文档刀耕火种。

官方文档： <https://help.dropbox.com/zh-cn/sync/ignored-files>

可见官方不是没有这个功能，而是不想把它拿出来给大众用。毕竟，早点填满，就需要开会员了呢。

这是本博客的配置，注意一定要用绝对路径。

```powershell
Set-Content -Path "C:\Users\${env:UserName}\Dropbox\blog" -Stream com.dropbox.ignored -Value 1
Set-Content -Path "C:\Users\${env:UserName}\Dropbox\blog\node_modules" -Stream com.dropbox.ignored -Value 1
Set-Content -Path "C:\Users\${env:UserName}\Dropbox\blog\.git" -Stream com.dropbox.ignored -Value 1
Set-Content -Path "C:\Users\${env:UserName}\Dropbox\blog\site" -Stream com.dropbox.ignored -Value 1
Clear-Content -Path "C:\Users\${env:UserName}\Dropbox\blog" -Stream com.dropbox.ignored
```
