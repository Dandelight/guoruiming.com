# 服务器本身

## 关于那个我自己写的服务器

[dandelight/enging](https://gitee.com/dandelight/enging)  是一个类 [`NginX`](https://nginx.org/)  的 `web`  静态页面服务器，但是出于未知的原因突然就只能 serve HTML 文件了，所以就用真正的 `NginX`  替代了。

## 关于 80 端口被`Welcome to nginx!`页面占用

找到`/etc/nginx/site-enabled/default`文件，把 80 端口的占用指向另一个端口。

> 提问：这样做还要占掉一个端口，有没有其他方法？
>
> 更新：把`site-enabled`当成主界面就好。

## `SSH`禁用密码登录

首先修改`/etc/ssh/sshd_config`

```
#禁用密码验证
PasswordAuthentication no
#启用密钥验证
RSAAuthentication yes
PubkeyAuthentication yes
```

然后重启`ssh`服务

```
service sshd restart #centos系统
service ssh restart #ubuntu系统
/etc/init.d/ssh restart #debian系统
```
