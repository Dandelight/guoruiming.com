# 服务器本身

现在网页在腾讯云对象存储（COS）上托管，自己参考 [`mkdocs`](https://github.com/mkdocs/mkdocs/) 源码写了个简单的持续集成，也是刚开始体会到直接看懂源码的乐趣。

开源软件的能量才刚刚显现。

## 关于那个我自己写的服务器

[dandelight/enging](https://gitee.com/dandelight/enging)  是一个类 [`NginX`](https://nginx.org/)  的 `web`  静态页面服务器，但是出于未知的原因突然就只能 serve HTML 文件了，所以就用真正的 `NginX`  替代了。

## 关于 80 端口被`Welcome to nginx!`页面占用

找到`/etc/nginx/site-enabled/default`文件，把 80 端口的占用指向另一个端口。

> 提问：这样做还要占掉一个端口，有没有其他方法？
>
> 更新：把`site-enabled`当成主界面就好。

## `SSH`禁用密码登录

首先修改`/etc/ssh/sshd_config`

```ssh
# 禁用密码验证
PasswordAuthentication no
# 启用密钥验证
PubkeyAuthentication yes
```

然后重启`ssh`服务

```shell
service sshd restart # centos
service ssh restart # ubuntu
/etc/init.d/ssh restart # debian
```
