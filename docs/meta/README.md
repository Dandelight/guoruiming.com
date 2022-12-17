## 项目概况

使用 [`mkdocs`](https://github.com/mkdocs/mkdocs/) 进行编译，采用 [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/) 主题，各项配置可在本项目仓库中查看。

项目发布流程如下：

```shell
mkdocs build # 编译
python scripts/add_meta.py # 手动增加一个 <meta> 标签
python upload_qcloud.py # 托管到腾讯云
```

编译后的 HTML 页面在腾讯云对象存储（COS）上托管。

`build.py` 文件是我原本打算自己写一个增量编译的脚本，但发现不能增量编译，`mkdocs` 这个框架甚至没留出并行的空间，必须全部文件打个包。后来了解到 [`Hugo`](https://gohugo.io/) 项目性能等都不错，但配置了一个晚上没配好，就没有迁移的动力了。

想自己搭建博客，还是推荐 [`Wordpress`](https://wordpress.org/)，便捷高效，主题和插件丰富，又方便管理。虽然自己没实际应用过就推荐是不太负责任的行为……

## 服务器

自己在学 `Linux` 网络编程时仿照 [`NginX`](https://nginx.org/)  写过一个 `web`  静态页面服务器，放在 [dandelight/enging](https://gitee.com/dandelight/enging)，但是出于未知的原因突然就只能 serve HTML 文件，`css` 和 `js` 无法加载，所以就用真正的 `NginX`  替代了。后来因为要上传图片，需要静态托管，就迁移到了对象存储上。

### 关于 80 端口被 `Welcome to nginx!` 页面占用

找到 `/etc/nginx/site-enabled/default` 文件，把 80 端口的占用指向另一个端口。

> 提问：这样做还要占掉一个端口，有没有其他方法？
>
> 更新：把 `site-enabled` 当成主界面就好。

## `SSH` 禁用密码登录

首先修改 `/etc/ssh/sshd_config`

```ssh
# 禁用密码验证
PasswordAuthentication no
# 启用密钥验证
PubkeyAuthentication yes
```

然后重启 `ssh` 服务

```shell
service sshd restart # centos
service ssh restart # ubuntu
/etc/init.d/ssh restart # debian
```
