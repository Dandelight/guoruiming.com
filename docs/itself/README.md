# 服务器本身

### 关于80端口被`Welcome to nginx!`页面占用

找到`/etc/nginx/site-enabled/default`文件，把80端口的占用指向另一个端口。

> 提问：这样做还要占掉一个端口，有没有其他方法？