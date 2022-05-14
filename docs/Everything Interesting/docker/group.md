# 将用户加入`docker`用户组

解决每次运行 docker 都要输入 sudo 的问题

```
# 方法一
sudo groupadd docker    #添加docker用户组
sudo gpasswd -a $USER docker    #将登陆用户加入到docker用户组中
newgrp docker    #更新用户组
# 方法二：
sudo groupadd docker    #添加docker用户组
sudo usermod -aG docker $USER         #将登陆用户加入到docker用户组中
log out/log in
```

refer to: https://blog.csdn.net/weixin_42800966/article/details/123304328
