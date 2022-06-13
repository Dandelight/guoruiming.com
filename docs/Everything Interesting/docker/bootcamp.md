# DOCKER BOOTCAMP

# 建立`docker`镜像的一些通用启动工作

```dockerfile
FROM # Base image name
# apt换源
RUN sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list &&\
        sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN apt-get update && apt-get install openssh-server -y
# pip换源
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
# 添加SSH公钥
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config &&\
        echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config &&\
        echo "AuthorizedKeysFile  .ssh/authorized_keys" >> /etc/ssh/sshd_config &&\
        /etc/init.d/ssh restart &&\
        mkdir -p ~/.ssh &&\
        echo $SSH_PUBKEY > ~/.ssh/authorized_keys &&\


ENTRYPOINT ["/usr/sbin/sshd", "-D"]
```

```yaml
version: "3"
services:
  pytorch: # 记住改service的名字，或者在下面加一个name字段
    build: "."
    ports:
      - "0:22" # 将容器内22映射到任意端口
    volumes:
      - $HOME:$HOME
      - /nvme:/nvme
    shm_size: "32gb" # PyTorch多线程加载数据
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"] # NVIDIA GPU支持
```
