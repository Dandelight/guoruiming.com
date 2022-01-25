# 基于`docker-compose`启动`MySQL`服务

## 安装`Docker`

首先安装`Docker`并开启服务：

```bash
systemctl start docker
```

如果没有添加国内源的话建议添加一下，新增/修改`/etc/docker/daemon.json`，添加如下内容：

```json
{
  "registry-mirrors": ["http://hub-mirror.c.163.com"]
}
```

这里用的是网易的源，其他源如下：

- 阿里云：https://help.aliyun.com/document_detail/60750.html
- 网易：http://hub-mirror.c.163.com
- 官方中国镜像：https://registry.docker-cn.com
- ustc ：https://docker.mirrors.ustc.edu.cn
- daocloud（需注册）：https://www.daocloud.io/mirror#accelerator-doc

## 拉取镜像

```shell
docker pull mysql:latest
docker images
```

## 运行容器

```shell
docker run -itd -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 --name mysql-latest mysql
docker ps
docker exec -it mysql-latest /bin/bash
mysql -u root -p
```

## 配置文件

根据[官网文档](https://hub.docker.com/_/mysql)，默认的配置文件位置为`/etc/mysql/my.cnf`，自定义的配置文件位置可以为`/etc/mysql/conf.d`或`/etc/mysql/mysql.conf.d`，因此将宿主机的`/etc/mysql`挂载到容器的`/etc/mysql/conf.d`，`MySQL`会自动合并默认的配置文件`/etc/mysql/my.cnf`与自定义的配置文件（这里是`/etc/mysql/conf.d/my.cnf`）。

## 使用`docker-compose`

1、创建工作目录

```text
mkdir -p /apps/mysql/{mydir,datadir,conf,source}
```

2、编写`docker-compose.yaml`

```text
version: '3'
services:
  mysql:
    restart: always
    image: mysql:5.7.18
    container_name: mysql-labe
    volumes:
      - /apps/mysql/mydir:/mydir
      - /apps/mysql/datadir:/var/lib/mysql
      - /apps/mysql/conf/my.cnf:/etc/my.cnf
      # 数据库还原目录 可将需要还原的sql文件放在这里
      - /apps/mysql/source:/docker-entrypoint-initdb.d
    environment:
      - "MYSQL_ROOT_PASSWORD=yourpassword"
      - "MYSQL_DATABASE=yourdbname"
      - "TZ=Asia/Shanghai"
    ports:
      # 使用宿主机的3306端口映射到容器的3306端口
      # 宿主机：容器
      - 3306:3306
```

3、编写数据库配置文件。

/apps/mysql/conf/my.cnf

```text
[mysqld]
user=mysql
default-storage-engine=INNODB
character-set-server=utf8
character-set-client-handshake=FALSE
collation-server=utf8_unicode_ci
init_connect='SET NAMES utf8'
[client]
default-character-set=utf8
[mysql]
default-character-set=utf8
```

4、启动

启动容器的时候，需要先检查所使用的端口是否被占用。

```text
$ ss -tunlp | grep 3306
$ docker-compose up -d
$ docker-compose ps
Name                 Command             State           Ports
--------------------------------------------------------------------------
mysql-lable   docker-entrypoint.sh mysqld   Up      0.0.0.0:3306->3306/tcp
```

5、测试

进入容器，使用密码登录数据库，并查看数据库有没有创建所指定的库，库里面有没有导入你的 sql 数据

```text
### docker exec -it {容器ID | 容器名(使用docker ps查看)} /bin/bash
$ docker exec -it e592ac9bfa70 /bin/bash
# root@e592ac9bfa70:/# mysql -uroot -p
Enter password:
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 31451
Server version: 5.7.18 MySQL Community Server (GPL)

Copyright (c) 2000, 2017, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>

# 查看数据
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql_data_test    |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.00 sec)

mysql> use mysql_data_test  #这个是我自己的恢复数据文件
mysql> show tables;
.......
```

---

参考：

https://zhuanlan.zhihu.com/p/266534015

https://zhuanlan.zhihu.com/p/384330120
