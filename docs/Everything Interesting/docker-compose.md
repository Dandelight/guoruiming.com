# `Docker Compose`

大三是项目巨多的一年，各种环境搭建工具都要用上，其中最香的莫过于`docker`；而实际生产中通常会用到许多组件，如数据库和页面逻辑分离，作为多个独立的`docker image`，要把它们组合（compose）到一起，`docker-compose`是最佳选择之一；其他还有`Kubernetes`等。

## 复习

### 如何用`Dockerfile`定制镜像

`Dockerfile`是由一行一行的指令构成的，在镜像构建过程中顺序执行。

#### `FROM`

```dockerfile
FROM <镜像名>
```

每个自定义的`Dockerfile`的第一句必定是`FROM`，先知道镜像是怎么来的，后面我们再讲镜像是怎么没的。

#### `RUN`

```dockerfile
RUN <命令>
```

一条`shell`命令，在相当于在被构建的镜像中的`shell`中执行，完成镜像的搭建工作；常见工作有修改配置文件、编译/下载安装软件包等。

#### `COPY`

```dockerfile
COPY [--chown=<user>:<group>] {"<源路径1>",...  "<目标路径>"}
```

`--chown`：将复制进容器中的文件所有者改成参数标明的用户和所有者

`<源路径>`：源文件或者源目录，这里可以是通配符表达式，其通配符规则要满足 Go 的 `filepath.Match` 规则

#### `ADD`

同`COPY`，不同点是会自动解压`tar`、`gzip`、`bzip2`和`xz`文件。

#### `CMD`

在`docker run`时默认执行，但`docker run`带有命令行参数时会被overridden。

#### `ENTRYPOINT`

类似于`CMD`，但不会被`docker run`时指定的命令行参数覆盖，而是会将命令行参数作为`ENTRYPOINT`的参数执行。但如果在`docker run`中使用了参数`–-entrypoint`，则被覆盖而只执行`–-entrypoint`的参数。

```dockerfile
ENTRYPOINT ["<executeable>","<param1>","<param2>",...]
```

如果同时指定了`CMD`和`ENTRYPOINT`，`CMD`会成为`ENTRYPOINT`的参数。注意`ENTRYPOINT`后参数不会被`docker run`的命令行参数覆盖，而`CMD`的会。

#### `ENV`

`shell`环境变量

```dockerfile
ENV {<key> <value>}
ENV {<key1>=<value1>} [<key2>=<value2> ...]
```

#### `ARG`

`Dockerfile`内部参数，仅在`docker build`过程中生效。

#### `VOLUME`

定义匿名卷。在启动容器时没有挂载数据卷时自动挂载。

#### `EXPOSE`

声明暴露端口，对外暴露的端口可用于端口映射。

#### `WORKDIR`

指定工作目录，在`docker build`期间此后的命令中一直有效。

#### `USER`

指定用户，在`docker build`期间此后的命令中一直有效。

#### `HEALTHCHECK`

### HEALTHCHECK

用于指定某个程序或者指令来监控 docker 容器服务的运行状态。

格式：

```dockerfile
HEALTHCHECK [选项] CMD <命令>：设置检查容器健康状况的命令
HEALTHCHECK NONE：如果基础镜像有健康检查指令，使用这行可以屏蔽掉其健康检查指令

HEALTHCHECK [选项] CMD <命令> : 这边 CMD 后面跟随的命令使用，可以参考 CMD 的用法。
```

#### `ONBUILD`

延迟构建命令。假设本镜像名为`image`，`ONBUILD`的参数在本次构建过程中不会执行，但如果另一镜像的`Dockerfile`第一条命令是`FROM image`，`ONBUILD`的参数会被执行。

#### `LABEL`

以键值对的形式添加一些元数据。

```dockerfile
LABEL <key>=<value> <key>=<value> <key>=<value> ...
```

## 起步

首先安装：

