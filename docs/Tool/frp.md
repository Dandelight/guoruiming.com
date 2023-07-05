# FRP

[FRP](https://github.com/fatedier/frp) 是一个开源的内网穿透软件，支持 TCP、UDP 等传输层协议，HTTP、HTTPS 等应用层协议，可以比较方便地将本地端口 forward 到远程计算机，将本地服务端口暴露在公网上（即内网穿透）。

## 下载

部署 FRP 需要客户端和服务器两部分。服务器端部署在公网服务器上，客户端部署在内网机器上。首先在服务端和客户端各自从 GitHub Releases 页面下载对应操作系统的程序包 <https://github.com/fatedier/frp/releases>，比如写作本文时最新的 `v0.50.0` 版本

```shell
curl -OL 'https://github.com/fatedier/frp/releases/download/v0.50.0/frp_0.50.0_linux_amd64.tar.gz'
```

解压后得到

```shell
$ ls
frpc  frpc_full.ini  frpc.ini  frps  frps_full.ini  frps.ini  LICENSE
```

其中

- `frpc` 和 `frps` 分别是客户端（Client）和服务端（Server）的可执行文件，
- `frpc.ini` 和 `frps.ini` 分别是客户端和服务器配置文件。
- `frpc_full.ini` 和 `frps_full.ini` 是配置文件的完整版，包含了所有可用的配置项。

## 配置与启动

首先，在编写服务端配置文件

```ini
[common]
bind_port = 7000
vhost_http_port = 8080
```

使用配置文件启动服务端

```shell
./frps -c frps.ini
```

然后，在编写客户端配置文件

```ini
[common]
server_addr = 182.92.221.30
server_port = 7000

[web]
type = tcp
local_ip = 127.0.0.1
local_port = 8001
remote_port = 8001
```

启动客户端

```shell
./frpc -c frpc.ini
```

## 配置文档说明

配置文件采用 `ini` 格式，结构如下：

```ini
[common]
server_addr = 182.92.221.30
server_port = 7000

[any_name]
type = {type}
```

其中 `common` 是共同的配置项，之后可以跟若干个服务，服务的名称 `any_name` 可以自定义，之后跟着以键值对方式设定的配置项。具体配置内容见官方文档。
