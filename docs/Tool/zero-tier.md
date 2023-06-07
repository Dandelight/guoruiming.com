# 异地虚拟局域网`ZeroTier`

`ZeroTier` 是一个优秀的异地组网程序，通过虚拟网卡的方式实现。竞品有 `Tailscale`，但 `tailscale` 的 Mac 端需要从 App Store 下载，中国区又下载不到，所以用 ZT。

但是，当加入网络之后迟迟卡在 `REQUESTING_CONFIGURATION`，管理端看不到新用户，就可能是连不上 ZT 官方的根服务器的原因。这时候自建根服务器（ROOT，以前叫 `moon`）可以缓解连不上、延迟大等情况。

## 基本概念

- `PLANET`：`ZeroTier`根节点，由官方维护，全球只有四个
- `MOON`：用户自建 `ZeroTier` 中继节点
- `LEAF`：`ZeroTier`用户端，也就是叶节点

## 安装、使用与管理

这个自不用多说，网络上资料很丰富

## 建立`MOON`

下面是配置`Moon`的步骤（Linux）：

### 安装`zerotier`

`zerotier`对 Linux 系统提供了一键安装脚本

```bash
curl -s https://install.zerotier.com/ | sudo bash
```

如果安装了 `GPG`，可以增加一步签名验证环节（可以省略）

```bash
curl -s 'https://raw.githubusercontent.com/zerotier/ZeroTierOne/master/doc/contact%40zerotier.com.gpg' | gpg --import && \
if z=$(curl -s 'https://install.zerotier.com/' | gpg); then echo "$z" | sudo bash; fi
```

### 生成 `moon` 配置文件

```bash
cd /var/lib/zerotier-one
su # 切换 root
zerotier-idtool initmoon identity.public > moon.json
```

### 修改配置文件中的公网 IP

修改配置文件`moon.json`，主要是添加公网 IP，修改内容如下，9993 是默认端口

```bash
vim moon.json # 找到对应行修改内容
```

```json
"stableEndpoints": [ "23.23.23.23/9993" ]
```

注:`23.23.23.23`为公网 ip, 一定要配置正确,`ZeroTier`依靠此配置去连接`moon`，后面的端口若没有改变则默认都是 UDP 协议的 9993 端口， ，此处在防火墙上需要开放 UDP，否则连接不上 Moon

### 签名

```shell
sudo zerotier-idtool genmoon moon.json
```

执行该命令后，`/var/lib/zerotier-one`目录下会生成一个类似`000000xxxxxxxxxx.moon`的文件

1. 使`moon`配置文件生效:

在`/var/lib/zerotier-one`目录下，新建一个`moons.d`文件夹，并将刚生成的`moon`配置文件放到该文件夹下

```bash
mkdir moons.d
mv 000000xxxxxxxxxx.moon moons.d/
```

### 重启 `ZeroTier`

重新启动 moon 服务器。由于使用命令安装时会自动注册为服务，所以可以依靠以下命令完成启动或重启

```bash
sudo service zerotier-one restart
```

经过以上配置,服务器上的`moon`即配置并应用完毕。

## 连接 `moon`

首先找到自己的 `moon` 的 `id`：

```bash
grep id /var/lib/zerotier-one/moon.json | head -n 1
```

在客户端执行：

```bash
zerotier-cli orbit <id> <id> # 注意要把id写两遍
```

~~直接在 zerotier 目录下,创建 moons.d 文件夹,并且将生成的 000000xxxxxxxx.moon 文件拷入,并重启服务即可~~

在服务器上执行`zerotier-cli listpeers`，发现多了一个`LEAF`节点，配置成功。
