# 异地虚拟局域网`ZeroTier`

## 基本概念

`PLANET`：`ZeroTier`根节点

`MOON`：用户自建`ZeroTier`中继节点

`LEAF`：`ZeroTier`用户端，也就是叶节点

## 安装、使用与管理

这个自不用多说，网络上资料很丰富

## 建立`MOON`

下面是配置`Moon`的步骤（Linux）：

1. 安装`zerotier`。`zerotier`对 Linux 系统提供了意见安装脚本

```bash
curl -s https://install.zerotier.com/ | sudo bash
```

2. 生成`moon`配置文件

```bash
cd /var/lib/zerotier-one
sudo zerotier-idtool initmoon identity.public > moon.json
```

3. 修改配置文件`moon.json`，主要是添加公网 IP，修改内容如下，9993 是默认端口

```bash
vim moon.json #找到对应行修改内容
```

```json
"stableEndpoints": [ "23.23.23.23/9993" ]
```

注:`23.23.23.23`为公网 ip, 一定要配置正确,`ZeroTier`依靠此配置去连接`moon`，后面的端口若没有改变则默认都是 UDP 协议的 9993 端口， ，此处在防火墙上需要开放 UDP，否则连接不上 Moon

4. 生成 moon 文件

> sudo zerotier-idtool genmoon moon.json

执行该命令后，`/var/lib/zerotier-one`目录下会生成一个类似`000000xxxxx.moon`的文件

5. 使`moon`配置文件生效:

在`/var/lib/zerotier-one`目录下，新建一个`moons.d`文件夹，并将刚生成的`moon`配置文件放到该文件夹下

```bash
mkdir moons.d
mv 00000018fasd2319.moon moons.d/
```

6. 重新启动 moon 服务器。由于使用命令安装时会自动注册为服务，所以可以依靠以下命令完成启动或重启

```bash
sudo service zerotier-one restart #服务重启命令
```

经过以上配置,服务器上的`moon`即配置并应用完毕。

7. 客户端连接并使用服务器上的 Moon

首先找到自己的`moon`的`id`：

```bash
grep id /var/lib/zerotier-one/moon.json | head -n 1
```

在客户端执行：

```bash
zerotier-cli orbit <id> <id> # 注意要把id写两遍
```

~~直接在 zerotier 目录下,创建 moons.d 文件夹,并且将生成的 000000xxxxxxxx.moon 文件拷入,并重启服务即可~~

8. 验证

在服务器上执行`zerotier-cli listpeers`，发现多了一个`LEAF`节点，配置成功。
