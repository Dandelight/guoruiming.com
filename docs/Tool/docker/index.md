内容大部分来源于**极客时间训练营**，点击链接免费领取视频：<https://u.geekbang.org/subject/intro/1000861>

```shell
docker run
```

* `-it` 交互
* `-d` daemon
* `-p` 端口映射
* `-v` 磁盘映射

```shell
docker pull
```

* 启动已经终止的容器 `docker start`
* 停止容器 `docker stop`
* 查看容器状态 `docker ps`

* 查看容器细节 `docker inspect <containerid>`
* 进入容器
  * `docker attach`
  * `docker exec`
* 通过 `nsenter`：
	```shell
	PID = $(docker inspect --format "{{.State.Pid}}" <containerid>)
	$ nsenter --target $PID --mount --uts --ipc --net --pid	
	```
* 拷贝文件到镜像里

  * `docker cp path <containerid>:/path`

# 构建镜像

```dockerfile
FROM ubuntu
ENV MY_SERVICE_PORT=80
LABEL multi.label1="value1" multi.label2="value2" other="value3"
ADD bin/amd64/httpserver /httpserver
EXPOSE 80
ENTRYPOINT /httpserver
```

> 提问：OCI 是什么？

# 三大关键技术

特性：

* 安全性
* 隔离性
* 便携性
* 可配额

## 资源隔离 Namespace

Linux Namespace 是 Linux Kernel 提供的资源隔离方案

* 系统可以为进程分配不同的 Namespace
* 并保证不同的 Namespace 资源独立分配、进程彼此隔离，即不同的 Namespace 下的进程互不干扰。

```c
struct task_struct {
  /* namespaces */
  struct nsproxy *nsproxy;
}
```

```c
struct nsproxy {
  atomic_t count;
  struct uts_namespace *uts_ns;
  struct ipc_namespace *ipc_ns;
  struct mnt_namespace *mnt_ns;
  struct pid_namespace *pid_ns_for_children;
  struct net *net_ns;
}
```

## Linux 对 Namespace 的操作方法

`fork` 的子进程默认继承父进程的 Namespace

* `clone`：创建新进程时，通过 flag 指定需要新建的 Namespace 类型
* `setns`：将调用进程移到某个已存在的 Namespace 下
* `unshare`：将调用进程移到新的 Namespace 下

| Namespace 类型 | 隔离类型                       | Kernel 最低版本 |
| -------------- | ------------------------------ | --------------- |
| IPC            | System V IPC 和 POSIX 消息队列 | 2.6.19          |
| Network        | 网络设备、网络协议栈、网络端口 | 2.6.29          |
| PID            | 进程                           | 2.6.14          |
| Mount          | 挂载点                         | 2.4.19          |
| UST            | 主机名和域名                   | 2.6.19          |
| USR            | 用户和用户组                   | 3.8             |

## Namespace 常见操作

查看当前系统的 Namespace

```shell
lsns -t net
```

查看某进程的 Namespace

```shell
ls -la /proc/<pid>/ns/
```

在某 Namespace 中执行命令

```shell
nsenter -t 51031 -n ip addr
```

用新的 Namespace 启动一些进程（`-fn` 启动网络进程）

```shell
unshare -fn sleep 120
ps -ef | grep slepp # 获取 pid
tsls -t net | grep <pid>
nsenter -t <ns> -n ip addr
```

可以看见，只有一个 loopback 网络，而 docker 启动的容器是有网络的，这是因为我们没有任何的配置，而 docker 的网络驱动帮我们配置好了网络。

## 资源配额 Cgroups

Control Groups 是 Linux 下用于对一个或者一组进程进行资源控制和监控的机制，可以对诸如 CPU 使用时间、内存、磁盘 IO 等进程所需资源进行限制。**不同资源的具体管理工作由相应的 cgroups 管理**。

Cgroups 以层级的方式组织管理，每个 Cgroups 可以包含子 Cgroups，因此一个 Cgroups 能使用的资源不仅受本 Cgroups 限制，还收到父 Cgroups 限制。

```c
struct task_struct {
  #ifdef CONFIG_CGROUPS
  struct css_set __rcu *cgroups; // Cgroups Subspace State
  struct list_head cg_list;
  #endif
}
```

```c
struct css_set {
  struct grpus_subsys_state *subsys[CGROUP_SUBSYS_COUNT];
}
```

### 可配置项目

Cgroups 实现了对资源的配额与度量

* `blkio` 块设备IO控制
* `cpu`
* `cpuacct`：产生 cgroup 任务的 CPU 资源报告，因为在 Namespace 里执行 `top`、`lscpu`、`free` 都是不准的，看到的都是主机的资源
* `cpuset`：指定在某核心和内存上运行
* `devices`：设备访问
* `freezer`：暂停和恢复 `cgroup` 任务
* `memory`：内存访问限制及资源报告
* `net_cls`：标记每个网络包以供 cgroups 使用
* `ns`：Namespace 子系统
* `pid`：进程标识子系统

### CPU 子系统

```shell
# Ubuntu 23.04
$ cd /sys/fs/cgroups
$ cat cpu.stat
usage_usec 54536000
user_usec 22876000
system_usec 31660000
nr_periods 0
nr_throttled 0
throttled_usec 0
nr_bursts 0
burst_usec 0
```

```shell
echo 62261 > cgroup.procs
echo 100000 > cpu.cfs_quota_us # 100%
echo 10000 > cpu.cfs_quota_us # 10%
```

CPU 是可压缩资源，但不可压缩资源

### Memory 子系统

## Union FS

将不同目录挂在到同一个虚拟文件系统下的文件系统，支持为每一个成员目录设为 readonly、readwrite、whiteout-able 权限。

* `bootfs`
* `rootfs`

Union FS 是一层一层叠加的。

* 写时复制
* 用时分配

OverlayFS 一个联合文件系统

```shell
mkdir upper lower
echo "from lower" > lower/in_lower.txt
echo "from upper" > upper/in_upper.txt
echo "from lower" > lower/in_both.txt
echo "from upper" > upper/in_both.txt
# Ubuntu 23.04 可能改过 API 了……
sudo mount -t overlay overlay -o lowerdir=`pwd`/lower,upperdir=`pwd`/upper,workdir=`pwd`/work `pwd`/merged
```

`work` 是一个临时目录

## 网络

* 容器里面网络怎么配置
* 容器网络和主机如何配置
* 容器网络如何与外界联通

docker 支持多种网络模式

* `Null`，后续自己配置
* `Host`，与主机共用网络
* `Container`：重用其他容器的网络
* `Bridge`：桥接
* `Overlay`
* `Remote`
