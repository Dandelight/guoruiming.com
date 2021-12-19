前往[ImageNet](https://image-net.org/)官网下载 ILSVRC 2012 数据集（需要使用学校的邮箱注册账号并同意使用协议）

下载之后，

递归解压文件

```
cd /path/to/your/dataset/train
tar -xf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
# 递归解压，看看如何实现才能不必将源文件删掉
# 提醒：执行脚本前一定要看一下脚本做了什么！否则，哭都没处哭去
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

解压验证集并移动至子文件夹

```
cd /path/to/your/dataset/val
tar -xvf ILSVRC2012_img_val.tar
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
# 或者
# wget -qO- https://files-cdn.cnblogs.com/files/luruiyuan/valprep.sh | bash
```

---

如何在真实的深度学习环境中测试自己的 GPU？

https://lambdalabs.com/gpu-workstations/vector

关于在`Docker`中训练使用`num_worker=4`导致如下错误：

```
ataLoader worker (pid 8639) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
```

解决方案：https://github.com/pytorch/pytorch/issues/2244

```
Okay. I think I solved it. Looks like the shared memory of the docker container wasn't set high enough. Setting a higher amount by adding --shm-size 8G to the docker run command seems to be the trick as mentioned here. Let me fully test it, if solved I'll close issue.
```

除了命令行之外，我们通常使用`docker-compose`部署自己的服务。`docker-compose`中这样设置：

If you're using docker-compose, you can set the `your_service.shm_size` value if you want your container to use that /dev/shm size when _running_ or `your_service.build.shm_size` when _building_.

Example:

```
version: '3.5'
services:
  your_service:
    build:
      context: .
      shm_size: '2gb' <-- this will set the size when BUILDING
    shm_size: '2gb' <-- when RUNNING
```

https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container
