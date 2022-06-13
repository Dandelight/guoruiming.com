最近发现，除了`ufoym/deepo`这个镜像，还有 [NVidia NGC](https://catalog.ngc.nvidia.com/) 项目提供了 PyTorch 和 TensorFlow 的镜像，详见：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

其实 `ufoym/deepo`，`tensorflow/tensorflow`  等都是基于 NGC 的镜像。

不想看就直接 pull：

```bash
docker run --gpus all --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:xx.xx-py3
```

> - `-it` means run in interactive mode
>
> - `--rm` will delete the container when finished
>
> - `-v` is the mounting directory
>
> - `local_dir` is the directory or file from your host system (absolute path) that you want to access from inside your container. For example, the `local_dir` in the following path is `/home/jsmith/data/mnist`.
>
>   -v /home/jsmith/data/mnist:/data/mnist
>
>   ```
>     If you are inside the container, for example, `ls /data/mnist`, you will see the same files as if you issued the `ls /home/jsmith/data/mnist` command from outside the container.
>   ```
>
> - `container_dir` is the target directory when you are inside your container. For example, `/data/mnist` is the target directory in the example:
>
>   ```
>     -v /home/jsmith/data/mnist:/data/mnist
>   ```
>
> - `xx.xx` is the container version. For example, `20.01`.
>
> - `command` is the command you want to run in the image.
>
> - Note: DIGITS uses shared memory to share data between processes. For example, if you use Torch multiprocessing for multi-threaded data loaders, the default shared memory segment size that the container runs with may not be enough. Therefore, you should increase the shared memory size by issuing either:
>
>   ```
>        --ipc=host
>   ```
>
>   or
>
>   ```
>        --shm-size=
>   ```
>
>   See **`/workspace/README.md`** inside the container for information on customizing your PyTorch image.

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
