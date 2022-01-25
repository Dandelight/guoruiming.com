最近发现，除了`ufoym/deepo`这个镜像，还有[NVidia NGC](https://catalog.ngc.nvidia.com/)项目提供了 PyTorch 和 TensorFlow 的镜像，详见：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

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
