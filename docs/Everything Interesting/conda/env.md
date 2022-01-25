#### 问题

听说`Anaconda`很好用, 但是更新了`2020.11`版本后启动`python`总是报一些警告, 比如在`powershell`里启动`Python`的时候:

```
Warning:
This Python interpreter is in a conda environment, but the environment has
not been activated.  Libraries may fail to load.  To activate this environment
please see https://conda.io/activation
```

所以`conda`环境是啥?

还有就是用`pip`安装需要的包的时候会被告知"目录没有写权限", 这是为啥?

总而言之, 都是`conda`环境的原因.

首先, 需要在`powershell`里启用`conda`环境:

```powershell
conda init powershell
```

此时`powershell`提示符从`PS E:\>`变成了`(base) PS E:\>`, 这说明我们成功进入了`conda`的`base`环境. 此时再启动`Python`就没有警告了.

但是, 在输入`conda install tensorflow`之后又报了一个`inconsistent`错误, 大意是`tensorflow`及其依赖包与现有环境中的 4 个包不相容, 这就需要一个全新的`conda`环境,

不急, 先把镜像配上, ~~省得下载速度把自己感动到脱发~~, 具体操作见https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
conda clean -i
```

重启`shell`, 键入以下命令生成新的`conda`环境

```powershell
conda create -n tensorflow python=3.8 tensorflow
```

`conda`会在安装目录下生成`envs\tensorflow`目录并开始下载`python`并安装`TensorFlow`及各项依赖. 注意这里的前一个`tensorflow`只是一个环境名, 后一个`tensorflow`才是真正要装的包. 如果你正确配置了镜像的话下载应该是很快的. 如果`conda`下载速度很慢, 请花些时间检查一下镜像配得对不对, 会给你省很多等待的时间和失败的次数.

不要把`python`版本设得太高, 也不要用`python 2.x`了.

**注意**这里又是一个大坑, 如果在安装`Anaconda`的时候勾选了了`Add to PATH`选项, 现在`conda`就不会正常工作了. 具体说来, `Add to PATH`向`PATH`中添加的是`base`环境的包, 而当使用`conda activate tensorflow`切换到`tensorflow`环境之后使用`python`命令, 优先解析`PATH`中的`python`(即`Anaconda`目录下的亦即`base`环境中的`python`) 而不是`tensorflow`环境下的`python`, 此时, 如果`base`里没有`tensorflow`, 那么`import tensorflow`就会找不到`module`, 这就是开头的警告产生的原因.

接下来再在新环境中安装一些开发用的包吧.

```powershell
conda install jupyterlab spyder
```

最后, 尽善尽美, 让`powershell`一打开就进入`tensorflow`环境:

```powershell
echo "conda activate tensorflow" >> $profile
```

总结:

1. 每次操作都要重启`shell`以使操作生效.

此外:

1. 可以考虑`Miniconda`, 毕竟`Anaconda`里有那么多包用不上, 而且换个环境还要重新装包. 不过`Miniconda`推荐给有经验的同学.

2. `conda`想重命名环境, 怎么办?

```
conda create --name newname --clone oldname
conda remove --name oldname --all
```
