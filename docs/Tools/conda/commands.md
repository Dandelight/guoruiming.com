# `conda`与`pip`常用命令

```shell
#查看虚拟环境
conda info -e#
创建虚拟环境
conda create -n your_env_name python=your_python_version
#删除虚拟环境：
conda remove -n your_env_name --all
#进入指定环境：
Conda activate your_env_name
#退出指定环境：
Conda deactivate your_env_name
#创建新环境想克隆部分旧的环境：
conda create -n your_env_name --clone oldname
#指定环境安装模块包：
conda install --name your_env_name package_name
#删除指定环境中的某个模块包：
conda remove --name your_env_name  package_name
#导出环境的配置，方便在其它地方部署相同环境：
conda env export >> environment.yml
#导入环境配置，部署相同环境：
conda env create -f environment.yml
```

安装模块包下载速度过慢或安装失败问题可使用国内 conda 源加速。

**国内知名 conda 源**：

- 清华开源软件镜像网站：https://mirror.tuna.tsinghua.edu.cn/
- 中科大开源软件镜像：https://mirrors.ustc.edu.cn/anaconda
- 阿里开源软件镜像：https://opsx.alibaba.com/mirror

```
# 切换清华源:
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/# 搜索时显示通道地址conda config --set show_channel_urls yes
```

```
#切换中科大源:
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/# 搜索时显示通道地址 conda config --set show_channel_urls yes
```

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

## pip 常用命令:

```shell
#安装模块包：
pip install package_name
#查看某个已经安装包：
pip show --files package_name
#升级模块包：
pip install --upgrade package_name
#卸载模块包：
pip uninstall package_name
#导出环境依赖包，requirements.txt记录项目所有的依赖包及其版本号，以便在其他的环境中部署：
pip freeze >> requirements.txt
#部署工程中requirements.txt依赖包:
pip install -r requirements.txt
#模块包下载速度慢或失败可切换包源下载：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
#忘记pip指令参数可用help，这是一个好的方式
pip --help
```

- 豆瓣(douban) https://pypi.douban.com/simple
- 清华大学 https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云 https://mirrors.aliyun.com/pypi/simple
- 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple
- 百度 https://mirror.baidu.com/pypi/simple
