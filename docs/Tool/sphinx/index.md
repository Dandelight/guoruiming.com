# `sphinx`：基于 `Python` 的文档生成工具

## Motivation

对于软件开发来说，**文档**是软件可维护性的重要保障。`sphinx` 是一款文档生成工具，以 `restructuredText` 为标记语言。更重点的是其 `autodoc` 插件支持从 `Python docstring` 生成文档，同一个模块、文件、类或者函数上进行说明，便于维护，并且非常 `Pythonic`。

顺便一提，`sphinx` 取名于埃及的狮身人面像斯芬克斯。

## 安装

基础包可以通过以下两种方式之一安装

```shell
pip install -U sphinx
conda install sphinx
```

安装完成后校验是否安装成功

```shell
 sphinx-build --version
```

优选地，安装 `sphinx-reload` 以实现监听文件变化并自动刷新 `html` 页面。

```shell
pip install sphinx-reload
```

## 使用

`sphinx` 软件包提供了一系列以 `sphinx-` 为前缀的命令行工具。对于项目初始化，我们需要使用 `sphinx-quickstart`。

首先切换到项目目录，然后

```shell
sphinx-quickstart docs
```

会提问几个问题，当前版本 `5.0.2` 主要问这几个问题：

- Separate source and build directories（如果选 `y`，`docs` 下会有两个文件夹 `source` 和 `build`；如果选 `n`，`docs` 下是文档的根目录，编译后文件放在 `_build` 中。
- Project name
- Author name(s)
- Project release
- Project language（自然语言）

按实际情况回答即可。

## 配置

配置主要在 `conf.py` 中

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'Project'
copyright = '2023, Ruiming Guo'
author = 'Ruiming Guo'

extensions = [
    "sphinx.ext.autodoc", # 重点，自动生成
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    'sphinx.ext.mathjax',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'
html_static_path = ['_static']
source_suffix = [".rst"]
```

## 运行

配置 `autodoc` 后，`sphinx-reload` 只会监听 `docs/` 下的文件，如果要监听 `python` 文件中 `docstring` 的变化，需要

```shell
sphinx-reload . --watch ../**/*.py
```

## 编译

```shell
make html
make latex
```
