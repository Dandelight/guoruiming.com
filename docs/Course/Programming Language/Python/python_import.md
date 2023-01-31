最近多了一些应用和思考，总结如下。

首先定义 `Python` 下 **namespace** 的概念。

namespace 是一个对象，代表一个运行环境，所有在该环境中创建的对象都会称为该 namespace 的一个属性。

Python 有三个 namespace：`builtin`、`global`、`local`。

namespace 不仅可以访问，还可以修改，修改 namespace 就相当于创建/修改该环境中的变量。

说回本文，每次 `import foo` 的时候，Python 会寻找 `foo.py` 这个 文件/包 并**在一个单独的 namespace 中**执行之。

所有在这个 namespace 中产生的对象，都将成为 `foo` 的方法，`foo` 成为一个包。

很容易想到，如果一个叫 `foo.py` 的包里有一句 `import foo`，会不会递归爆栈？

不会。

因为 Python 会缓存已经导入过的 `module`，只会执行一次。

所以 `foo.py` 会被当成主脚本运行一次，当做 `module` 运行一次。

那么想到：如果在 `a.py` 中 `import b`，在 `b.py` 中 `import c`，再在 `a.py` 中 `import c`，那么 `c` 会被 `import` 几次？

一次。

---

**只记是什么，不问为什么。**

转载自：https://zhuanlan.zhihu.com/p/156774410

`import`绝对是我们在使用 python 时最常用的语句之一了，但其实关于`import`，需要注意的地方还真不少，如导入第三方库，导入自己写的库，导入相对路径下文件中的方法，在包内部的相对与绝对导入等导入源；有导入的顺序；有 Lazy Load 惰性导入方法；有已经导入后的重载等等。本文就旨在讲清楚这些问题，并提供足够的示例以供参考。

## `import` 已安装的第三方库

```python
import <PKG>
import <PKG> as <ABBR>
from <PKG> import <SUBMODULE>
from <PKG> import *
from <PKG>.<SUBMODULE> import *
```

## `import` 相对路径下的文件（非 package）

只能引用本目录下的，子目录，孙目录等下的文件，不能引用父目录中的内容。

### √ 以下是正确的：

```python
import <FILE_STEM>
from <FILE_STEM> import <METHOD>
from <DIR>.<FILE_STEM> import <METHOD>
from <DIR1>.<DIR2>.<FILE_STEM> import <METHOD>
```

### **× **以下是错误的**：**

```python
import .<FILE_STEM>
from .<FILE_STEM> import <METHOD>
from . import <FILE_STEM>
from .. import <FILE_STEM>
```

### 当希望 `import` 非这些路径下的文件时：

先把这些文件的父文件夹 append 到`sys.path`中，然后直接`import`它们的名字。

```python
import sys
sys.path.append(<TARGET_PARENT_PATH>)
import <FILE_STEM>
```

## 在 package 内部`import`包相对路径下的文件

包其实可以看作是一种特殊的模块。例如常规包的目录中需要包含 `__init__.py` 文件，当包被导入时，该文件的顶层代码被隐式执行，就如同模块导入时顶层代码被执行，该文件就像是包的代码一样。所以 **包是一种特殊的模块**。需要记住的是，**所有的包都是模块，但不是所有的模块都是包**。包中子包和模块都有 `__path__` 属性，具体地说，任何包含 `__path__` 属性的模块被认为是包。所有的模块都有一个名称，类似于标准属性访问语法，子包与他们父包的名字之间用点隔开。

所有含有**包内引用**的脚本都**不能**直接被运行（`python <name>.py`），而**只能作为包的一部分**被**导入包外部的其他文件中**使用（如`from mlib.xxx.xxx import xxx`），或者**作为包的一部分运行**（如`python -m ./mlib/utils/test.py`。这里以包名字为`mlib`为例：

### **绝对路径引用**（包外脚本调用）

```python
import mlib.<FILE_STEM>
import mlib.<DIR>.<FILE_STEM>
from mlib.<FILE_STEM> import <METHOD>
```

### 相对路径引用（包内脚本调用）

```python
import .<FILE_STEM>
import ..<FILE_STEM>
import ..<DIR>.<FILE_STEM>
from .<FILE_STEM> import <METHOD>
from .<DIR>.<FILE_STEM> import <METHOD>
from ..<DIR>.<FILE_STEM> import <METHOD>
```

### × 错误引用

```python
import <FILE_STEM>
from <FILE_STEM> import <METHOD>
```

### 若想运行包内某个含有包引用的（相对或绝对都算）脚本：

1. 首先，不论如何，你不能试着在包内部目录下运行这个脚本。也就是说，如果你的包叫`mlib`，那你需要先在命令行中`cd`到其外部文件夹，只要不在包内，其他哪里都行。
2. `python -m <SCRIPT_PATH>`， 如：`python -m ./mlib/utils/test.py`。`-m` flag 允许了用户运行包内部的 python 脚本。
3. 但注意，即使是这样，依然有一定可能出现相对导入的问题，这个要视具体情况而定。

## Lazy Load

如果你不一定会用到某个模块，但后续开发时候又很可能会用到他们，而你既不想每次都手动`import`， 又不想一次性导入一大堆可能使用的 package，有没有解决方案？

还真有！这时候 lazy load 将是一个很好的选择。

下面是一份 TensorFlow 中包含的 Lazy Load 的代码。它可以做到并不真正`import`一个包，但在用户真正调用该包、该包的子模块，或是使用`Tab`自动补全时候把它真正导入。

### 代码

```python
import importlib
import types


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `contrib`, and `ffmpeg` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
```

### 使用方法：

```python
from <PATH> import LazyLoader
os = LazyLoader("os", globals(), "os")
op = LazyLoader("op", globals(), "os.path")
np = LazyLoader("np", globals(), "numpy")
```

或是如果你想更加优雅地一次性导入多个包而不用写 N 行重复代码：

```python
_import_dict = {
    "os": "os",
    "sys": "sys",
    "time": "time",
    "math": "math",
    "yaml": "yaml",
    "random": "random",
    "op": "os.path",
    "np": "numpy",
    "pd": "pandas",
    "pkl": "pickle",
    "glob": "glob",

    "pf": "mlib.file.path_func",
    "lang": "mlib.lang"
}

for key, value in _import_dict.items():
    exec(f"{key}=LazyLoader('{key}', globals(), '{value}')")
```

你可以自定义你常用的一些模块和它们的简称，然后每次直接调用这份代码即可做到迅速而无痛`import`。

此部分参考了[Lazily Importing Python Modules](https://wil.yegelwel.com/lazily-importing-python-modules/)。

## **Re-import**

如果你已经 load 了一个模块，但是由对这个模块本身的代码做出了修改，此时你也许就需要`reload`了，尤其是在`jupyter`环境下，这将是非常有用的功能。

```python
import <PKG>
from importlib import reload
reload(<PKG>)
```

`jupyter` 还提供了拓展包 `autoreload`，可以监听文件变化并自动 `reload`。

```python
# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
```

## 在代码中通过包的字符串名称导入包

`__import__`或`importlib.__import__`都可以完成该任务，二者完全等价，但根据 python 官方文档建议，个人在代码中最好不要使用这个函数，而是使用其替代品`importlib.import_module(name)`。

### `__import__`

```text
__import__(name[, globals[, locals[, fromlist[, level]]]])
```

### 参数介绍：

- name (required): 被加载 module 的名称
- globals (optional): 包含全局变量的字典，该选项很少使用，采用默认值 global()
- locals (optional): 包含局部变量的字典，内部标准实现未用到该变量，采用默认值 - local()
- fromlist (Optional): 被导入的 submodule 名称
- level (Optional): 导入路径选项，Python 2 中默认为 -1，表示同时支持 absolute import 和 relative import。Python 3 中默认为 0，表示仅支持 absolute import。如果大于 0，则表示相对导入的父目录的级数，即 1 类似于 ‘.’，2 类似于 ‘..’。

### 使用示例：

```python
# import spam
spam = __import__('spam')

# import spam.ham
spam = __import__('spam.ham')

# from spam.ham import eggs, sausage as saus
_temp = __import__('spam.ham', fromlist=['eggs', 'sausage'])
eggs = _temp.eggs
saus = _temp.sausage
```

### `import_module`

`importlib.import_module`(_name_, _package=None_)

它最大的优点是方便，易于控制，与常见的`import`语法几乎完全一致，且支持绝对和相对`import`。例如：`basic=importlib.import_module(".utils.basic", "mlib")`。注意当`name`为相对路径时，`package`需要指定其父模块。

> Import a module. The _name_ argument specifies what module to import in absolute or relative terms (e.g. either `pkg.mod` or `..mod`). If the name is specified in relative terms, then the _package_ argument must be set to the name of the package which is to act as the anchor for resolving the package name (e.g. `import_module('..mod', 'pkg.subpkg')` will import `pkg.mod`).
> The `import_module()` function acts as a simplifying wrapper around `importlib.__import__()`. This means all semantics of the function are derived from `importlib.__import__()`. The most important difference between these two functions is that `import_module()` returns the specified package or module (e.g. `pkg.mod`), while `__import__()` returns the top-level package or module (e.g. `pkg`).
> If you are dynamically importing a module that was created since the interpreter began execution (e.g., created a Python source file), you may need to call `invalidate_caches()` in order for the new module to be noticed by the import system.

## Import 的顺序

**加载 python 时默认导入的标准库 > 同级目录下的文件(程序根目录) > PYTHONPATH 环境变量设置的目录 > 标准库 > 第三方库**

关于第一个“加载 python 时默认导入的标准库 ”，可以参见[这篇文章](https://segmentfault.com/q/1010000017357057)中的解释。
