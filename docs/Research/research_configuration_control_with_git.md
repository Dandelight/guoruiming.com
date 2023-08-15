# 使用 `Git` 进行科研实验版本控制

版本控制系统（Version Control System），或配置管理系统（Configuration Management System）是软件项目管理的重要组成部分，用于管理软件这种无形资产的增加、删除、修改。计算机学科的科研实验代码也属于软件系统，但科研实验项目相比于工程项目，一些新的需求：

6. 要求实验结果可复现，不能因为换个时间或换台机器就跑不出相同的结果了
7. 单人或小团队维护，一般不超过五人
8. 每次改动都很小，但 `merge`、`rebase` 都很痛苦，因为很难验证 `merge` 对结果的影响
9. 最终得出一个结论

对版本控制系统的需求也更细化：

1. 需要知道每个版本的结果是什么，而不需要重跑一遍
2. 需要知道在两个版本之间修改了什么，以总结出导致结果改变的原因
3. 反复修改，对实验结果进行频繁、反复的修改才能得出结论
4. 对于一些良好的实验结果
5. 控制变量，才能调研出真正有影响的部分

传统的版本控制范式能否满足这些呢（因为如果满足了就没必要创造一个新的了）

## 传统版本控制范式

### PR-merge

PR-merge 是 GitHub 的模式，开发者提出 `issue` 后，从 `HEAD` 中 `branch` 出去，几个 `commit` 解决掉 `issue`，再提出 `Pull Request`，将分支 `merge` 回去。如果管理更严格，`Pull Request` 还需要 Review 和 CI（持续集成）之后才能合入。

显然，即使没有 Review & CI，PR-merge 模式也是行不通的。一方面，研究人员会在同一个 `main` 分支上创建大量 `branch` 进行实验，这些实验大多相互冲突（修改同一部分代码）无法同时 `merge`，最多只能选择一个 `merge`。对代码的公共修改（指与核心逻辑无关，单纯重构）从一堆 `branch` 中提取出来，`merge` 入 `main`，其他分支也难以同步修改，因为很可能出现冲突。

### Upstream first

Upstream first 适合于 Android、Linux 需要多个参与方的项目。上游厂商开发并发布新版本后，下游厂商跟进。如果下游厂商提出了有价值的 `commit`，最好是能提交给上游厂商，而下游厂商通过 `cheery-pick` 来更新自己的代码库。

这种模式并不适合科研用代码的编写，因为科研代码很难做到“一堆优点合在一起都是优点”，所以维护一个 long running 的 `production` 分支并不合适。

## 新的模式

在运行科研代码时，有以下特点：

1. 一份源代码，可能在多台主机上运行，不断变化
2. 需要保存以保证可复现性

`VS Code` 可以使用 [`SFTP` 插件](https://marketplace.visualstudio.com/items?itemName=Natizyskunk.sftp) ，将本地文件同步到服务器（只需要服务器开启 `ssh`，缺点是只能同时同步到一台），配置文件如：

```json
{
  "name": "nameOfServer",
  "host": "192.168.0.1",
  "username": "dandelight",
  "privateKeyPath": "/Users/dandelight/.ssh/id_ed25519",
  "remotePath": "/home/dandelight/workspace",
  "uploadOnSave": true,
  "ignore": ["__pycache__", ".DS_Store", "wandb", "logs", ".git"]
}
```

`ignore` 掉 `.git` 是因为可能有 `Permission denied` 问题。

基于 [`worktree`](https://git-scm.com/docs/git-worktree/zh_HANS-CN) 的模式，分多个目录，每个目录下都有一个 `.git` 文件，其中一个目录下的 `.git` 是完整的 `git` 文件夹，其他的都是一个文本文件，指向 `.git`：类比一下就是一个人，他有很多分身，对每个分身的修改都会影响到本体。

在本体中，我们主要做一些对下游不会产生本质影响的修改，如重构、`fix typo` 等；而在分身中，我们做一些 `task specific` 的修改，同时在实验数据表格中记录 `commit hash`。

对于问题2，可以使用如下的一个 `infopak.py` 文件打印信息（`print_fn` 是为了在 `DistributedDataParallel` 环境下只在 `rank 0` 打印）。

```python
from traceback import print_tb
import warnings
import os
import sys
import subprocess
import socket


def infopak(entry_point, print_fn=None):
    if print_fn is None:
        print_fn = print

    if git_has_uncommitted_changes():
        print_fn(
            """
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@ You have uncommitted git changes. Please commit them  @
@ to maintain reproducibility.                          @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

""",
            file=sys.stderr,
        )

    print_fn(f"========== Task info ==========")
    print_fn(f"Commit Hash: {git_commit_hash()}")
    print_fn(f"Branch: {git_branch()}")
    print_fn(f"Working dir: {os.getcwd()}")
    print_fn(f"Hostname: {get_hostname()}")
    print_fn(f"Entry point: {entry_point}")
    print_fn(f"Command line: {' '.join(sys.argv)}")
    print_fn(f"===============================")


def git_has_uncommitted_changes() -> bool:
    try:
        git_output = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"Possibly not a git repository."
            f"Error message: {e}"
        )
        return False

    return len(git_output.stdout) > 0


def git_commit_hash() -> str:
    try:
        git_output = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        return ""
    return git_output.stdout.decode("utf-8").strip()


def git_branch() -> str:
    try:
        git_output = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        return ""
    return git_output.stdout.decode("utf-8").strip()


def get_hostname() -> str:
    return socket.gethostname()
```