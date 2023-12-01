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

如果有多个目标服务器，可以使用 `profile` 属性指定。按照如下文件配置 `sftp.json` 后，

```json
{
  "name": "projectName",
  "username": "dandelight",
  "privateKeyPath": "/Users/dandelight/.ssh/id_ed25519",
  "ignore": ["__pycache__", ".DS_Store", "wandb", "logs", ".git"],
  "uploadOnSave": true,
  "useTempFile": true,
  "openSsh": true,
  "profiles": {
    "Titan 1": {
      "host": "10.244.0.1",
      "remotePath": "/mount/raymond/projectDir"
    },
    "3090 2": {
      "host": "10.244.0.2",
      "remotePath": "/workspace/raymond/projectDir"
    },
    "3090 3": {
      "host": "10.244.0.3",
      "remotePath": "/content/raymond/projectDir"
    }
  }
}
```

然后，<kbd>F1</kbd> 打开 `Command Palette`，执行 `SFTF: Set Profile`，选择一个 profile 之后，再执行 `SFTP: Sync Local -> Remote`。

## Worktree

基于 [`worktree`](https://git-scm.com/docs/git-worktree/zh_HANS-CN) 的模式，分多个目录，每个目录下都有一个 `.git` 文件，其中一个目录下的 `.git` 是完整的 `git` 文件夹，其他的都是一个文本文件，指向 `.git`：类比一下就是一个人，他有很多分身，对每个分身的修改都会影响到本体。

在本体中，我们主要做一些对下游不会产生本质影响的修改，如重构、`fix typo` 等；而在分身中，我们做一些 `task specific` 的修改，同时在实验数据表格中记录 `commit hash`。

对于问题 2，可以使用如下的一个 `infopak.py` 文件打印信息（`print_fn` 是为了在 `DistributedDataParallel` 环境下只在 `rank 0` 打印）。

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

这也是为了一些问题而提出的：

1. 试图把所有代码塞进仓库，尤其是 `tex` 文件和研究用代码文件；还拖了很多 `submodule` 下来
2. 加 `submodule` 就加吧，但为了一时方便，放得到处都是；不光放得到处都是还乱改，最后一个项目拖了十多个 `submodule`s，`VS Code` 的 `Git` 插件都要罢工了。
3. 数据乱放，不知道该放哪，统统提交到仓库里吧，仓库瞬间增肥十几兆。

关于 Submodule, Git submodules are a powerful tool, but they can also be tricky to work with. It's best to use submodules for third-party libraries and avoid using them for your own code [1]. Additionally, you should avoid nesting submodules, and when creating a submodule, it's best to strongly name the assembly. When updating a submodule, it's recommended to squash the commits so that only the head commit is included in the main repository. Finally, if possible, try to avoid using submodules altogether and instead use a package manager to install the necessary libraries.

Git submodules are a powerful tool for managing code dependencies. They allow you to keep track of multiple repositories within a single repository, making it easier to manage and maintain code. However, they can be tricky to use and can lead to a lot of confusion if not used properly.

In this article, we’ll discuss 10 best practices for using Git submodules. We’ll cover topics such as how to set up a submodule, how to keep them up to date, and how to avoid common pitfalls. By following these best practices, you can ensure that your code is well-organized and easy to maintain.

1. Use submodules for third-party libraries
   When you use a third-party library, it’s important to keep track of the version that your project is using. This ensures that if there are any changes or updates to the library, you can easily update your project accordingly. With git submodules, you can easily add and commit the library as its own repository within your main project. This way, you can always refer back to the exact version of the library that your project is using.

Additionally, when you use git submodules for third-party libraries, you don’t have to worry about including the entire library in your project. Instead, only the necessary files will be included, which helps reduce the size of your project and makes it easier to manage.

2. Don’t use submodules for your own code
   Submodules are designed to be used for external code, such as third-party libraries or frameworks. When you use submodules for your own code, it can become difficult to keep track of changes and ensure that all the necessary files are included in each commit. This can lead to errors and confusion when trying to deploy your code.

Instead, it’s best to use git branches for your own code. This allows you to easily switch between different versions of your code without having to worry about managing multiple repositories. It also makes it easier to collaborate with other developers since everyone will have access to the same version of the code.

3. Avoid using nested submodules
   Nested submodules can quickly become difficult to manage and maintain. When you have multiple levels of nested submodules, it becomes increasingly hard to keep track of which version of each module is being used in the project. This can lead to unexpected errors or bugs due to mismatched versions.

It’s also important to note that git does not support recursive submodule operations, so if you need to update a nested submodule, you’ll need to manually run the command for each level of nesting.

For these reasons, it’s best to avoid using nested submodules whenever possible. If you do find yourself needing to use them, make sure to document your setup thoroughly so that other developers can easily understand how everything works.

4. Keep the .gitmodules file in sync with the working tree
   The .gitmodules file is a text file that contains the configuration for each submodule. It stores information such as the URL of the repository, the branch to use, and other settings. If this file gets out of sync with the working tree, then git won’t be able to properly track changes in the submodules. This can lead to unexpected behavior and errors when trying to commit or push changes.

To keep the .gitmodules file in sync, make sure you always run `git submodule update` after making any changes to the submodules. This will ensure that the .gitmodules file reflects the current state of the working tree.

5. Commit changes to a submodule only from within that submodule
   When you commit changes to a submodule, the parent repository will not be aware of those changes. This means that if someone else pulls your code and tries to use it, they won’t have access to the latest version of the submodule. To ensure everyone has access to the most up-to-date version of the submodule, make sure to commit any changes from within the submodule itself.

6. Make sure everyone is on the same page
   When you use git submodules, each team member needs to be aware of the changes that have been made in order for them to pull and push their own changes. This means that everyone should be familiar with the structure of the project and how it works.

It’s also important to make sure that all team members are using the same version of the submodule. If someone is using an older version, they may not be able to access the latest features or bug fixes. To ensure this doesn’t happen, it’s best practice to keep track of which versions of the submodule everyone is using.

7. Use git submodule update –remote when possible
   When you use git submodule update –remote, it will fetch the latest changes from the remote repository and merge them into your local copy. This ensures that your local version of the submodule is always up-to-date with the remote version. Without this command, you would have to manually pull in any new changes from the remote repository each time you wanted to update your local version.

Using git submodule update –remote also helps ensure that all developers on a project are working off of the same version of the codebase. This can help reduce conflicts when merging branches or pushing changes.

8. Use tags instead of branches as much as you can
   When you use a branch as the source for your submodule, it can be difficult to keep track of which version of the code is being used. This is because branches are constantly changing and updating, so if you don’t check in regularly, you may end up with an outdated version of the code.

On the other hand, tags provide a snapshot of the code at a specific point in time. This makes them much easier to manage and ensures that you always have access to the exact version of the code you need.

9. Consider alternatives to Git Submodules

Git Submodules can be difficult to manage and maintain, as they require extra steps when cloning or updating repositories. Additionally, submodules are not always the most efficient way of sharing code between projects.

Alternatives such as Git subtrees, which allow you to embed a repository within another one without creating a separate submodule, may be more suitable for your needs. You should also consider using package managers like NPM or Yarn if you need to share code between multiple projects.

10. Automate everything!

When you use git submodules, it’s important to keep track of all the changes that are made in each repository. This can be a tedious and time-consuming task if done manually. Automating this process ensures that all changes are tracked accurately and efficiently.

Automation also helps ensure that your code is always up-to-date with the latest version of the submodule. By automating the process, you don’t have to worry about forgetting to update the submodule or having outdated versions of the code.

Finally, automation makes it easier to manage multiple repositories at once. With automated processes, you can easily set up triggers for when certain events occur, such as when a new commit is pushed to a repository. This allows you to quickly respond to any changes without having to manually check every repository.

1. 10 Git Submodule Best Practices - CLIMB https://climbtheladder.com/10-git-submodule-best-practices/
   Git submodules best practices GitHub https://gist.github.com/slavafomin/08670ec0c0e75b500edbaa5d43a5c93c
   Mastering Git submodules. Hair-pulling. Helpless rage | by ... https://medium.com/@porteneuve/mastering-git-submodules-34c65e940407

Assume I have a repository which is hosted on GitHub. I add another public repository (not belongs to me) as a submodule. I make changes to the submodule and commits it locally. I want to push the commit in submodule to my repository without disturbing the original author of the submodule. How can I do this?

> In order to commit changes to a submodule without disturbing the original author, you can use the git subtree command. This command allows you to add a submodule as a subtree to your repository, which will allow you to commit changes to the submodule without pushing them to the original author. To do this, run the following commands:
>
> ```shell
> git subtree add --prefix=<path> <remote> <branch>
> git commit -am "Commit message"
> git subtree push --prefix=<path> <remote> <branch>
> ```
>
> Replace `<path>` with the path of the submodule, `<remote>` with the remote repository URL, and `<branch>` with the branch name of the submodule. This will add the submodule as a subtree to your repository, commit your changes to the submodule, and push them to your repository without pushing them to the original author.
>
> Git Submodules basic explanation GitHub https://gist.github.com/gitaarik/8735255
> How to Utilize Submodules within Git Repos | by Paige ... https://blog.bitsrc.io/how-to-utilize-submodules-within-git-repos-5dfdd1c62d09
> How to un-submodule a Git submodule? - Stack Overflow https://stackoverflow.com/questions/1759587/how-to-un-submodule-a-git-submodule
