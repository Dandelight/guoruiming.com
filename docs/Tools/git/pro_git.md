_Pro GIt_ 阅读笔记

# 跟着*Pro Git*重新学一遍`Git`

## 创建仓库

### 自己创建

```bash
cd /path/to/repo
git init
```

## `clone`

```bash
git clone [url]
```

`[url]`选项：

- 本地文件/网络存储：`/srv/repo`
- `HTTP`：分为 smart HTTP 和
- `SSH`：推荐，配置简单，但只有鉴权用户才能使用
- `GIT`：没有鉴权机制，监听端口，只推荐公开访问

```bash
git clone https://github.com/git/git git-fork # 重命名
```

## 添加文件

```bash
git add -- [file]
```

## 查看

### stage 详情

```bash
git status
git status -s # shorter
```

### 修改情况

```bash
git diff
git diff --staged
```

`git diff`本身`diff`的是`staged`和`unstaged`的文件，不会涉及上次提交的文件；`git diff --cached`（或者同义词`--staged`）

或者可以通过设置`git difftool`使用外部工具进行`diff`。

### 提交修改

```bash
git commit
git commit -v # 详细的diff信息
git commit -a # commit前自动stage所有tracked文件
```

### 删除文件

##### 使用`rm`和`git add`

```bash
rm README.md
git add README.md
```

##### 使用`git rm`

```bash
git rm README.md # 硬删除
git rm --staged README.md
```

### 通配符

```bash
git rm log/\*.log # 删除log/文件夹下所有后缀名为.log的文件
```

### 移动文件

```bash
git mv file_from file_to
```

值得注意的是，`git`里并没有`mv`这个状态，而是`rename`。这是通过`git`的文件跟踪机制实现的。

### 查看`commit`历史

```bash
git log
git log -p -2 # -p --patch 显示每次的patch; -2 显示最近两条
git log --stat # 查看简单的统计信息
git log pretty=oneline # --oneline
git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
```

### 后悔药

```bash
git commit --amend
git reset HEAD CONTRIBUTING.md # 把CONTRIBUTION.md文件拉回HEAD提交的样子
git checkout -- <file> # 很危险，会丢失当前修改
git restore --staged <file> # 用于不小心多stage了一个文件的情况，unstage文件但不丢失修改
git restore <file> # Restore a modified file
```

### 分支

每一个`git object`中，除了文件的`snapshot`之外还会有指向父`commit`的指针，可能有零个（`root commit`）、一个、两个及以上（`merge commit`），指针情况可以通过`git log --oneline --decorate`查看。

```bash
git log --oneline --decorate --graph --all
```

#### 创建分支

```bash
git branch new_branch
```

#### 切换分支

```bash
git checkout new_branch
```

#### 注意

在`git 2.23`之后，可以使用`git switch`来：

- 切换到已存在分支：`git switch test-branch`
- 创建分支并切换：`git switch -c new-branch`（或`--create`）
- 切回上一个分支：`git switch -`

### Merge

#### Fast-forward

在两个分支的`tag`在同一主线上时

#### Merge

利用两个`commit`的内容和公共祖先来合并分支
