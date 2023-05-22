# `git hooks` 的设置与跳过

在 `git` 仓库下的 `.git/hooks` 下加上这一句

```shell
#!/usr/bin/env sh
FORMATTER="./node_modules/.bin/prettier"
FILES=$(git diff --cached --name-only --diff-filter=ACMR | sed 's| |\\ |g')

[ -z "$FILES" ] && exit 0

# Prettify all selected files
echo "$FILES" | xargs $FORMATTER --ignore-unknown --write

# Add back the modified/prettified files to staging
echo "$FILES" | xargs git add

exit 0
```

即可实现格式化产生 `diff` 的文件。

但是也会有一些情况需要跳过格式化，比如解决掉一些`prettier`造成的 Markdown 格式问题。

在 `commit`  时增加 `--no-verify`  指令来跳过`git hooks`：`git commit --no-verify`
