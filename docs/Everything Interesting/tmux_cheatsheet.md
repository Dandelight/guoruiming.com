# `TMUX` 快捷键对照

`tmux` (Terminal MUltipleXer) 是一款便捷的**终端会话管理工具**。与 `GNU Screen` ；类似，但不同之处在于可以分屏。

**注意：以下所有命令默认以 `tmux` 的子命令，所有快捷键以 <kbd>ctrl</kbd>+<kbd>B</kbd> 开头**（可以通过 `.tmux.conf` 配置为其他快捷键）

| 功能             | 命令              | 快捷键       |
| ---------------- | ----------------- | ------------ |
| **Session 管理** |                   |              |
| 新建             |                   |              |
| 离开             | `detach`          | <kbd>D</kbd> |
| 列表             | `ls`              | <kbd>S</kbd> |
| 关闭             | `kill-session`    |              |
| 切换             | `switch`          |              |
| 重命名           | `rename-session`  | <kbd>$</kbd> |
| **窗格操作**     |                   |              |
| 上下切割窗格     | `split-window`    | <kbd>"</kbd> |
| 左右切割窗格     | `split-window -h` | <kbd>%</kbd> |
