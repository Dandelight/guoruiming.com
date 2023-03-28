# macOS 下 `defaults` 配置

macOS 下一个统一的配置管理工具是 `defaults`，可以完成系统的诸多配置。具体用法之后再说，这里先列举本人常用的几项配置。

```shell
# 调节按键重复速度
defaults write NSGlobalDomain KeyRepeat -int 1
# 调节按键重复其实速度
defaults write NSGlobalDomain InitialKeyRepeat -int 15

# 显示隐藏文件
defaults write com.apple.finder AppleShowAllFiles -bool true
# 禁止自动生成 .DS_store 文件
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool TRUE
```

修改 <kbd>HOME</kbd> 和 <kbd>END</kbd> 按键为调到行首/行尾

```shell
# https://discussions.apple.com/thread/251108215
mkdir -p $HOME/Library/KeyBindings
echo > $HOME/Library/KeyBindings/DefaultKeyBinding.dict <<- DONE
{
/* Remap Home / End keys to be correct */
"\UF729" = "moveToBeginningOfLine:"; /* Home */
"\UF72B" = "moveToEndOfLine:"; /* End */
"$\UF729" = "moveToBeginningOfLineAndModifySelection:"; /* Shift + Home */
"$\UF72B" = "moveToEndOfLineAndModifySelection:"; /* Shift + End */
"^\UF729" = "moveToBeginningOfDocument:"; /* Ctrl + Home */
"^\UF72B" = "moveToEndOfDocument:"; /* Ctrl + End */
"$^\UF729" = "moveToBeginningOfDocumentAndModifySelection:"; /* Shift + Ctrl + Home */
"$^\UF72B" = "moveToEndOfDocumentAndModifySelection:"; /* Shift + Ctrl + End */
}
DONE
```
