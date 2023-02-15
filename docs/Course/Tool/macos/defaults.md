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

