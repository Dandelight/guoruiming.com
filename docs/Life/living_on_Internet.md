# 活在互联网上

如题，我是一个每天高强度上网的人，可以说除了吃饭睡觉走路，都生活在网上。

## 一些工具

### 命令行工具

- `homebrew`: macOS 的软件管理工具
- `CocoaPods`: Swift/Objective-C 的包管理工具

### VS Code

平台：Windows、macOS、Linux

### IntelliJ IDEA

不用介绍了吧，代码静态分析的神，动态调试也很好用。

## Mac 上软件的替代品

mac 定义了一套刚开始感觉很难用，但习惯了之后特别愉悦的流程，所以我把用得不爽的原生 APP 都替换了

### Raycast 替代 Spotlight

Raycast 可以基于 node 编写扩展，但是代码包本身却并不大。可惜了，只支持 macOS。

### QSpace Pro 替代 Finder

设置里甚至可以将 Reveal in Finder 设置为使用 QSpace Pro

### AltTab 替代原生 Alt+Tab

当我在 macOS 上使用原生 Alt+Tab 时，我有两大痛点：

1. 显示窗口截图。Windows 10 已经做到了，macOS 没跟进啊。
2. macOS 从设计之初就不是以窗口区分 APP，而是以文档区分 APP，这么多年从没改过。所以假设打开了两个 Word，Alt+Tab 里只有一个 Word 的图标。

AltTab 解决了这两个痛点，真的很好用。

### `Royal TSX`

Windows 上用惯了 `MobaXTerm`（超级强大的网络工程师必备软件），但是 mac 上没有对标软件。后来发现 `Royal TSX` 勉强够用，虽然免费版只支持 10 个连接，但勉强够用。

## mac 专用

### `Texifier`

`Texifier` 是一个 `LaTeX` 写作环境。相较于 `Overleaf` 以及自建的 `VS Code` 写作环境，其集成的 `texpadTex` 的巨大优势是实时响应变化，正如官网上所述，

> Research is hard. Writing should be easy.

`VS Code` 里的 `LaTeX.js` 可以勉强做到类似效果，但是体验不如 `Texifier`。

### `Arc`

一款被吹上天的浏览器，但实际使用体验一般。对于最核心的使用场景——浏览，没啥优化，优化的都是边边角角，还和我多年的使用习惯不符（比如 `ctrl+tab` 不是跳转到下一个 tab，而是跳转到刚刚打开的标签页；`ctrl+s` 也不是复制了……

### HiddenBar + iBar

我觉得苹果推出刘海屏时，也没想清楚，如果 APP 太多被刘海挡住了怎么处理。HiddenBar 在外接屏上使用，iBar 在没用外接屏的时候用。

### 外设管理类

- `MonitorControl`：外接显示屏亮度控制
- `stats`: 在状态栏上显示统计信息

### 磁盘清理

- `Tencent Lemon` ：磁盘清理（还算好用）
- `CleanMyMac` ：磁盘清理
- `Disk Space Analyzer` 可视化查看文件大小、磁盘占比

### 其他

- 键指如飞：快捷键管理
- 全能解压：压缩、解压、局域网分享
- `AnyGo`：`iPhone` 免越狱改定位

### Time Machine

系统自带，还算方便，我 1T 多数据，备份完只有 500M 的包。换机非常好用，因为我 mac 的硬盘是换过的，所以还是有必要保存备份。

![image-20240117143305876](./assets/living_on_Internet/image-20240117143305876.png)

### ~~华强北~~

并不推荐每个人都这么折腾自己的电脑，但作为 Flutter 开发者，为了打包出两个各不到 100MB 的安装包，我需要安装近 50GB 的软件（IntelliJ IDEA + Android Studio + Android SDK + Maven、XCode + Cocoapods），256 GB 真的不够用，所以花 1400 送到华强北改成了 2T 硬盘。但，问题不是一点没有，电脑有的时候掀起屏幕，屏幕还是黑的，只能强制重启。
