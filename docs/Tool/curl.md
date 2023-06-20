# `cURL` 命令行并行下载

```shell
curl --parallel --parallel-immediate -k -L -C - -o code 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64'
```

参数：

- `--parallel` 并行下载
- `--parallel-immediate` When doing parallel transfers, this option will instruct curl that it should rather prefer opening up more connections in parallel at once rather than waiting to see if new transfers can be added as multiplexed streams on another connection.
- `-k`: `--insecure`
- `-L`: `--location`，允许重定向
- `-C -`: 断点续传，`-C -` 表示从上次下载的位置继续下载
- `-o code`: 输出文件名
