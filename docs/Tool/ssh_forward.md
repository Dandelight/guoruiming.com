将一台没有公网的服务器的 22 端口映射到 `example.com` 的 54216 端口。

```shell
ssh -CNfR 54216:localhost:22 username@example.com
```
