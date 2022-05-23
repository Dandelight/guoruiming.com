错误：

The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 3B4FE6ACC0B21F32

原因：

更换三方源没有对应的 Key

解决：

直接执行如下命令，比如我没有 `3B4FE6ACC0B21F32`，就需要执行

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
```
