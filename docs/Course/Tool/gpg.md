## 因为 GPG 没有找到 Key 而导致 apt update 报错

### 错误

The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 3B4FE6ACC0B21F32

### 原因

更换三方源没有对应的 Key

### 解决

从 `keyserver` 上获取对应的 `key` 即可。比如，假设我的服务器上没有 `3B4FE6ACC0B21F32` 这个密钥，就需要执行

```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
```
