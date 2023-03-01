# 使用 GPG 对提交签名

## 原理

GPG 是 GNU Privacy Guard 的缩写，是一个开源的加密工具，可以用来加密和签名文件。Git 也可以使用 GPG 来对提交进行签名，以验证提交的真实性。其签名算法是著名的 非对称加密算法 RSA。

## 下载

macOS 用户前往 <https://gpgtools.org/>

## 安装

装就行了

## 配置

### 生成密钥

```bash
gpg --full-generate-key
```

### 查看密钥

```bash
gpg --list-secret-keys --keyid-format LONG
```

### 导出公钥

```bash
gpg --armor --export 你的密钥ID
```

### 导出私钥

```bash
gpg --armor --export <邮箱或者指纹字符串或者长密钥 ID> //查看完整的公钥
gpg --armor --export-secret-keys <邮箱或者指纹字符串或者长密钥 ID> //查看完整的私钥
```

### 导入公钥

```bash
gpg --import 公钥文件
```

### 导入私钥

```bash
gpg --allow-secret-key-import --import 私钥文件
```

### 配置 Git

```bash
git config --global gpg.program gpg
git config --global user.signingkey 你的密钥ID
git config --global commit.gpgsign true
```

## GitHub 上配置 GPG

打开 Settings，找到 GPG keys，点击 New GPG key，将公钥粘贴进去，点击 Add GPG key。

## 验证

```bash
git commit -S -m "test"
git push
```

## 参考

* <https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification>
* <https://juejin.cn/post/7047440793920864287>
* 部分内容由 GitHub Copilot 生成
