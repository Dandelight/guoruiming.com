# 使用 certbot 获取证书并部署到阿里云 SAE next

```shell
brew install certbot
certbot certonly --manual
```

会生成几个证书文件，分别是：

```plaintext
README        cert.pem      chain.pem     fullchain.pem privkey.pem
```

其中，

- `fullchain.pem` 是证书 PEM 文件
- `privkey.pem` 是私钥 PEM 文件

但注意，阿里云上传证书，私钥的第一行需要是

```plaintext
-----BEGIN RSA PRIVATE KEY-----
```

而生成的 `privkey.pem` 第一行是

```plaintext
-----BEGIN PRIVATE KEY-----
```

简单，把 `RSA` 三个字加上。注意 `END` 那里也要加上。

设置好证书之后就可以随便 CNAME 了。
