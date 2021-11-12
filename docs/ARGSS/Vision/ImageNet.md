前往[ImageNet](https://image-net.org/)官网下载ILSVRC 2012数据集（需要使用学校的邮箱注册账号并同意使用协议）

下载之后，

递归解压文件

```
cd /path/to/your/dataset/train
tar -xf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
# 递归解压，看看如何实现才能不必将源文件删掉
# 提醒：执行脚本前一定要看一下脚本做了什么！否则，哭都没处哭去
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

解压验证集并移动至子文件夹

```
cd /path/to/your/dataset/val
tar -xvf ILSVRC2012_img_val.tar
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
# 或者
# wget -qO- https://files-cdn.cnblogs.com/files/luruiyuan/valprep.sh | bash
```

------

如何在真实的深度学习环境中测试自己的GPU？

https://lambdalabs.com/gpu-workstations/vector