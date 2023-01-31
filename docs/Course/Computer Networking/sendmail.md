伯克利大学的系列作品`Dex-Net`，号称能够达到 95%的准确率。

`dex-net 4.0`官网在[https://berkeleyautomation.github.io/dex-net/](https://berkeleyautomation.github.io/dex-net/)，训练使用的数据集可以在[这里](https://berkeley.app.box.com/s/6mnb2bzi5zfa7qpwyn7uq5atb7vbztng)下载到，文件极大（训练一个模型 8GB 数据，一共十多个模型），请在网络优良的情况下下载。

---

书归正传，谈一谈`dex-net`论文的阅读体会。

`dex-net 2.0`的主要工作是这个

![image-20211118015402381](image-20211118015402381.png)

一个卷积神经网络

我也不知道为什么`dex-net 4.0`没有把

我觉得可以做一个多 head 的输出。

![image-20211118014501533](image-20211118014501533.png)

所以从这张图来看，`dex-net 4.0`最主要的改进是增加了`Abibdextrous Policy`以确定以何种机械臂执行抓取。从其目的来看很像一个 Ensemble 方法。

至于代码，洋洋洒洒几千行代码，大部分却是在填`TensorFlow`的坑。唯一看来有用的，就只有以下一段定义神经网络的`YAML`文件了：

```yaml
### GQCNN CONFIG ###
gqcnn:
  # basic data metrics
  im_height: 96
  im_width: 96
  im_channels: 1
  debug: *debug
  seed: *seed

  # needs to match input data mode that was used for training, determines the pose dimensions for the network
  gripper_mode: parallel_jaw

  # method by which to integrate depth into the network
  input_depth_mode: im_depth_sub

  # used for training with multiple angular predictions
  angular_bins: 16

  # prediction batch size, in training this will be overriden by the val_batch_size in the optimizer's config file
  batch_size: *val_batch_size

  # architecture
  architecture:
    im_stream:
      conv1_1:
        type: conv
        filt_dim: 9
        num_filt: 16
        pool_size: 1
        pool_stride: 1
        pad: VALID
        norm: 0
        norm_type: local_response
      conv1_2:
        type: conv
        filt_dim: 5
        num_filt: 16
        pool_size: 2
        pool_stride: 2
        pad: VALID
        norm: 0
        norm_type: local_response
      conv2_1:
        type: conv
        filt_dim: 5
        num_filt: 16
        pool_size: 1
        pool_stride: 1
        pad: VALID
        norm: 0
        norm_type: local_response
      conv2_2:
        type: conv
        filt_dim: 5
        num_filt: 16
        pool_size: 2
        pool_stride: 2
        pad: VALID
        norm: 0
        norm_type: local_response
      fc3:
        type: fc
        out_size: 128
      fc4:
        type: fc
        out_size: 128
      fc5:
        type: fc
        out_size: 32

  # architecture normalization constants
  radius: 2
  alpha: 2.0e-05
  beta: 0.75
  bias: 1.0

  # leaky relu coefficient
  relu_coeff: 0.0

```
