伯克利大学研发的`dex-net`系列宣称对不规则物体的抓取率达到了99%，在`dex-net 4.0`称对不规则物体的抓取成功率达到了95%，对于仓储物流业等需要分拣大量物体的行业来说无疑是个非常重要的技术突破。

`dex-net 1.0`采用多视角卷积神经网络。

![image-20211118020650518](media/dex-net/image-20211118020650518.png)

`dex-net 2.0`采用了`Grasp Quality CNN`，使用卷积神经网络学习图像特征，使用全连接神经网络学习深度特征，

![image-20211118020616463](media/dex-net/image-20211118020616463.png)

`dex-net 3.0`对吸盘建立了模型，其网络结构依然是`GQ-CNN`。

![image-20211118021036457](media/dex-net/image-20211118021036457.png)

`dex-net 4.0`作为集成，提出了`Ambidextrous grasping`，在分别训练的`Parallel Jaw`和`Suction`机械臂头的基础上提出了`Ambidextrous Policy`，从两种机械臂中选取一种抓取成功率更高的进行抓取。

![image-20211118021145462](media/dex-net/image-20211118021145462.png)

至于`GQ-CNN`的代码，虽然说洋洋洒洒几千行，其实大多是给`TensorFlow`填本来该开发者填的坑。`TensorFlow 1`暴露了太多底层`API`，大部分代码都是用来编写自己的类的。我仔细阅读了一下，发现有用的只有下面一段`YAML`：

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

