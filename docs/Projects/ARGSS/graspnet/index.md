# GraspNet 通用物品抓取数据集

GraspNet 是一个通用物品抓取的开源项目，现包含

- [GraspNet-1Biliion](https://graspnet.net/datasets.html)：使用平行爪进行抓取
- [SuctionNet-1Billion](https://graspnet.net/suction)：使用吸盘进行抓取

下面就 GraspNet-1Billion 展开介绍

## GraspNet-1Billion

物体抓取计算机视觉领域很具有挑战性的研究课题，也是人工智能可以直接影响现实生活的应用。目前对于简单背景、单个物体的研究精度基本达标，但对于拥挤场景研究较为缺乏。究其原因，一是训练数据不足，而是没有一个统一的评价标准。针对这项问题，GraspNet 论文提供了**一个大规模抓取姿态检测的数据集 GraspNet-1Billion**、**一个统一的评估系统 GraspNet API**以及**一个 baseline 模型**，供研究者研究学习~~快乐打榜~~。

## 数据集

![image-20220130112957734](media/index/image-20220130112957734.png)

GraspNet-1Billion 数据集包含 97280 张 RGB-D 图像和超过 10 亿个抓取姿势（故称 1Billion)。

该数据集包含 97280 张 RGB-D 图像，这些图像取自 190 多个杂乱场景的不同视角，其中每一个场景 512 张 RGB-D 图像（$512\times 190=97280$），对于数据集中 88 个物体提供了精确的 3D 网格模型，每个场景都密集标注物体的 6D pose 和抓取姿态。

数据集中共有 88 种不同物体，从 YCB 数据集中选取适合抓取的 32 个物体，从 DexNet 2.0 中选取 13 个对抗性对象，并收集自己的 43 个对象来构建整个物体集，包括洗发水、塑料盆、纸质包装盒等。每种物体均有准确的 CAD 模型。对每个物体的 CAD 模型，都可以通过受力分析来使用计算机自动进行标注。

数据集一共有$190$个场景，对于每个场景，本文从整个物体集中随机挑选大约 10 个物体，将它们随意地堆在一起。数据集将场景$1\sim100$定为训练集，$101\sim 130$为“见过的“物体，$131\sim160$为”熟悉的”物体，$161\sim 190$为“没见过”的物体。每个场景由**两款相机**各拍摄的**256 张图片**构成，这 512 张图片是一个机械臂上绑了两个相机拍出来的，机械臂沿着固定的轨迹移动，覆盖四分之一球体上的 256 个不同视点，因为运动轨迹是确定的，所以只需要标注第一张图片，其他都可以通过三维空间中的**投影变换**得到；而对于第一张图片，也只需要标注物体的姿态，将场景中的物体与模型对齐后，就可以将 CAD 模型的标注信息**投影到 RGB-D 图像上**了。之后需要进行**碰撞检查**，去掉会产生碰撞的标注。

考虑到所有物体是已知的，标注抓取姿态分为两个阶段来进行，如图 3 所示。首先，对每个物体都要标注抓取姿势。

除此之外，采用解析计算的方法对每个抓取进行评分，力封闭度量在抓取评价中被证明是有效的：给定一个抓取姿态、关联物体和摩擦系数$\mu$，力封闭度量输出二元标签，表示抓取在该系数下是否可以，评分表示为：$s = 1.1 - \mu$。（$\mu$最小值为$0.1$）

通过这两个步骤，我们可以为每个场景生成密集的抓取集$\mathbb{G}_{(w)}$。数据集中正标签和负标签的比例大约是 1:2。[^raywit]

### 训练图片

训练图片共 100 个场景，每个场景由 Kinect 和 RealSense 相机分别拍摄 256 张 RGB-D 图片。一个场景中的信息如下

```
scene_0000
|-- object_id_list.txt              # 场景中的object-id
|-- rs_wrt_kn.npy                   # RealSense相对Kinect相机的位置shape: 256x(4x4)
|-- kinect                          # kinect相机数据
|   |-- rgb
|   |   |-- 0000.png to 0255.png    # 256 rgb images
|   `-- depth
|   |   |-- 0000.png to 0255.png    # 256 depth images
|   `-- label
|   |   |-- 0000.png to 0255.png    # 256 object mask images, 0 is background, 1-88 denotes each object (1-indexed), same format as YCB-Video dataset
|   `-- annotations
|   |   |-- 0000.xml to 0255.xml    # 256 object 6d pose annotation. ‘pos_in_world' and'ori_in_world' denotes position and orientation w.r.t the camera frame.
|   `-- meta
|   |   |-- 0000.mat to 0255.mat    # 256 object 6d pose annotation, same format as YCB-Video dataset for easy usage
|   `-- rect
|   |   |-- 0000.npy to 0255.npy    # 256 2D planar grasp labels
|   |
|   `-- camK.npy                    # camera intrinsic, shape: 3x3, [[f_x,0,c_x], [0,f_y,c_y], [0,0,1]]
|   `-- camera_poses.npy            # 256 camera poses with respect to the first frame, shape: 256x(4x4)
|   `-- cam0_wrt_table.npy          # first frame's camera pose with respect to the table, shape: 4x4
|
`-- realsense
    |-- same structure as kinect
```

`train_1.zip`到`train_4.zip`四个压缩包中是采集的 99 个场景 Ground Truth 及其描述信息，下载下来之后将其中的内容全部解压到一个`graspnet/scenes`文件夹下，解压完成后目录结构如图：

```
|-- graspnet
    |-- scenes
    |   |-- scene_0000/
    |   |-- scene_0001/
    |   |-- ... ...
    |   `-- scene_0099/
```

紧接着是测试图片，其目录结构与训练图片相同，也解压到`graspnet/scenes`文件夹下：

```
|-- graspnet
    |-- scenes
    |   |-- scene_0000/
    |   |-- scene_0001/
    |   |-- ... ...
    |   `-- scene_0189/
```

6 DoF grasp labels 是抓取的标签，需要解压到`graspnet/grasp_label`文件夹下，

## 评价系统

同时，提供的评价系统通过分析计算可以直接反映抓取是否成功，做到不用详尽标注，就可以对任何抓取姿势进行评价。

对每一个预测的 pose$\hat{\mathbf{P}}_i$，通过检查位于 pose 夹爪之间的点云来确定其抓取的是哪个物体。然后，和标注的过程一样，通过力封闭度量（force-closure metric[^force-closure]）和摩擦系数$\mu$确定一个二元的 label。

对于抓取，我们对准确率更加关注，所以以$Precision@k$作为评价指标，$Prescision@k$定义为前$k$个抓取的准确性。$AP_\mu$表示对于$k$从 1 到 50 在摩擦系数为$\mu$的情况下的平均准确率。并且定义$\mathbf{AP}$为$\mu = 0.2$到$1.0$步长$0.2$的平均$AP$。（有点 COCO mAP 的意思）

为了避免全部抓到同一个点上，在评估前会跑一个 pose-NMS（非极大值抑制）。

## baseline 模型

此外，还提出了一个端到端抓取姿态预测网络，点云作为输入，采用解耦的方式来学习 接近方向和操作参数(approaching direction and operation parameters)，例如平面旋转、抓取宽度。

[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-rSnENbmT-1646122082502)(media/graspnet-baseline/image-20220301143028250.png)]

### 抓取姿势表示

作者使用夹爪式机械臂进行抓取，抓取姿势$\mathbf G$表示为

$$
\mathbf{G}=[\mathbf{R}\, \mathbf{t}\, w]
$$

其中$\mathbf R \in \mathbb R^{3\times 3}$标志夹爪的方向，$\mathbf t \in \mathbb R^{3\times 1}$为抓取的重心，$w \in \mathbb R$为夹爪的宽度。但是，$\mathbf R$不是旋转矩阵：因为旋转矩阵必须是正交阵，约束其为正交阵不利于神经网络的训练。具体表示方式参考[^ssd-6d]。

### 网络结构

网络结构分为三部分：ApproachNet、OperationNet、ToleranceNet。

#### Approach Network

Approach Network 预测 approach vector 和可行的抓取点。

##### Base Network

骨干网络为 PointNet++，输入$N \times 3$的点云，输出一个通道数为$C$的特征集。从中使用[^farthest-point]方法采样出$M$个点。

##### Output Head

文章将 approaching vector 视为分类任务，将其分到$V$个预定义的 viewpoints 中。对于每个 point，Approach Network 输出两个值表示抓取成功度。因此 Proposal Generation Network 的输出为$M \times (2 + V)$，$2$是抓取的成功度，$V$是每个 viewpoint 的成功度。

##### Loss Function

对于每一个预测的抓取位姿，没有落在物体上的抓取成功度为$0$。对于此外的点，如果在 5mm 范围内有一个正样本，那么其抓取成功度为$1$。没有找到对应 ground truth 的点被忽略。

对于每一个可行的点采样$V$个 virtual approaching vectors。设在第$i$个 virtual view 下第$j$个可抓取点为$v_{i, j}$。检查其对应的 ground truth $\hat{v}_{i, j}$，只考虑 5 度之内的 reference vector。损失函数如下：

$$
L^A(\{c_i\}, \{s_{ij}\}) = \frac{1}{N_{cls}}\sum_i L_{cls}(c_i, c_i^*) + \lambda_1\frac{1}{N_{reg}}\sum_i \sum_j c_i^* \mathbf 1(|v_{ij}, v^*_{ij}|<5^\circ)L_{reg}(s_{ij}s^*_{ij})
$$

#### Operation Network

获取 approaching vector 后，模型进一步 1)预测平面内的转动、2)approaching distance、3)夹爪宽度和 4)抓取置信度。

##### Cylinder Region Transformation

模型对每一个 candidate grasp 建立一个统一的表达。因为 approaching distance 不是很敏感，所以我们将其分 bin。对于给定的距离$d_k$，将 approaching vector 附近的圆柱体中包含的点云采样出来。为了便于学习，通过转移矩阵$\mathbf O_{ij}$将所有点转换到原点为抓取中心、$z$轴为$\mathbf v_{ij}$的坐标系中。

$$
\mathbf O_{ij} = [\mathbf{o_{ij}^1}, [0, -\mathbf{v_ij}^{(3)}]^\mathbf{T}, \mathbf{V_{ij}}] \\
\text{where}\quad \mathbf{o_{ij}^1} = [0, -\mathbf{V_{ij}}^{(3)}, \mathbf{v_{ij}}^{(2)}]^\mathbf{T} \times \mathbf{v_{ij}}
$$

$\mathbf{v_{ij}}^{(k)}$为$\mathbf{v_{ij}}$的第$k$个元素。

##### Loss function

$$
L^{R}\left(R_{i j}, S_{i j}, W_{i j}\right)=\sum_{d=1}^{K}\left(\frac{1}{N_{c l s}} \sum_{i j} L_{c l s}^{d}\left(R_{i j}, R_{i j}^{*}\right)\right.
+\lambda_{2} \frac{1}{N_{r e g}} \sum_{i j} L_{r e g}^{d}\left(S_{i j}, S_{i j}^{*}\right)
+\left.\lambda_{3} \frac{1}{N_{r e g}} \sum_{i j} L_{r e g}^{d}\left(W_{i j}, W_{i j}^{*}\right)\right),
$$

#### Tolerance Network

在之前的工作之后，端到端的模型已经可以准确预测 grasp 了。更进一步，作者提出了 Grasp Affinity Field 来提升预测的鲁棒性。

对于每一个 ground truth 抓取位姿，在附近的球形空间搜索$s > 0.5$的最远点，将其设为 GAF 的目标，loss 如下：

$$
L^{F}\left(A_{i j}\right)=\frac{1}{N_{r e g}} \sum_{d=1}^{K} \sum_{i j} L_{r e g}^{d}\left(T_{i j}, T_{i j}^{*}\right)
$$

#### 总 loss

加权求和

$$
L=L^{A}\left(\left\{c_{i}\right\},\left\{s_{i j}\right\}\right)+\alpha L^{R}\left(R_{i j}, S_{i j}, W_{i j}\right)+\beta L^{F}\left(T_{i j}\right)
$$

作者把[ArUco](https://www.uco.es/investiga/grupos/ava/node/26)标志贴在物体上来确定物体在相机坐标系内的位置，做了真实的抓取实验。

模型超参数设置：

> For our method, rotation angle is divided into 12 bins and approaching distance is divided into 4 bins with the value of 0.01, 0.02, 0.03, 0.04 meter. We set M = 1024 and V = 300. PointNet++ has four set abstraction layers with the radius of 0.04, 0.1, 0.2, 0.3 in meters and grouping size of 64, 32, 16 and 16, by which the point set is down-sampled to the size of 2048, 1024, 512 and 256 respectively. Then the points are up-sampled by two feature propagation layers to the size 1024 with 256-dim features. ApproachNet, OperationNet and ToleranceNet is composed of MLPs with the size of (256, 302, 302), (128, 128, 36) and (128, 64, 12) respectively. For the loss function, we set λ1, λ2, λ3, α, β = 0.5, 1.0, 0.2, 0.5, 0.1.
>
> Our model is implemented with PyTorch and trained with Adam optimizer [18] on one Nvidia RTX 2080 GPU. During training, we randomly sample 20k points from each scene. The initial learning rate is 0.001 and the batch size is 4. The learning rate is decreased to 0.0001 after 60 epochs and then decreased to 0.00001 after 100 epochs.

[^ssd-6d]: Wadim Kehl, Fabian Manhardt, Federico Tombari, Slobodan Ilic, and Nassir Navab. Ssd-6d: Making rgb-based 3d detection and 6d pose estimation great again. In Proceedings of the IEEE International Conference on Computer Vision, pages 1521–1529, 2017.
[^farthest-point]: Yuval Eldar, Michael Lindenbaum, Moshe Porat, and Yehoshua Y Zeevi. The farthest point strategy for progressive image sampling. IEEE Transactions on Image Processing, 6(9):1305–1315, 1997.
[^force-closure]: Van-Duc Nguyen. Constructing force-closure grasps. The International Journal of Robotics Research, 7(3):3–16, 1988.
[^raywit]: 数据集 2020（一）GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping https://blog.csdn.net/qq_40520596/article/details/107751346
