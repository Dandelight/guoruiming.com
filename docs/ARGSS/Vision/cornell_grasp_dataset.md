# Cornell Grasp Dataset

描述该数据集的文章https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

## 数据集获取

[官方的下载地址](http://pr.cs.cornell.edu/deepgrasping)连它的域名一起挂掉了，只有[Kaggle 上可以下载到](https://www.kaggle.com/oneoneliu/cornell-grasp)。

## 数据格式

数据集文件夹组织如下：

```
.
├─depth_image
| ├─pcd0100.txt
| ├─pcd0101.txt
| ...
| └─pcd1034.txt
├─image
| ├─pcd0100r.png
| ├─pcd0101r.png
| ...
| └─pcd1034r.png
├─neg_label
| ├─pcd0100cneg.txt
| ├─pcd0101cneg.txt
| ...
| └─pcd1034cneg.txt
└─pos_label
  ├─pcd0100cpos.txt
  ├─pcd0101cpos.txt
  ...
  └─pcd1034cpos.txt
```

`depth_image`文件夹下保存有`PCD v.7`格式的点云信息，`image`文件夹下是正常的`RGB`图片，`neg_label`是……是啥我也不清楚。

## 数据格式

### 深度信息

深度信息采用`Point Cloud Data file format v.7`编码，详细信息见[]

以下，拿出`depth_image/pcd0100.txt`作个分析

```pointcloud
# .PCD v.7 - Point Cloud Data file format
FIELDS x y z rgb index
SIZE 4 4 4 4 4
TYPE F F F F U
COUNT 1 1 1 1 1
WIDTH 253674
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 253674
DATA ascii
1924.064 -647.111 -119.4176 0 25547
1924.412 -649.678 -119.7147 0 25548
1919.929 -650.5591 -116.9839 0 25549
1920.276 -653.1166 -117.2799 0 25550
...以下每行五个数，略。
```

可以看到本文件很明显地分成三个部分：

#### 第一部分：版本号

```
# .PCD v.7 - Point Cloud Data file format
```

很明显这一行是在标识版本为`0.7`

#### 第二部分：头信息

```
FIELDS x y z rgb index
SIZE 4 4 4 4 4
TYPE F F F F U
COUNT 1 1 1 1 1
WIDTH 253674
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 253674
DATA ascii
```

这一部分构成了数据的定义，逐行分析如下：

`FIELD`：以后的每一行都有什么域，相当于电子表格中的表头

`SIZE`：该`field`中每一列的数据占用的字节数

`TYPE`：每列数据的数据类型：`Float`、`Integer`或者`Unsigned integer`.

`COUNT`：每个`field`的数据数，也就是对应多少列

`WIDTH`和`HEIGHT`：如果`height`为 1，则`width`为数据点的数目；否则，两个数是正常的图片宽高。

`VIEWPOINT`：视点，观测者的位置，以`translation (tx ty tz) + quaternion (qw qx qy qz)`的形式表达，默认值是`0 0 0 1 0 0 0`。

`POINTS`：数据点数目

`DATA`：有三种顾名思义的取值：`ascii`、`binary`、`binary_compressed`

#### 第三部分：数据

上边讲的都是瓶子，下面要给您上酒了，您请好：

```
1924.064 -647.111 -119.4176 0 25547
1924.412 -649.678 -119.7147 0 25548
1919.929 -650.5591 -116.9839 0 25549
1920.276 -653.1166 -117.2799 0 25550
```

好吧，很多时候瓶子比酒有意思，that's life。

---

聊了这么多，下面就是字符串处理时间了？

别想了，自己做字符串处理是最费力不讨好的事情。想一想，这么有影响力的数据格式，怎么可能没有专门**进行读写的库**呢？

[`MATLAB pcread`](https://ww2.mathworks.cn/help/vision/ref/pcread.html)只支持`PCD 0.7`，有`pcread`就有`pcwrite`。

`Python`有`open3d`库做三维图，还有`pypcd`（停更了）

### `MATLAB`中处理`pcd`文件

`MATLAB`中存在一系列以`pc`开头的一系列命令

[`pointCloud`](https://ww2.mathworks.cn/help/vision/ref/pointcloud.html) | [`pcplayer`](https://ww2.mathworks.cn/help/vision/ref/pcplayer.html) | [`pcshow`](https://ww2.mathworks.cn/help/vision/ref/pcshow.html) | [`pcwrite`](https://ww2.mathworks.cn/help/vision/ref/pcwrite.html) | [`pcmerge`](https://ww2.mathworks.cn/help/vision/ref/pcmerge.html) | [`pcfitplane`](https://ww2.mathworks.cn/help/vision/ref/pcfitplane.html) | [`planeModel`](https://ww2.mathworks.cn/help/vision/ref/planemodel.html) | [`pctransform`](https://ww2.mathworks.cn/help/vision/ref/pctransform.html) | [`pcdownsample`](https://ww2.mathworks.cn/help/vision/ref/pcdownsample.html) | [`pcdenoise`](https://ww2.mathworks.cn/help/vision/ref/pcdenoise.html) | [`pcregistericp`](https://ww2.mathworks.cn/help/vision/ref/pcregistericp.html)

`MATLAB`范例如是说：

```matlab
ptCloud = pcread('teapot.ply');
pcshow(ptCloud);
```

在`MATLAB`中，点云处理属于于`Computer Vision Toolbox`的一部分，详见https://ww2.mathworks.cn/help/vision/point-cloud-processing.html

`pcread`可读取`ply`或`pcd`格式的点云文件，但要求点云文件以`ply`或`pcd`为后缀名；`pcwrite`亦可输出`ply`或`pcd`文件。

在`Cornell Grasping Dataset`中，数据集作者提供了背景，我们可以通过减除背景的方式得到物体的图像。注意减除背景之后可能得到$0$值，可能需要进行特殊处理。
