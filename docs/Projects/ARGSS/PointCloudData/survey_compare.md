# 开源点云标注工具调研与对比

## 1. `labelCloud`

地址：https://github.com/ch-sa/labelCloud

技术栈：Python + PyQt + Open3D + PyOpenGL

![image-20220129163308085](media/Untitled/image-20220129163308085.png)

较为简单的工具，可以设置点云文件夹和标注文件夹。左上角显示了当前的文件名和下一个文件的文件名；其下方是对标定框的控制；工具有“点击框定”和”扩展“两种标定模式，可以在最下角选取；右侧有标签的详细信息。

代码方面，该工具很好地遵循了”边界-实体-控制“的设计，在标定模式的设置上采取了策略模式，还包含单元测试。

## PCAT_open_source

https://github.com/halostorm/PCAT_open_source

技术栈：ROS rviz

改自https://github.com/RMonica/rviz_cloud_annotation

不想多说，只是在这里做个记录，项目看起来是标注车辆用的，但是需要 ubuntu，需要 rviz，需要 ROS，有些复杂了说实话。

## 3d-bat

https://github.com/walzimmer/3d-bat

Window npm 装不上

## Label6DPose

也是 ROS 写的

https://github.com/cmitash/Label6DPose

## Pointcloud_Labeling_Tool

https://github.com/Jensssen/Pointcloud_Labeling_Tool

## SceneLabel

https://github.com/panyunyi97/SceneLabel
