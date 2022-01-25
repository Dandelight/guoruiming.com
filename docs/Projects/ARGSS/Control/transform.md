> tf is a package that lets the user keep track of multiple coordinate frames over time. tf maintains the relationship between coordinate frames in a tree structure buffered in time, and lets the user transform points, vectors, etc between any two coordinate frames at any desired point in time.

> ROS 进二阶学习笔记（1） TF 学习笔记 2 -- TF Broadcaster
> Ref: http://wiki.ros.org/tf/Tutorials#Learning_tf
>
> > > Ref: Writing a tf broadcaster (Python)
>
> This tutorial teaches you how to broadcast the state of a robot to tf.
>
> Ref 内的东西请读者自行研读，这里要做点总结：
>
> /tf topic 上 是有一个发布器 broadcaster 发布 /tf 消息才能被 /tf 的 listener 收听到。
> /tf 的发布器类：python: tf.TransformBroadcaster 所以我们要理解它里面的方法： sendTransform(translation, rotation, time, child, parent)
>
> 1. 先看看 tf broadcaster 的示例代码：

```python
> #!/usr/bin/env python
> import roslib
> roslib.load_manifest('learning_tf')
> import rospy
>
> import tf
> import turtlesim.msg
>
> def handle_turtle_pose(msg, turtlename):
>     br = tf.TransformBroadcaster()
>     br.sendTransform((msg.x, msg.y, 0), #the translation of the transformtion as a tuple (x, y, z)
>                      tf.transformations.quaternion_from_euler(0, 0, msg.theta),
>                                                 #the rotation of the transformation as a tuple (x, y, z, w)
>                      rospy.Time.now(), #the time of the transformation, as a rospy.Time()
>                      turtlename, #child frame in tf, string
>                      "world") #parent frame in tf, string
>
> if __name__ == '__main__':
>     rospy.init_node('turtle_tf_broadcaster')
>     turtlename = rospy.get_param('~turtle')
>                      #takes parameter "turtle", which specifies a turtle name, e.g. "turtle1" or "turtle2"
>     rospy.Subscriber('/%s/pose' % turtlename, # subscribe to turtlename's /pose topic
>                      turtlesim.msg.Pose,      # Pose message data structure
>                      handle_turtle_pose,      # callback function,
>                      turtlename)              # argument for callback function
>     rospy.spin()
>
```

> 上面代码的框架就是：
> 通过参数知道 turtle 是 1 号还是 2 号，比如： turtle1。 赋值给 turtlename 变量
> 收听 turtle1/pose 主题，上面是 turtle1 的位姿数据。
> 收听到主题上有 message 就调用 handle_turtle_pose 这个函数。后面跟了一个 turtlename 参数，正好对应 def handle_turtle_pose()函数定义时的第 2 个参数：turtlename
> 补充一点：
> 这里要补充一点 python 回调函数调用时，传参的知识：
>
> #define the callback method:
> def callback(arg1, arg2, arg3):
> ...
>
> #next, Subscriber:
> rospy.Subscriber(topicName, data_type, callback, arg2)
> 上面 callback 的 arg1 一般是 subscriber 读到的数据，直接作为第一个参数传入 callback 函数。arg2 传入 callback 的 arg2 位置。 2. 重点是 TransformBroadcaster() 类
> 参考：http://mirror.umd.edu/roswiki/doc/diamondback/api/tf/html/python/tf_python.html#transformbroadcaster
>
> 读一读上面的定义：
>
> translation - 元素组形式的，变换的坐标的转换。tuple(元素组) 据我理解，就是 tf 的位置变换（x，y，z）形式。
> rotation - 四元数数据结构，元素组形式的变换，的旋转。方向变换，上面实例调用了欧拉旋转（Roll, Pitch, Yaw）, 比如有人这样用：
> quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
> time - rospy.Time() 数据结构的时间，变换时的时间
> child - 子坐标系; 这里，我们的子坐标系是：turtlename = "turtle1";
> parent - 父坐标系 ; 父坐标系是“world”.
> 再来看这个源代码：
>
>     br.sendTransform((msg.x, msg.y, 0), #the translation of the transformtion as a tuple (x, y, z)
>                      tf.transformations.quaternion_from_euler(0, 0, msg.theta),
>                      #the rotation of the transformation as a tuple (x, y, z, w)
>                      rospy.Time.now(), #the time of the transformation, as a rospy.Time()
>                      turtlename, #child frame in tf, string
>                      "world") #parent frame in tf, string
>
> 就比较清楚了，原料是 msg.x; msg.y; msg.theta，要完成坐标变换的发布：
> 平移变换，用了 msg.x 、 msg.y
> 旋转变换，调了 tf.transformations.quaternion_from_euler 函数，将（0,0，msg.theta）转换成了 quaternion 四元数。
> =========
>
> 总结
> 至此，我们就解读清楚了 tf 发布器的编写方法。
> 发布器只是把 child_frame 与 parent_frame 之间的平移和旋转关系发布在 /tf 主题上，TF_Listener 怎么解读和利用，是更重要的。
> 下一节我们将看看如何写 TF_Listener 来完成简单的任务。
>
> 通过 launch 文件启动这个 TF Broadcaster
> 要测试，我们还得借助 ROS 提供的 turtlesim 工具，看看 launch 文件：
>
>   <launch>
>     <!-- Turtlesim Node-->
>     <node pkg="turtlesim" type="turtlesim_node" name="sim"/>
>     <node pkg="turtlesim" type="turtle_teleop_key" name="teleop" output="screen"/>
>
>     <node name="turtle1_tf_broadcaster" pkg="learning_tf" type="turtle_tf_broadcaster.py" respawn="false" output="screen" >
>       <param name="turtle" type="string" value="turtle1" />
>     </node>
>     <node name="turtle2_tf_broadcaster" pkg="learning_tf" type="turtle_tf_broadcaster.py" respawn="false" output="screen" >
>       <param name="turtle" type="string" value="turtle2" />
>     </node>
>
>   </launch>
> 做个简单解读：
>  turtlesim/turtlesim_node启动，名：sim -- 这是打开带海龟的窗口。这个turtlesim_node节点会publish 一个turtleX/pose 主题，会被我们turtle_tf_broadcaster接收用来发布tf信息。
>  turtlesim/turtle_teleop_key , 名：teleop -- 这是启动接收键盘指令的节点。
>  pkg="learning_tf" type="turtle_tf_broadcaster.py" , 启动一个TF Broadcaster, 发布 turtle1 相对于world的tf信息。
>  pkg="learning_tf" type="turtle_tf_broadcaster.py" , 启动一个TF Broadcaster, 发布 turtle2 相对于world的tf信息。
> Checking the results
> Now, use the tf_echo tool to check if the turtle pose is actually getting broadcast to tf:
>
> $ rosrun tf tf_echo /world /turtle1
> This should show you the pose of the first turtle. Drive around the turtle using the arrow keys (make sure your terminal window is active, not your simulator window). If you run tf_echo for the transform between the world and turtle 2, you should not see a transform, because the second turtle is not there yet. However, as soon as we add the second turtle in the next tutorial, the pose of turtle 2 will be broadcast to tf.
>
> ---
>
> 版权声明：本文为 CSDN 博主「Sonictl」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/sonictl/article/details/52183461

相机内参矩阵，进行空间坐标和图像坐标之间的相互转换。

- 内参矩阵表示
  $$
  M=\left[\begin{array}{ccc}
  f x & 0 & u x \\
  0 & f y & u y \\
  0 & 0 & 1
  \end{array}\right]
  $$
- 另图像坐标为 $\mathrm{s}$, 空间坐标为 $\mathrm{t}$,
  $$
  s=\left[\begin{array}{l}
  u \\
  v \\
  1
  \end{array}\right] \quad t=\left[\begin{array}{l}
  x \\
  y \\
  z
  \end{array}\right]
  $$
- 其转换关系可以表示为
  $$
  z \cdot s=M \cdot t
  $$
- 拆成一般式为:

1. 由图像坐标转换至空间坐标:
   $$
   \begin{aligned}
   &x=z \cdot(u-u x) / f x \quad \text { equ1 } \\
   &y=z \cdot(v-u y) / f y \quad \text { equ2 }
   \end{aligned}
   $$
2. 由空间坐标转换为图像坐标为:
   $$
   \begin{aligned}
   &u=(f x \cdot x+u x \cdot z) / z \\
   &v=(f y \cdot y+u y \cdot z) / z
   \end{aligned}
   $$
   根据 equ1 和 equ2 来判断图像上一个像素点在空间中对应的距离差 另 $u=u+1, \% u 5 E 26 \% u 5165 \% u 5 F 97$ 代入得, $x^{\prime}=z \cdot(u+1-u x) / f x=z \cdot(u-u x) / f x+z / f x$
   对比 equ1 发现， $\Delta x=z / f x$
   因此对应空间分辨率是个范围值，一般常用数据集中 $\mathrm{z}$ 值在 $500 \sim 1000$ 之间， fx 在 $200 \sim 400$ 之间，故空间分辨率大致在 $1 \sim 5 \mathrm{~mm}$ 之间。

还有一篇讲得云里雾里

https://blog.csdn.net/qq_40369926/article/details/89251296
