# 获取 RealSense 相机内参

## 软件方法：`rs-sensor-control`获取

打开`C:\Program Files (x86)\Intel RealSense SDK 2.0\tools`

```shell
$ ./rs-sensor-control
======================================================

Found the following devices:

  0 : Intel RealSense D435 #817612070563

Select a device by index: 0

Device information:
  Name                 : Intel RealSense D435
  Serial Number        : 817612070563
  Firmware Version     : 05.10.06.00
  Recommended Firmware Version : 05.10.03.00
  Physical Port        : /sys/devices/pci0000:00/0000:00:14.0/usb2/2-8/2-8:1.0/video4linux/video0
  Debug Op Code        : 15
  Advanced Mode        : YES
  Product Id           : 0B07
  Camera Locked        : N/A
  Usb Type Descriptor  : 3.2

======================================================

Device consists of 2 sensors:

  0 : Stereo Module
  1 : RGB Camera

Select a sensor by index: 1

======================================================

What would you like to do with the sensor?

0 : Control sensor's options
1 : Control sensor's streams
2 : Show stream intrinsics
3 : Display extrinsics

Select an action:
```

注意第二项，这个就是内参的内容了。选择 2

```
======================================================

Sensor consists of 1 streams:
  - Color #0
Sensor provides the following stream profiles:
0  : Color #0 (Video Stream: RGB8 1920x1080@ 30Hz)
1  : Color #0 (Video Stream: RAW16 1920x1080@ 30Hz)
2  : Color #0 (Video Stream: Y16 1920x1080@ 30Hz)
3  : Color #0 (Video Stream: BGRA8 1920x1080@ 30Hz)
4  : Color #0 (Video Stream: RGBA8 1920x1080@ 30Hz)
5  : Color #0 (Video Stream: BGR8 1920x1080@ 30Hz)
6  : Color #0 (Video Stream: YUYV 1920x1080@ 30Hz)
... ...

Please select the desired streaming profile:
```

所有流都在这里了，比如，选择 0

```
Please select the desired streaming profile: 0

Principal Point         : 966.617, 532.934
Focal Length            : 1395.12, 1395.28
Distortion Model        : Brown Conrady
Distortion Coefficients : [0,0,0,0,0]
```

相机的内参矩阵是

```
K = fx		s		x0
   	  0		fy		y0
	  0		0		1
```

fx,fy 为焦距，一般情况下，二者相等。
x0,y0 为主坐标（相对于成像平面）。
s 为坐标轴倾斜参数，理想情况下为 0。

**python pipeline**

```
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

time.sleep(1)
frames = pipeline.wait_for_frames()
# time.sleep(3)
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#
# # Stack both images horizontally
# images = np.hstack((color_image, depth_colormap))
#
# # Show images
# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('RealSense', images)
# cv2.waitKey(1)

fig, axes = plt.subplots(1, 2)
for ax, im in zip(axes, [color_image, depth_image]):
    ax.imshow(im)
    ax.axis('off')
plt.show()
pipeline.stop()
```

```
import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
cfg = pipeline.start(config)
time.sleep(1)
profile = cfg.get_stream(rs.stream.depth)
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)  # 获取内参 width: 640, height: 480, ppx: 319.115, ppy: 234.382, fx: 597.267, fy: 597.267, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]
319.1151428222656
print(intr.ppx)  # 获取指定某个内参
```

注意：通过 python script 获取内参一定要看好自己到底用了哪个 video stream，每个 video stream 的内参都是不一样的

## 硬件方法
