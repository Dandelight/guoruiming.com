# 基于`Qt` 的 `RealSense` 相机数据采集软件

This is a very simple example, that uses only Qt and Intel Realsense SDK.

We start by writing a class that handles our camera:

```cpp
#ifndef CAMERA_H
#define CAMERA_H

// Import QT libs, one for threads and one for images
#include <QThread>
#include <QImage>

// Import librealsense header
#include <librealsense2/rs.hpp>

// Let's define our camera as a thread, it will be constantly running and sending frames to
// our main window
class Camera : public QThread
{
    Q_OBJECT
public:
    // We need to instantiate a camera with both depth and rgb resolution (as well as fps)
    Camera(int rgb_width, int rgb_height, int depth_width, int depth_height, int fps);
    ~Camera() {}

    // Member function that handles thread iteration
    void run();

    // If called it will stop the thread
    void stop() { camera_running = false; }

private:
    // Realsense configuration structure, it will define streams that need to be opened
    rs2::config cfg;

    // Our pipeline, main object used by realsense to handle streams
    rs2::pipeline pipe;

    // Frames returned by our pipeline, they will be packed in this structure
    rs2::frameset frames;

    // A bool that defines if our thread is running
    bool camera_running = true;

signals:
    // A signal sent by our class to notify that there are frames that need to be processed
    void framesReady(QImage frameRGB, QImage frameDepth);
};
// A function that will convert realsense frames to QImage
QImage realsenseFrameToQImage(const rs2::frame& f);

#endif // CAMERA_H
```

To fully understand what this class do I redirect you to these two pages: [Signals & Slots](https://doc.qt.io/qt-5/signalsandslots.html) and [QThread](https://doc.qt.io/qt-5/qthread.html). This class is a QThread, that means that it can run parallelly with our main window. When a couple of frames is ready, the signal framesReady will be emitted and the window will show the images.

Let's start by saying how to open the camera streams with librealsense:

```cpp
Camera::Camera(int rgb_width, int rgb_height, int depth_width, int depth_height, int fps)
{
    // Enable depth stream with given resolution. Pixel will have a bit depth of 16 bit
    cfg.enable_stream(RS2_STREAM_DEPTH, depth_width, depth_height, RS2_FORMAT_Z16, fps);

    // Enable RGB stream as frames with 3 channel of 8 bit
    cfg.enable_stream(RS2_STREAM_COLOR, rgb_width, rgb_height, RS2_FORMAT_RGB8, fps);

    // Start our pipeline
    pipe.start(cfg);
}
```

As you can see our constructor is very simple and it will only open the pipeline with the given stream.

Now that the pipeline is started we only need to get the corresponding frames. We'll do it in our 'run' method, the method that will be launched when the QThread starts:

```cpp
void Camera::run()
{
    while(camera_running)
    {
        // Wait for frames and get them as soon as they are ready
        frames = pipe.wait_for_frames();

        // Let's get our depth frame
        rs2::depth_frame depth = frames.get_depth_frame();
        // And our rgb frame
        rs2::frame rgb = frames.get_color_frame();

        // Let's convert them to QImage
        auto q_rgb = realsenseFrameToQImage(rgb);
        auto q_depth = realsenseFrameToQImage(depth);

        // And finally we'll emit our signal
        emit framesReady(q_rgb, q_depth);
    }
}
```

The function that does the conversion is the following:

```cpp
QImage realsenseFrameToQImage(const rs2::frame &f)
{
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = QImage((uchar*) f.get_data(), w, h, w*3, QImage::Format_RGB888);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        // only if you have Qt > 5.13
        auto r = QImage((uchar*) f.get_data(), w, h, w*2, QImage::Format_Grayscale16);
        return r;
    }

    throw std::runtime_error("Frame format is not supported yet!");
}
```

Our Camera is done.

Now we'll define our main window. We'll need a slot that receives our frames and two labels where we will put our images:

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);

public slots:
    // Slot that will receive frames from the camera
    void receiveFrame(QImage rgb, QImage depth);

private:
    QLabel *rgb_label;
    QLabel *depth_label;
};

#endif // MAINWINDOW_H
```

We create a simple view for the window, with the images that will be shown vertically.

```cpp
#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    // Creates our central widget that will contain the labels
    QWidget *widget = new QWidget();

    // Create our labels with an empty string
    rgb_label = new QLabel("");
    depth_label = new QLabel("");

    // Define a vertical layout
    QVBoxLayout *widgetLayout = new QVBoxLayout;

    // Add the labels to the layout
    widgetLayout->addWidget(rgb_label);
    widgetLayout->addWidget(depth_label);

    // And then assign the layout to the central widget
    widget->setLayout(widgetLayout);

    // Lastly assign our central widget to our window
    setCentralWidget(widget);
}
```

And now we need to define the slot function. The only job assigned to that function is to change the images related to the labels:

```cpp
void MainWindow::receiveFrame(QImage rgb, QImage depth)
{
    rgb_label->setPixmap(QPixmap::fromImage(rgb));
    depth_label->setPixmap(QPixmap::fromImage(depth));
}
```

主线程和主函数

```cpp
#include <QApplication>
#include "mainwindow.h"
#include "camera.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow window;
    Camera camera(640, 480, 320, 240, 30);

    // Connect the signal from the camera to the slot of the window
    QApplication::connect(&camera, &Camera::framesReady, &window, &MainWindow::receiveFrame);

    window.show();

    camera.start();

    return a.exec();
}
```

然而在运行代码的时候遇到了比较大的问题：`qmake`  如何链接 `RealSense`？结果就是两个库链不上。
