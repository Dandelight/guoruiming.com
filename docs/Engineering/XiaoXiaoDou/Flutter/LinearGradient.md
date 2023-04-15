```dart
Positioned.fill(  // 使用绝对定位可全局渐变（可不用）
 child: Container(
    decoration: BoxDecoration(
      gradient: LinearGradient(     // 渐变位置
         begin: Alignment.topRight, // 右上
         end: Alignment.bottomLeft, // 左下
         stops: [0.0, 1.0],         // [渐变起始点, 渐变结束点]
         // 渐变颜色[始点颜色, 结束颜色]
         colors: [Color.fromRGBO(63, 68, 72, 1), Color.fromRGBO(36, 41, 46, 1)]
      )
    ),
  ),
)
```

关于从`web`开发过渡到`Flutter`，官方教程：<https://flutter.cn/get-started/flutter-for/web-devs>

> 一时嵌套一时爽，一直嵌套，会怎样……

[^ref]: https://blog.csdn.net/qq_41614928/article/details/107282159
