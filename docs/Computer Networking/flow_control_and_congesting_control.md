# 计算机网络中的流量控制与拥塞控制

TCP（Transmission Control Protocol）是面向连接的传输层协议，不仅具有多进程复用网络、差错校验功能，还有可靠数据传输、流量控制和拥塞控制功能。流量控制和拥塞控制都是通过控制发送方发送未被 ACK 的数据包的数量控制的。

## 流量控制

```

    0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |          Source Port          |       Destination Port        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                        Sequence Number                        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                    Acknowledgment Number                      |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |  Data |           |U|A|P|R|S|F|                               |
   | Offset| Reserved  |R|C|S|S|Y|I|            Window             |
   |       |           |G|K|H|T|N|N|                               |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |           Checksum            |         Urgent Pointer        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                    Options                    |    Padding    |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                             data                              |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

                            TCP Header Format

          Note that one tick mark represents one bit position.

                               Figure 3.
```

这是 TCP 的报文结构，来自 RFC793，可以看到有一个 16 位的 Window 字段，RFC 中是这样解释的：

> The number of data octets beginning with the one indicated in the acknowledgment field which the sender of this segment is willing to accept.

也就是说，Window 大小是接收方愿意接受的数据的大小；这个数据会有一个放大倍数，是客户端和服务协商好的，比如下边这个是 256。

![image-20211228132700764](C:\Users\Min\AppData\Roaming\Typora\typora-user-images\image-20211228132700764.png)

现在的问题是，如何计算自己的 Window size 呢？

TCP 是有缓冲区的，建立 TCP 连接之后，操作系统会分配缓冲区，缓冲区大小是有限的，数据的处理需要时间，中间丢包还需要暂存数据。一般来说，TCP 的一方（称其为 B）在接收时会维护两个状态：

`LastByteRead`：B 的应用层进程读出的最后一个字节的序号

`LastByteRcvd`：B 的网络层传到传输层的最后一个字节的序号

TCP 缓冲区不能溢出，所以我们规定：`LastByteRcvd - LastByteRead <= RecvBuffer`，注意这里的减法是$\mod 2$意义下的，永远都是自然数。

因此接收窗口`Receive Window`由公式计算得到：`rwnd = RecvBuffer - [LastByteRcvd - LastByteRead]`

和 B 建立连接的 A 也维护着 B 的两个状态：

`LastByteSent`：A 的传输层传到网络层的最后一个字节的需要

`LastByteAcked`：A 接收到最大的`ACK`的序号

在发送的时候，A 掌握着主动权，在传输的全过程中，它需要维持以下不等式成立：`LastByteSent - LastByteAcked <= rwnd`

有另外一个小问题：如果 B 的缓冲区满了，它送回一个 0，那么 A 怎么知道 B 的缓冲区什么时候空了呢？事实上，标准规定 A 在得知 B 的`rwnd`为 0 时，依然会持续发送一字节的报文，直到 B 返回的 ACK 报文中的`Window size`成为正值。

## 拥塞控制
