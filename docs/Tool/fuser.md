
`fuser` 是 Linux 下查询文件或 socket 占用的程序，因为 Linux 下一切系统资源都是文件，所以 `fuser` 是检查系统资源占用的强大工具。

## 用例

`torch.distributed.DistributedDataParallel` 训练失败多次后，留下了很多占用显卡内存的进程，如下：

```
Tue Aug 15 20:37:50 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:4F:00.0 Off |                  N/A |
| 30%   39C    P8    24W / 350W |     20MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:52:00.0 Off |                  N/A |
| 30%   27C    P8    26W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce ...  Off  | 00000000:56:00.0 Off |                  N/A |
| 30%   26C    P8    24W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce ...  Off  | 00000000:57:00.0 Off |                  N/A |
| 30%   28C    P8    29W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA GeForce ...  Off  | 00000000:CE:00.0 Off |                  N/A |
| 35%   47C    P2   145W / 350W |  11084MiB / 24576MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA GeForce ...  Off  | 00000000:D1:00.0 Off |                  N/A |
| 32%   45C    P2   145W / 350W |  11084MiB / 24576MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA GeForce ...  Off  | 00000000:D5:00.0 Off |                  N/A |
| 36%   47C    P2   155W / 350W |  11082MiB / 24576MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA GeForce ...  Off  | 00000000:D6:00.0 Off |                  N/A |
| 30%   29C    P8    25W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      3282      G   /usr/bin/gnome-shell                6MiB |
|    1   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    2   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    3   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    4   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    4   N/A  N/A      6956      C   ...da3/envs/torch/bin/python     2382MiB |
|    4   N/A  N/A      8949      C   ...da3/envs/torch/bin/python     2382MiB |
|    4   N/A  N/A     10902      C   ...da3/envs/torch/bin/python     2382MiB |
|    4   N/A  N/A     12880      C   ...da3/envs/torch/bin/python     3926MiB |
|    5   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    5   N/A  N/A      6957      C   ...da3/envs/torch/bin/python     2382MiB |
|    5   N/A  N/A      8950      C   ...da3/envs/torch/bin/python     2382MiB |
|    5   N/A  N/A     10903      C   ...da3/envs/torch/bin/python     2382MiB |
|    5   N/A  N/A     12881      C   ...da3/envs/torch/bin/python     3926MiB |
|    6   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    6   N/A  N/A      6958      C   ...da3/envs/torch/bin/python     2382MiB |
|    6   N/A  N/A      8951      C   ...da3/envs/torch/bin/python     2382MiB |
|    6   N/A  N/A     10904      C   ...da3/envs/torch/bin/python     2382MiB |
|    6   N/A  N/A     12882      C   ...da3/envs/torch/bin/python     3926MiB |
|    7   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

（这是不应该出现的，具体为什么出现了有待排查）

所以当我把能看到的 `python` 进程全部 `kill` 掉：

```
Tue Aug 15 20:56:04 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:4F:00.0 Off |                  N/A |
| 30%   26C    P8    20W / 350W |     20MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:52:00.0 Off |                  N/A |
| 30%   26C    P8    27W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce ...  Off  | 00000000:56:00.0 Off |                  N/A |
| 30%   25C    P8    24W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce ...  Off  | 00000000:57:00.0 Off |                  N/A |
| 30%   27C    P8    35W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA GeForce ...  Off  | 00000000:CE:00.0 Off |                  N/A |
| 30%   40C    P2   106W / 350W |   2394MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA GeForce ...  Off  | 00000000:D1:00.0 Off |                  N/A |
| 30%   26C    P8    24W / 350W |     10MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA GeForce ...  Off  | 00000000:D5:00.0 Off |                  N/A |
| 30%   27C    P8    28W / 350W |     10MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA GeForce ...  Off  | 00000000:D6:00.0 Off |                  N/A |
| 30%   26C    P8    27W / 350W |      8MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      3282      G   /usr/bin/gnome-shell                6MiB |
|    1   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    2   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    3   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    4   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    5   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    6   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
|    7   N/A  N/A      2868      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

诡异的事情出现了，卡 4 上只有一个 `Xorg` 进程运行，但显存占用却达到 `2374MiB`。这时可以用 `fuser` 排查（具体为什么 `fuser` 看得到 `nvidia-smi` 看不到我也不知道）

```shell
sudo fuser -v /dev/nvidia*
```

```
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia1:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia2:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia3:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia4:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia5:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia6:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia7:        root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidiactl:      root       2868 F...m Xorg
                     gdm        3282 F...m gnome-shell
                     ruiming   14810 F...m python
/dev/nvidia-modeset: root       2868 F.... Xorg
                     gdm        3282 F.... gnome-shell
/dev/nvidia-uvm:     ruiming   14810 F...m python
```

发现这些 `GPU` 都被 `14810` 进程占用，`kill` 之，问题解决。
