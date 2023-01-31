# TensorBoard

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
```

```shell
pip install tensorboard
tensorboard --logdir=runs
```

具有相同 `prefix/` 的绘图会被归到同一个标签下

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

## 构造器

```python
class torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
```

- **log_dir** (_string_) – Save directory location. Default is runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run. Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
- **comment** (_string_) – Comment log_dir suffix appended to the default `log_dir`. If `log_dir` is assigned, this argument has no effect.
- **purge_step** ([_int_](https://docs.python.org/3/library/functions.html#int)) – When logging crashes at step T+X*T*+_X_ and restarts at step T*T*, any events whose global_step larger or equal to T*T* will be purged and hidden from TensorBoard. Note that crashed and resumed experiments should have the same `log_dir`.
- **max_queue** ([_int_](https://docs.python.org/3/library/functions.html#int)) – Size of the queue for pending events and summaries before one of the ‘add’ calls forces a flush to disk. Default is ten items.
- **flush_secs** ([_int_](https://docs.python.org/3/library/functions.html#int)) – How often, in seconds, to flush the pending events and summaries to disk. Default is every two minutes.
- **filename_suffix** (_string_) – Suffix added to all event filenames in the log_dir directory. More details on filename construction in tensorboard.summary.writer.event_file_writer.EventFileWriter.

```python
from torch.utils.tensorboard import SummaryWriter

# create a summary writer with automatically generated folder name.
writer = SummaryWriter()
# folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

# create a summary writer using the specified folder name.
writer = SummaryWriter("my_experiment")
# folder location: my_experiment

# create a summary writer with comment appended.
writer = SummaryWriter(comment="LR_0.1_BATCH_16")
# folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
```

## 方法

- `add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)`
- `add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)`：与上一个不同就是传进去个 `dict`
- `add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)`
- `add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`：至少不同一次弹出个窗口来看图片效果
- `add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')`：比上边多了一维 `Batch`，依赖 `Pillow`
- `add_figure(tag, figure, global_step=None, close=True, walltime=None)`：贴一个 `matplotlib` `figure` 上去，依赖 `matplotlib`
- `add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)`：贴一段视频上去，依赖 `moviepy`
- `add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)`：贴一段音频上去
- `add_text(tag, text_string, global_step=None, walltime=None)`：文字
- `add_graph(model, input_to_model=None, verbose=False, use_strict_trace=True)`：计算图，通过 `trace` 方式得到 `Module` 的计算图并显示在 `TensorBoard` 中。`input_to_model` 是模型的伪输入，可以是任意内容，只要形状对就可以
- `add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)`
- `add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)`：这就更高级了……直接把 Precision-Recall 曲线贴上去
- `add_custom_scalars(layout)`
- `add_mesh(tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None)`：贴一块三维点云上去，使用 `Three.js` 进行可视化
- `add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)`：把一堆超参数贴上去进行比较。~~**调参必备**~~<https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams>
- `flush()`
- `close()`
