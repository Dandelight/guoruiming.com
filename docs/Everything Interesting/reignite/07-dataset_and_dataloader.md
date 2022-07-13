# Dataset 和 DataLoader

本节讲解 PyTorch 中最重要的两个类：Dataset 和 DataLoader

## Dataset

Dataset 分为 map 类 dataset 和 iterable 类 dataset。

map 类 dataset 需要继承 `torch.utils.data.Dataset` 类，重写 `__getitem__` 和 `__len__` （可选）方法。

iterable dataset 需要继承 `torch.utils.data.IterableDataset`，额外重写 `__iter__` 方法。

```python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

# Mult-process loading with two worker processes
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
```

## DataLoader

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

首先 DataLoader 接收一个 Dataset，对其进行 `batch`、`shuffle`、`sample` 等操作。

### `batch_size`

如果 `batch_size` 为 `None`，则不会产生 `batch` 维；否则会自动进行 `batch`。

### `shuffle`

若 `shuffle` 为 `True`，自动构造一个 `Sequential Sampler` 和 `Shuffle Sampler`。

### `batch_sampler`

可以在程序内指定 [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)

### `num_workers`

设置为正数以启用多线程加载。

tl;dr replace your lists / dicts in Dataloader `__getitem__` with numpy arrays, pandas dataframes or PyArrow objects.

### `collate_fn`

若为 `None`，使用默认函数；如果 `batch` 为正数，`DataLoader` 调用该函数来对数据进行 `batch`；否则，`DataLoader` 对每个数据调用该函数。

### `pin_memory`

If you load your samples in the `Dataset` on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling `pin_memory`.
This lets your `DataLoader` allocate the samples in page-locked memory, which speeds-up the transfer.
You can find more information on the [NVIDIA blog](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/)

### `drop_last`

数据集不够切分为 `batch` 的时候把最后的去掉。

### `timeout`

如果是正数，`worker` 超时时间

### `worker_init_fn`

If not `None`, this will be called on each worker subprocess with the worker id (an int in `[0, num_workers - 1]`) as input, after seeding and before data loading.

### `generator`

If not `None`, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers. (default: `None`)

### `prefetch_factor`

Number of batches loaded in advance by each worker. `2` means there will be a total of 2 \* num_workers batches prefetched across all workers. (default: `2`)

### `persistent_workers`

If `True`, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: `False`)

### `pin_memory_device`

The data loader will copy Tensors into device pinned memory before returning them if pin_memory is set to true.

### 其他内置 `Dataset`

- `torch.utils.data.TensorDataset(*tensors)`
- `torch.utils.data.ConcatDataset(datasets)`
- `torch.utils.data.ChainDataset(datasets)`
- `torch.utils.data.Subset(dataset, indices)`

## `Sampler`

- `torch.utils.data.Sampler(data_source)`

每个 `Sampler` 需要实现 `__iter__` 方法，可以实现 `__len__` 方法。

- `torch.utils.data.SequentialSampler(data_source)`
- `torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)`
- `torch.utils.data.SubsetRandomSampler(indices, generator=None)`
- `torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)`
- `torch.utils.data.BatchSampler(sampler, batch_size, drop_last)`
- `torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)`

## 实用工具

- `torch.utils.data.random_split(dataset, lengths, generator=<torch._C.Generator object>)`

```python
random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
```

- `torch.utils.data.get_worker_info()`
