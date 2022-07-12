# Dataset 和 DataLoader

本节讲解 PyTorch 中最重要的两个类：Dataset 和 DataLoader

` torch.utils.data.``Dataset `(**args*, ***kwds\*)[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset)

An abstract class representing a [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite `__getitem__()`, supporting fetching a data sample for a given key. Subclasses could also optionally overwrite `__len__()`, which is expected to return the size of the dataset by many [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) implementations and the default options of [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) by default constructs a index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided.

` torch.utils.data.``IterableDataset `(**args*, ***kwds\*)[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset)

An iterable Dataset.

All datasets that represent an iterable of data samples should subclass it. Such form of datasets is particularly useful when data come from a stream.

All subclasses should overwrite `__iter__()`, which would return an iterator of samples in this dataset.

When a subclass is used with [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), each item in the dataset will be yielded from the [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) iterator. When `num_workers > 0`, each worker process will have a different copy of the dataset object, so it is often desired to configure each copy independently to avoid having duplicate data returned from the workers. [`get_worker_info()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info), when called in a worker process, returns information about the worker. It can be used in either the dataset’s `__iter__()` method or the [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) ‘s `worker_init_fn` option to modify each copy’s behavior.

Example 1: splitting workload across all workers in `__iter__()`:

```
>>> class MyIterableDataset(torch.utils.data.IterableDataset):
...     def __init__(self, start, end):
...         super(MyIterableDataset).__init__()
...         assert end > start, "this example code only works with end >= start"
...         self.start = start
...         self.end = end
...
...     def __iter__(self):
...         worker_info = torch.utils.data.get_worker_info()
...         if worker_info is None:  # single-process data loading, return the full iterator
...             iter_start = self.start
...             iter_end = self.end
...         else:  # in a worker process
...             # split workload
...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
...             worker_id = worker_info.id
...             iter_start = self.start + worker_id * per_worker
...             iter_end = min(iter_start + per_worker, self.end)
...         return iter(range(iter_start, iter_end))
...
>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
>>> ds = MyIterableDataset(start=3, end=7)

>>> # Single-process loading
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
[3, 4, 5, 6]

>>> # Mult-process loading with two worker processes
>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
[3, 5, 4, 6]

>>> # With even more workers
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
[3, 4, 5, 6]
```

Example 2: splitting workload across all workers using `worker_init_fn`:

```
>>> class MyIterableDataset(torch.utils.data.IterableDataset):
...     def __init__(self, start, end):
...         super(MyIterableDataset).__init__()
...         assert end > start, "this example code only works with end >= start"
...         self.start = start
...         self.end = end
...
...     def __iter__(self):
...         return iter(range(self.start, self.end))
...
>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
>>> ds = MyIterableDataset(start=3, end=7)

>>> # Single-process loading
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
[3, 4, 5, 6]
>>>
>>> # Directly doing multi-process loading yields duplicate data
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
[3, 3, 4, 4, 5, 5, 6, 6]

>>> # Define a `worker_init_fn` that configures each dataset copy differently
>>> def worker_init_fn(worker_id):
...     worker_info = torch.utils.data.get_worker_info()
...     dataset = worker_info.dataset  # the dataset copy in this worker process
...     overall_start = dataset.start
...     overall_end = dataset.end
...     # configure the dataset to only process the split workload
...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
...     worker_id = worker_info.id
...     dataset.start = overall_start + worker_id * per_worker
...     dataset.end = min(dataset.start + per_worker, overall_end)
...

>>> # Mult-process loading with the custom `worker_init_fn`
>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
[3, 5, 4, 6]

>>> # With even more workers
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
[3, 4, 5, 6]
```

https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler

## Single- and Multi-process Data Loading

A [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) uses single-process data loading by default.

Within a Python process, the [Global Interpreter Lock (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) prevents true fully parallelizing Python code across threads. To avoid blocking computation code with data loading, PyTorch provides an easy switch to perform multi-process data loading by simply setting the argument `num_workers` to a positive integer.

### Single-process data loading (default)

In this mode, data fetching is done in the same process a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) is initialized. Therefore, data loading may block computing. However, this mode may be preferred when resource(s) used for sharing data among processes (e.g., shared memory, file descriptors) is limited, or when the entire dataset is small and can be loaded entirely in memory. Additionally, single-process loading often shows more readable error traces and thus is useful for debugging.

### Multi-process data loading

Setting the argument `num_workers` as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.

For data loading, passing `pin_memory=True` to a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.

The default memory pinning logic only recognizes Tensors and maps and iterables containing Tensors. By default, if the pinning logic sees a batch that is a custom type (which will occur if you have a `collate_fn` that returns a custom batch type), or if each element of your batch is a custom type, the pinning logic will not recognize them, and it will return that batch (or those elements) without pinning the memory. To enable memory pinning for custom batch or data type(s), define a `pin_memory()` method on your custom type(s).

See the example below.

Example:

```
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```

` torch.utils.data.``DataLoader `(_dataset_, _batch_size=1_, _shuffle=None_, _sampler=None_, _batch_sampler=None_, _num_workers=0_, _collate_fn=None_, _pin_memory=False_, _drop_last=False_, _timeout=0_, _worker_init_fn=None_, _multiprocessing_context=None_, _generator=None_, *\*\*, *prefetch_factor=2*, *persistent_workers=False*, *pin_memory_device=''\*)[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader)

Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

The [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.

See [`torch.utils.data`](https://pytorch.org/docs/stable/data.html#module-torch.utils.data) documentation page for more details.

- Parameters

  **dataset** ([_Dataset_](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) – dataset from which to load the data.**batch_size** ([_int_](https://docs.python.org/3/library/functions.html#int)_,_ _optional_) – how many samples per batch to load (default: `1`).**shuffle** ([_bool_](https://docs.python.org/3/library/functions.html#bool)_,_ _optional_) – set to `True` to have the data reshuffled at every epoch (default: `False`).**sampler** ([_Sampler_](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) _or_ _Iterable\*\*,_ _optional_) – defines the strategy to draw samples from the dataset. Can be any `Iterable` with `__len__` implemented. If specified, `shuffle` must not be specified.**batch_sampler** ([_Sampler_](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) _or_ _Iterable\*\*,_ _optional_) – like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.**num_workers** ([_int_](https://docs.python.org/3/library/functions.html#int)_,_ _optional_) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)**collate_fn** (_callable\*\*,_ _optional_) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.**pin_memory** ([_bool_](https://docs.python.org/3/library/functions.html#bool)_,_ _optional_) – If `True`, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your `collate_fn` returns a batch that is a custom type, see the example below.**drop_last** ([_bool_](https://docs.python.org/3/library/functions.html#bool)_,_ _optional_) – set to `True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If `False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: `False`)**timeout** (_numeric\*\*,_ _optional_) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: `0`)**worker_init_fn** (_callable\*\*,_ _optional_) – If not `None`, this will be called on each worker subprocess with the worker id (an int in `[0, num_workers - 1]`) as input, after seeding and before data loading. (default: `None`)**generator** ([_torch.Generator_](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)_,_ _optional_) – If not `None`, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers. (default: `None`)**prefetch_factor** ([_int_](https://docs.python.org/3/library/functions.html#int)_,_ _optional\*\*,_ _keyword-only arg_) – Number of batches loaded in advance by each worker. `2` means there will be a total of 2 * num_workers batches prefetched across all workers. (default: `2`)**persistent_workers** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,\* _optional_) – If `True`, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: `False`)**pin_memory_device** ([_str_](https://docs.python.org/3/library/stdtypes.html#str)_,_ _optional_) – the data loader will copy Tensors into device pinned memory before returning them if pin_memory is set to true.
