# `torch` 中保证随机性的最佳实践

保证随机性主要分为两部分：

- 随机数据：比如 `torch.random` 生成的随机数，在随机初始化张量、随机采样、随机增强等过程中会用到。这种随机性可以通过设置随机种子来控制。
- 随机算法：一些算法本身具有随机性，可以通过使用确定性算法来回避随机性（注意，一些算法是没有确定性实现的，如果非要 `use_deterministic_algorithms`，会报 `RuntimeError`）。

## 野生方案

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

有网友提供了解决方案：

```python
# https://github.com/pytorch/pytorch/pull/56488#issuecomment-825128350
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

一位 `numpy.random` 代码贡献者提供了设置 `NumPy` 的更好方案：

```python
# https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
def worker_init_fn(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

但这个方案没有设置 `PyTorch` 的种子。下面叙述 `lightning` 的方案。

## `lightning` 的方案

如果使用 `lightning`，保证可复现性会比较简单，只需要：（ref: <https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility>

```python
from lightning.pytorch import Trainer, seed_everything

# Sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)
model = Model()

# Enable deterministic training
trainer = Trainer(deterministic=True)
```

究其原理，`seed_everything` 做了以下几件事：

```python
# https://github.com/Lightning-AI/lightning/blob/017262e5e0c65215e9e75121d155d7a07cd9e7bf/src/lightning/fabric/utilities/seed.py#L19
def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.
    """
    # 处理种子的设置
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # 正剧开始
    log.info(rank_prefixed_message(f"Global seed set to {seed}", _get_rank()))
    # 1) 设置全局种子，在不同的 spawn 进程之间共享
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    # 2) python.random
    random.seed(seed)
    # 3) numpy.random
    np.random.seed(seed)
    # 4) torch
    torch.manual_seed(seed)
    # 5) torch.cuda
    torch.cuda.manual_seed_all(seed)
    # 6) 设置环境变量，该环境变量用于 pl_worker_init_fn 为 DataLoader 的 workers 设置种子
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
```

主进程设置了环境变量 `PL_SEED_WORKERS` ，当子进程检测到该环境变量 `==1` 且 `dataloader` 没有设置 `worker_inif_fn` 时，将其 `worker_init_fn` 设置为 `pl_worker_init_fn`。

```python
# https://github.com/Lightning-AI/lightning/blob/017262e5e0c65215e9e75121d155d7a07cd9e7bf/src/lightning/fabric/utilities/data.py#L247C15-L247C29
def _auto_add_worker_init_fn(dataloader: object, rank: int) -> None:
    if not hasattr(dataloader, "worker_init_fn"):
        return
    if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
        dataloader.worker_init_fn = partial(pl_worker_init_function, rank=rank)
```

而其官方文档也说明，`DataLoader` 种子的设置是通过 `worker_init_fn` 实现的，如果用户已经自行设置了 `worker_init_fn`，则 `worker=True` 不起作用。`pl_worker_init_fn` 的实现是这样的，和上文中叙述并无大的差异。

```python
# https://github.com/Lightning-AI/lightning/blob/017262e5e0c65215e9e75121d155d7a07cd9e7bf/src/lightning/fabric/utilities/seed.py#L81
def pl_worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
    """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.

    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    log.debug(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)
```

而另一部分 `deterministic=True`，最终被传到了 `_set_torch_flags`，其中关键是：

- `torch.cuda.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`
- `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`

```python
def _set_torch_flags(
    *, deterministic: Optional[Union[bool, _LITERAL_WARN]] = None, benchmark: Optional[bool] = None
) -> None:
    if deterministic:
        if benchmark is None:
            # Set benchmark to False to ensure determinism
            benchmark = False
        elif benchmark:
            rank_zero_warn(
                "You passed `deterministic=True` and `benchmark=True`. Note that PyTorch ignores"
                " torch.backends.cudnn.deterministic=True when torch.backends.cudnn.benchmark=True.",
            )
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark

    if deterministic == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif isinstance(deterministic, bool):
        # do not call this if deterministic wasn't passed
        torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```
