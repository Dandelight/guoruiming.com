# Profiler

https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

## `torch.utils.bottleneck`

torch.utils.bottleneck is a tool that can be used as an initial step for debugging bottlenecks in your program. It summarizes runs of your script with the Python profiler and PyTorch’s autograd profiler.

Run it on the command line with

```
python -m torch.utils.bottleneck /path/to/source/script.py [args]
```

where [args] are any number of arguments to script.py, or run `python -m torch.utils.bottleneck -h` for more usage instructions.

## `torch.utils.benchmark.Timer`

https://pytorch.org/tutorials/recipes/recipes/benchmark.html
