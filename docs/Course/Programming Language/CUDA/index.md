参考：<https://blog.csdn.net/dcrmg/article/details/54867507>

![](./media/Pasted%20image%2020230916094720.png)
![](./media/Pasted%20image%2020230916094734.png)

```cuda
int blockId = blockIdx.x + blockIdx.y * gridDim.x
                     + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                       + (threadIdx.z * (blockDim.x * blockDim.y))
                       + (threadIdx.y * blockDim.x) + threadIdx.x;
```
