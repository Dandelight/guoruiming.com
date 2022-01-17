$\operatorname{IoU}$即交并比，又名 Jaccard Index，在语义分割中一直广为使用。

$$
\operatorname{IoU} = \frac{target \cup prediction}{target \cap prediction}
$$

$\mathrm{IoU}$一般基于类计算，也有基于图片计算[^csdn87901365]，一定要看清数据集的评价标准。

对于$\mathrm{mIoU}$，就是基于类计算的$\mathrm{IoU}$对每一类的值进行累加后取平均。

具体到计算上，设 Ground Truth 为$A_{GT} \in \mathbb{R}^{2}$，预测结果$A_{PE} \in \mathbb{R}^2$，则对于$A_{GT}$中的每一个像素的真实值值$A_{GT}[x, y]$（代表一个类），与预测值作混淆矩阵（confusion matrix）。我们考虑一个简单的情况：

$$
A_{GT} = \begin{bmatrix}
 1 & 1 \\ 2 & 2
\end{bmatrix},\quad
A_{PE} = \begin{bmatrix}
2 & 1 \\ 2 & 2
\end{bmatrix}
$$

则其混淆矩阵$M$（横向为 GT，纵向为 PE）

$$
M = \begin{bmatrix}
	1 & 0 \\ 1 & 2
\end{bmatrix}
$$

> 或许画个正字会好一些？

那么可知，$\mathrm{IoU}_1 = \frac{1}{2} = 0.5$，$\mathrm{IoU}_2=\frac{2}{3} \approx 0.667$，$\mathrm{mIoU}$为二者的平均。

从向量的角度理解，$\mathrm{IoU}_i$即为主对角线上第$i$个元素除以（第$i$行所有值的和），即$\mathrm{IoU}_i = \frac{M[i, i]}{M[i, :]}$。进而，$\mathrm{mIoU} = \frac{\sum_{i=1}^n{\mathrm{IoU_i}}}{n}$。

$\mathrm{IoU}$用`PyTorch`实现可以这样做：

```python
def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def _list_tensor(x, y, sigmoid = False):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(x)
        y = torch.tensor(y)
        if sigmoid:
            x = m(x)
    else:
        x, y = x, y
        if sigmoid:
            x = m(x)
    return x, y

def iou(pr, gt, eps=1e-7, threshold = 0.5, sigmoid = False):
    pr_, gt_ = _list_tensor(pr, gt, sigmoid = sigmoid)
    pr_ = _threshold(pr_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_)
    union = torch.sum(gt_) + torch.sum(pr_) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()
```

$\mathrm{mIoU}$的难点主要在混淆矩阵的高效计算上

```python
def miou(gt, pe, num_classes):
    iou = []
    for cls in range(1, num_classes+1):
        cls_gt = gt == cls
        cls_pe = pe == cls
        intersection = cls_pe & cls_gt
        union = cls_pe | cls_gt
        iou_i = (intersection.sum() + 1e-9) / (union.sum() + 1e-9)
        iou.append(iou_i)
    return numpy.array(iou).mean()
```

混淆矩阵可以如下计算：

```python
# https://github.com/hualin95/Deeplab-v3plus/blob/master/utils/eval.py
def generate_matrix(gt_image, pre_image,num_class):
    """ This function calculates the confusion matrix. """
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class) #21 * 21(for pascal)
    return confusion_matrix

calc_cm(np.array([[0, 0], [1, 1]]), np.array([[1, 0], [1, 1]]), 2)
# output:
# array([[1, 1],
#        [0, 2]])
```

一个`bitcount`可谓神来之笔。

## `FWIoU`

Frequency Weighted Intersection over Union，公式如下：

$$
\operatorname{FwIoU}=\frac{1}{\sum_{j=1}^{k} t_{j}} \sum_{j=1}^{k} t_{j} \frac{n_{j j}}{n_{i j}+n_{j i}+n_{j j}}, \quad i \neq j
$$

```python
# https://github.com/hualin95/Deeplab-v3plus/blob/master/utils/eval.py
def Frequency_Weighted_Intersection_over_Union(self):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU
```

[^csdn87901365]: https://blog.csdn.net/lingzhou33/article/details/87901365
