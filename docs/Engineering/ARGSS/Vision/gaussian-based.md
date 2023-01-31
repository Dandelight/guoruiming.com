https://arxiv.org/abs/2101.10226v1

绷不住了，又有人刷榜了

在这项工作中，Hu Cao 等人建立了一个高效且鲁棒的全卷积神经网络，从来自真实场景的 n 通道图像中获取抓取信息。

- lightweight generative architecture
- a grasping representation based on Gaussian kernel
- a Receptive Field Block is assembled to the bottleneck of grasping detection architecture
- Combined pixel attention and channel attention

Grasping is a challenge task for robots: perception, planning and extection.

作者贡献如下：

- We propose a Gaussian-based grasping representation, which relects the maximum grasping score at the center point location and can signigicantly improve the grasping detection accuracy.
- We develope a lightweight generative architecture which achieves high detection accuracy and real-time running speed with small network parameters.
- A receptive field block module is embedded in the bottleneck of the network to enhance its feature discriminability and robustness, and a multi-dimensional attention fusion network is developed to suppress redundant features and enhance target features in the fusion process.
- Evaluation on the public Cornell and Jacquard grasping datasets demonstrate that the proposed generative based grasping detection algorithm achieves state-of-the-art performance of both speed and detection accuracy

## Related work

### Oriented rectangle-based representation

- Analytic methods: use mathematical and physical models in geometry, motion and dynamics to carry out the calculation for grasping
- Empirical methods: deep learning
  - Classification-based: Proposals, GQ-CNN, Spatial Transformer Network
  - Regression-based: Multi-model fusion, ROI => more inclined to learn the mean value of the ground truth grasps
  - Vision and tactic sensing are fuse

### Point-based Grasp representation

GGCNN

Orientation Attentive Grasping Detection

## Gaussian-based

$$
G_{K}=\left\{\Phi, W, Q_{K}\right\} \in \mathbb{R}^{3 \times W \times H}
$$

where,

$$
Q_{K}=K(x, y)=\exp \left(-\frac{\left(x-x_{0}\right)^{2}}{2 \sigma_{x}^{2}}-\frac{\left(y-y_{0}\right)^{2}}{2 \sigma_{y}^{2}}\right)
$$

where,

$$
\sigma_{x}=T_{x}, \sigma_{y}=T_{y}
$$

### Questions

What about classification? In some of the implementations they also output class probabilities.

What about implementation?

Are you conducting related researches?

Multiple objects?
