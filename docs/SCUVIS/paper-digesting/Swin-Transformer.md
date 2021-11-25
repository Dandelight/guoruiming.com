```tex
@InProceedings{Liu_2021_ICCV,
    author    = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    title     = {Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10012-10022}
}
```

Transformer是在自然语言处理领域大获成功的RNN模型，并且在计算机视觉任务中也表现出优异的成绩。但是，因为CV和NLP两个领域中的不同点，比如，CV中视觉实体的尺度变化更大、像素的分辨率比文字更高等，都给Transformer在CV的应用带来了挑战。为了解决此问题，作者提出了一种层级化的Transformer网络，并使用Shifted Windows方法进行representation的计算。Shifted Windows（简称Swin）方法将自注意力的计算限制在不重叠的局部窗口中，通过shift（位移）操作实现跨窗口信息连接。网络的层级化结构使得网络能够适应多种尺度的视觉目标，并且Swin的计算只有线性时间复杂度。以Swin Transformer作为骨干网络的模型在多个计算机视觉任务中取得了SOTA的成绩。