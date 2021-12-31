![image-20211230200448872](media/AoAnet/image-20211230200448872.png)

Tech Report

# 在 COCO 2014 数据集上训练 AoAnet

基于[AoAnet](https://github.com/husthuaan/AoANet)进行训练，参考文献：https://arxiv.org/abs/1908.06954

```tex
@inproceedings{huang2019attention,
  title={Attention on Attention for Image Captioning},
  author={Huang, Lun and Wang, Wenmin and Chen, Jie and Wei, Xiao-Yong},
  booktitle={International Conference on Computer Vision},
  year={2019}
}
```

## 数据

### 数据集的下载

#### COCO 图片

项目需要 COCO 2014 数据集的 train 和 val，可以从官网下载：

```bash
wget -c http://images.cocodataset.org/zips/train2014.zip http://images.cocodataset.org/annotations/image_info_val2014.zip
```

> 全部链接：
>
> http://images.cocodataset.org/zips/train2014.zip
>
> http://images.cocodataset.org/annotations/annotations_trainval2014.zip
>
> http://images.cocodataset.org/zips/val2014.zip
>
> http://images.cocodataset.org/annotations/image_info_val2014.zip
>
> http://images.cocodataset.org/zips/test2014.zip
>
> http://images.cocodataset.org/annotations/image_info_test2014.zip

也可以通过 Redmon 的镜像站：https://pjreddie.com/projects/coco-mirror/

**注意`train2014`中有一张图像坏掉了，需要替换**，[参考](https://github.com/karpathy/neuraltalk2/issues/4)，替换方式：

```bash
cd train2014
curl https://msvocds.blob.core.windows.net/images/262993_z.jpg >> COCO_train2014_000000167126.jpg
```

#### 预处理后的描述

```bash
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
```

### 代码仓库

```bash
git clone https://github.com/husthuaan/AoANet.git --recursive
```

注意两个`submodule`是不是都下载下来了。

### 数据预处理

#### 预处理标注

下面这个预处理需要`python2`的环境，搭建环境：

```bash
conda create -n py2 python<3
conda activate py2
conda install pytorch cpuonly -c pytorch
conda install six
```

```bash
python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

本项目使用 ResNet 提取特征，预处理流程如下。

下载模型：

```bash
curl https://download.pytorch.org/models/resnet101-63fe2227.pth >> ./data/imagenet_weights/resnet101.pth
```

创建环境，此处使用`docker`

```yaml
# docker-compose.yml
version: "3"
services:
  pytorch:
    image: "ufoym/deepo:pytorch-cu111"
    ports:
      - "0:22"
    volumes:
      - $HOME:$HOME
      - /nvme:/nvme
      # 记得把代码所在目录映射进来
    shm_size: "32gb"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
```

```bash
docker-compose up -d
docker exec -it AoAnet_pytorch bash # 进入容器
cd # 代码所在目录
pip install h5py yacs lmdbdict pyemd
export PYTHONPATH=`pwd`
```

然后就可以开始进行预处理

```bash
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

```bash
python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```

预处理速度很慢，耐心等待

在训练时遇到了问题，没库，没依赖，作者也不提供一下参考

```bash
sudo apt install openjdk-8-jdk # 依赖
pip install gensim # NLP库

# 下载Word2Vec的一个模型
cd coco-caption/pycocoevalcap/wmd/data
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
```

```python
from gensim import models
w = models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
```

作者怎么不把依赖写全，非要等到跑到一半扔个异常出来。气。抓异常只抓`(RuntimeError, KeyboardInterrupt)`，您就没考虑过有人可能没装全依赖吗。
