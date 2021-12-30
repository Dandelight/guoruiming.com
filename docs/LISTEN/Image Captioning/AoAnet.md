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
pip install h5py
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
