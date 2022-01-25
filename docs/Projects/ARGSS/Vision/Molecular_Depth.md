```tex
@article{9594697,
  author   = {Wu, Lin and Wang, Teng and Sun, Changyin},
  journal  = {IEEE Signal Processing Letters},
  title    = {Multi-Modal Visual Place Recognition in Dynamics-Invariant Perception Space},
  year     = {2021},
  volume   = {28},
  number   = {},
  pages    = {2197-2201},
  abstract = {Visual place recognition is one of the essential and challenging problems in the fields of robotics. In this letter, we for the first time explore the use of multi-modal fusion of semantic and visual modalities in dynamics-invariant space to improve place recognition in dynamic environments. We achieve this by first designing a novel deep learning architecture to generate the static semantic segmentation and recover the static image directly from the corresponding dynamic image. We then innovatively leverage the spatial-pyramid-matching model to encode the static semantic segmentation into feature vectors. In parallel, the static image is encoded using the popular Bag-of-words model. On the basis of the above multi-modal features, we finally measure the similarity between the query image and target landmark by the joint similarity of their semantic and visual codes. Extensive experiments demonstrate the effectiveness and robustness of the proposed approach for place recognition in dynamic environments.},
  keywords = {},
  doi      = {10.1109/LSP.2021.3123907},
  issn     = {1558-2361},
  month    = {}
}

@inproceedings{Li2019,
   abstract = {We propose a self-supervised learning framework for visual odometry (VO) that incorporates correlation of consecutive frames and takes advantage of adversarial learning. Previous methods tackle self-supervised VO as a local structure from motion (SfM) problem that recovers depth from single image and relative poses from image pairs by minimizing photometric loss between warped and captured images. As single-view depth estimation is an ill-posed problem, and photometric loss is incapable of discriminating distortion artifacts of warped images, the estimated depth is vague and pose is inaccurate. In contrast to previous methods, our framework learns a compact representation of frame-to-frame correlation, which is updated by incorporating sequential information. The updated representation is used for depth estimation. Besides, we tackle VO as a self-supervised image generation task and take advantage of Generative Adversarial Networks (GAN). The generator learns to estimate depth and pose to generate a warped target image. The discriminator evaluates the quality of generated image with high-level structural perception that overcomes the problem of pixel-wise loss in previous methods. Experiments on KITTI and Cityscapes datasets show that our method obtains more accurate depth with details preserved and predicted pose outperforms state-of-the-art self-supervised methods significantly.},
   author = {Shunkai Li and Fei Xue and Xin Wang and Zike Yan and Hongbin Zha},
   doi = {10.1109/ICCV.2019.00294},
   isbn = {9781728148038},
   issn = {15505499},
   journal = {Proceedings of the IEEE International Conference on Computer Vision},
   month = {10},
   pages = {2851-2860},
   publisher = {Institute of Electrical and Electronics Engineers Inc.},
   title = {Sequential adversarial learning for self-supervised deep visual odometry},
   volume = {2019-October},
   year = {2019},
}

@inproceedings{Wofk2019,
   abstract = {Depth sensing is a critical function for robotic tasks such as localization, mapping and obstacle detection. There has been a significant and growing interest in depth estimation from a single RGB image, due to the relatively low cost and size of monocular cameras. However, state-of-the-art single-view depth estimation algorithms are based on fairly complex deep neural networks that are too slow for real-time inference on an embedded platform, for instance, mounted on a micro aerial vehicle. In this paper, we address the problem of fast depth estimation on embedded systems. We propose an efficient and lightweight encoder-decoder network architecture and apply network pruning to further reduce computational complexity and latency. In particular, we focus on the design of a low-latency decoder. Our methodology demonstrates that it is possible to achieve similar accuracy as prior work on depth estimation, but at inference speeds that are an order of magnitude faster. Our proposed network, FastDepth, runs at 178 fps on an NVIDIA Jetson TX2 GPU and at 27 fps when using only the TX2 CPU, with active power consumption under 10 W. FastDepth achieves close to state-of-the-art accuracy on the NYU Depth v2 dataset. To the best of the authors' knowledge, this paper demonstrates real-time monocular depth estimation using a deep neural network with the lowest latency and highest throughput on an embedded platform that can be carried by a micro aerial vehicle.},
   author = {Diana Wofk and Fangchang Ma and Tien Ju Yang and Sertac Karaman and Vivienne Sze},
   doi = {10.1109/ICRA.2019.8794182},
   issn = {10504729},
   journal = {Proceedings - IEEE International Conference on Robotics and Automation},
   title = {FastDepth: Fast monocular depth estimation on embedded systems},
   volume = {2019-May},
   year = {2019},
}

@article{Zaki2017,
   abstract = {Recognizing semantic category of objects and scenes captured using vision-based sensors is a challenging yet essential capability for mobile robots and UAVs to perform high-level tasks such as long-term autonomous navigation. However, extracting discriminative features from multi-modal inputs, such as RGB-D images, in a unified manner is non-trivial given the heterogeneous nature of the modalities. We propose a deep network which seeks to construct a joint and shared multi-modal representation through bilinearly combining the convolutional neural network (CNN) streams of the RGB and depth channels. This technique motivates bilateral transfer learning between the modalities by taking the outer product of each feature extractor output. Furthermore, we devise a technique for multi-scale feature abstraction using deeply supervised branches which are connected to all convolutional layers of the multi-stream CNN. We show that end-to-end learning of the network is feasible even with a limited amount of training data and the trained network generalizes across different datasets and applications. Experimental evaluations on benchmark RGB-D object and scene categorization datasets show that the proposed technique consistently outperforms state-of-the-art algorithms.},
   author = {Hasan F.M. Zaki and Faisal Shafait and Ajmal Mian},
   doi = {10.1016/j.robot.2017.02.008},
   issn = {09218890},
   journal = {Robotics and Autonomous Systems},
   title = {Learning a deeply supervised multi-modal RGB-D embedding for semantic scene and object category recognition},
   volume = {92},
   year = {2017},
}

@article{Wang2021,
   abstract = {Depth estimation is crucial to understanding the geometry of a scene in robotics and computer vision. Traditionally, depth estimators can be trained with various forms of self-supervised stereo data or supervised ground-truth data. In comparison to the methods that utilize stereo depth perception or ground-truth data from laser scans, determining depth relation using an unlabeled monocular camera proves considerably more challenging. Recent work has shown that CNN-based depth estimators can be learned using unlabeled monocular video. Without needing the stereo data or ground-truth depth data, learning with monocular self-supervised strategies can utilize much larger and more varied image datasets. Inspired by recent advances in depth estimation, in this paper, we propose a novel objective that replaces the use of explicit ground-truth depth or binocular stereo depth with unlabeled monocular video sequence data. No assumptions about scene geometry or pre-trained information are used in the proposed architecture. To enable a better pose prediction, we propose the use of an improved differentiable direct visual odometry (DDVO), which is fused with an appearance-matching loss. The auto-masking approach is introduced in the DDVO depth predictor to filter out the low-texture area or occlusion area, which can easily reduce matching error, from one frame to the subsequent frame in the monocular sequence. Additionally, we introduce a self-supervised loss function to fuse the auto-masking segment and the depth-prediction segment accordingly. Our method produces state-of-the-art results for monocular depth estimation on the KITTI driving dataset, even outperforming some supervised methods that have been trained with ground-truth depth.},
   author = {Haixia Wang and Yehao Sun and Q. M.Jonathan Wu and Xiao Lu and Xiuling Wang and Zhiguo Zhang},
   doi = {10.1016/j.neucom.2020.10.025},
   issn = {18728286},
   journal = {Neurocomputing},
   title = {Self-supervised monocular depth estimation with direct methods},
   volume = {421},
   year = {2021},
}

@inproceedings{Gordon2019,
   abstract = {We present a novel method for simultaneous learning of depth, egomotion, object motion, and camera intrinsics from monocular videos, using only consistency across neighboring video frames as supervision signal. Similarly to prior work, our method learns by applying differentiable warping to frames and comparing the result to adjacent ones, but it provides several improvements: We address occlusions geometrically and differentiably, directly using the depth maps as predicted during training. We introduce randomized layer normalization, a novel powerful regularizer, and we account for object motion relative to the scene. To the best of our knowledge, our work is the first to learn the camera intrinsic parameters, including lens distortion, from video in an unsupervised manner, thereby allowing us to extract accurate depth and motion from arbitrary videos of unknown origin at scale. We evaluate our results on the Cityscapes, KITTI and EuRoC datasets, establishing new state of the art on depth prediction and odometry, and demonstrate qualitatively that depth prediction can be learned from a collection of YouTube videos. The code will be open sourced once anonymity is lifted.},
   author = {Ariel Gordon and Hanhan Li and Rico Jonschkowski and Anelia Angelova},
   doi = {10.1109/ICCV.2019.00907},
   issn = {15505499},
   journal = {Proceedings of the IEEE International Conference on Computer Vision},
   title = {Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras},
   volume = {2019-October},
   year = {2019},
}

```

```
arXiv:2110.02178v1 MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER
```

```tex

```
