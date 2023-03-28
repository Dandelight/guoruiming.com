# Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks

Oscar is a novel method for explicitly learning correlations between images and text with salient object as anchor points. It utilized the features extracted by image model and text model to pertrain the network. Then the pretrained network is fine-tuned on a variaty of downstream tasks.

## Problem statement

Given a dataset $\mathcal{D} = \{(\mathbf{I}_i, \boldsymbol{w}\}_{i=1}^N$, with image $\mathbf{I}$ and text sequence $\boldsymbol{w}$. With an object detection model $f$, we obtain vision region features $\boldsymbol{v}=\{v_1, \ldots, v_K\} = f(\mathbf{I})$. Similarly we get a sequence of word embeddings $\boldsymbol{w} = \{w_1, \ldots, w_T\}$. Previous vision-language pertraining (VLP) methods employs multi-layer self-attention Transformers to learn cross-modal _contextualized_ representations based on teh _sigular_ embedding of each modality.

$$
\mathit{output} = \operatorname{Transformer}(\boldsymbol{v}, \boldsymbol{w}).
$$

Though simple and effective, existing methods suffer from two issues:

1. **Ambiguity**: Vision region features extracted by Faster R-CNN inevitably overlaps with among image regions at different positions. This renders ambiguities for the extracted visual embeddings.
2. **Lack of grounding**: As a weakly-supervised task, there is no explicitly labeled alignments between $\boldsymbol{v}$ and $\boldsymbol{w}$. However, the authors observe that salient objects can be used as anchor points for learning semantic alignments.

## Method

Oscar introduces the Word-Tag-Image triple $(\boldsymbol{w}, \boldsymbol{q}, \boldsymbol{v})$, where

- $\boldsymbol{v}=(v', z)$, where $v'\in \mathbb{R}^p$ is the feature vectors from Faster R-CNN, and $z\in\mathbb{R}^R$ is the region vector ($R=4\text{ or }6$, for top-left, bottom-right, and/or height & width)
- $\boldsymbol{q}$ is the word embedding sequence of object tags (in text) detected from the image.
- $\boldsymbol{w}$ is the text emhedding sequence.

$$
\boldsymbol{x} \triangleq[\underbrace{\boldsymbol{w}}_{\text {language }}, \underbrace{\boldsymbol{q}, \boldsymbol{v}}_{\text {image }}]=[\underbrace{\boldsymbol{w}, \boldsymbol{q}}_{\text {language }}, \underbrace{\boldsymbol{v}}_{\text {image }}] \triangleq \boldsymbol{x}^{\prime},
$$

where $x$ is a _modality_ view to distinguish the representations between a text and an image; while $x'$ is a _dictionary_ view to distinguish the two different semantic spaces in which the input is represented.

**Modality view**: Contrastive Loss. For each triple, we group $\boldsymbol{h}^{\prime} \triangleq[\boldsymbol{q}, \boldsymbol{v}]$ to represent the image modality, and consider the $\boldsymbol{w}$ as the text modality. We then randomly replace 50% of $\boldsymbol{q}$ with **a** different tag sequence randomly sampled from the dataset $\mathcal{D}$. On the `[CLS]` token, the model outputs the fused vision-language representation of $(\boldsymbol{h}', \boldsymbol{w})$, the authors apply a fully-connected layer as a binary classifier to predict whether the pair contains the original image representation $(y=1)$ or **any** polluted ones $(y=0)$.

$$
\mathcal{L}_{\mathrm{C}}=-\mathbb{E}_{\left(\boldsymbol{h}^{\prime}, \boldsymbol{w}\right) \sim \mathcal{D}} \log p\left(y \mid f\left(\boldsymbol{h}^{\prime}, \boldsymbol{w}\right)\right)
$$

**Dictionary view**: Masked Token Loss. Each word randomly masked (replaced with a special token `[MASK]`) with a probability of 15%. The model is required to predict the masked word. This actually follows the setting of BERT.

$$
\mathcal{L}_{\mathrm{MTL}}=-\mathbb{E}_{(\boldsymbol{v}, \boldsymbol{h}) \sim \mathcal{D}} \log p\left(h_i \mid \boldsymbol{h}_{\backslash i}, \boldsymbol{v}\right)
$$

This Oscar method is illustrated as following:

![image-20230328113856788](./assets/Oscar/image-20230328113856788.png)

The full-pretraining objective of Oscar is:

$$
\mathcal{L}_{\text {Pre-training }}=\mathcal{L}_{\mathrm{MTL}}+\mathcal{L}_{\mathrm{C}}
$$

## Appendix

### vision and language models

Oscar method requires good features. In the paper, the authors use Faster R-CNN to extract vision features and use BERT to extract text features. We briefly introduce these methods to gain a deeper understanding of the Oscar method.

#### BERT

the training of BERT share some similarities with Oscar.

### Weakly-supervised methods

### Why it is called Oscar

Because BERT and Oscar are friends.
