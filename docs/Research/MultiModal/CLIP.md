# Learning Transferable Visual Models from Natural Language Supervision

State-of-the-art computer vision models are trained on a fixed set of predefined object categories. This restricted form of supervision limits their generality and usability since labeled data is required to learn any visual concept. This requires lots of crowd labeling. Learning directly from raw text about images is a promising alternative which leverages a much boarder source of supervision.

The authors propose a model called Contrastive Language-Image Pre-training (CLIP), an efficient method of learning from natural language supervision.

## Model

I think the method is modeled after the SimCLR. The loss is the same, except that in CLIP, the temprature $\tau$ is a trainable parameter.;
