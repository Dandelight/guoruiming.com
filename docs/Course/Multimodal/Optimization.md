For some basic knowledge you can see [Deep Learning](../Deep Learning/cs231n/01).

## Adaptive Learning Rate

Saddle points can be very challenging -- Saddles

- To detect saddle points

No.1 Overfit you training data -- you model has the flexibility to get there.

## Regularization

- Parameter Regularization
  - $\ell_1$ forces some variables to be zero to preserve sparsity.
- Maximum a posteriori (MAP) estimation
- Structural Regularization
  - Go for simpler models and slowly go more and more complex.
  - User tasks specific models

## Co-adaptation

- Dropout: multiply the output of a hidden layer with a mask of 0s and 1s
  - Backward: multiply the weights by $1-p_i$.
  - Stop co-adaptation and learn ensemble
- Other variations
  - Gaussian dropout: multiply with a Gaussian with mean 1
  - Swapout: Allow skip connections to happen (?)

## Multimodal optimization

- Biggest challenge
- Pretraining
