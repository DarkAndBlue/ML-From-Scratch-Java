package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.NDArray;

public abstract class Optimizer {
  public double learning_rate;
  public NDArray w_updt;
  
  abstract NDArray update(NDArray w, NDArray gradWrtW);
}