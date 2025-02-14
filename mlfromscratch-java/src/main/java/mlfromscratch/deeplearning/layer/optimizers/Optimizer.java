package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.ndarray.NDArray;

public abstract class Optimizer {
  public float learning_rate;
  public NDArray w_updt;
  
  public abstract NDArray update(NDArray w, NDArray gradWrtW);
  
  public abstract Optimizer copy();
}