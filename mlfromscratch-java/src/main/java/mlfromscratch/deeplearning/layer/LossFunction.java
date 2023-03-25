package mlfromscratch.deeplearning.layer;

import mlfromscratch.math.NDArray;

public abstract class LossFunction {
  public abstract NDArray acc(NDArray y, NDArray yPred);
  
  public abstract NDArray loss(NDArray y, NDArray yPred);
  
  public abstract NDArray gradient(NDArray y, NDArray yPred);
}