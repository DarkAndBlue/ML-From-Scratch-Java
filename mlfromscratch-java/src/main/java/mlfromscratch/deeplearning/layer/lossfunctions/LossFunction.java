package mlfromscratch.deeplearning.layer.lossfunctions;

import mlfromscratch.deeplearning.layer.Layer;
import mlfromscratch.math.NDArray;

public abstract class LossFunction extends Layer {
  public abstract NDArray acc(NDArray y, NDArray yPred);
  
  public abstract NDArray loss(NDArray y, NDArray yPred);
  
  public abstract NDArray gradient(NDArray y, NDArray yPred);
}