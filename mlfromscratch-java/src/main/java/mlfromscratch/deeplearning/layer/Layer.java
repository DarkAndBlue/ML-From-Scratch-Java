package mlfromscratch.deeplearning.layer;

import mlfromscratch.deeplearning.layer.optimizers.Optimizer;
import mlfromscratch.math.ndarray.NDArray;

public abstract class Layer {
  public int input_shape;
  
  public abstract void initialize(Optimizer optimizer);
  
  public abstract void setInputShape(int[] shape);
  
  public abstract int[] outputShape();
  
  // indicates if the layer weights are freezed
  public boolean trainable;
  
  public abstract NDArray forward_pass(NDArray layerOutput, boolean training);
  
  public abstract NDArray backward_pass(NDArray lossGrad);
  
  public String layer_name() {
    return this.getClass().getSimpleName(); // TODO: might be not the right class
  }
  
  public abstract int parameters();
  
  public abstract NDArray output_shape();
}