package mlfromscratch.deeplearning.layer;

import mlfromscratch.math.ndarray.NDArray;

public abstract class Layer {
  public int[] input_shape;
  public NDArray layer_input;
  
  public Layer(boolean trainable) {
    this.trainable = trainable;
  }
  
  public void set_input_shape(int[] shape) {
    this.input_shape = shape;
  }
  
  public abstract int[] output_shape();
  
  // indicates if the layer weights are freezed
  public boolean trainable;
  
  public abstract NDArray forward_pass(NDArray layerOutput, boolean training);
  
  public abstract NDArray backward_pass(NDArray lossGrad);
  
  public String layer_name() {
    return this.getClass().getSimpleName(); // TODO: might be not the right class
  }
  
  // Returns the number of parameters for the layer. TODO: make it abstract when refactoring because it could be missed when implementing a new layer. 
  public int parameters() {
    return 0;
  }
}