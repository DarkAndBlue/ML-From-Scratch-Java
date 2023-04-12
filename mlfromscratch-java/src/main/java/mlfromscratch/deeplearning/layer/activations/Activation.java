package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.deeplearning.layer.Layer;
import mlfromscratch.math.ndarray.NDArray;

public abstract class Activation extends Layer {
//  enum Activation {
//    sigmoid(new Sigmoid()),
//    relu(new ReLU());
//    
//    private final Activation activationFunction;
//    
//    Activation(Activation activationFunction) {
//      this.activationFunction = activationFunction;
//    }
//  }
  
/*
  A layer that applies an activation operation to the input.
  
  Parameters:
  -----------
  name: string
  The name of the activation function that will be used.
*/
  
  public final String name;
  
  public Activation(String name) {
    super(true);
    this.name = name;
  }
  
  @Override
  public NDArray forward_pass(NDArray X, boolean training) {
    this.layer_input = X;
    return __call__(X);
  }
  
  public abstract NDArray __call__(NDArray x);
  
  @Override
  public NDArray backward_pass(NDArray accum_grad) {
    return accum_grad.multiply(activation_func_gradient(this.layer_input));
  }
  
  @Override
  public int[] output_shape() {
    return this.input_shape;
  }
  
  public abstract NDArray activation_func_gradient(NDArray x);
  
  
  public String layer_Name() {
    return "Activation (" + this.getClass().getSimpleName() + ")";
  }
}