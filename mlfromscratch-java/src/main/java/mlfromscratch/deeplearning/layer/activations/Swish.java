package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class Swish extends Activation {
  float beta = 1;
  
  public Swish(String name) {
    super(name);
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return x.multiply(this.__sigmoid(NDArray.of(this.beta).multiply(x)));
  }
  
  private NDArray __sigmoid(NDArray x) {
    return NDArray.ONE.divide(NDArray.ONE.add(Numpy.exp(x.invert())));
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    return null;
  }
}