package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class Sigmoid extends Activation {
  
  public Sigmoid(String name) {
    super(name);
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return NDArray.ONE.divide(NDArray.ONE.add(Numpy.exp(x.invert())));
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray X) {
    return null;
  }
}