package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class ReLU extends Activation {
  public ReLU(String name) {
    super(name);
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return Numpy.where((a) -> a >= 0, x, 0);
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    return Numpy.where((a) -> a >= 0, x, 1, 0);
  }
}