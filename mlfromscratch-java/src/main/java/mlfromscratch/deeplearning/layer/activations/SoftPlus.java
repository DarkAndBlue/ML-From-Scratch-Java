package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class SoftPlus extends Activation {
  public SoftPlus(String name) {
    super(name);
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return Numpy.log(NDArray.ONE.add(Numpy.exp(x)));
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    return NDArray.ONE.divide(NDArray.ONE.add(Numpy.exp(x.invert())));
  }
}