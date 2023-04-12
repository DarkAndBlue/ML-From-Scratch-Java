package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class SELU extends Activation {
  float alpha = 1.6732632423543772848170429916717f;
  float scale = 1.0507009873554804934193349852946f;
  
  public SELU(String name) {
    super(name);
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return this.scale * Numpy.where((a) -> a >= 0.0, x, NDArray.of(this.alpha).multiply(Numpy.exp(x).subtract(NDArray.ONE)))
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    return this.scale * Numpy.where((a) -> a >= 0.0, 1, NDArray.of(this.alpha).multiply(Numpy.exp(x)));
  }
}