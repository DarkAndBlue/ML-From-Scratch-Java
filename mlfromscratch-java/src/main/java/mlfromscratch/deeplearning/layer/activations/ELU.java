package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class ELU extends Activation {
  float alpha = 0.1f;
  public ELU(String name) {
    super(name);
  }
  public ELU(String name, float alpha) {
    super(name);
    this.alpha = alpha;
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return Numpy.where((a) -> a >= 0, x, NDArray.of(this.alpha).multiply((Numpy.exp(x).subtract(NDArray.ONE))));
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    return null;
  }
}