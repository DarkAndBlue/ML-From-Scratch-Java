package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class LeakyReLU extends Activation {
  float alpha = 0.2f;
  public LeakyReLU(String name) {
    super(name);
  }
  
  public LeakyReLU(String name, float alpha) {
    super(name);
    this.alpha = alpha;
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    return Numpy.where((a) -> a >= 0, x, (b) -> b * alpha);
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    return null;
  }
}