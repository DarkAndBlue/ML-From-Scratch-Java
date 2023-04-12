package mlfromscratch.deeplearning.layer.activations;

import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class TanH extends Activation {
  public TanH(String name) {
    super(name);
  }
  
  @Override
  public NDArray __call__(NDArray x) {
    NDArray e_x = Numpy.exp(x.subtract(Numpy.max(x, /*axis=*/-1, /*keepdims=*/true)));
    return e_x.divide(Numpy.sum(e_x, /*axis=*/-1, /*keepdims=*/true));
  }
  
  @Override
  public NDArray activation_func_gradient(NDArray x) {
    NDArray p = this.__call__(x);
    return p.multiply(NDArray.ONE.subtract(p));
  }
}