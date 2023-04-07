package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.ndarray.NDArray;
import mlfromscratch.math.Numpy;

import java.util.function.Function;

// Nesterov Accelerated Gradient
public class NesterovAcceleratedGradient extends Optimizer {
  private NDArray momentum;
  
  public NesterovAcceleratedGradient(float learning_rate, NDArray momentum) {
    this.learning_rate = learning_rate;
    this.momentum = momentum;
    this.w_updt = null;
  }
  
  public NDArray update(NDArray w, Function<NDArray, NDArray> grad_func) {
    // Calculate the gradient of the loss a bit further down the slope from w
    NDArray approx_future_grad = Numpy.clip(grad_func.apply(w.subtract(this.momentum.multiply(this.w_updt))), -1, 1);
    // Initialize on first update
    if(!this.w_updt.any()) {
      this.w_updt = Numpy.zeros(Numpy.shape(w));
    }
  
    this.w_updt = this.momentum.multiply(this.w_updt).add(NDArray.of(this.learning_rate).multiply(approx_future_grad));
    // Move against the gradient to minimize loss
    return w.subtract(this.w_updt);
  }
  
  @Override
  NDArray update(NDArray w, NDArray gradWrtW) {
    return null;
  }
}