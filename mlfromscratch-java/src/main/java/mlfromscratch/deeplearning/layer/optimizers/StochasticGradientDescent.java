package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.ndarray.NDArray;
import mlfromscratch.math.Numpy;

public class StochasticGradientDescent extends Optimizer {
  private NDArray momentum;
  
  public StochasticGradientDescent(float learning_rate, NDArray momentum) {
    this.learning_rate = learning_rate;
    this.momentum = momentum;
    this.w_updt = null;
  }
  
  public NDArray update(NDArray w, NDArray gradWrtW) {
    // If not initialized
    if (this.w_updt == null) {
      this.w_updt = Numpy.zeros(Numpy.shape(w));
    }
    // Use momentum if set
    this.w_updt = this.momentum.multiply(this.w_updt).add((NDArray.ONE.subtract(this.momentum).multiply(gradWrtW)));
    // Move against the gradient to minimize loss
    return w.subtract(NDArray.of(this.learning_rate).multiply(this.w_updt));
  }
}