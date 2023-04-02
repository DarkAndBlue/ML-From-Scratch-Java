package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.ndarray.NDArray;
import mlfromscratch.math.Numpy;

public class StochasticGradientDescent extends Optimizer {
  private NDArray momentum;
  
  public StochasticGradientDescent(double learning_rate, NDArray momentum) {
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
    this.w_updt = this.momentum/* * */.dot(this.w_updt).add((NDArray.of(1).subtract(this.momentum)).dot(gradWrtW));
    // Move against the gradient to minimize loss
    return w.subtract(NDArray.of(this.learning_rate).dot(this.w_updt));
  }
}