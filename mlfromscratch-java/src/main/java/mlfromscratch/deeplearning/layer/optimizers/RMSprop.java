package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.NDArray;
import mlfromscratch.math.Numpy;

public class RMSprop extends Optimizer {
  NDArray Eg;
  double eps;
  double rho;
  
  public RMSprop(double learning_rate/*=0.01*/, double rho/*=0.9*/) {
    this.learning_rate = learning_rate;
    this.Eg = null; //Running average of the square gradients at w
    this.eps = 1e-8;
    this.rho = rho;
  }
  
  public NDArray update(NDArray w, NDArray grad_wrt_w) {
    // If not initialized
    if (this.Eg == null) {
      this.Eg = Numpy.zeros(Numpy.shape(grad_wrt_w));
    }
  
    this.Eg = NDArray.of(this.rho).dot(this.Eg).add(NDArray.of(1 - this.rho).dot(Numpy.power(grad_wrt_w, 2)));
  
    // Divide the learning rate for a weight by a running average of the magnitudes of recent
    // gradients for that weight
    return w.subtract(NDArray.of(this.learning_rate).dot(grad_wrt_w).divide(Numpy.sqrt(this.Eg.add(NDArray.of(this.eps)))));
  }
}