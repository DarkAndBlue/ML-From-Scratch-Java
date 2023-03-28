package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.NDArray;
import mlfromscratch.math.Numpy;

public class Adagrad extends Optimizer {
  private NDArray G; // Sum of squares of the gradients
  private final double eps = 0.00000001d;
  
  public Adagrad(double learning_rate) {
    this.learning_rate = learning_rate;
  }
  
  public NDArray update(NDArray w, NDArray grad_wrt_w) {
    // If not initialized
    if (this.G == null) {
      this.G = Numpy.zeros(Numpy.shape(w));
    }
    // Add the square of the gradient of the loss function at w
    this.G = this.G.add(Numpy.power(grad_wrt_w, 2));
    // Adaptive gradient with higher learning rate for sparse data
    return w.subtract(NDArray.of(this.learning_rate).dot(grad_wrt_w).divide(Numpy.sqrt(this.G.add(NDArray.of(this.eps)))));
  }
}