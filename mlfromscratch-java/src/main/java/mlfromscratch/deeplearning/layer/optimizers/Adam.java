package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.ndarray.NDArray;
import mlfromscratch.math.Numpy;

public class Adam extends Optimizer {
  NDArray m;
  double eps;
  NDArray v;
  double b1;
  double b2;
  
  public Adam(double learning_rate/*=0.001*/, double b1/*=0.9*/, double b2/*=0.999*/) {
    this.learning_rate = learning_rate;
    this.eps = 1e-8;
    this.m = null;
    this.v = null;
    // Decay rates
    this.b1 = b1;
    this.b2 = b2;
  }
  
  NDArray update(NDArray w, NDArray grad_wrt_w) {
    // If not initialized
    if (this.m == null) {
      this.m = Numpy.zeros(Numpy.shape(grad_wrt_w));
      this.v = Numpy.zeros(Numpy.shape(grad_wrt_w));
    }
    
    this.m = NDArray.of(this.b1).dot(this.m).add(NDArray.of(1 - this.b1).dot(grad_wrt_w));
    this.v = NDArray.of(this.b2).dot(this.v).add(NDArray.of(1 - this.b2).dot(Numpy.power(grad_wrt_w, 2)));
    
    NDArray m_hat = this.m.divide(NDArray.of(1 - this.b1));
    NDArray v_hat = this.v.divide(NDArray.of(1 - this.b2));
    
    this.w_updt = NDArray.of(this.learning_rate).dot(m_hat).divide(Numpy.sqrt(v_hat).add(NDArray.of(this.eps)));
    
    return w.subtract(this.w_updt);
  }
}