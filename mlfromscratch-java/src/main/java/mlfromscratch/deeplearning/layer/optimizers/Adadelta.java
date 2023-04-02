package mlfromscratch.deeplearning.layer.optimizers;

import mlfromscratch.math.ndarray.NDArray;
import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.Vector;

public class Adadelta extends Optimizer {
  NDArray E_w_updt;
  NDArray E_grad;
  NDArray w_updt;
  double eps;
  double rho;
  
  public Adadelta(double rho/*=0.95*/, double eps/*=1e-6*/) {
    this.E_w_updt = null; //Running average of squared parameter updates
    this.E_grad = null;   //Running average of the squared gradient of w
    this.w_updt = null;   //Parameter update
    this.eps = eps;
    this.rho = rho;
  }
  
  NDArray update(NDArray w, NDArray grad_wrt_w) {
    // If not initialized
    if (this.w_updt == null) {
      this.w_updt = Numpy.zeros(Numpy.shape(w));
      this.E_w_updt = Numpy.zeros(Numpy.shape(w));
      this.E_grad = Numpy.zeros(Numpy.shape(grad_wrt_w));
    }
  
    // Update average of gradients at w
    this.E_grad = this.E_grad.multiply(this.rho).add(NDArray.of(1 - this.rho).multiply(Numpy.power(grad_wrt_w, 2)));
  
    NDArray RMS_delta_w = Numpy.sqrt(this.E_w_updt.add(NDArray.of(this.eps)));
    NDArray RMS_grad = Numpy.sqrt(this.E_grad.add(NDArray.of(this.eps)));
    
    // Adaptive learning rate
    NDArray adaptive_lr = RMS_delta_w.divide(RMS_grad);
    
    // Calculate the update
    this.w_updt = adaptive_lr.dot(grad_wrt_w);
    
    // Update the running average of w updates
    this.E_w_updt = NDArray.of(this.rho).dot(this.E_w_updt).add(NDArray.of(1 - this.rho).dot(Numpy.power(this.w_updt, 2)));
    
    return w.subtract(this.w_updt);
  }
}