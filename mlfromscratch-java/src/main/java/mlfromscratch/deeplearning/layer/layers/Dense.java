package mlfromscratch.deeplearning.layer.layers;

import mlfromscratch.deeplearning.layer.Layer;
import mlfromscratch.deeplearning.layer.optimizers.Optimizer;
import mlfromscratch.math.Numpy;
import mlfromscratch.math.ndarray.NDArray;

public class Dense extends Layer implements Initializable {
  // n_units = number of nodes per layer
  int n_units;
  NDArray W;
  NDArray w0;
  Optimizer W_opt;
  Optimizer w0_opt;
  
  public Dense(int n_units, int... input_shape) {
    super(true);
    this.layer_input = null;
    this.input_shape = input_shape;
    this.n_units = n_units;
  }
  
  @Override
  public void initialize(Optimizer optimizer) {
    // Initialize the weights
    
    float limit = (float) (1f / Math.sqrt(this.input_shape[0]));
    this.W = Numpy.random.uniform(-limit, limit, this.input_shape[0], this.n_units); // TODO: correct translation? Numpy.random.uniform(-limit, limit, (this.input_shape[0], this.n_units))
    this.w0 = Numpy.zeros(1, this.n_units); // TODO: is (()) same as ()? ... Numpy.zeros((1, self.n_units));
    // Weight optimizers
    this.W_opt = optimizer.copy();
    this.w0_opt = optimizer.copy();
  }
  
  @Override
  public int parameters() {
    return Numpy.prod(this.W.shape) + Numpy.prod(this.w0.shape);
  }
  
  @Override
  public NDArray forward_pass(NDArray X, boolean training) {
    this.layer_input = X;
    return X.dot(this.W).add(this.w0);
  }
  
  @Override
  public NDArray backward_pass(NDArray accum_grad) {
    // Save weights used during forwards pass
    NDArray W = this.W.copy();
    
    if (this.trainable) {
      // Calculate gradient w.r.t layer weights
      NDArray grad_w = this.layer_input.transpose().dot(accum_grad);
      NDArray grad_w0 = Numpy.sum(accum_grad, /*axis = */0, /*keepdims = */true);
      
      // Update the layer weights
      this.W = this.W_opt.update(this.W, grad_w);
      this.w0 = this.w0_opt.update(this.w0, grad_w0);
    }
    
    // Return accumulated gradient for next layer
    // Calculated based on the weights used during the forward pass
    accum_grad = accum_grad.dot(W.transpose());
    return accum_grad;
  }
  
  @Override
  public int[] output_shape() {
    return new int[] { this.n_units }; // TODO check if correct for:   return (self.n_units, )
  }
}