package mlfromscratch.deeplearning.layer.layers;

import mlfromscratch.deeplearning.layer.optimizers.Optimizer;
import mlfromscratch.math.ndarray.NDArray;

public interface Initializable {
  // If the layer has weights that needs to be initialized
  public abstract void initialize(Optimizer optimizer);
}