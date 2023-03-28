package mlfromscratch.deeplearning;

import mlfromscratch.StringUtil;
import mlfromscratch.deeplearning.layer.Layer;
import mlfromscratch.deeplearning.layer.lossfunctions.LossFunction;
import mlfromscratch.deeplearning.layer.optimizers.Optimizer;
import mlfromscratch.math.NDArray;
import mlfromscratch.math.Numpy;
import mlfromscratch.rendering.ProgressBar;
import mlfromscratch.rendering.WidgetsEnum;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class NeuralNetwork {
  Optimizer optimizer;
  List<Layer> layers;
  HashMap<String, ArrayList<Double>> errors; // "training" and "validation" errors inside NDArray
  LossFunction loss_function;
  ProgressBar progressbar;
  HashMap<String, NDArray> val_set;
  
  /*Neural Network. Deep Learning base model.

 Parameters:
 -----------
 optimizer: class
     The weight optimizer that will be used to tune the weights in order of minimizing
     the loss.
 loss: class
     Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
 validation: tuple
     A tuple containing validation data and labels (X, y)
 */
  public NeuralNetwork(Optimizer optimizer, LossFunction loss, NDArray X_test, NDArray y_test) {
    this.optimizer = optimizer;
    this.layers = new ArrayList<>();
    this.errors = new HashMap<>();
    this.loss_function = loss;
    this.progressbar = new ProgressBar(WidgetsEnum.BAR_WIDGET);
  
    HashMap<String, NDArray> val_set = new HashMap<>();
    val_set.put("X", X_test);
    val_set.put("y", y_test);
    this.val_set = val_set;
  }
  
  public NeuralNetwork(Optimizer optimizer, LossFunction loss) {
    this(optimizer, loss, null);
  }
  
  // TODO: maybe replace the trainable boolean with stopping the training threads later
  void set_trainable(boolean trainable) {
    // Method which enables freezing of the weights of the network's layers.
    for (Layer layer : this.layers) {
      layer.trainable = trainable;
    }
  }
  
  public void add(Layer layer) {
    /* Method which adds a layer to the neural network */
    // If this is not the first layer added then set the input shape
    // to the output shape of the last added layer
    if (!this.layers.isEmpty())
      layer.setInputShape(this.layers.get(this.layers.size() - 1).outputShape());
    
    // If the layer has weights that needs to be initialized 
    if (hasAttr(layer, "initialize"))
      layer.initialize(this.optimizer);
    
    // Add layer to the network
    this.layers.add(layer);
  }
  
  Object[] test_on_batch(NDArray X, NDArray y) {
    // Evaluates the model over a single batch of samples
    NDArray y_pred = this._forward_pass(X, false/*training = False*/);
    double loss = Numpy.mean(this.loss_function.loss(y, y_pred));
    NDArray acc = this.loss_function.acc(y, y_pred);
    
    return new Object[] { loss, acc };
  }
  
  Object[] train_on_batch(NDArray X, NDArray y) {
    // Single gradient update over one batch of samples
    NDArray y_pred = this._forward_pass(X, true);
    double loss = Numpy.mean(this.loss_function.loss(y, y_pred));
    NDArray acc = this.loss_function.acc(y, y_pred);
    // Calculate the gradient of the loss function wrt y_pred
    NDArray loss_grad = this.loss_function.gradient(y, y_pred);
    // Backpropagate. Update weights
    this._backward_pass(loss_grad);
    
    return new Object[] { loss, acc };
  }
  
  public List<Double>[] fit(NDArray X, NDArray y, int n_epochs, int batch_size) {
    // Trains the model for a fixed number of epochs
    for (int i = 0; i < n_epochs; i++) {
      this.progressbar.setState(i, n_epochs);
      NDArray batch_error = new NDArray();
      for(NDArray[] element : DataManipulation.batch_iterator(X, y, batch_size)) {
        NDArray X_batch = element[0];
        NDArray y_batch = element[1];
        double loss = (Double) this.train_on_batch(X_batch, y_batch)[0];
        batch_error.append(loss);
      }
      
      this.errors.get("training").add(Numpy.mean(batch_error));
      
      if(this.val_set != null) {
        Object[] result = this.test_on_batch(this.val_set.get("X"), this.val_set.get("y"));
        double val_loss = (Double) result[0];
        this.errors.get("validation").add(val_loss);
      }
    }
    
    List<Double>[] lists = new ArrayList[] { this.errors.get("training"), this.errors.get("validation") };
    return lists;
  }
  
  NDArray _forward_pass(NDArray X, boolean training /*=True*/) {
    // Calculate the output of the NN
    NDArray layer_output = X;
    for (Layer layer : this.layers)
      layer_output = layer.forward_pass(layer_output, training);
    
    return layer_output;
  }
  
  void _backward_pass(NDArray loss_grad) {
    // Propagate the gradient 'backwards' and update the weights in each layer
    for (int i = this.layers.size() - 1; i >= 0; i--) {
      Layer layer = layers.get(i);
      loss_grad = layer.backward_pass(loss_grad);
    }
  }
  
  public void summary(String name/*="Model Summary"*/) {
    // Print model name
    System.out.println(StringUtil.createStringTable(name));
    // Network input shape (first layer's input shape)
    System.out.println("Input Shape: " + this.layers.get(0).input_shape);
    // Iterate through network and get each layer's configuration
    String[][] table_data = new String[this.layers.size()][];
    table_data[0] = new String[] { "Layer Type", "Parameters", "Output Shape" };
    int tot_params = 0;
    for (int i = 0; i < this.layers.size(); i++) {
      Layer layer = this.layers.get(i);
      String layer_name = layer.layer_name();
      int params = layer.parameters();
      NDArray out_shape = layer.output_shape();
      table_data[i] = new String[] { layer_name, "" + params, out_shape.toString() };
      tot_params += params;
    }
    // Print network configuration table
    System.out.println(StringUtil.createStringTable(table_data));
    System.out.println("Total Parameters: " + tot_params);
  }
  
  NDArray predict(NDArray X) {
    // Use the trained model to predict labels of X
    return this._forward_pass(X, false/*training=False*/);
  }
}