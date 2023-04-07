package mlfromscratch.examples;

import mlfromscratch.deeplearning.Datasets;
import mlfromscratch.deeplearning.NeuralNetwork;
import mlfromscratch.deeplearning.layer.activations.ReLU;
import mlfromscratch.deeplearning.layer.layers.*;
import mlfromscratch.deeplearning.layer.optimizers.Optimizer;
import mlfromscratch.deeplearning.layer.lossfunctions.CrossEntropy;
import mlfromscratch.deeplearning.layer.optimizers.Adam;
import mlfromscratch.math.ndarray.NDArray;
import mlfromscratch.rendering.Plot;

import java.util.List;


public class ConvolutionalNeuralNetwork {
  
  // ----------
  // Conv Net
  // ----------
  
  public static void main(String[] args) {
    Optimizer optimizer = new Adam(0.001f, 0.9f, 0.999f);
    
    Datasets.Data data = Datasets.load_digits();
    NDArray X = data.data;
    NDArray y = data.target;
    
    // Convert to one-hot encoding
    y = DataManipulation.to_categorical(y.astype("int"));
    
    NDArray[] split = DataManipulation.train_test_split(X, y, /*test_size=*/0.4, /*seed=*/1);
    NDArray X_train = split[0];
    NDArray X_test = split[1];
    NDArray y_train = split[2];
    NDArray y_test = split[3];
    
    // Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape(-1, 1, 8, 8);
    X_test = X_test.reshape(-1, 1, 8, 8);
    
    NeuralNetwork clf = new NeuralNetwork(
      optimizer,
      /*loss=*/ new CrossEntropy(),
      /*validation_data=*/ X_test, y_test
    );
    
    clf.add(new Conv2D(/*n_filters=*/16, /*filter_shape=*/new float[] { 3f, 3f }, /*stride=*/1, /*input_shape=*/new float[] { 1f, 8f, 8f }, /*padding=*/"same"));
    clf.add(new ReLU());
    clf.add(new Dropout(0.25));
    clf.add(new BatchNormalization());
    clf.add(new Conv2D(/*n_filters=*/32, /*filter_shape=*/new float[] { 3f, 3f }, /*stride*/1, /*padding*/"same"));
    clf.add(new ReLU());
    clf.add(new Dropout(0.25));
    clf.add(new BatchNormalization());
    clf.add(new Flatten());
    clf.add(new Dense(256));
    clf.add(new ReLU());
    clf.add(new Dropout(0.4));
    clf.add(new BatchNormalization());
    clf.add(new Dense(10));
    clf.add(new Softmax());
    
    System.out.println();
    clf.summary(/*name=*/"ConvNet");
    
    List<Double>[] errors = clf.fit(X_train, y_train, /*n_epochs=*/50, /*batch_size=*/256);
    List<Double> train_err = errors[0];
    List<Double> val_err = errors[1];
    
    // Training and validation error plot
    int n = len(train_err);
    training, =plt.plot(range(n), train_err, label = "Training Error");
    validation, =plt.plot(range(n), val_err, label = "Validation Error");
    Plot plt = new Plot();
    plt.legend(/*handles=[*/training, validation);
    plt.title("Error Plot");
    plt.ylabel('Error');
    plt.xlabel('Iterations');
    plt.show();
    
    _, accuracy = clf.test_on_batch(X_test, y_test)
    print("Accuracy:", accuracy)
    
    y_pred = np.argmax(clf.predict(X_test), axis = 1)
    X_test = X_test.reshape(-1, 8 * 8)
    // Reduce dimension to 2D using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title = "Convolutional Neural Network", accuracy = accuracy, legend_labels = range(10))
  }
}