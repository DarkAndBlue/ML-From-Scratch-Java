package mlfromscratch.math;

import mlfromscratch.math.ndarray.NDArray;

import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

public class Numpy {
  public static Random random;
  public class Random {
    public static NDArray uniform(float from, float to, float idk, float idk2) {
      
    }
  }
  
  public static int prod(int[] array) { // TODO: is the input "int[]" correct?
    
  }
  
  // for importing the class staticly
  //public static Numpy np = new Numpy();
  
  public static NDArray exp(NDArray input) {
    
  }
  public static float mean(NDArray input) {
    // TODO: implement this method
  }
  
  // First number is width, second height
  public static NDArray zeros(int... shape) {
    
  }
  
  public static int[] shape(NDArray ndArray) {
    
  }
  
  public static NDArray clip(NDArray apply, int i, int i1) {
  }
  
  public static NDArray power(NDArray gradWrtW, double power) {
  }
  
  public static NDArray sqrt(NDArray add) {
  }
  
  public static NDArray subtract(Number number, NDArray other) {
    
  }
  
  public static float mean(List<Float> values) {
  }
  
  public static NDArray max(NDArray x, int i, boolean b) {
  }
  
  public static NDArray sum(NDArray eX, int i, boolean b) {
  }
  
  public static NDArray where(Predicate<Float> condition, NDArray x, int i) {
  }
  public static NDArray where(Predicate<Float> condition, NDArray x, int i, int j) {
  }
  
  /*
  Assuming that "calculate" calculates the result from each element inside the NDArray
   */
  public static NDArray where(Predicate<Float> condition, NDArray x, Function<Float, Float> calculate) {
  }
  
  public static NDArray log(NDArray add) {
  }
}