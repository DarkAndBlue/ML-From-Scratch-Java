package mlfromscratch.math.ndarray;

import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Stream;

public class NDArray {
  public final int rows;
  public final int cols;
  public final float[][] data;
  public static NDArray ONE = new NDArray(1);
  public static NDArray ZERO = new NDArray(1);
  public final int[] shape; // for NDArray parity
  
  // TODO: most classes have .dot instead of .multiply because I was stupid (compare again with all python code and replace * with .multiply and not .dot)
  public NDArray(int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    
    this.data = new float[rows][cols];
    
    this.shape = new int[] { rows, cols };
  }
  
  public NDArray(float[][] data) {
    this.data = data;
    
    this.rows = data.length;
    this.cols = data[0].length;
    
    this.shape = new int[] { rows, cols };
  }
  
  public NDArray(float... array1d) {
    this(new float[][] {array1d});
  }

//  public NDArray(Number... data) {
//    this(Arrays.stream(data).map(Number::floatValue).toArray(float[][]::new));
//  }

//  public static NDArray of(Number... array) {
//    return new NDArray(array);
//  }
  
  public static NDArray of(float[][] array2D) {
    return new NDArray(array2D);
  }
  
  // TODO: at refector time create for each NDArray operator like "multiply" or "divide" a new function for using primitive floats instead of NDArray.ONE or NDArray.of(1)
  public static NDArray of(float... array1d) {
    return new NDArray(array1d);
  }
  
  public NDArray flat() { // TODO: check if this is correct. The flat function convets [[ 1, 2 ], [[4]] to [1, 2, 4] but idk if one dimensional ndarray is a [[1, 2, 4]] in my implementation
    float[] newData = new float[rows * cols];
    int index = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        newData[index++] = data[i][j];
      }
    }
    return new NDArray(new float[][] { newData });
  }
  
  public boolean any() {
    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
      for (int colIndex = 0; colIndex < cols; colIndex++) {
        if (data[rowIndex][colIndex] != 0)
          return true;
      }
    }
    return false;
  }
  
  /*
  Reshape the NDArray and return the new NDArray.
  https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape
  https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html
   */
  public NDArray reshape(int newRows, int newCols) { // TODO: the function is untested
    if(newCols == -1 && newCols == -1)
      throw new RuntimeException("newRows and newCols can't be -1 at the same time.");
    int cells = rows * cols;
    
    if(newRows == -1) {
      if(cells % newCols == 0) {
        newRows = cells / newCols;
      } else {
        throw new RuntimeException("Cell count is not dividable by newCols.");
      }
    } else if(newCols == -1) {
      if(cells % newRows == 0) {
        newCols = cells / newRows;
      } else {
        throw new RuntimeException("Cell count is not dividable by newRows.");
      }
    } else {
      if(cells != newRows * newCols) {
        throw new RuntimeException("Cell count is not the same as current cell count.");
      }
    }
    
    float[][] newData = new float[newRows][newCols];
    for (int i = 0; i < cells; i++) {
      int lastIndex = i % rows + i / rows;
      int newIndex = i % newRows + i / newRows;
      newData[newIndex] = data[lastIndex];
    }
    return new NDArray(newData);
  }
  
  // TODO: A problem is that the python code uses 4 dimensional arrays which is hard to implement with this self made library
//  public NDArray reshape(int i, int i1, int i2, int i3) {
//    
//  }
  
  /*
    https://numpy.org/doc/stable/reference/generated/numpy.multiply.html#numpy-multiply
    Same as * with NDArrays in Python.
    Supports:
    ndarray * number
    ndarray * ndarray
    vector * ndarray
    ndarray * vector
   */
  public NDArray multiply(NDArray other) {
    NDArray a = rows < other.rows ? this : other;
    NDArray b = rows < other.rows ? other : this;
    
    if (a.rows == 1) {
      if (a.cols == 1) { // ndarray * number
        float[][] newData = new float[b.rows][b.cols];
        for (int colIndex = 0; colIndex < b.cols; colIndex++) {
          for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
            newData[rowIndex][colIndex] = a.data[0][0] * b.data[rowIndex][colIndex];
          }
        }
        return new NDArray(newData);
      }
      
      if (a.rows == b.rows) { // ndarray * ndarray(vector)
        float[][] newData = new float[b.rows][b.cols];
        for (int colIndex = 0; colIndex < b.cols; colIndex++) {
          for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
            newData[rowIndex][colIndex] = a.data[0][colIndex] * b.data[rowIndex][colIndex];
          }
        }
        return new NDArray(newData);
      }
    }
  
    // matrix scalar multiplication matrix[x][y] * matrix[x][y]
    if (a.rows == b.rows && a.cols == b.cols) { // ndarray * ndarray
      float[][] newData = new float[rows][cols];
      for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int colIndex = 0; colIndex < cols; colIndex++) {
          newData[rowIndex][colIndex] = a.data[rowIndex][colIndex] * b.data[rowIndex][colIndex];
        }
      }
      return new NDArray(newData);
    }
    
    if (a.rows == b.rows) {
      a = cols < other.cols ? this : other;
      b = cols < other.cols ? other : this;
      
      if (a.cols == 1) {
        float[][] newData = new float[rows][b.cols];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
          for (int colIndex = 0; colIndex < b.cols; colIndex++) {
            newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] * a.data[rowIndex][0];
          }
        }
        return new NDArray(newData);
      }
    }
    
    throw new RuntimeException("Multiply dimension must be [n][1] or [n][n] " + a.dimension() + ", " + b.dimension());
  }
  
  /*
    https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    Used for Matrix multiplication only with Vectors and Matrix.
   */
//  public NDArray matmul(NDArray other) 
  /*
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    Matrix multiplication between Vector and Matrix.
    Also supports element wise multiplication with Number.
   */
  public NDArray dot(NDArray other) {
    NDArray a = rows < other.rows ? this : other;
    if(a.cols == 1 && a.rows == 1) { // ndarray.dot(number)
      NDArray b = rows < other.rows ? other : this;
      float[][] newData = new float[b.rows][b.cols];
      for (int colIndex = 0; colIndex < b.cols; colIndex++) {
        for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
          newData[rowIndex][colIndex] = a.data[0][0] * b.data[rowIndex][colIndex];
        }
      }
      return new NDArray(newData);   
    }
  
    if (cols != other.rows) {
      throw new RuntimeException("No dot product function found for dimension: " + this.dimension() + " / " + other.dimension());
    }
    // Matrix multiplication (matrix * matrix)
    float[][] newData = new float[rows][other.cols];
    for (int rowIndex = 0; rowIndex < rows; rowIndex++) { // ndarray.dot(ndarray)
      for (int otherColIndex = 0; otherColIndex < cols; otherColIndex++) {
        for (int colIndex = 0; colIndex < other.cols; colIndex++) {
          newData[rowIndex][colIndex] += data[rowIndex][otherColIndex] * other.data[otherColIndex][colIndex];
        }
      }
    }
    return new NDArray(newData);
  }
  
  // https://numpy.org/doc/stable/reference/generated/numpy.add.html
  public NDArray add(NDArray other) {
    NDArray a = rows < other.rows ? this : other;
    NDArray b = rows < other.rows ? other : this;
  
    if(a.rows == 1) {
      // matrix|vector + 1
      if (a.cols == 1) {
        float[][] newData = new float[b.rows][b.cols];
        for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
          for (int colIndex = 0; colIndex < b.cols; colIndex++) {
            newData[rowIndex][colIndex] = a.data[0][0] + b.data[rowIndex][colIndex];
          }
        }
        return new NDArray(newData);
      }
  
      // matrix + vector
      if (a.rows == 1 && a.cols == b.cols) {
        float[][] newData = new float[b.rows][a.cols];
        for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
          for (int colIndex = 0; colIndex < a.cols; colIndex++) {
            newData[rowIndex][colIndex] = a.data[0][colIndex] + b.data[rowIndex][colIndex];
          }
        }
        return new NDArray(newData);
      }
    }
  
    // matrix + matrix addition
    if (a.rows == b.rows && a.cols == b.cols) {
      float[][] newData = new float[rows][cols];
      for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int coldIndex = 0; coldIndex < cols; coldIndex++) {
          newData[rowIndex][coldIndex] = a.data[rowIndex][coldIndex] + b.data[rowIndex][coldIndex];
        }
      }
      return new NDArray(newData);
    }
  
    throw new RuntimeException("Couldn't find a add( function for array dimensions " + dimension() + " / " + other.dimension());
  }
  
  // https://numpy.org/doc/stable/reference/generated/numpy.subtract.html
  public NDArray subtract(NDArray other) {
    // reverse when other has more rows (is a matrix)
    boolean reversed = !(rows < other.rows);
    NDArray a = rows < other.rows ? this : other;
    NDArray b = rows < other.rows ? other : this;
  
    if(a.rows == 1) {
      // matrix|vector - 1
      if (a.cols == 1) {
        float[][] newData = new float[b.rows][b.cols];
        for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
          for (int colIndex = 0; colIndex < b.cols; colIndex++) {
            if (reversed) {
              newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] - a.data[0][0];
            } else {
              newData[rowIndex][colIndex] = a.data[0][0] - b.data[rowIndex][colIndex];
            }
          }
        }
        return new NDArray(newData);
      }
  
      // matrix - vector
      if (a.cols == b.cols) {
        float[][] newData = new float[b.rows][a.cols];
        for (int rowIndex = 0; rowIndex < b.rows; rowIndex++) {
          for (int colIndex = 0; colIndex < a.cols; colIndex++) {
            if (reversed) {
              newData[rowIndex][colIndex] = b.data[rowIndex][colIndex] - a.data[0][colIndex];
            } else {
              newData[rowIndex][colIndex] = a.data[0][colIndex] - b.data[rowIndex][colIndex];
            }
          }
        }
        return new NDArray(newData);
      }
    }
  
    // matrix - matrix subtraction
    if (a.rows == b.rows && a.cols == b.cols) {
      float[][] newData = new float[rows][cols];
      for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int coldIndex = 0; coldIndex < cols; coldIndex++) {
          if (reversed) {
            newData[rowIndex][coldIndex] = b.data[rowIndex][coldIndex] - a.data[rowIndex][coldIndex];
          } else {
            newData[rowIndex][coldIndex] = a.data[rowIndex][coldIndex] - b.data[rowIndex][coldIndex];
          }
        }
      }
      return new NDArray(newData);
    }
    throw new RuntimeException("Couldn't find a subtract() function for array dimensions " + dimension() + " / " + other.dimension());
  }
  
  // https://numpy.org/doc/stable/reference/generated/numpy.divide.html
  public NDArray divide(NDArray other) {
    if (other.cols == 1) {
      if (other.rows == 1) {
        // single value
      } else {
        // vector
      }
    } else {
      // matrix
    }
  }
  
  public NDArray matmul(NDArray other) {
    if (other.cols == 1) {
      if (other.rows == 1) {
        // single value
      } else {
        // vector
      }
    } else {
      // matrix
    }
  }
  
  public String dimension() {
    return "[[" + rows + "], [" + cols + "]]";
  }
  
  // Does -NDArray or NDArray * -1
  public NDArray invert() {
  }
  
  
  /*
  Represents the ndarray.T feature of Numpy
  https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html
  */
  public NDArray transpose() {
  }
  
  // Makes a deep Copy of the object
  public NDArray copy() {
  }
}