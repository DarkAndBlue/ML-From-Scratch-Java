package mlfromscratch;

public class StringUtil {
  
  public static String createStringTable(Object... objects) {
    return createStringTable(new Object[] { objects });
  }
  /* 
  Uses a two-dimensional array of objects and creates a table
  with strings. The column count must be the same on every row.
   */
  public static String createStringTable(Object[]... objects) {
    String[][] strings = new String[objects.length][];
    int maxWidth = 0;
    int[] maxWordLength = new int[objects[0].length];
    
    for (int y = 0; y < objects.length; y++) {
      int width = 0;
      int[] wordLength = new int[objects[0].length];
      for (int x = 0; x < objects[0].length; x++) {
        String string = objects[y][x].toString();
        strings[y][x] = string;
        width += string.length();
        wordLength[x] = string.length();
      }
      if (width > maxWidth) {
        maxWidth = width;
        maxWordLength = wordLength;
      }
    }
    
    StringBuilder stringBuilder = new StringBuilder();
    for (int y = 0; y < strings.length; y++) {
      // new row or start of table +-------+------+
      stringBuilder.append('+');
      for (int i = 0; i < maxWordLength.length; i++) {
        int wordLength = maxWordLength[i];
        for (int j = 0; j < wordLength + 2; j++) {
          stringBuilder.append('-');
        }
        stringBuilder.append('+');
      }
      stringBuilder.append('\n');
      
      for (int x = 0; x < strings[y].length; x++) {
        if (x == 0) { // table border before
          stringBuilder.append("| ");
        } else {
          stringBuilder.append(" ");
        }
        String stringToAdd = strings[y][x];
        stringBuilder.append(stringToAdd);
        
        // adding spaces to fill the cell
        for (int i = 0; i < maxWordLength[x] - stringToAdd.length(); i++) {
          stringBuilder.append(" ");
        }
        
        if(x == strings[y].length - 1) { // border + linebreak
          stringBuilder.append(" |\n");
        } else {
          stringBuilder.append(" |");
        }
      }
    }
    // end of table +-------+------+
    stringBuilder.append('+');
    for (int i = 0; i < maxWordLength.length; i++) {
      int wordLength = maxWordLength[i];
      for (int j = 0; j < wordLength + 2; j++) {
        stringBuilder.append('-');
      }
      stringBuilder.append('+');
    }
    stringBuilder.append('\n');
    
    return stringBuilder.toString();
  }
}