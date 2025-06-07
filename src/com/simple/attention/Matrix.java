package com.simple.attention;

public class Matrix {
	
	public static float[][] transpose(float[][] matrix) {
	    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
	        return new float[0][0];
	    }

	    int rows = matrix.length;
	    int cols = matrix[0].length;
	    float[][] transposed = new float[cols][rows];

	    for (int i = 0; i < rows; i++) {
	        for (int j = 0; j < cols; j++) {
	            transposed[j][i] = matrix[i][j];
	        }
	    }

	    return transposed;
	}
	
	public static float[][] divideAllElemntsBy(float[][] matrix, float divisor) {
		if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
			return new float[0][0];
		}
		if (divisor == 0) {
			throw new IllegalArgumentException("Divisor cannot be zero.");
		}
		int rows = matrix.length;
		int cols = matrix[0].length;
		float[][] result = new float[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = matrix[i][j] / divisor;
			}
		}
		return result;
	}
	
	public static float[][] doCrossProduct(float[][] matrix1, float[][] matrix2) {
		int row1 = matrix1.length;
		int col1 = matrix1[0].length;
		int row2 = matrix2.length;
		int col2 = matrix2[0].length;
		System.out.println("matrix1 dim: " + row1 + " x " + col1);
		System.out.println("matrix2 dim: " + row2 + " x " + col2);
		System.out.println("Multilplying " + ArrayPrinter.prettyPrint2D(matrix1) + "\n and " + ArrayPrinter.prettyPrint2D(matrix2));
		if ((row1 == 0 || col1 == 0 || row2 == 0 || col2 == 0) || col1 != row2) {
			throw new IllegalArgumentException("Invalid input dimensions for matrix1 and matrix2.");
		}
		float[][] result = new float[row1][col2];
		for (int i = 0; i < row1; i++) {
			for (int j = 0; j < col2; j++) {
				float sum = 0.0f;
				String multiplications = "";
				for (int k = 0; k < col1; k++) {
					sum += matrix1[i][k] * matrix2[k][j];
					if(multiplications.equals("")) {
						multiplications = String.format("%.2f * %.2f", matrix1[i][k], matrix2[k][j]);
					} else {
						multiplications += " + " + String.format("%.2f * %.2f", matrix1[i][k], matrix2[k][j]);
					}
				}
				//System.out.println(multiplications + " = " + sum);
				result[i][j] = sum;
			}
		}
		return result;
	}
}
