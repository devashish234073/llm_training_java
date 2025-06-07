package com.simple.attention;

import java.util.Arrays;

public class QueryKeyValueProcessor {
	public float[][] performScaledDotProduct(int[] tokenIds, float[][] query, float[][] key, float[][] value) {
		int numQueries = query.length;
		int numKeys = key.length;
		if (numQueries == 0 || numKeys == 0 || query[0].length != key[0].length) {
			throw new IllegalArgumentException("Invalid input dimensions for query and key.");
		}
		int embedDim = query[0].length;
		if (embedDim != key[0].length) {
			throw new IllegalArgumentException("Query and key must have the same embedding dimension.");
		}
		float sqrtVal = (float) Math.sqrt(embedDim);
		System.out.println("calculating scores by multiplying query and key(transposed) and dividing by sqrt(embedDim): " + sqrtVal);
		float[][] qv = Matrix.doCrossProduct(query, Matrix.transpose(key));
		float[][] scores = Matrix.divideAllElemntsBy(qv, sqrtVal);
		ArrayPrinter.printAttentionScores("scores after multiplying query and key(transposed) and dividing by sqrt(embedDim): " + sqrtVal,tokenIds, scores);

		float[][] weights = softmax(scores);
		ArrayPrinter.printAttentionScores("weights after doing softmax on scores: ",tokenIds, weights);
		
		System.out.println("Multiplying weights with value to get attention scores");
		float[][] attention = Matrix.doCrossProduct(weights,value);
		System.out.println("Attention after multiplying weights with value: "+ArrayPrinter.prettyPrint2D(attention));

		return attention;
	}

	private float[][] softmax(float[][] scores) {
		int numRows = scores.length;
		int numCols = scores[0].length;
		float[][] softmaxScores = new float[numRows][numCols];
		System.out.println("calculating softmax..");
		for (int row = 0; row < numRows; row++) {
			float maxScoreInTheRow = Float.NEGATIVE_INFINITY;
			//check all the columns in the row to find the max score
			for (int col = 0; col < numCols; col++) {
				if (scores[row][col] > maxScoreInTheRow) {
					maxScoreInTheRow = scores[row][col];
				}
			}

			float sumExp = 0.0f;
			for (int col = 0; col < numCols; col++) {
				softmaxScores[row][col] = (float) Math.exp(scores[row][col] - maxScoreInTheRow);
				sumExp += Math.exp(scores[row][col] - maxScoreInTheRow);
			}

			for (int col = 0; col < numCols; col++) {
				softmaxScores[row][col] /= sumExp;
			}
			
			System.out.println("Row: " + row + ", maxScoreInTheRow: " + maxScoreInTheRow + ", sumExp: " + sumExp + ", softmaxScores: " + Arrays.toString(softmaxScores[row]));
		}

		return softmaxScores;
	}
}
