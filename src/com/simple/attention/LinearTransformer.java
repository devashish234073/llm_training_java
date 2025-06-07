package com.simple.attention;

import java.util.Random;

public class LinearTransformer {
	
	public static float[][] linearTransform(float[][] input, int outputSize) {
		if (outputSize <= 0) {
			throw new IllegalArgumentException("Output size must be positive");
		}
		float[][] transformed = new float[input.length][outputSize];
		for (int row = 0;row<input.length;row++) {
			float[] rowContent = input[row];
			transformed[row] = linearTransform(rowContent, outputSize);
		}
		return transformed;
	}
    
    public static float[] linearTransform(float[] input, int outputSize) {
        // Input validation
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }
        if (outputSize <= 0) {
            throw new IllegalArgumentException("Output size must be positive");
        }
        
        int inputSize = input.length;
        Random random = new Random();
        float[] output = new float[outputSize];
        
        // Initialize weights and biases, then compute output
        double stdv = Math.sqrt(2.0 / (inputSize + outputSize));
        
        for (int i = 0; i < outputSize; i++) {
            // Initialize bias (typically starts at 0)
            float bias = 0;
            
            // Compute weighted sum
            float sum = 0;
            for (int j = 0; j < inputSize; j++) {
                // Initialize weight randomly (Xavier/Glorot initialization)
                double weight = random.nextGaussian() * stdv;
                sum += input[j] * weight;
            }
            
            // Add bias
            output[i] = sum + bias;
        }
        
        return output;
    }
}