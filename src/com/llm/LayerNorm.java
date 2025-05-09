package com.llm;

import java.io.Serializable;

public class LayerNorm implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private final int size;
	private float[][][] lastInput3d; // Cache for 3D backward pass
	private float[][] lastInput2d; // Cache for 2D backward pass
	private float[] lastMean; // Cache mean for backward pass
	private float[] lastStd; // Cache std for backward pass

	public LayerNorm(int size) {
		this.size = size;
	}

	public float[][] forward(float[][] x) {
		this.lastInput2d = x; // Store input for backward pass
		this.lastMean = new float[x.length];
		this.lastStd = new float[x.length];

		float[][] output = new float[x.length][size];
		for (int i = 0; i < x.length; i++) {
			// Calculate mean
			float mean = 0f;
			for (float val : x[i]) {
				mean += val;
			}
			mean /= size;
			lastMean[i] = mean;

			// Calculate variance
			float variance = 0f;
			for (float val : x[i]) {
				variance += (val - mean) * (val - mean);
			}
			variance /= size;

			// Calculate standard deviation
			float std = (float) Math.sqrt(variance + 1e-5);
			lastStd[i] = std;

			// Normalize
			for (int j = 0; j < size; j++) {
				output[i][j] = (x[i][j] - mean) / std;
			}
		}
		return output;
	}
	
	public float[][][] forward3d(float[][][] x) {
        this.lastInput3d = x;
        float[][][] output = new float[x.length][x[0].length][size];
        for (int i = 0; i < x.length; i++) {
            output[i] = forward(x[i]);
        }
        return output;
    }

	public float[][] backward(float[][] gradOutput) {
		float[][] gradInput = new float[gradOutput.length][size];

		for (int i = 0; i < gradOutput.length; i++) {
			float mean = lastMean[i];
			float std = lastStd[i];
			float stdInv = 1.0f / std;
			float[] x = lastInput2d[i];

			// Calculate sum of gradients
			float sumGrad = 0;
			float sumGradX = 0;
			for (int j = 0; j < size; j++) {
				sumGrad += gradOutput[i][j];
				sumGradX += gradOutput[i][j] * (x[j] - mean);
			}

			// Compute gradient for each element
			for (int j = 0; j < size; j++) {
				gradInput[i][j] = (gradOutput[i][j] - sumGrad / size - (x[j] - mean) * sumGradX / (size * std * std))
						* stdInv;
			}
		}

		return gradInput;
	}
	
	public float[][][] backward3d(float[][][] gradOutput) {
        float[][][] gradInput = new float[gradOutput.length][gradOutput[0].length][size];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = backward(gradOutput[i]);
        }
        return gradInput;
    }
}