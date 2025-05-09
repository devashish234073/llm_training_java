package llm;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class Linear implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private final float[][] weights;
    private final float[] bias;
    private float[][] gradWeights;  // Gradient for weights
    private float[] gradBias;       // Gradient for bias
    private float[][] lastInput;    // Cache of last input for backpropagation

    public Linear(int inFeatures, int outFeatures) {
        this.weights = new float[outFeatures][inFeatures];
        this.bias = new float[outFeatures];
        this.gradWeights = new float[outFeatures][inFeatures];
        this.gradBias = new float[outFeatures];
        
        // Initialize weights and bias
        Random rand = new Random();
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                weights[i][j] = (float) (rand.nextGaussian() * 0.02);
            }
            bias[i] = 0f;
        }
    }

    public float[][] forward(float[][] x) {
        this.lastInput = x;  // Store input for backpropagation
        int batchSize = x.length;
        int inFeatures = x[0].length;
        int outFeatures = weights.length;
        
        float[][] output = new float[batchSize][outFeatures];
        
        for (int b = 0; b < batchSize; b++) {
            for (int o = 0; o < outFeatures; o++) {
                float sum = bias[o];
                for (int i = 0; i < inFeatures; i++) {
                    sum += x[b][i] * weights[o][i];
                }
                output[b][o] = sum;
            }
        }
        
        return output;
    }

    public float[][] backward(float[][] gradOutput) {
        // gradOutput shape: [batchSize][outFeatures]
        int batchSize = gradOutput.length;
        int outFeatures = weights.length;
        int inFeatures = weights[0].length;
        
        float[][] gradInput = new float[batchSize][inFeatures];
        
        // Compute gradient of weights and bias
        for (int o = 0; o < outFeatures; o++) {
            for (int i = 0; i < inFeatures; i++) {
                float sum = 0;
                for (int b = 0; b < batchSize; b++) {
                    sum += lastInput[b][i] * gradOutput[b][o];
                }
                gradWeights[o][i] += sum;  // Accumulate gradient
            }
            
            float biasSum = 0;
            for (int b = 0; b < batchSize; b++) {
                biasSum += gradOutput[b][o];
            }
            gradBias[o] += biasSum;  // Accumulate gradient
        }
        
        // Compute gradient of input
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inFeatures; i++) {
                float sum = 0;
                for (int o = 0; o < outFeatures; o++) {
                    sum += weights[o][i] * gradOutput[b][o];
                }
                gradInput[b][i] = sum;
            }
        }
        
        return gradInput;
    }

    public void zeroGrad() {
        // Reset gradients to zero
        for (int o = 0; o < gradWeights.length; o++) {
            Arrays.fill(gradWeights[o], 0f);
        }
        Arrays.fill(gradBias, 0f);
    }

    // Getters for parameters and gradients
    public float[][] getWeights() { return weights; }
    public float[] getBias() { return bias; }
    public float[][] getGradWeights() { return gradWeights; }
    public float[] getGradBias() { return gradBias; }
}