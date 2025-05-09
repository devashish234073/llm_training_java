package com.llm;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class SimpleGPT implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final int vocabSize;
	private final int embedDim;
	private final int maxLength;
	private final int numHeads;
	private final int numLayers;

	private final SimpleEmbedding tokenEmbedding;
	private final SimpleEmbedding posEmbedding;
	private final LayerNorm layerNorm;
	private final TransformerEncoder transformer;
	private final Linear head;
	private SimpleTokenizer tokenizer;
	private Map<String, float[][]> gradients;
	private int[][] lastInput; // Store last input for backward pass

	public SimpleGPT(SimpleTokenizer tokenizer, int vocabSize, int embedDim, int maxLength) {
		this.vocabSize = vocabSize;
		this.embedDim = embedDim;
		this.maxLength = maxLength;
		this.numHeads = 8;
		this.numLayers = 4;

		// Initialize components
		this.tokenEmbedding = new SimpleEmbedding(vocabSize, embedDim);
		this.posEmbedding = new SimpleEmbedding(maxLength, embedDim);
		this.layerNorm = new LayerNorm(embedDim);
		this.transformer = new TransformerEncoder(embedDim, numHeads, numLayers);
		this.tokenizer = tokenizer;
		this.head = new Linear(embedDim, vocabSize);
		this.gradients = new HashMap<>();
		this.lastInput = null; // Initialize as null
	}

	public Map<String, float[][]> getParameters() {
		Map<String, float[][]> params = new HashMap<>();
		params.put("token_embedding.weights", tokenEmbedding.getWeights());
		params.put("pos_embedding.weights", posEmbedding.getWeights());
		// Add all other parameters...
		return params;
	}

	public Map<String, float[][]> getGradients() {
		Map<String, float[][]> grads = new HashMap<>();
		Map<String, float[][]> params = getParameters();

		for (String key : params.keySet()) {
			float[][] param = params.get(key);
			if (!gradients.containsKey(key)) {
				// Initialize gradient if it doesn't exist
				gradients.put(key, new float[param.length][param[0].length]);
			}
			grads.put(key, gradients.get(key));
		}
		return grads;
	}

	public void zeroGrad() {
		// Reset all gradients to zero
		for (float[][] grad : gradients.values()) {
			for (float[] row : grad) {
				Arrays.fill(row, 0f);
			}
		}
	}

	public String generate(String prompt, int maxLength, float temperature) {
		// Tokenize the prompt using List<Integer>
		List<Integer> inputTokens = tokenizer.encode(prompt);

		// Convert List<Integer> to int[][] for model input (batch size 1)
		int[][] input = new int[1][inputTokens.size()];
		for (int i = 0; i < inputTokens.size(); i++) {
			input[0][i] = inputTokens.get(i);
		}

		// Generate tokens one by one
		for (int i = 0; i < maxLength; i++) {
			// Forward pass
			float[][][] logits = forward(input);

			// Get last token's logits
			float[] nextTokenLogits = logits[0][logits[0].length - 1];

			// Apply temperature and sample
			int nextToken = sampleToken(nextTokenLogits, temperature);

			// Append to input for next iteration
			int[][] newInput = new int[1][input[0].length + 1];
			System.arraycopy(input[0], 0, newInput[0], 0, input[0].length);
			newInput[0][newInput[0].length - 1] = nextToken;
			input = newInput;
		}

		// Convert final output to List<Integer> for decoding
		List<Integer> outputTokens = new ArrayList<>();
		for (int token : input[0]) {
			outputTokens.add(token);
		}

		// Decode tokens back to text
		return tokenizer.decodeToString(outputTokens);
	}

	private int sampleToken(float[] logits, float temperature) {
		// Apply temperature scaling
		float[] scaledLogits = new float[logits.length];
		for (int i = 0; i < logits.length; i++) {
			scaledLogits[i] = logits[i] / temperature;
		}

		// Softmax
		float[] probs = softmax(scaledLogits);

		// Sample from distribution
		float rand = new Random().nextFloat();
		float cumProb = 0.0f;
		for (int i = 0; i < probs.length; i++) {
			cumProb += probs[i];
			if (rand <= cumProb) {
				return i;
			}
		}
		return probs.length - 1; // fallback
	}

	public float[][][] forward(int[][] x) {
		int B = x.length; // batch size
		int T = x[0].length; // sequence length

		if (T > maxLength) {
			throw new IllegalArgumentException("Sequence length " + T + " > max_length " + maxLength);
		}

		// Cache the input for backward pass
		this.lastInput = new int[B][T];
		for (int b = 0; b < B; b++) {
			System.arraycopy(x[b], 0, this.lastInput[b], 0, T);
		}

		// Token embeddings
		float[][][] tokenEmb = new float[B][T][embedDim];
		for (int i = 0; i < B; i++) {
			tokenEmb[i] = tokenEmbedding.call(x[i]);
		}

		// Position embeddings
		int[] positions = new int[T];
		for (int i = 0; i < T; i++)
			positions[i] = i;
		float[][] posEmb = posEmbedding.call(positions);

		// Add token and position embeddings
		float[][][] xTensor = new float[B][T][embedDim];
		for (int b = 0; b < B; b++) {
			for (int t = 0; t < T; t++) {
				for (int d = 0; d < embedDim; d++) {
					xTensor[b][t][d] = tokenEmb[b][t][d] + posEmb[t][d];
				}
			}
		}

		// Layer normalization
		for (int b = 0; b < B; b++) {
			xTensor[b] = layerNorm.forward(xTensor[b]);
		}

		// Transformer expects [T, B, D] so we need to permute
		float[][][] xPermuted = new float[T][B][embedDim];
		for (int t = 0; t < T; t++) {
			for (int b = 0; b < B; b++) {
				System.arraycopy(xTensor[b][t], 0, xPermuted[t][b], 0, embedDim);
			}
		}

		// Create causal mask
		boolean[][] mask = generateCausalMask(T);

		// Transformer
		float[][][] xTransformed = transformer.forward(xPermuted, mask);

		// Permute back to [B, T, D]
		float[][][] xOutput = new float[B][T][embedDim];
		for (int b = 0; b < B; b++) {
			for (int t = 0; t < T; t++) {
				System.arraycopy(xTransformed[t][b], 0, xOutput[b][t], 0, embedDim);
			}
		}

		// Final linear layer
		float[][][] logits = new float[B][T][vocabSize];
		for (int b = 0; b < B; b++) {
			logits[b] = head.forward(xOutput[b]);
		}

		return logits;
	}

	public void backward(float[][][] gradOutput, int[][] targetIds) {
		if (lastInput == null) {
			throw new IllegalStateException("Forward pass must be called before backward pass");
		}
		int B = gradOutput.length; // batch size
		int T = gradOutput[0].length; // sequence length

		if (T > maxLength) {
			throw new IllegalArgumentException("Sequence length " + T + " > max_length " + maxLength);
		}
		// gradOutput shape: [B][T][vocabSize] (gradient of loss w.r.t. logits)
		float[][][] gradLoss = computeGradLoss(gradOutput, targetIds);

		// Initialize gradient tensors
		float[][][] gradHeadInput = new float[B][T][embedDim];
		float[][][] gradTransformerOutput = new float[T][B][embedDim];
		float[][][] gradLayerNormInput = new float[B][T][embedDim];
		float[][][] gradEmbeddingsSum = new float[B][T][embedDim];
		float[][][] gradTokenEmb = new float[B][T][embedDim];
		float[][] gradPosEmb = new float[maxLength][embedDim];

		// 1. Backward through final linear layer (head)
		for (int b = 0; b < B; b++) {
			float[][] headGrad = head.backward(gradOutput[b]);
			System.arraycopy(headGrad, 0, gradHeadInput[b], 0, headGrad.length);
		}

		// 2. Permute gradient for transformer (B,T,D) -> (T,B,D)
		for (int t = 0; t < T; t++) {
			for (int b = 0; b < B; b++) {
				System.arraycopy(gradHeadInput[b][t], 0, gradTransformerOutput[t][b], 0, embedDim);
			}
		}

		// 3. Backward through transformer
		float[][][] gradTransformerInput = transformer.backward(gradTransformerOutput);

		// 4. Permute back (T,B,D) -> (B,T,D)
		float[][][] gradLayerNormOutput = new float[B][T][embedDim];
		for (int b = 0; b < B; b++) {
			for (int t = 0; t < T; t++) {
				System.arraycopy(gradTransformerInput[t][b], 0, gradLayerNormOutput[b][t], 0, embedDim);
			}
		}

		// 5. Backward through layer norm
		for (int b = 0; b < B; b++) {
			gradEmbeddingsSum[b] = layerNorm.backward(gradLayerNormOutput[b]);
		}

		// 6. Backward through embedding addition
		for (int b = 0; b < B; b++) {
			for (int t = 0; t < T; t++) {
				for (int d = 0; d < embedDim; d++) {
					gradTokenEmb[b][t][d] = gradEmbeddingsSum[b][t][d];
					gradPosEmb[t][d] += gradEmbeddingsSum[b][t][d]; // Accumulate across batch
				}
			}
		}

		// 7. Backward through position embeddings
		posEmbedding.backward(gradPosEmb);

		// 8. Backward through token embeddings
		for (int b = 0; b < B; b++) {
			tokenEmbedding.backward(lastInput[b], gradTokenEmb[b]);
		}
	}

	private float[][][] computeGradLoss(float[][][] logits, int[][] targetIds) {
		int B = logits.length;
		int T = logits[0].length;
		float[][][] gradLoss = new float[B][T][vocabSize];

		for (int b = 0; b < B; b++) {
			for (int t = 0; t < T; t++) {
				// Softmax gradient: ∂L/∂z = p - y
				float[] probs = softmax(logits[b][t]);
				int target = targetIds[b][t];
				System.arraycopy(probs, 0, gradLoss[b][t], 0, vocabSize);
				gradLoss[b][t][target] -= 1.0f;
			}
		}
		return gradLoss;
	}

	private float[] softmax(float[] logits) {
		float[] result = new float[logits.length];

		// Find max logit for numerical stability
		float maxLogit = Float.NEGATIVE_INFINITY;
		for (float logit : logits) {
			if (logit > maxLogit) {
				maxLogit = logit;
			}
		}

		// Compute exponentials and sum
		float sum = 0.0f;
		for (int i = 0; i < logits.length; i++) {
			result[i] = (float) Math.exp(logits[i] - maxLogit);
			sum += result[i];
		}

		// Normalize
		for (int i = 0; i < result.length; i++) {
			result[i] /= sum;
		}

		return result;
	}

	private boolean[][] generateCausalMask(int size) {
		boolean[][] mask = new boolean[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				mask[i][j] = j > i; // Upper triangular mask
			}
		}
		return mask;
	}

	public static SimpleGPT loadModel(String path) throws IOException, ClassNotFoundException {
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
			@SuppressWarnings("unchecked")
			//Map<String, float[][]> params = (Map<String, float[][]>) ois.readObject();
			SimpleGPT model = (SimpleGPT) ois.readObject();
			System.out.println("Model loaded from " + path);
			return model;
		}
	}
}