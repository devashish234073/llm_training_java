package com.llm;

import java.util.*;
import java.io.*;

public class GPTTraining {
	private final SimpleGPT model;
	private final int batchSize;
	private final int epochs;
	private final float learningRate;
	private final int vocabSize;
	private final int maxLength;
	private final SimpleTokenizer tokenizer;

	public GPTTraining(SimpleTokenizer tokenizer, int embedDim, int maxLength, int batchSize, int epochs, float learningRate) {
		this.tokenizer = tokenizer;
		this.vocabSize = tokenizer.getnVocab();
		this.maxLength = maxLength;
		this.batchSize = batchSize;
		this.epochs = epochs;
		this.learningRate = learningRate;
		this.model = new SimpleGPT(tokenizer, vocabSize, embedDim, maxLength);
	}

	public void train(GPTDatasetV1 dataset) {
		// Initialize optimizer
		AdamW optimizer = new AdamW(model, learningRate);

		for (int epoch = 0; epoch < epochs; epoch++) {
			float totalLoss = 0;
			int batchCount = 0;

			// Create batches from dataset
			List<int[][]> batches = dataset.createBatches(batchSize);

			for (int[][] batch : batches) {
				// Prepare input and target (shifted by one)
				int[][] inputIds = new int[batch.length][maxLength];
				int[][] targetIds = new int[batch.length][maxLength];

				for (int i = 0; i < batch.length; i++) {
					// Copy tokens to input (all except last)
					int copyLength = Math.min(batch[i].length - 1, maxLength);
					System.arraycopy(batch[i], 0, inputIds[i], 0, copyLength);

					// Copy tokens to target (shifted by one)
					if (batch[i].length > 1) {
						System.arraycopy(batch[i], 1, targetIds[i], 0, copyLength);
					}
				}

				// Forward pass
				float[][][] logits = model.forward(inputIds);

				// Calculate loss
				float loss = crossEntropyLoss(logits, targetIds);

				// Backward pass
				model.zeroGrad();
				model.backward(logits, targetIds);

				// Update parameters
				optimizer.step();

				totalLoss += loss;
				batchCount++;

				System.out.printf("Epoch %d - Batch %d - Loss: %.4f%n", epoch + 1, batchCount, loss);
			}

			float avgLoss = totalLoss / batchCount;
			System.out.printf("Epoch %d/%d - Avg Loss: %.4f%n", epoch + 1, epochs, avgLoss);
		}
		saveModel("simple_gpt.bin");
	}

	private int[] tokenize(String text) {
		// Simplified tokenization - replace with actual tokenizer
		return text.chars().toArray();
	}

	private float crossEntropyLoss(float[][][] logits, int[][] targets) {
		float totalLoss = 0;
		int totalTokens = 0;

		for (int b = 0; b < targets.length; b++) {
			for (int t = 0; t < targets[b].length; t++) {
				if (targets[b][t] == 0)
					continue; // Skip padding

				// Softmax
				float[] probs = softmax(logits[b][t]);

				// Cross-entropy
				int target = targets[b][t];
				totalLoss += -Math.log(probs[target]);
				totalTokens++;
			}
		}

		return totalLoss / totalTokens;
	}

	private float[] softmax(float[] logits) {
		float max = Float.NEGATIVE_INFINITY;
		for (float val : logits) {
			if (val > max)
				max = val;
		}

		float sum = 0;
		float[] exps = new float[logits.length];
		for (int i = 0; i < logits.length; i++) {
			exps[i] = (float) Math.exp(logits[i] - max);
			sum += exps[i];
		}

		for (int i = 0; i < exps.length; i++) {
			exps[i] /= sum;
		}

		return exps;
	}

	public void saveModel(String filename) {
		try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
			out.writeObject(model);
			System.out.println("Model saved to " + filename);
		} catch (IOException e) {
			System.err.println("Failed to save model: " + e.getMessage());
			e.printStackTrace();
		}
	}
}