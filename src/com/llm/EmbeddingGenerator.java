package com.llm;

import java.util.Random;

public class EmbeddingGenerator {
	
	private int vocabSize;
	private int tokenVectorDimension;
	private double[][] embeddingMatrix;
	
	public EmbeddingGenerator(int vocabSize, int tokenVectorDimension) {
		this.vocabSize = vocabSize;
		this.tokenVectorDimension = tokenVectorDimension;
		initializeEmbeddings();
	}

	public double[] getEmbedding(int tokenId) {
		if (tokenId < 0 || tokenId >= vocabSize) {
			throw new IllegalArgumentException("Token ID out of range");
		}
		return embeddingMatrix[tokenId];
	}

	private void initializeEmbeddings() {
		Random rand = new Random(42); // Fixed seed for reproducibility
		embeddingMatrix = new double[vocabSize][tokenVectorDimension];

		for (int i = 0; i < vocabSize; i++) {
			for (int j = 0; j < tokenVectorDimension; j++) {
				embeddingMatrix[i][j] = rand.nextGaussian() * 0.02; // Small random values
			}
		}
	}

}
