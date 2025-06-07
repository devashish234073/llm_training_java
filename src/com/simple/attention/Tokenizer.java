package com.simple.attention;

import java.util.Random;

public class Tokenizer {
	private int vocabSize;
	private int embedDim;
	private int tokenIds[];
	
	public Tokenizer(int vocabSize, int enbedDim) {
		this.vocabSize = vocabSize;
		this.embedDim = enbedDim;
		this.tokenIds = new int[vocabSize];
		for(int i = 0; i < vocabSize; i++) {
			this.tokenIds[i] = i;
		}
	}
	
	public int[] tokenize(String[] tokens) {
		int[] tokenIds = new int[tokens.length];
		for (int i = 0; i < tokens.length; i++) {
			tokenIds[i] = getTokenId(tokens[i]);
		}
		return tokenIds;
		
	}
	
	private int getTokenId(String text) {
		int hash = hashText(text);
		return hash;
	}

	private int hashText(String token) {
		token = token.toLowerCase();
		int hash = 0;
		for (int i = 0; i < token.length(); i++) {
			hash = (31 * hash + token.charAt(i)) % vocabSize;
		}
		return hash;
	}
	
	public float[] getTokenEmbedding(String token) {
		int tokenId = getTokenId(token);
		return getTokenEmbedding(tokenId);
	}
	
	public float[][] getTokenEmbedding(int[] tokenIds) {
		float[][] embeddings = new float[tokenIds.length][embedDim];
		for (int i = 0; i < tokenIds.length; i++) {
			embeddings[i] = getTokenEmbedding(tokenIds[i]);
		}
		return embeddings;
	}
	
	public float[] getTokenEmbedding(int tokenId) {
		float[] embedding = new float[embedDim];
		Random random = new Random(tokenId);
		for (int i = 0; i < embedDim; i++) {
			embedding[i] = (random.nextInt(200) - 100) / 100.0f;;
		}
		return embedding;
	}

	public String[] getTokens(String text) {
		text = text.toLowerCase();
		return text.split(" ");
	}
	
}
