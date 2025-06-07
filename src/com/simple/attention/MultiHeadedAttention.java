package com.simple.attention;

import java.util.Arrays;

public class MultiHeadedAttention {

	public static void main(String[] args) {
		int EMBED_DIM = 3; //Embedding dimension
		int VOCAB_SIZE = 1000; //Vocabulary size
		
		String text = "Time flies like an arrow";
		System.out.println("Input text: " + text);
		Tokenizer tokenizer = new Tokenizer(VOCAB_SIZE, EMBED_DIM);
		String[] tokens = tokenizer.getTokens(text);
		System.out.println("Tokens: "+Arrays.toString(tokens));
		int[] tokenIds = tokenizer.tokenize(tokens);
		System.out.println("Token IDs: "+Arrays.toString(tokenIds));
		float[][] token_embeddings = tokenizer.getTokenEmbedding(tokenIds);
		System.out.println("Token Embeddings: "+Arrays.deepToString(token_embeddings));
		
		int NUM_HEADS = 3;
		int HEAD_DIM = EMBED_DIM / NUM_HEADS; //Dimension of each head
		float[][][] multiHeadedEmbeddings = new float[NUM_HEADS][token_embeddings.length][HEAD_DIM];
		for(int head=0;head<NUM_HEADS;head++) {
			System.out.println("Processing head: "+head);
			//Keep query and key the same for simplicity
			float[][] query = LinearTransformer.linearTransform(token_embeddings, HEAD_DIM);
			float[][] key = LinearTransformer.linearTransform(token_embeddings, HEAD_DIM);
			float[][] value = LinearTransformer.linearTransform(token_embeddings, HEAD_DIM);
			System.out.println("query = key = value: "+Arrays.deepToString(query));
			
			QueryKeyValueProcessor dotProduct = new QueryKeyValueProcessor();
			float[][] attentionWeights = dotProduct.performScaledDotProduct(tokenIds, query, key, value);
			copyToMultiHeadedEmbeddings(multiHeadedEmbeddings, head, attentionWeights);
		}
		System.out.println("Combined: "+Arrays.deepToString(multiHeadedEmbeddings));
	}

	private static void copyToMultiHeadedEmbeddings(float[][][] multiHeadedEmbeddings, int head,
			float[][] attentionWeights) {
		for (int i = 0; i < attentionWeights.length; i++) {
			for (int j = 0; j < attentionWeights[i].length; j++) {
				multiHeadedEmbeddings[head][i][j] = attentionWeights[i][j];
			}
		}
		
	}

}
