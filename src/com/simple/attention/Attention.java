package com.simple.attention;

import java.util.Arrays;

public class Attention {

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
		System.out.println("Token Embeddings: "+ArrayPrinter.prettyPrint2D(token_embeddings));
		
		//Keep query and key the same for simplicity
		float[][] query = token_embeddings;
		float[][] key = token_embeddings;
		float[][] value = token_embeddings;
		System.out.println("query = key = value: "+ArrayPrinter.prettyPrint2D(query));
		
		QueryKeyValueProcessor dotProduct = new QueryKeyValueProcessor();
		float[][] attentionWeights = dotProduct.performScaledDotProduct(tokenIds, query, key, value);
	}

}