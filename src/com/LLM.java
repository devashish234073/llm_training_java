package com;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import com.llm.EmbeddingGenerator;
import com.llm.PositionalEncoder;
import com.llm.SimpleTokenizer;
import com.llm.TransformerModel;
import com.util.UrlContentReader;

public class LLM {

	public static void main(String[] args) {
		String rawText = UrlContentReader.read("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt", "data.txt");
		//System.out.println(rawText);
		SimpleTokenizer tokenizer = new SimpleTokenizer(rawText);
		System.out.println("Vocabulary Size: " + tokenizer.getVocabSize());
		List<Integer> encoded = tokenizer.encode("The quick brown fox jumps over the lazy dog");
		System.out.println("Encoded: " + encoded);
		List<String> decoded = tokenizer.decode(encoded);
		System.out.println("Decoded: " + decoded);
		System.out.println("Decoded String: " + tokenizer.decodeToString(encoded));
		int tokenVectorDimension = 128;
		EmbeddingGenerator embeddingGenerator = new EmbeddingGenerator(tokenizer.getVocabSize(), tokenVectorDimension);
		for (int positionInSequence = 0; positionInSequence < encoded.size(); positionInSequence++) {
            int tokenId = encoded.get(positionInSequence);
            double[] embedding = embeddingGenerator.getEmbedding(tokenId);
            double[] positionalEncoding = PositionalEncoder.getPositionalEncoding(positionInSequence, tokenVectorDimension);
            System.out.println("Token ID: " + tokenId + 
                               ", Position: " + positionInSequence + 
                               ", Embedding: " + Arrays.toString(embedding) + 
                               "\n     Positional Encoding: " + Arrays.toString(positionalEncoding));
        }
		TransformerModel transformerModel = new TransformerModel(tokenizer, tokenVectorDimension);
		
		List<String> trainingSequences = new ArrayList<>();
        String[] sentences = rawText.split("\\. "); // Simple sentence splitting
        System.out.println("Total sentences: " + sentences.length);
        int totalAddedSentencesPrinted = 0;
        int totalSkippedSentencesPrinted = 0;
        for (String sentence : sentences) {
            if (sentence.length() > 20) { // Ignore very short sentences
            	if(totalAddedSentencesPrinted<3) {
            	    System.out.println("adding sentence: " + sentence);
            	    totalAddedSentencesPrinted++;
            	}
                trainingSequences.add(sentence);
            } else if(totalSkippedSentencesPrinted<3) {
				totalSkippedSentencesPrinted++;
				System.out.println("skipping sentence: " + sentence);
			}
        }
        
        System.out.println("Starting training with " + trainingSequences.size() + " sequences.");
        int epochs = 5;
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("\nEpoch " + (epoch + 1));
            
            for (String sequence : trainingSequences) {
                List<Integer> tokenIds = tokenizer.encode(sequence);
                if (tokenIds.size() < 2) continue;
                
                // Split into input and target (predict next token)
                List<Integer> inputs = tokenIds.subList(0, tokenIds.size()-1);
                List<Integer> targets = tokenIds.subList(1, tokenIds.size());
                
                // Train on this sequence
                transformerModel.train(inputs, targets);
            }
            
            // Test generation after each epoch
            System.out.println("Sample generation after epoch " + (epoch+1) + ":");
            System.out.println(transformerModel.generate("I think", 20));
        }
		
		//transformerModel.train(decoded, tokenVectorDimension);
		String prompt = "";
		Scanner scanner = new Scanner(System.in);
		while (!prompt.equals("exit") && !prompt.equals("quit") && !prompt.equals("bye")) {
			System.out.print("Enter a prompt: ");
			prompt = scanner.nextLine();
			if (prompt.equals("exit") || prompt.equals("quit") || prompt.equals("bye")) {
				break;
			}
			String generated = transformerModel.generate(prompt, 30);
			System.out.println("Generated: " + generated);
		}
	}

}
