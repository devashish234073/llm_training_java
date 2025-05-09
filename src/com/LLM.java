package com;

import java.util.Arrays;
import java.util.List;

import com.llm.CustomDataLoader;
import com.llm.DataLoaderFactory;
import com.llm.GPTDatasetV1;
import com.llm.GPTTraining;
import com.llm.Sample;
import com.llm.SimpleTokenizer;
import com.llm.UrlContentReader;

public class LLM {

	public static void main(String[] args) {
		String rawText = UrlContentReader.read("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt", "data.txt");
		//System.out.println(rawText);
		SimpleTokenizer tokenizer = new SimpleTokenizer(rawText);
		List<Integer> encoded = tokenizer.encode("The quick brown fox jumps over the lazy dog");
		System.out.println("Encoded: " + encoded);
		List<String> decoded = tokenizer.decode(encoded);
		System.out.println("Decoded: " + decoded);
		int maxLength = 128;
		int embedDim = 128;
		int stride = 64;
		int batchSize = 32;
		int epochs = 10;
		float learningRate = 0.001f;
		GPTDatasetV1 dataset = new GPTDatasetV1(rawText, maxLength, stride, tokenizer);
		List<Sample> samples = dataset.getSamples();
		System.out.println("Number of samples: " + samples.size());
		for (int i = 0; i < 5; i++) {
			Sample sample = samples.get(i);
			System.out.println("Sample " + i + ":");
			System.out.println("Input: " + Arrays.toString(sample.getInputChunk()));
			System.out.println("Target: " + Arrays.toString(sample.getTargetChunk()));
		}
		CustomDataLoader dataloader = DataLoaderFactory.createDataloaderV1(rawText, stride, false, false, stride, maxLength, stride, tokenizer);
		GPTTraining trainer = new GPTTraining(tokenizer, embedDim, maxLength, batchSize, epochs, learningRate);
        trainer.train(dataset);
	}

}
