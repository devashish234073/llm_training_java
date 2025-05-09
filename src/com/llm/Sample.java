package com.llm;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Sample {
	private int[] inputChunk;
	private int[] targetChunk;
	private SimpleTokenizer tokenizer;

	public Sample(SimpleTokenizer tokenizer, int[] inputChunk, int[] targetChunk) {
		this.inputChunk = inputChunk;
		this.targetChunk = targetChunk;
		this.tokenizer = tokenizer;
	}

	public int[] getInputChunk() {
		return inputChunk;
	}

	public int[] getTargetChunk() {
		return targetChunk;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("Input: ");
		sb.append(tokenizer.decode(Arrays.stream(inputChunk).boxed().collect(Collectors.toList())));
		sb.append("\nTarget: ");
		sb.append(tokenizer.decode(Arrays.stream(targetChunk).boxed().collect(Collectors.toList())));
		return sb.toString();
	}
}