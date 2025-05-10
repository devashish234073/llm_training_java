package com;

import java.util.Arrays;
import java.util.List;

import com.llm.SimpleTokenizer;
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
	}

}
