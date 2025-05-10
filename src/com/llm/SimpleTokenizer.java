package com.llm;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class SimpleTokenizer implements Serializable {
	private static final long serialVersionUID = 1L;
	
	private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final Set<Character> punctuationChars = Set.of('.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', '\'');
    int unkTokenId;
    int nVocab;
    
	public SimpleTokenizer(String rawText) {
		List<String> tokens = tokenize(rawText);
		
		Set<String> uniqueTokens = new LinkedHashSet<>(tokens);// Preserve insertion order
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        
        int id = 0;
        for (String token : uniqueTokens) {
            this.vocab.put(token, id);
            this.reverseVocab.put(id, token);
            id++;
        }
        
        // Special tokens
        this.unkTokenId = id;
        this.nVocab = id + 1;
	}
	
	private List<String> tokenize(String text) {
        String processed = text.toLowerCase();;
        
        // Add spaces around punctuation
        StringBuilder sb = new StringBuilder();
        for (char c : processed.toCharArray()) {
            if (punctuationChars.contains(c)) {
                sb.append(' ').append(c).append(' ');
            } else {
                sb.append(c);
            }
        }
        
        // Split on whitespace and filter out empty strings
        return Arrays.stream(sb.toString().split("\\s+"))
                     .filter(s -> !s.isEmpty())
                     .collect(Collectors.toList());
    }
	
	public List<Integer> encode(String text) {
        return tokenize(text).stream()
                .map(token -> vocab.getOrDefault(token, unkTokenId))
                .collect(Collectors.toList());
    }
    
    public List<String> decode(List<Integer> tokenIds) {
        return tokenIds.stream()
                .map(id -> reverseVocab.getOrDefault(id, "<unk>"))
                .collect(Collectors.toList());
    }
    
    public String decodeToString(List<Integer> tokenIds) {
        return tokenIds.stream()
                .map(id -> reverseVocab.getOrDefault(id, "<unk>"))
                .collect(Collectors.joining(" "));
    }

	public int getVocabSize() {
		return this.nVocab;
	}
}
