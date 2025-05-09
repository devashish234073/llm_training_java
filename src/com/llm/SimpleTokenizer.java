package com.llm;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class SimpleTokenizer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final int unkTokenId;
    private final int nVocab;
    private final Set<Character> punctuationChars;
    private final boolean lowerCase;
    
    public SimpleTokenizer(String text) {
        this(text, true, defaultPunctuation());
    }
    
    public SimpleTokenizer(String text, boolean lowerCase, Set<Character> punctuationChars) {
        this.lowerCase = lowerCase;
        this.punctuationChars = Collections.unmodifiableSet(new HashSet<>(punctuationChars));
        
        // Pre-process and tokenize text
        List<String> tokens = tokenize(text);
        
        // Build vocabulary
        Set<String> uniqueTokens = new LinkedHashSet<>(tokens); // Preserve insertion order
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
    
    private static Set<Character> defaultPunctuation() {
        return Set.of('.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', '\'');
    }
    
    public List<String> tokenize(String text) {
        String processed = text;
        if (lowerCase) {
            processed = processed.toLowerCase();
        }
        
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
        List<String> tokens = decode(tokenIds);
        
        // Reconstruct original spacing around punctuation
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            if (token.length() == 1 && punctuationChars.contains(token.charAt(0))) {
                // Punctuation - don't add space before
                sb.append(token);
                if (i < tokens.size() - 1 && !tokens.get(i + 1).isEmpty()) {
                    // Add space after unless it's the last token
                    sb.append(" ");
                }
            } else {
                // Regular word
                if (i > 0 && !tokens.get(i - 1).isEmpty()) {
                    sb.append(" ");
                }
                sb.append(token);
            }
        }
        
        return sb.toString();
    }

    // Getters (no setters to maintain immutability)
    public Map<String, Integer> getVocab() {
        return Collections.unmodifiableMap(vocab);
    }

    public Map<Integer, String> getReverseVocab() {
        return Collections.unmodifiableMap(reverseVocab);
    }

    public int getUnkTokenId() {
        return unkTokenId;
    }

    public int getVocabSize() {
        return nVocab;
    }
    
    public int getnVocab() {
        return nVocab;
    }
    
    public boolean isLowerCase() {
        return lowerCase;
    }
    
    public Set<Character> getPunctuationChars() {
        return punctuationChars;
    }
}