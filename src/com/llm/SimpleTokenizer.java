package com.llm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class SimpleTokenizer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private Map<String,Integer> vocab;
	private Map<Integer,String> reverseVocab;
	private int unkTokenId;
	private int nVocab;
	
	public SimpleTokenizer(String text) {
		List<String> words = Arrays.asList(text.toLowerCase().split("\\s+"));
		Set<String> uniqueWordsSet = new HashSet<>(words);
		List<String> uniqueWords = new ArrayList<>(uniqueWordsSet);
		Collections.sort(uniqueWords);
		
		this.vocab = new HashMap<>();
		this.reverseVocab = new HashMap<>();
		for(int i=0;i<uniqueWords.size();i++) {
			String uniqueWord = uniqueWords.get(i);
			this.vocab.put(uniqueWord, i);
			this.reverseVocab.put(i, uniqueWord);
			
			this.unkTokenId = this.vocab.size();
			this.nVocab = this.vocab.size()+1;
		}
	}
	
	public List<Integer> encode(String text) {
		List<String> tokens = Arrays.asList(text.toLowerCase().split("\\s+"));
		List<Integer> encodedTokens = new ArrayList<>();
		for(String token : tokens) {
			encodedTokens.add(this.vocab.getOrDefault(token, this.unkTokenId));
		}
		return encodedTokens;
	}
	
	public List<String> decode(List<Integer> tokenIds) {
		List<String> tokens = new ArrayList<>();
		for(Integer tokenId : tokenIds) {
			tokens.add(this.reverseVocab.getOrDefault(tokenId, "<unk>"));
		}
		return tokens;
	}
	
	public String decodeToString(List<Integer> tokenIds) {
		return this.decode(tokenIds).stream().collect(Collectors.joining(" "));
		/*StringBuilder sb = new StringBuilder();
		for(Integer tokenId : tokenIds) {
			sb.append(this.reverseVocab.getOrDefault(tokenId, "<unk>")).append(" ");
		}
		return sb.toString().trim();*/
	}

	public Map<String, Integer> getVocab() {
		return vocab;
	}

	public void setVocab(Map<String, Integer> vocab) {
		this.vocab = vocab;
	}

	public Map<Integer, String> getReverseVocab() {
		return reverseVocab;
	}

	public void setReverseVocab(Map<Integer, String> reverseVocab) {
		this.reverseVocab = reverseVocab;
	}

	public int getUnkTokenId() {
		return unkTokenId;
	}

	public void setUnkTokenId(int unkTokenId) {
		this.unkTokenId = unkTokenId;
	}

	public int getnVocab() {
		return nVocab;
	}

	public void setnVocab(int nVocab) {
		this.nVocab = nVocab;
	}
}
