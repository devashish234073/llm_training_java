package com.llm;

import java.util.ArrayList;
import java.util.List;

public class GPTDatasetV1 {
    private List<Sample> samples;
    private final int maxLength;
    private final int stride;
    private final SimpleTokenizer tokenizer;
    
    public GPTDatasetV1(String text, int maxLength, int stride, SimpleTokenizer tokenizer) {
        this.samples = new ArrayList<>();
        this.maxLength = maxLength;
        this.stride = stride;
        this.tokenizer = tokenizer;
        
        List<Integer> tokenIds = tokenizer.encode(text);
        for (int i = 0; i < tokenIds.size() - maxLength; i += stride) {
            int[] inputChunk = new int[maxLength];
            int[] targetChunk = new int[maxLength];
            
            for (int j = 0; j < maxLength; j++) {
                inputChunk[j] = tokenIds.get(i + j);
                targetChunk[j] = tokenIds.get(i + j + 1);
            }
            
            this.samples.add(new Sample(inputChunk, targetChunk));
        }
    }
    
    public List<int[][]> createBatches(int batchSize) {
        List<int[][]> batches = new ArrayList<>();
        List<int[]> currentBatch = new ArrayList<>();
        
        for (Sample sample : samples) {
            currentBatch.add(sample.getInputChunk());
            
            if (currentBatch.size() == batchSize) {
                batches.add(currentBatch.toArray(new int[0][]));
                currentBatch.clear();
            }
        }
        
        if (!currentBatch.isEmpty()) {
            batches.add(currentBatch.toArray(new int[0][]));
        }
        
        return batches;
    }

    public List<Sample> getSamples() {
        return samples;
    }
}