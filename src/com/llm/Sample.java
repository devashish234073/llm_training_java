package com.llm;

public class Sample {
    private int[] inputChunk;
    private int[] targetChunk;

    public Sample(int[] inputChunk, int[] targetChunk) {
        this.inputChunk = inputChunk;
        this.targetChunk = targetChunk;
    }
    
    public int[] getInputChunk() {
        return inputChunk;
    }
    
    public int[] getTargetChunk() {
        return targetChunk;
    }
}