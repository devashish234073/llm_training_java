package llm;

public class Sample {
    int[] inputChunk;
    int[] targetChunk;

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