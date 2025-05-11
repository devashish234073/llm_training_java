package com.llm;

import java.io.*;
import java.util.*;

public class TransformerModel implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final SimpleTokenizer tokenizer;
    private final EmbeddingGenerator embeddings;
    private final int tokenVectorDimension;
    
    // Model parameters (simplified for example)
    private transient double[][] attentionWeights;
    private transient double[][] outputLayer;
    
    public TransformerModel(SimpleTokenizer tokenizer, int tokenVectorDimension) {
        this.tokenizer = tokenizer;
        this.tokenVectorDimension = tokenVectorDimension;
        this.embeddings = new EmbeddingGenerator(tokenizer.getVocabSize(), tokenVectorDimension);
        initializeModel();
    }
    
    private void initializeModel() {
        Random rand = new Random(42);
        
        // Simplified attention weights (in real model, this would be more complex)
        this.attentionWeights = new double[tokenVectorDimension][tokenVectorDimension];
        for (int i = 0; i < tokenVectorDimension; i++) {
            for (int j = 0; j < tokenVectorDimension; j++) {
                attentionWeights[i][j] = rand.nextGaussian() * 0.02;
            }
        }
        
        // Output layer
        this.outputLayer = new double[tokenVectorDimension][tokenizer.getVocabSize()];
        for (int i = 0; i < tokenVectorDimension; i++) {
            for (int j = 0; j < tokenizer.getVocabSize(); j++) {
                outputLayer[i][j] = rand.nextGaussian() * 0.02;
            }
        }
    }
    
    public void train(List<Integer> inputs, List<Integer> targets) {
        // 1. Forward pass
        double[][] hiddenStates = processInput(inputs);
        
        // 2. Calculate gradients (simplified)
        double[][] gradients = new double[outputLayer.length][outputLayer[0].length];
        for (int i = 0; i < targets.size(); i++) {
            int target = targets.get(i);
            for (int j = 0; j < tokenVectorDimension; j++) {
                gradients[j][target] += hiddenStates[i][j];
            }
        }
        
        // 3. Update weights (simplified SGD)
        double learningRate = 0.01;
        for (int i = 0; i < outputLayer.length; i++) {
            for (int j = 0; j < outputLayer[i].length; j++) {
                outputLayer[i][j] -= learningRate * gradients[i][j];
            }
        }
    }
    
    private double[][] processInput(List<Integer> tokenIds) {
        double[][] hiddenStates = new double[tokenIds.size()][tokenVectorDimension];
        
        for (int pos = 0; pos < tokenIds.size(); pos++) {
            int tokenId = tokenIds.get(pos);
            double[] tokenEmbedding = embeddings.getEmbedding(tokenId);
            double[] positionalEncoding = PositionalEncoder.getPositionalEncoding(pos, tokenVectorDimension);
            
            // Combine embedding + positional encoding
            for (int i = 0; i < tokenVectorDimension; i++) {
                hiddenStates[pos][i] = tokenEmbedding[i] + positionalEncoding[i];
            }
        }
        
        // Simplified attention (real implementation would use proper attention)
        return applyAttention(hiddenStates);
    }
    
    private double[][] applyAttention(double[][] hiddenStates) {
        // Simplified attention operation
        double[][] output = new double[hiddenStates.length][tokenVectorDimension];
        for (int i = 0; i < hiddenStates.length; i++) {
            for (int j = 0; j < tokenVectorDimension; j++) {
                for (int k = 0; k < tokenVectorDimension; k++) {
                    output[i][j] += hiddenStates[i][k] * attentionWeights[k][j];
                }
            }
        }
        return output;
    }
    
    public String generate(String prompt, int maxLength) {
        List<Integer> tokenIds = tokenizer.encode(prompt);
        Random random = new Random();
        
        for (int i = 0; i < maxLength; i++) {
            double[][] hiddenStates = processInput(tokenIds);
            double[] logits = calculateLogits(hiddenStates[hiddenStates.length-1]);
            
            // Add some randomness instead of always taking argmax
            int nextToken = sampleFromLogits(logits, random, 0.7);
            tokenIds.add(nextToken);
        }
        
        return tokenizer.decodeToString(tokenIds);
    }

    private int sampleFromLogits(double[] logits, Random random, double temperature) {
        // Apply temperature and convert to probabilities
        double[] probs = softmaxWithTemperature(logits, temperature);
        
        // Sample according to probabilities
        double r = random.nextDouble();
        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
            if (r <= sum) return i;
        }
        return probs.length - 1;
    }
    
    private double[] calculateLogits(double[] hiddenState) {
        double[] logits = new double[tokenizer.getVocabSize()];
        
        // Matrix multiplication: hiddenState × outputLayer
        for (int i = 0; i < tokenizer.getVocabSize(); i++) {
            logits[i] = 0;
            for (int j = 0; j < tokenVectorDimension; j++) {
                logits[i] += hiddenState[j] * outputLayer[j][i];
            }
        }
        
        return logits;
    }
    
    private double[] softmaxWithTemperature(double[] logits, double temperature) {
        double[] probs = new double[logits.length];
        double maxLogit = Double.NEGATIVE_INFINITY;
        
        // Find maximum logit for numerical stability
        for (double logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }
        
        // Apply temperature and calculate exponentials
        double sumExp = 0.0;
        for (int i = 0; i < logits.length; i++) {
            double scaledLogit = (logits[i] - maxLogit) / temperature;
            probs[i] = Math.exp(scaledLogit);
            sumExp += probs[i];
        }
        
        // Normalize to probabilities
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }
        
        return probs;
    }
    
    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        }
    }
    
    public static TransformerModel loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (TransformerModel) ois.readObject();
        }
    }
}