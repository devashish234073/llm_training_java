package com.llm;

public class PostionalEncoder {
	public static double[] getPositionalEncoding(int pos, int tokenVectorDimension) {//tokenVectorDimension is also called embedding dimension
        double[] encoding = new double[tokenVectorDimension];
        
        for (int i = 0; i < tokenVectorDimension; i++) {
            double denominator = Math.pow(10000, 2.0 * (i / 2) / tokenVectorDimension);
            double angle = pos / denominator;
            
            if (i % 2 == 0) {
                encoding[i] = Math.sin(angle);
            } else {
                // Odd index: use cosine
                encoding[i] = Math.cos(angle);
            }
        }
        return encoding;
    }
}
