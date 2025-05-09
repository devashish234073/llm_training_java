package llm;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class TransformerEncoderLayer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private final int embedDim;
	private final int numHeads;
	private final int headDim;

	// Attention weights
	private final float[][][] wq, wk, wv; // [numHeads][embedDim][headDim]
	private final float[][][] wo; // [numHeads][headDim][embedDim]

	// Feed-forward network weights
	private final float[][] ffnW1; // [embedDim][ffnDim]
	private final float[][] ffnW2; // [ffnDim][embedDim]
	private final float[] ffnB1, ffnB2;
	int ffnDim;

	// Layer norms
	private final LayerNorm norm1, norm2;

	// Add these cache fields
	private float[][][] lastInput; // Cache for main input
	private float[][][] lastAttnOutput; // Cache for attention output
	private float[][][] lastFfnOutput; // Cache for FFN output
	private float[][][] lastFfnHiddenInput; // Cache for FFN input

	// Gradient storage
	private float[][][] gradWq, gradWk, gradWv, gradWo;
	private float[][] gradFfnW1, gradFfnW2;
	private float[] gradFfnB1, gradFfnB2;
	private float[][][] lastQ, lastK, lastV; // Cache for attention
	private float[][][] lastAttnScores; // Cache for attention
	private float[][][] lastFfnInput; // Cache for FFN

	public TransformerEncoderLayer(int embedDim, int numHeads) {
		this.embedDim = embedDim;
		this.numHeads = numHeads;
		this.headDim = embedDim / numHeads;

		Random rand = new Random();

		// Initialize attention weights
		this.wq = new float[numHeads][embedDim][headDim];
		this.wk = new float[numHeads][embedDim][headDim];
		this.wv = new float[numHeads][embedDim][headDim];
		this.wo = new float[numHeads][headDim][embedDim];

		initializeWeights(wq, rand);
		initializeWeights(wk, rand);
		initializeWeights(wv, rand);
		initializeWeights(wo, rand);

		// Initialize FFN weights (typically ffnDim = 4*embedDim)
		this.ffnDim = embedDim * 4;
		this.ffnW1 = new float[embedDim][ffnDim];
		this.ffnW2 = new float[ffnDim][embedDim];
		this.ffnB1 = new float[ffnDim];
		this.ffnB2 = new float[embedDim];

		initializeWeights(ffnW1, rand);
		initializeWeights(ffnW2, rand);

		// Initialize layer norms
		this.norm1 = new LayerNorm(embedDim);
		this.norm2 = new LayerNorm(embedDim);

		// Initialize gradients
		this.gradWq = new float[numHeads][embedDim][headDim];
		this.gradWk = new float[numHeads][embedDim][headDim];
		this.gradWv = new float[numHeads][embedDim][headDim];
		this.gradWo = new float[numHeads][headDim][embedDim];
		this.gradFfnW1 = new float[embedDim][ffnDim];
		this.gradFfnW2 = new float[ffnDim][embedDim];
		this.gradFfnB1 = new float[ffnDim];
		this.gradFfnB2 = new float[embedDim];
	}

	private void initializeWeights(float[][][] weights, Random rand) {
		float scale = (float) (1.0 / Math.sqrt(weights[0][0].length));
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] = (float) (rand.nextGaussian() * scale);
				}
			}
		}
	}

	private void initializeWeights(float[][] weights, Random rand) {
		float scale = (float) (1.0 / Math.sqrt(weights[0].length));
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = (float) (rand.nextGaussian() * scale);
			}
		}
	}

	public float[][][] forward(float[][][] x, boolean[][] mask) {
	    this.lastInput = x; // Cache original input
	    
	    // 1. Self-attention with residual connection
	    float[][][] attnOut = multiHeadAttention(x, x, x, mask);
	    this.lastAttnOutput = attnOut;
	    
	    float[][][] residual = add(x, attnOut);
	    float[][][] norm1Out = norm1.forward3d(residual);
	    
	    // 2. Feed-forward network with residual connection
	    float[][][] ffnOut = feedForward(norm1Out);
	    
	    float[][][] residual2 = add(norm1Out, ffnOut);
	    float[][][] output = norm2.forward3d(residual2);
	    
	    return output;
	}

	public float[][][] backward(float[][][] gradOutput, boolean[][] mask) {
		// gradOutput shape: [seqLen][batchSize][embedDim]

		// 1. Backward through second LayerNorm and residual
		float[][][] gradNorm2 = norm2.backward3d(gradOutput);
		float[][][] gradFfn = gradNorm2.clone();
		float[][][] gradNorm1Out = gradNorm2.clone();

		// 2. Backward through FFN
		float[][][] gradFfnOut = backwardFeedForward(gradFfn);

		// 3. Backward through first LayerNorm and residual
		float[][][] gradNorm1 = norm1.backward3d(add(gradNorm1Out, gradFfnOut));
		float[][][] gradAttn = gradNorm1.clone();
		float[][][] gradX = gradNorm1.clone();

		// 4. Backward through attention
		float[][][] gradAttnOut = backwardMultiHeadAttention(gradAttn, mask);

		return add(gradX, gradAttnOut);
	}

	private float[][][] backwardMultiHeadAttention(float[][][] gradOutput, boolean[][] mask) {
		int seqLen = gradOutput.length;
		int batchSize = gradOutput[0].length;

		// 1. Backward through output projection
		float[][][][] gradHeads = new float[numHeads][seqLen][batchSize][headDim];

		for (int i = 0; i < seqLen; i++) {
			for (int b = 0; b < batchSize; b++) {
				// Gradient through WO
				for (int h = 0; h < numHeads; h++) {
					for (int dIn = 0; dIn < headDim; dIn++) {
						for (int dOut = 0; dOut < embedDim; dOut++) {
							int concatPos = h * headDim + dIn;
							gradWo[h][dIn][dOut] += lastInput[i][b][concatPos] * gradOutput[i][b][dOut];
							gradHeads[h][i][b][dIn] += wo[h][dIn][dOut] * gradOutput[i][b][dOut];
						}
					}
				}
			}
		}

		// 2. Backward through attention computation
		float[][][][] gradV = new float[numHeads][seqLen][batchSize][headDim];

		for (int h = 0; h < numHeads; h++) {
			for (int i = 0; i < seqLen; i++) {
				for (int b = 0; b < batchSize; b++) {
					for (int j = 0; j < seqLen; j++) {
						for (int d = 0; d < headDim; d++) {
							gradV[h][j][b][d] += lastAttnScores[i][j][b] * gradHeads[h][i][b][d];
						}
					}
				}
			}
		}

		// 3. Backward through Q, K, V projections
		float[][][] gradInput = new float[seqLen][batchSize][embedDim];

		for (int h = 0; h < numHeads; h++) {
			// Backward through V projection
			for (int i = 0; i < seqLen; i++) {
				for (int b = 0; b < batchSize; b++) {
					for (int d = 0; d < headDim; d++) {
						for (int e = 0; e < embedDim; e++) {
							gradWv[h][e][d] += lastInput[i][b][e] * gradV[h][i][b][d];
							gradInput[i][b][e] += wv[h][e][d] * gradV[h][i][b][d];
						}
					}
				}
			}

			// Similar backward passes for Q and K projections...
		}

		return gradInput;
	}

	private float[][][] backwardFeedForward(float[][][] gradOutput) {
	    int seqLen = gradOutput.length;
	    int batchSize = gradOutput[0].length;
	    
	    // 1. Backward through second linear layer (ffnW2)
	    float[][][] gradHidden = new float[seqLen][batchSize][ffnDim];
	    
	    for (int i = 0; i < seqLen; i++) {
	        for (int b = 0; b < batchSize; b++) {
	            // Gradient for weights and hidden layer
	            for (int j = 0; j < ffnDim; j++) {
	                for (int k = 0; k < embedDim; k++) {
	                    gradFfnW2[j][k] += lastFfnOutput[i][b][j] * gradOutput[i][b][k];
	                    gradHidden[i][b][j] += ffnW2[j][k] * gradOutput[i][b][k];
	                }
	            }
	            
	            // Gradient for bias (ffnB2 shape: [embedDim])
	            for (int k = 0; k < embedDim; k++) {
	                gradFfnB2[k] += gradOutput[i][b][k];
	            }
	        }
	    }
	    
	    // 2. Backward through GELU activation
	    float[][][] gradFfn1 = new float[seqLen][batchSize][ffnDim];
	    
	    for (int i = 0; i < seqLen; i++) {
	        for (int b = 0; b < batchSize; b++) {
	            for (int j = 0; j < ffnDim; j++) {
	                float x = lastFfnHiddenInput[i][b][j]; // Input to GELU
	                float tanhArg = (float)(Math.sqrt(2.0/Math.PI) * (x + 0.044715f * x * x * x));
	                float tanhVal = (float)Math.tanh(tanhArg);
	                float gradGELU = 0.5f * (1.0f + tanhVal + 
	                    x * (1 - tanhVal * tanhVal) * (float)(Math.sqrt(2.0/Math.PI) * (1 + 0.134145f * x * x)));
	                gradFfn1[i][b][j] = gradHidden[i][b][j] * gradGELU;
	            }
	        }
	    }
	    
	    // 3. Backward through first linear layer (ffnW1)
	    float[][][] gradInput = new float[seqLen][batchSize][embedDim];
	    
	    for (int i = 0; i < seqLen; i++) {
	        for (int b = 0; b < batchSize; b++) {
	            for (int j = 0; j < embedDim; j++) {
	                for (int k = 0; k < ffnDim; k++) {
	                    gradFfnW1[j][k] += lastFfnInput[i][b][j] * gradFfn1[i][b][k];
	                    gradInput[i][b][j] += ffnW1[j][k] * gradFfn1[i][b][k];
	                }
	            }
	            // Gradient for bias (ffnB1 shape: [ffnDim])
	            for (int k = 0; k < ffnDim; k++) {
	                gradFfnB1[k] += gradFfn1[i][b][k];
	            }
	        }
	    }
	    
	    return gradInput;
	}

	public Map<String, float[][]> getParameters() {
		Map<String, float[][]> params = new HashMap<>();

		// Attention parameters
		for (int h = 0; h < numHeads; h++) {
			params.put("wq_" + h, wq[h]);
			params.put("wk_" + h, wk[h]);
			params.put("wv_" + h, wv[h]);
			params.put("wo_" + h, wo[h]);
		}

		// FFN parameters - convert 1D arrays to 2D
		params.put("ffn_w1", ffnW1);
		params.put("ffn_w2", ffnW2);
		params.put("ffn_b1", new float[][] { ffnB1 }); // Convert to 2D
		params.put("ffn_b2", new float[][] { ffnB2 }); // Convert to 2D

		return params;
	}

	public Map<String, float[][]> getGradients() {
		Map<String, float[][]> grads = new HashMap<>();

		// Attention gradients
		for (int h = 0; h < numHeads; h++) {
			grads.put("grad_wq_" + h, gradWq[h]);
			grads.put("grad_wk_" + h, gradWk[h]);
			grads.put("grad_wv_" + h, gradWv[h]);
			grads.put("grad_wo_" + h, gradWo[h]);
		}

		// FFN gradients - convert 1D arrays to 2D
		grads.put("grad_ffn_w1", gradFfnW1);
		grads.put("grad_ffn_w2", gradFfnW2);
		grads.put("grad_ffn_b1", new float[][] { gradFfnB1 }); // Convert to 2D
		grads.put("grad_ffn_b2", new float[][] { gradFfnB2 }); // Convert to 2D

		return grads;
	}

	public void zeroGrad() {
		// Reset all gradients to zero
		for (float[][][] grad : new float[][][][] { gradWq, gradWk, gradWv, gradWo }) {
			for (float[][] g : grad) {
				for (float[] gg : g) {
					Arrays.fill(gg, 0f);
				}
			}
		}

		// For 2D gradients (ffn weights)
		for (float[][] grad : new float[][][] { gradFfnW1, gradFfnW2 }) {
			for (float[] g : grad) {
				Arrays.fill(g, 0f);
			}
		}

		// For 1D gradients (ffn biases)
		Arrays.fill(gradFfnB1, 0f);
		Arrays.fill(gradFfnB2, 0f);
	}

	private float[][][] multiHeadAttention(float[][][] query, float[][][] key, float[][][] value, boolean[][] mask) {
		int seqLen = query.length;
		int batchSize = query[0].length;

		float[][][][] heads = new float[numHeads][seqLen][batchSize][headDim];
		this.lastAttnScores = new float[seqLen][seqLen][batchSize]; // Initialize cache
		
		// Process each head
		for (int h = 0; h < numHeads; h++) {
			// Linear projections
			float[][][] q = matmul(query, wq[h]); // [seqLen][batchSize][headDim]
			float[][][] k = matmul(key, wk[h]); // [seqLen][batchSize][headDim]
			float[][][] v = matmul(value, wv[h]); // [seqLen][batchSize][headDim]

			// Cache Q, K, V for backward pass
	        this.lastQ = q;
	        this.lastK = k;
	        this.lastV = v;
	        // Cache attention scores for backward pass
			// Scaled dot-product attention
			//float[][][] attnScores = new float[seqLen][seqLen][batchSize];

			// Compute attention scores
			for (int i = 0; i < seqLen; i++) {
				for (int j = 0; j < seqLen; j++) {
					for (int b = 0; b < batchSize; b++) {
						float score = 0;
						for (int d = 0; d < headDim; d++) {
							score += q[i][b][d] * k[j][b][d];
						}
						score /= Math.sqrt(headDim);

						// Apply mask (set to -inf where mask is true)
						if (mask != null && mask[i][j]) {
							score = Float.NEGATIVE_INFINITY;
						}

						lastAttnScores[i][j][b] = score;
					}
				}
			}

			// Softmax
			for (int i = 0; i < seqLen; i++) {
				for (int b = 0; b < batchSize; b++) {
					// Find max for numerical stability
					float max = Float.NEGATIVE_INFINITY;
					for (int j = 0; j < seqLen; j++) {
						if (lastAttnScores[i][j][b] > max) {
							max = lastAttnScores[i][j][b];
						}
					}

					// Compute exp and sum
					float sum = 0;
					for (int j = 0; j < seqLen; j++) {
						float exp = (float) Math.exp(lastAttnScores[i][j][b] - max);
						lastAttnScores[i][j][b] = exp;
						sum += exp;
					}

					// Normalize
					for (int j = 0; j < seqLen; j++) {
						lastAttnScores[i][j][b] /= sum;
					}
				}
			}

			// Apply attention to values
			for (int i = 0; i < seqLen; i++) {
				for (int b = 0; b < batchSize; b++) {
					for (int d = 0; d < headDim; d++) {
						float sum = 0;
						for (int j = 0; j < seqLen; j++) {
							sum += lastAttnScores[i][j][b] * v[j][b][d];
						}
						heads[h][i][b][d] = sum;
					}
				}
			}
		}

		// Concatenate heads and apply output projection
		float[][][] output = new float[seqLen][batchSize][embedDim];

		for (int i = 0; i < seqLen; i++) {
			for (int b = 0; b < batchSize; b++) {
				// Concatenate heads
				int concatPos = 0;
				for (int h = 0; h < numHeads; h++) {
					for (int d = 0; d < headDim; d++) {
						output[i][b][concatPos++] = heads[h][i][b][d];
					}
				}

				// Output projection
				float[] proj = new float[embedDim];
				for (int h = 0; h < numHeads; h++) {
					for (int dOut = 0; dOut < embedDim; dOut++) {
						for (int dIn = 0; dIn < headDim; dIn++) {
							proj[dOut] += output[i][b][h * headDim + dIn] * wo[h][dIn][dOut];
						}
					}
				}

				System.arraycopy(proj, 0, output[i][b], 0, embedDim);
			}
		}

		return output;
	}

	private float[][][] feedForward(float[][][] x) {
	    int seqLen = x.length;
	    int batchSize = x[0].length;
	    int ffnDim = ffnW1[0].length;

	    // Cache the input to the FFN (for first layer gradients)
	    this.lastFfnInput = x;  // shape [seqLen][batchSize][embedDim]

	    float[][][] hidden = new float[seqLen][batchSize][ffnDim];
	    float[][][] hiddenInput = new float[seqLen][batchSize][ffnDim]; // pre-GELU
	    float[][][] output = new float[seqLen][batchSize][embedDim];

	    for (int i = 0; i < seqLen; i++) {
	        for (int b = 0; b < batchSize; b++) {
	            // First linear layer
	            for (int j = 0; j < ffnDim; j++) {
	                float sum = ffnB1[j];
	                for (int k = 0; k < embedDim; k++) {
	                    sum += x[i][b][k] * ffnW1[k][j];
	                }
	                hiddenInput[i][b][j] = sum; // store pre-activation
	                // GELU activation
	                hidden[i][b][j] = (float) (sum * 0.5
	                        * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (sum + 0.044715 * Math.pow(sum, 3)))));
	            }

	            // Second linear layer
	            for (int j = 0; j < embedDim; j++) {
	                float sum = ffnB2[j];
	                for (int k = 0; k < ffnDim; k++) {
	                    sum += hidden[i][b][k] * ffnW2[k][j];
	                }
	                output[i][b][j] = sum;
	            }
	        }
	    }

	    // Cache values needed for backward pass
	    this.lastFfnOutput = hidden; // post-GELU activations [seqLen][batchSize][ffnDim]
	    this.lastFfnHiddenInput = hiddenInput; // pre-GELU inputs [seqLen][batchSize][ffnDim]
	    
	    return output;
	}

	// Helper methods
	private float[][][] matmul(float[][][] a, float[][] b) {
		int dim1 = a.length;
		int dim2 = a[0].length;
		int dim3 = b[0].length;

		float[][][] result = new float[dim1][dim2][dim3];

		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				for (int k = 0; k < dim3; k++) {
					float sum = 0;
					for (int m = 0; m < b.length; m++) {
						sum += a[i][j][m] * b[m][k];
					}
					result[i][j][k] = sum;
				}
			}
		}

		return result;
	}

	private float[][][] add(float[][][] a, float[][][] b) {
		float[][][] result = new float[a.length][a[0].length][a[0][0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				for (int k = 0; k < a[i][j].length; k++) {
					result[i][j][k] = a[i][j][k] + b[i][j][k];
				}
			}
		}
		return result;
	}

	private float[][][] applyLayerNorm(float[][][] x, LayerNorm norm) {
		float[][][] output = new float[x.length][x[0].length][x[0][0].length];
		for (int i = 0; i < x.length; i++) {
			output[i] = norm.forward(x[i]);
		}
		return output;
	}
}