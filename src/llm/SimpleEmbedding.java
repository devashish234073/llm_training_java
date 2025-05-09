package llm;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

public class SimpleEmbedding implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private final int vocabSize;
	private final int embedDim;
	private final float[][] weights; // Embedding matrix: [vocabSize][embedDim]

	public int getVocabSize() {
		return vocabSize;
	}

	public int getEmbedDim() {
		return embedDim;
	}

	public float[][] getWeights() {
		return weights;
	}

	public SimpleEmbedding(int vocabSize, int embedDim) {
		this.vocabSize = vocabSize;
		this.embedDim = embedDim;
		this.weights = new float[vocabSize][embedDim];

		// Random initialization (similar to np.random.randn)
		Random rand = new Random();
		for (int i = 0; i < vocabSize; i++) {
			for (int j = 0; j < embedDim; j++) {
				weights[i][j] = (float) rand.nextGaussian(); // Gaussian distribution
			}
		}
	}

	/**
	 * Get embeddings for token IDs (input as int[]).
	 * 
	 * @param tokenIds Array of token IDs (indices into the embedding matrix).
	 * @return 2D array of embeddings: shape [tokenIds.length][embedDim].
	 */
	public float[][] call(int[] tokenIds) {
		float[][] embeddings = new float[tokenIds.length][embedDim];
		for (int i = 0; i < tokenIds.length; i++) {
			int tokenId = tokenIds[i];
			if (tokenId < 0 || tokenId >= vocabSize) {
				throw new IllegalArgumentException("Token ID " + tokenId + " is out of bounds.");
			}
			System.arraycopy(weights[tokenId], 0, embeddings[i], 0, embedDim);
		}
		return embeddings;
	}

	/**
	 * Get embeddings for token IDs (input as List<Integer>).
	 * 
	 * @param tokenIds List of token IDs.
	 * @return 2D array of embeddings: shape [tokenIds.size()][embedDim].
	 */
	public float[][] call(List<Integer> tokenIds) {
		return call(tokenIds.stream().mapToInt(i -> i).toArray());
	}

	// For token embeddings (with token IDs)
	public void backward(int[] tokenIds, float[][] gradOutput) {
		for (int t = 0; t < tokenIds.length; t++) {
			int tokenId = tokenIds[t];
			for (int d = 0; d < gradOutput[t].length; d++) {
				weights[tokenId][d] += gradOutput[t][d];
			}
		}
	}

	// For position embeddings (without token IDs)
	public void backward(float[][] gradOutput) {
		for (int pos = 0; pos < gradOutput.length; pos++) {
			for (int d = 0; d < gradOutput[pos].length; d++) {
				weights[pos][d] += gradOutput[pos][d];
			}
		}
	}
}