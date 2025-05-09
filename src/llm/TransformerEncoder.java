package llm;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class TransformerEncoder implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private final TransformerEncoderLayer[] layers;
    private boolean[][] lastMask; // Cache for backward pass
    private float[][][] lastInput; // Cache for backward pass
    
    public TransformerEncoder(int embedDim, int numHeads, int numLayers) {
        this.layers = new TransformerEncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new TransformerEncoderLayer(embedDim, numHeads);
        }
    }
    
    public float[][][] forward(float[][][] x, boolean[][] mask) {
        this.lastInput = x; // Store input for backward pass
        this.lastMask = mask; // Store mask for backward pass
        
        for (TransformerEncoderLayer layer : layers) {
            x = layer.forward(x, mask);
        }
        return x;
    }
    
    public float[][][] backward(float[][][] gradOutput) {
        float[][][] gradInput = gradOutput;
        
        // Backward through layers in reverse order
        for (int i = layers.length - 1; i >= 0; i--) {
            gradInput = layers[i].backward(gradInput, lastMask);
        }
        
        return gradInput;
    }
    
    // Helper method to get all parameters
    public Map<String, float[][]> getParameters() {
        Map<String, float[][]> params = new HashMap<>();
        for (int i = 0; i < layers.length; i++) {
            Map<String, float[][]> layerParams = layers[i].getParameters();
            for (Map.Entry<String, float[][]> entry : layerParams.entrySet()) {
                params.put("layer_" + i + "." + entry.getKey(), entry.getValue());
            }
        }
        return params;
    }
    
    // Helper method to get all gradients
    public Map<String, float[][]> getGradients() {
        Map<String, float[][]> grads = new HashMap<>();
        for (int i = 0; i < layers.length; i++) {
            Map<String, float[][]> layerGrads = layers[i].getGradients();
            for (Map.Entry<String, float[][]> entry : layerGrads.entrySet()) {
                grads.put("layer_" + i + "." + entry.getKey(), entry.getValue());
            }
        }
        return grads;
    }
    
    // Helper method to zero all gradients
    public void zeroGrad() {
        for (TransformerEncoderLayer layer : layers) {
            layer.zeroGrad();
        }
    }
}