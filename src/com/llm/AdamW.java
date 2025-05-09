package com.llm;

import java.util.HashMap;
import java.util.Map;

class AdamW {
	private final Map<String, float[][]> parameters;
	private final Map<String, float[][]> m; // First moment estimate
	private final Map<String, float[][]> v; // Second moment estimate
	private final float lr;
	private final float beta1 = 0.9f;
	private final float beta2 = 0.999f;
	private final float eps = 1e-8f;
	private final float weightDecay = 0.01f;
	private int t = 0;
	private final SimpleGPT model; // Add reference to the model

	public AdamW(SimpleGPT model, float lr) {
		this.model = model;
		this.parameters = model.getParameters(); // Get parameters from model
		this.lr = lr;
		this.m = new HashMap<>();
		this.v = new HashMap<>();

		// Initialize moment estimates
		for (String key : parameters.keySet()) {
			float[][] param = parameters.get(key);
			m.put(key, new float[param.length][param[0].length]);
			v.put(key, new float[param.length][param[0].length]);
		}
	}

	public void step() {
		t++;
		Map<String, float[][]> gradients = model.getGradients(); // Get gradients from model

		for (String key : parameters.keySet()) {
			float[][] param = parameters.get(key);
			float[][] grad = gradients.get(key);
			float[][] mEst = m.get(key);
			float[][] vEst = v.get(key);

			// Skip if gradient is null (shouldn't happen after our fix above)
			if (grad == null) {
				System.err.println("Warning: Null gradient for parameter " + key);
				continue;
			}

			for (int i = 0; i < param.length; i++) {
				for (int j = 0; j < param[i].length; j++) {
					// Update moments
					mEst[i][j] = beta1 * mEst[i][j] + (1 - beta1) * grad[i][j];
					vEst[i][j] = beta2 * vEst[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];

					// Bias correction
					float mHat = mEst[i][j] / (1 - (float) Math.pow(beta1, t));
					float vHat = vEst[i][j] / (1 - (float) Math.pow(beta2, t));

					// Update with weight decay
					param[i][j] -= lr * (mHat / (Math.sqrt(vHat) + eps) + weightDecay * param[i][j]);
				}
			}
		}
	}
}