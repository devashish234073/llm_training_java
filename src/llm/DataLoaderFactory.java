package llm;

import java.util.List;

public class DataLoaderFactory {

	public static CustomDataLoader createDataloaderV1(String text, int batchSize, boolean shuffle, boolean dropLast,
			int numWorkers, // Not used in this Java version
			int maxLength, int stride, SimpleTokenizer tokenizer) {
		GPTDatasetV1 dataset = new GPTDatasetV1(text, maxLength, stride, tokenizer);
		List<Sample> samples = dataset.getSamples();
		return new CustomDataLoader(samples, batchSize, shuffle, dropLast);
	}
}
