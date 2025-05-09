package com.llm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class CustomDataLoader implements Iterable<Batch> {
    private final List<Sample> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final List<Integer> indices;

    public CustomDataLoader(List<Sample> dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.indices = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            indices.add(i);
        }
    }

    @Override
    public Iterator<Batch> iterator() {
        return new Iterator<Batch>() {
            private int currentIndex = 0;
            private final List<Integer> shuffledIndices;

            {
                shuffledIndices = new ArrayList<>(indices);
                if (shuffle) {
                    Collections.shuffle(shuffledIndices);
                }
            }

            @Override
            public boolean hasNext() {
                return currentIndex < shuffledIndices.size();
            }

            @Override
            public Batch next() {
                int endIndex = Math.min(currentIndex + batchSize, shuffledIndices.size());
                List<Integer> batchIndices = shuffledIndices.subList(currentIndex, endIndex);

                if (dropLast && batchIndices.size() < batchSize) {
                    currentIndex = shuffledIndices.size(); // skip remaining
                    return null;
                }

                List<int[]> inputBatch = new ArrayList<>();
                List<int[]> targetBatch = new ArrayList<>();

                for (int idx : batchIndices) {
                    Sample sample = dataset.get(idx);
                    inputBatch.add(sample.getInputChunk());
                    targetBatch.add(sample.getTargetChunk());
                }

                currentIndex += batchSize;
                return new Batch(inputBatch, targetBatch);
            }
        };
    }

    public int size() {
        int fullBatches = dataset.size() / batchSize;
        if (!dropLast && dataset.size() % batchSize != 0) {
            fullBatches += 1;
        }
        return fullBatches;
    }
}

