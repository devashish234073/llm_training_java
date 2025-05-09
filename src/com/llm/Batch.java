package com.llm;

import java.util.List;

public class Batch {
    private final List<int[]> inputs;
    private final List<int[]> targets;

    public Batch(List<int[]> inputs, List<int[]> targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public List<int[]> getInputs() {
        return inputs;
    }

    public List<int[]> getTargets() {
        return targets;
    }
}
