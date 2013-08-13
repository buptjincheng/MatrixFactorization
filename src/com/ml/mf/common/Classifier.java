package com.ml.mf.common;

import com.ml.mf.data.Instance;


public interface Classifier {
    public double predict(Instance instance);
}
