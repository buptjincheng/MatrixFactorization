package com.ml.mf.common;

import com.ml.mf.data.Instance;



public interface SGDLearner {
    public void update(Instance instance) throws Exception;  
}
