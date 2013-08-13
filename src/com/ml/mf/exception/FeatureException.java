package com.ml.mf.exception;

/**
 * Exception class used to record any exception related to
 * feature vector..
 *
 */
public class FeatureException extends Exception{
    public FeatureException(String message){
        super(message);
    }
    
    public FeatureException(String message, Throwable e){
        super(message, e);
    }
    
    public FeatureException(Exception ex){
        super(ex);
    }
}
