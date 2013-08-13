package com.ml.mf.data;

import java.util.Random;

public class User extends Entity{
    protected double[] latentFeatures;
    public static final int numOfFeatures = 20;
   
    public User(){
        super();
        latentFeatures = new double[numOfFeatures];
        // randomly initialize the latent value
        for (int i = 0; i < numOfFeatures; i++){
            latentFeatures[i] = new Random().nextDouble();
        }
    }
    
    public double[] getLatentFeatures(){
        return latentFeatures;
    }
    
    public void setLatentFeatures(double[] latentFeatures){
        this.latentFeatures = latentFeatures;
    }
}
