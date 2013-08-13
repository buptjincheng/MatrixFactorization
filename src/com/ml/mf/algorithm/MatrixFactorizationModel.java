package com.ml.mf.algorithm;

import java.util.HashMap;
import java.util.Map;

import com.ml.mf.common.Classifier;
import com.ml.mf.common.SGDLearner;
import com.ml.mf.data.Instance;
import com.ml.mf.data.Item;
import com.ml.mf.data.User;
import com.ml.mf.exception.FeatureException;
import com.ml.mf.utils.FeatureVectorUtils;
import com.ml.mf.data.UserItem;

/**
 * Matrix factorization model for recommendation. 
 * The model only uses latent features. (please note that it suffers from "cold start" problem )
 * 
 * @author wenzhe
 *
 */
public class MatrixFactorizationModel implements SGDLearner, Classifier{
    
    // free-parameters to control to model, should be learned from cross validation
    // TODO need to set this in config. 
    public double stepSize = 0.01;
    public double regularizationRate = 1;
    
    public double squareLoss = 0;
    
    /** ItemMF id (index) to ItemMF object map */
    Map<Integer, Item> items = new HashMap<Integer, Item>();
    /** UserMF id (index) to UserMF object map */
    Map<Integer, User> users = new HashMap<Integer, User>();
    
    @Override
    public void update(Instance instance) throws FeatureException {
        int userIndex = ((UserItem)instance).getUserIndex();
        int itemIndex = ((UserItem)instance).getItemIndex();
        double rating = ((UserItem)instance).getRating();
        
        if (!users.containsKey(userIndex)){
            // initialize new UserMF 
            users.put(userIndex, new User());
        }
        
        if (!items.containsKey(itemIndex)){
            // initialize new ItemMF. 
            items.put(itemIndex, new Item());
        }
        
        User user = users.get(userIndex);
        Item item = items.get(itemIndex);
        
        // update the weights. 
        double epsilon = FeatureVectorUtils.calInnerProduct(user.getLatentFeatures(), 
                item.getLatentFeatures()) - rating;
        squareLoss += epsilon * epsilon;
        
        // update User latent feature vector. 
        // u(t) = (1-\eta * lamda)u(t-1)-\eta * epsilon * v(t-1). 
        // u(t) is the User for time t,  and v(t) is Item for time t. 
        double[] firstTerm = FeatureVectorUtils.calMultiply(1-stepSize*regularizationRate, user.getLatentFeatures()); 
        double[] secondTerm = FeatureVectorUtils.calMultiply(stepSize * epsilon, item.getLatentFeatures()) ;
        double[] updatedUserFeatures = FeatureVectorUtils.calMinus(firstTerm, secondTerm);
        
        // update Item latent feature vector
        // v(t) = (1-\eta * lamda)v(t-1) - \eta * epsilon * u(t-1)
        firstTerm = FeatureVectorUtils.calMultiply(1-stepSize*regularizationRate, item.getLatentFeatures());
        secondTerm = FeatureVectorUtils.calMultiply(stepSize * epsilon, user.getLatentFeatures());
        double[] updatedItemFeatures = FeatureVectorUtils.calMinus(firstTerm, secondTerm);
        
        user.setLatentFeatures(updatedUserFeatures);
        item.setLatentFeatures(updatedItemFeatures);
        users.put(userIndex, user);
        items.put(itemIndex, item);
    }

    @Override
    public double predict(Instance instance) {
        User user = users.get(instance.getUserIndex());
        Item item = items.get(instance.getItemIndex());
        
        if (user == null || item == null){
            System.out.println("cold start for standard matrix factorization!");
            return -1;
        }
        
        double predictedValue = 0;
        try{ 
            predictedValue = FeatureVectorUtils.calInnerProduct(user.getLatentFeatures(), 
                    item.getLatentFeatures());
        } catch(Exception e){
            System.out.println("Cannot prodict the rating for given instance. Exception" +  
                    " occurs when compute the dot product of two feature vectors");
            return -1;
        }
        return predictedValue;
    }  
}
