import numpy as np
import math
import random

in_dim = 785 # input dimension
out_dim = 10 # number of classes (0-9)
eta = 1 # Learning rate. You might try different rates (between 0.001 and 1) to maximize the accuracy

def Weight_update(feature, label, weight_i2o):
	##
	#Update the weights for a train feature.
		# Inputs:
			# feature: feature vector (ndarray) of a data point with 785 dimensions. Here, the feature represents a handwritten digit 
			         # of a 28x28 pixel grayscale image, which is flattened into a 785-dimensional vector (include bias)
			# label: Actual label of the train feature 
			# weight_i2o: current weights with shape (in_dim x out_dim) from input (feature vector) to output (digit number 0-9)
		# Return: updated weight
	##
	#"*** YOUR CODE HERE ***"
    
    #getting a highest score as predicted label for the given feature:
    wT = np.transpose(weight_i2o)
    scores = wT.dot(feature)
    predicted = scores.argmax()

    label = int(label)
    
    #if predicted label is not equal to true label, we need to update the weight:
    if label != predicted:
        
        #vector result of mapped true label
        tx = np.zeros((out_dim, 1))
        tx[label] = 1
        
        #vector result of mapped predicted label
        yx = np.zeros((out_dim, 1))
        yx[predicted] = 1
        
        #reshaping the feature for calculation
        feature = np.reshape(feature, (in_dim, 1))
        
        #updating part for readilibility:
        change = np.transpose(tx - yx)
        
        #update of weight:
        #heigher true score & lower wrong score updates for weight, implementation structured together:
        weight_i2o = weight_i2o + ( eta * feature.dot(change) )
        
    return weight_i2o

def get_predictions(dataset, weight_i2o):
	#"""
	#Calculates the predicted label for each feature in dataset.
		# Inputs:
			# dataset: a set of feature vectors with shape  
			# weight_i2o: current weights with shape (in_dim x out_dim)
		# Return: list (or ndarray) of predicted labels from given dataset
	#"""
	#"*** YOUR CODE HERE ***"
    
    predictions = []
    
    #going through each feature x in the dataset:
    for x in dataset:
        
        #calculating a predicted label for each x:
        wT = np.transpose(weight_i2o)
        scores = wT.dot(x)
        
        #taking only the heighest valued score (max activation value):
        label = scores.argmax()
        predictions.append(label)
        
    return predictions


def train(train_set, labels, weight_i2o):
	#"""
	#Train the perceptron until convergence.
	# Inputs:
		# train_set: training set (ndarray) with shape (number of data points x in_dim)
		# labels: list (or ndarray) of actual labels from training set
		# weight_i2o:
	# Return: the weights for the entire training set
	#"""
    for i in range(0, train_set.shape[0]):        
        weight_i2o = Weight_update(train_set[i, :], labels[i], weight_i2o)        
    return weight_i2o
