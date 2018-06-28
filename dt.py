import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
 
# Function importing Dataset
def importdata():
    haberman_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/haberman/haberman.data',
    sep= ',', header = None)
     
    # Printing the dataswet shape
    print ("Dataset Lenght: ", len(haberman_data))
    print ("Dataset Shape: ", haberman_data.shape)
     
    # Printing the dataset obseravtions
    print ("Dataset: ",haberman_data.head())
    return haberman_data
 
# Function to split the dataset
def splitdataset(haberman_data):
 
    # Seperating the target variable
    X = haberman_data.values[:, 0:2]
    Y = haberman_data.values[:, 3]
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.6, random_state = 100)
     
    return X, Y, X_train, X_test, y_train, y_test
     
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 10)
 
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 
 
# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
     
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
 
# Driver code
def main():
     
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
     
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
     
     
# Calling main function
if __name__=="__main__":
    main()