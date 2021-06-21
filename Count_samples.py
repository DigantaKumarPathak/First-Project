#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 05:31:47 2018

@author: rusa
"""
import numpy as np
windowSize = 11
numPCAcomponents = 30
testRatio = 0.7
import csv

def Count_Array_Element(X, file_csv):
    unique, counts = np.unique(X, return_counts=True)
    train_labels_count=dict(zip(unique, counts))
    with open(file_csv, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in train_labels_count.items():
            writer.writerow([key, value])
       
    ''' 
    f=open(file_csv, 'wb')
    w=csv.DictWriter(f, train_labels_count)
    w.writerow(train_labels_count)  
    f.close()
    '''
    return train_labels_count
#-------------------------------------------------------------------------------------------------------------------------------
# Load The Train Data
X_train = np.load("/home/diganta/Diganta_DeepLearning/Classification-of-Hyperspectral-Image-master/XtrainWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")

y_train = np.load("/home/diganta/Diganta_DeepLearning/Classification-of-Hyperspectral-Image-master/ytrainWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
#--------------------------------------------------------------------------------------------------------------------------------
# Load The Validate Data
X_validate = np.load("/home/diganta/Diganta_DeepLearning/Classification-of-Hyperspectral-Image-master/XvalidateWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")

y_validate = np.load("/home/diganta/Diganta_DeepLearning/Classification-of-Hyperspectral-Image-master/yvalidateWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
#--------------------------------------------------------------------------------------------------------------------------------
# Load The Test Data
X_test = np.load("/home/diganta/Diganta_DeepLearning/Classification-of-Hyperspectral-Image-master/XtestWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")

y_test = np.load("/home/diganta/Diganta_DeepLearning/Classification-of-Hyperspectral-Image-master/ytestWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

#--------------------------------------------------------------------------------------------------------------------------------
count_label_train=Count_Array_Element(y_train, file_csv='train_set.csv')
count_label_validate=Count_Array_Element(y_validate, file_csv='validate_set.csv')
count_label_test=Count_Array_Element(y_test, file_csv='test_set.csv')
print(count_label_train)
print(count_label_test)
print(count_label_validate)        