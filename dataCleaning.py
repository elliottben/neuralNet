#!/usr/bin/python
import csv as csv
import numpy as np
import sys
import os
import operator
import re

class AiData:
    def __init__(self, batchSize):
        csv_file_train = csv.reader(open('data/reducedData.csv', 'rb'))
        header = csv_file_train.next()

        #read training data into list
        training_data = list(csv_file_train)
        rows = len(training_data)
        cols = len(training_data[0])
        self.dataDim = (rows, cols)
        
        #initialize the numpy arrays for the data
        self.rawData = np.zeros([1, self.dataDim[1]])
        self.featureData = np.zeros([1, self.dataDim[1]-1])
        self.solutionData = np.zeros([1, 1])
        
        #populating the numpy arrays, only training over 100 at first
        count = 0
        for dataPoint in training_data:
            self.rawData = np.vstack((self.rawData, dataPoint))
            self.featureData = np.vstack((self.featureData, dataPoint[0:21]))
            self.solutionData = np.vstack((self.solutionData, dataPoint[21:22]))
            if count % 1000 == 0 and count !=0:
                print "gathered 1000 data points..."
            count += 1
        
        self.rawData = np.delete(self.rawData, (0), axis=0)
        self.featureData = np.delete(self.featureData, (0), axis=0)
        self.solutionData = np.delete(self.solutionData, (0), axis=0)

        #modifying data ouput given batchSize
        rowsEnd = rows % batchSize
        self.rawData = self.rawData[0:rows - rowsEnd]
        self.featureData = self.featureData[0:rows - rowsEnd]
        self.solutionData = self.solutionData[0:rows - rowsEnd]

        self.iterations = rows / batchSize

        self.rawData = np.split(self.rawData, self.iterations)
        self.featureData = np.split(self.featureData, self.iterations)
        self.solutionData = np.split(self.solutionData, self.iterations)            