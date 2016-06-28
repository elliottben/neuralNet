#!/usr/bin/python
import csv as csv
import numpy as np
import sys
import os
import operator
import re
#import dataset
import dataCleaning

class net:
    def __init__(self, layers, batchSize, wRows, wCols):
        self.data = dataCleaning.AiData(batchSize)
        print "Data successfully read with batch size..."
        self.batchSize = batchSize
        self.step = 100

        #create the neuron layers, indexing from zero
        self.layers = layers
        self.layer = []
        for count in range(layers):
            self.layer.append(self.neuronLayer(wRows[count], wCols[count], batchSize))
        print "Neuron Layers successfully created..."

    #forward propagate the data
    def updateInput(self, iteration):
        self.layer[0].input = self.data.featureData[iteration]
        output = self.layer[0].sigmoid()
        #for the rest populate
        for count in range(1, self.layers, 1):
            self.layer[count].input = output
            output = self.layer[count].sigmoid()

    #score per batch
    def NLLScore(self, iteration):
        #predictions produced in the last layer
        predictionsFor = self.layer[self.layers-1].sigmoid()
        predictionsAgainst = 1 - predictionsFor
        solutions = self.data.solutionData[iteration].astype('float64')
        lossPerItem = -1 * (solutions * np.log(predictionsFor) + (1-solutions)*np.log(predictionsAgainst))
        return np.sum(lossPerItem)/self.batchSize

    def NLLDeriv(self, iteration):
        solutions = self.data.solutionData[iteration].astype('float64')
        predictions = self.layer[self.layers-1].sigmoid()
        #dJdzAbove = (solutions / predictions) * self.layer[self.layers-1].sigmoidDeriv()
        dJdzAbove = -1 * (solutions - predictions) * self.layer[self.layers-1].sigmoidDeriv()
        delta = [dJdzAbove]
        for count in range(self.layers-2, -1, -1):
            weightsT = self.layer[count+1].weights.transpose()
            dadz = self.layer[count].sigmoidDeriv()
            dJda = np.dot(dJdzAbove, weightsT)
            dJdzAbove = dJda * dadz
            delta.append(dJdzAbove)
        #produces deltas in reverse
        delta = delta[::-1]
        """
        print delta[0].shape, delta[1].shape, delta[2].shape
        """
        #multiply deltas by respective inputs to produce deltas
        derivativeW = []
        derivativeB = []
        for count in range(len(delta)):
            inputT = self.layer[count].input.astype('float64').transpose()
            der = np.dot(inputT, delta[count])
            derivativeW.append(der)
            derivativeB.append(delta[count])
        """
        print derivativeW[0].shape, derivativeW[1].shape, derivativeW[2].shape
        print derivativeB[0].shape, derivativeB[1].shape, derivativeB[2].shape
        """
        return (derivativeW, derivativeB)
    
    #perform gradient on givven input iteration
    def Optimize(self, iteration):
        derivatives = self.NLLDeriv(iteration)
        for count in range(self.layers):
            self.layer[count].weights -= derivatives[0][count]
            self.layer[count].biases -= derivatives[1][count]
        return

    class neuronLayer:
        def __init__(self, wRows, wColumns, batchSize):
            self.input = np.array([])
            np.random.seed()
            self.weights = np.random.random((wRows, wColumns))
            self.biases = np.random.random((batchSize, wColumns))

        def z(self):
            return np.dot(self.input.astype('float64'), self.weights) + self.biases

        def sigmoid(self):
            score = self.z()
            return  1/(1+(np.exp(-1*score)))

        def sigmoidDeriv(self):
            return self.sigmoid() * (1-self.sigmoid())
        
        



def main():
    """
    SAMPLE NET CREATION:
    predictAi = net(2, 1, [21, 1], [1, 1])
    for index in range(1000):
        NLL = 0
        for iteration in range(predictAi.data.iterations):
            predictAi.updateInput(iteration)
            NLL += predictAi.NLLScore(iteration)
            predictAi.Optimize(iteration)
        print NLL / predictAi.data.iterations, "on index", index
    return
    """
    

if __name__ == "__main__":
    main()