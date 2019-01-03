"""
Program for showing the implementation 
of Guassian Naive Bayes algorithm for Pima Indian Diabetes
problem.
https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
"""

import csv
import math
import random

"""
This class encapsulates all the data related to describe the guassian distribution
observed in the data set for each of the attributes.
We make an assumption that all attributes are of continuous type here
"""
class GuassianData:
    def __init__(self, dataSet):
        # Extract data for each label . Here we will also ensure that we 
        # have at first index in dataByLabel[label] the values for first attribute
        # and at second index in dataByLabel[label] the values for second attribute
        # and so on. This will be handy for mean and variance calculation
        self.dataByLabel = self.getDataByLabel(dataSet)
        # probData is the probability data for each labels each column(mean and variance)
        # which will give us the power to get the distribution of each attribute for each label
        self.probData = self.getProbData(self.dataByLabel)
        
    """
    Function to return the mean and variances for each attribute for each label
    """
    def getProbData(self, dataByLabel):
        probData = dict()
        for label in dataByLabel.keys():
            probDataForLabel = [(self.getMean(numbers), self.getVariance(numbers)) for numbers in dataByLabel[label]]
            probData[label] = probDataForLabel
        return probData
         
    """
    Function to return the data grouped by label and then grouped by attribute.
    0 : DataForLabel0[(col1val1, col1val2,col1val3 ....), (col2val1, col2val2, ....)]
    1 : DataForLabel1[(col1val1, col1val2,col1val3 ....), (col2val1, col2val2, ....)]
    """   
    def getDataByLabel(self, dataSet):
        dataSetByLabel = dict()
        for row in dataSet:
            rowLabel = row[len(row)-1]
            if rowLabel in dataSetByLabel.keys():
                dataSetByLabel[rowLabel].append(row[:-1])
            else:
                dataSetByLabel[rowLabel] = [row[:-1]]
        for label in dataSetByLabel.keys():
            dataSetByLabel[label] = zip(*dataSetByLabel[label])
            for i in range(len(dataSetByLabel[label])):
                dataSetByLabel[label][i] = [ float(elem) for elem in dataSetByLabel[label][i]]
        return dataSetByLabel
            
    def getMean(self, numbers):
        return sum(numbers)/float(len(numbers))
    
    def getVariance(self, numbers):
        mean = self.getMean(numbers)
        return math.sqrt(sum([pow(x-mean,2) for x in numbers])/float(len(numbers)-1))
        

"""
Function to calculate the probability of a number , given 
a mean and standard deviation
"""
def getProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
        
"""
Function to get the data from CSV file. 
Assumption made here is that csv file does NOT have a header row and 
the dataSet generated should be a list  
"""
def getDataFromCSV(fileName):
    dataSet = []
    with open(fileName) as file_obj:
        reader = csv.reader(file_obj, delimiter=',')
        for line in reader:
            dataSet.append(line)
    return dataSet

"""
Function to get the probabilities for each label for the input using
the guassian data.
"""
def getProbabilityForLabel(guassianData, inputData):
    probForLabelDict = dict()
    for label in guassianData.probData.keys():
        probDataForLabel = guassianData.probData[label]
        probForLabel = 1
        for colIndex in range(len(probDataForLabel)):
            mean , stdev = probDataForLabel[colIndex]
            probForLabel = probForLabel * getProbability(float(inputData[colIndex]), mean, stdev)
        probForLabelDict[label] = probForLabel
    return probForLabelDict

"""
Function to return the classLabel and the confidence in estimate of that class 
for a given input data
"""
def getClassLabelAndProbability(guassianData, inputData):
    probForLabelDict = getProbabilityForLabel(guassianData, inputData)
    normalizationFactor = sum(probForLabelDict.values())
    for label in probForLabelDict.keys():
        probForLabelDict[label] = (probForLabelDict[label])/normalizationFactor
    return max(probForLabelDict, key=probForLabelDict.get) , max(probForLabelDict.values())
    
"""
Function to partition the dataSet into train and testData.
testRatio -> ratio of test examples:dataSet size. Default = 0.2
shouldShuffle -> whether the data needs to be shuffled or not. Default = false
"""
def partitionTrainAndTest(dataSet, testRatio = 0.2, shouldShuffle = True):
    dataSize = len(dataSet)
    if shouldShuffle:
        random.shuffle(dataSet)
    trainDataSize = int(dataSize*(1-testRatio))
    trainData = dataSet[:trainDataSize]
    testData = dataSet[trainDataSize:]
    return (trainData,testData)

"""
Function to get the predictions according to guassian naive bayes 
for each input in test data
"""
def getPredictions(guassianData, testData):
    predictions = []
    for row in testData:
        label, confidence = getClassLabelAndProbability(guassianData, row)
        predictions.append(label)
    return predictions

"""
Function to get the accuracy by counting the errors in the predictions
"""
def getAccuracy(predictions, testData):
    #for row in testData:
    #    if predictions[testIndex] != row[len(row)-1]:
    #        errorCount = errorCount + 1
    #    testIndex = testIndex + 1
    #return (1 - errorCount/float(len(testData))) * 100

    errorCount = 0
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    index = 0
    for row in testData:
        rowLabel = row[len(row)-1]
        if predictions[index] == "1" and rowLabel == "0":
            errorCount = errorCount + 1
            falsePositives = falsePositives + 1
        if predictions[index] == "0" and rowLabel == "1":
            errorCount = errorCount + 1
            falseNegatives = falseNegatives + 1
        if predictions[index] == "0" and rowLabel == "0":
            trueNegatives = trueNegatives + 1
        if predictions[index] == "1" and rowLabel == "1":
            truePositives = truePositives + 1
        index = index + 1
    
    outputDict = {"FP" : falsePositives, "FN" : falseNegatives, "TP" : truePositives, "TN" : trueNegatives, "accuracy" : ((1 - float(errorCount)/len(testData)) * 100)}
    return outputDict

if __name__ == "__main__":
    #1. Get the data from the CSV file
    dataSet = getDataFromCSV("../data/pima-indians-diabetes.csv")
    
    #2. Partition dataSet into training and test dataSet
    trainData, testData = partitionTrainAndTest(dataSet)
    
    #3. Create a GuassianData class for the data set 
    guassianData = GuassianData(trainData)
    
    #4. Get the predictions for testData rows according to the guassian data 
    predictions = getPredictions(guassianData, testData)
    
    #6. Get the accuracy data  
    confusionDict = getAccuracy(predictions, testData)
    
    #Number of Actual Diabetic patients 
    print("Actual Diabetic persons : " + str(confusionDict["TP"] + confusionDict["FN"]))
    print("Predicted Diabetic persons : " + str(confusionDict["TP"] + confusionDict["FP"]))
    
    print("Actual Non-Diabetic persons : " + str(confusionDict["TN"] + confusionDict["FP"]))
    print("Predicted Non-Diabetic persons : " + str(confusionDict["TN"] + confusionDict["FN"]))
    
    print("Correctly Predicted Diabetic persons : " + str(confusionDict["TP"]))
    print("Correctly Predicted Non-Diabetic persons : " + str(confusionDict["TN"]))
    
    print("Incorrectly Predicted Diabetic persons : " + str(confusionDict["FP"]))
    print("Incorrectly Predicted Non-Diabetic persons : " + str(confusionDict["FN"]))
    
    print("Accuracy of classifier [% of errors] = " + str(confusionDict["accuracy"]))
    
