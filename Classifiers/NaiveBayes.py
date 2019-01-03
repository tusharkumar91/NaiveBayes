"""
Program for showing the implementation 
of Naive Bayes algorithm for Spam classification.
"""
from nltk.stem.snowball import SnowballStemmer
import nltk
import csv
from nltk.tokenize import word_tokenize
import random
import math

"""
This class holds the data for each label present in the data set.
It encapsulates the words and their relative frequency  and probability 
for each label
"""
class CorpusForLabel:
    def __init__(self, label, corpusStatements, stemWords = True, language = "english", encoding = "utf-8"):
        #label for which this class is instantiated (ham/spam)
        self.label = label
        self.numberOfStatements = len(corpusStatements)
        #wordsAndFreq is a dictionary of words in text corresponding to label and their corresponding counts
        self.wordsAndFreq = self.extractWordsAndFreq(corpusStatements, stemWords, language, encoding)
        #corpusCount holds the total number of words in corpus of this label
        self.corpusCount = self.getWordCount()
        self.stemWords = stemWords
        self.language = language
        self.encoding = encoding
        
    """
    Function to extract words and their frequencies in the given set of statements
    """
    def extractWordsAndFreq(self, corpusStatements, stemWords, language, encoding):
        wordsAndFreqDict = {}
        #Iterate over all data in the corpusStatements
        for line in corpusStatements:
            line = unicode(line, encoding)
            #Iterate over each sentence in current data row
            for sentence in nltk.sent_tokenize(line):
                #Iterate over the words
                for word in word_tokenize(sentence):
                    #Use the stemmed version of the word if said so
                    if stemWords:
                        stemmer = SnowballStemmer(language)
                        word = stemmer.stem(word)
                        #Fetch the frequency only of word is alpha numeric
                        if word.isalnum():
                            if word in wordsAndFreqDict.keys():
                                wordsAndFreqDict[word.lower()] = wordsAndFreqDict[word.lower()] + 1
                            else:
                                wordsAndFreqDict[word.lower()] = 1
        return wordsAndFreqDict
    
    """
    Function to return the total number of words
    (taking each occurrence of same word as different) in this corpus
    """    
    def getWordCount(self):
        count = 0
        for wordCount in self.wordsAndFreq.values():
            count = count + wordCount
        return count

     
"""
Function to get the data from CSV file. 
Assumption made here is that csv file has a header row and 
the dataSet generated should be a list of dictionaries where 
each row is represented by a dictionary 
"""
def getDataFromCSV(fileName):
    dataSet = []
    with open(fileName) as file_obj:
        reader = csv.DictReader(file_obj, delimiter=',')
        ignoredHeader = 0
        for line in reader:
            #We don't really need the header of the file
            if ignoredHeader == 0:
                pass
            else:
                dataSet.append(line)
            ignoredHeader = 1
    return dataSet

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
Given a dataset , this function creates a CorpusForLabel object 
for each of the classLabels present in the data
"""
def createCorpusForData(data, classLabelName, textLabelName):
    corpusLabelDict = {}
    rowsForLabel = {}
    for row in data:
        if row[classLabelName] in rowsForLabel:
            rowsForLabel[row[classLabelName]].append(row[textLabelName])
        else:
            rowsForLabel[row[classLabelName]] = [row[textLabelName]]
    for label in rowsForLabel:
        corpusLabelDict[label] = CorpusForLabel(label, rowsForLabel[label])
    return corpusLabelDict

"""
Function to return the probability of the word in the corpus and smoothing 
in cases when word is unseen
"""
def getProbabilityForLabel(word, corpus):
    if word in corpus.wordsAndFreq.keys():
        wordCount = corpus.wordsAndFreq[word]
    else:
        wordCount = 0
    return((1 + wordCount)/float(1 + corpus.corpusCount)) 

"""
Function to predict the class label probability by 
using conditional independence.
The probability can drop to very low numbers and fall below the limit of float value in python, 
hence we move to using log probabilities.
"""
def predictClassLabel(data, textLabel, corpusForLabelDict):
    predictedClassLabelProbDict = {}
    index = 0
    #Repeat the steps for both labels spam and ham
    for label in corpusForLabelDict.keys():
        corpusForLabel = corpusForLabelDict[label]
        predictedClassLabelProbDict[label] = []
        index = 0
        #Iterate over data for current label
        for row in data:
            line = unicode(row[textLabel], corpusForLabel.encoding)
            probClassLabel = 0
            for sentence in nltk.sent_tokenize(line):
                for word in word_tokenize(sentence):
                    if corpusForLabel.stemWords:
                        stemmer = SnowballStemmer(corpusForLabel.language)
                        word = stemmer.stem(word)                
                    if word.isalnum():
                        probClassLabel = probClassLabel + math.log(getProbabilityForLabel(word.lower(), corpusForLabel))            
            predictedClassLabelProbDict[label].append(probClassLabel)
            index = index + 1
    return predictedClassLabelProbDict

"""
Function to return the error rate for the classifier and also
return the number of false positives and false negatives
"""
def getAccuracy(labelName, testData, predictedLabels, spamLabel, hamLabel):
    errorCount = 0
    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0
    index = 0
    for row in testData:            
        if row[labelName] == spamLabel and predictedLabels[index] == spamLabel:
            truePositives = truePositives + 1
        if row[labelName] == hamLabel and predictedLabels[index] == hamLabel:
            trueNegatives = trueNegatives + 1
        if row[labelName] == hamLabel and predictedLabels[index] == spamLabel:
            errorCount = errorCount + 1
            falsePositives = falsePositives + 1
        if row[labelName] == spamLabel and predictedLabels[index] == hamLabel:
            errorCount = errorCount + 1
            falseNegatives = falseNegatives + 1
        index = index + 1
        
    confusionDict = {"FP" : falsePositives, "FN" : falseNegatives, "TP" : truePositives, "TN" : trueNegatives, "accuracy" : ((1 - float(errorCount)/len(testData)) * 100)}
    return confusionDict
 
"""
Function to get the predictions for spam by using the P(w|spam) and P(spam) 
"""
def getPredictedLabels(predictedClassLabelProb, spamLabelName, hamLabelName, probOfSpam, probOfHam):
    predictedLabels = []
    for i in range(len(predictedClassLabelProb[spamLabelName])):
        probForSpam = predictedClassLabelProb[spamLabelName][i] + math.log(probOfSpam) 
        probForHam = predictedClassLabelProb[hamLabelName][i] + math.log(probOfHam)
        if probForSpam > probForHam:
            labelForIndex = spamLabelName
        else:
            labelForIndex = hamLabelName
        predictedLabels.append(labelForIndex)
    return predictedLabels

if __name__ == "__main__":
    classLabelName = "type"
    textLabelName = "text"
    spamLabelName = "spam"
    hamLabelName = "ham"
    
    #1. Get the data from the CSV file
    dataSet = getDataFromCSV("../data/sms_spam.csv")
    
    #2. Partition dataSet into training and test dataSet
    trainData, testData = partitionTrainAndTest(dataSet)
    
    #3.Create corpus from trainData for each label 
    corpusLabelDict = createCorpusForData(trainData, classLabelName, textLabelName)

    #3.Get LIkelihood for spam and ham , P(w|Spam) and P(w|Ham) for each row in test data
    predictedClassLabelProb = predictClassLabel(testData, textLabelName, corpusLabelDict)
    
    #4 Get Priors for spam and ham
    probOfSpam = corpusLabelDict[spamLabelName].numberOfStatements
    probOfHam = corpusLabelDict[hamLabelName].numberOfStatements
    
    #5. Get the probability that the row is spam/ham. We are just interested in class so we do not normalize
    predictedLabels = getPredictedLabels(predictedClassLabelProb, spamLabelName, hamLabelName, probOfSpam, probOfHam)
    
    #6. Get the accuracy data 
    confusionDict = getAccuracy(classLabelName, testData, predictedLabels, spamLabelName, hamLabelName)
    
    print("Actual Spam number : " + str(confusionDict["TP"] + confusionDict["FN"]))
    print("Predicted Spam number : " + str(confusionDict["TP"] + confusionDict["FP"]))
    
    print("Actual Ham number : " + str(confusionDict["TN"] + confusionDict["FP"]))
    print("Predicted Ham number : " + str(confusionDict["TN"] + confusionDict["FN"]))
    
    print("Correctly Predicted Spam number : " + str(confusionDict["TP"]))
    print("Correctly Predicted Ham number : " + str(confusionDict["TN"]))
    
    print("Incorrectly Predicted Spam number : " + str(confusionDict["FP"]))
    print("Incorrectly Predicted Ham number : " + str(confusionDict["FN"]))
    
    print("Accuracy of classifier [% of errors] = " + str(confusionDict["accuracy"]))
 
    # ================ Accuracy for 5 iterations==================== #
    #            Iteration        |            Accuracy              #
    #                1            |             96.04                #
    #                2            |             95.5                 #
    #                3            |             96.76                #
    #                4            |             96.22                #
    #                5            |             95.86                #
    
