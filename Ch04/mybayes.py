# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:44:44 2016

@author: chen.jiankai

朴素贝叶斯分类器
预知识：
P(c|x)=P(x|c)*P(c)/P(x)
分类器作用
无法实现精确分类的情况下，返回概率最大的分类结果
此处使用分类器对文本进行分类
"""
from pylab import *
import random


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # ‘|’用于set时，取并集
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    #trainMatrix和trainCategory类型是array
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/numTrainDocs
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2
    p1Denom = 2
    #以下为计算训练矩阵中每个单词的条件概率P(Wi|Ci)
    #即文档中每个单词在对应文档类别中出现的概率
    #计算方式为遍历所有文档单词，如果属于某类别，
    #则该单词量加1，该类别单词量加1
    #最后每个单体的数量除以该类别总单词量得到每个单词的条件概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #vec2Classify类型为array
    p0 = sum(vec2Classify*p0Vec)+log(1-pClass1)
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    return 1 if p1>p0 else 0
def testingNB():
    listOfPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    
#以下，使用分类器对email目录下的邮件进行分类
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
def spamTest():
    docList=[];classList = [];fullText=[]
    #以下，读取email目录下spam和ham两个子目录里的文本文件
    #并解析成字符串列表
    #每篇文档的字符串列表保存在docList中，对应的分类保存在classList中
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
#        print(i)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #生成完整字符串表
    vocabList = createVocabList(docList)
    #50表示两个子目录中共有50个文件
    trainingSet = list(range(50)); testSet = []
    #选取10个作为测试集，剩下的作为训练集
    for i in range(10):
        randIndex = random.randint(0,len(trainingSet)-1)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #得到训练结果
    p0V,p1V,pAb = trainNB0(array(trainMat),array(trainClasses))
    
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWords2Vec(vocabList,docList[docIndex])
        result = classifyNB(array(wordVec),p0V,p1V,pAb)
        if result != classList[docIndex]:
            errorCount+=1
            print('error classify:',docList[docIndex],docIndex)
    print('the error rate is: ',errorCount/len(testSet))
    
        
