# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:24:04 2016

@author: Club_
构造决策树
基本思路是：根据特征，划分数据集，对每个划分出来的子集继续划分，
直到所有子集同属一类
为了更好的划分数据集，需要找出最适合用来划分的特征
最适合用来划分的特征由信息增益决定，信息增益越多的划分越好
信息增益即前后信息熵的差值
信息熵的计算根据信息论由calcShannonEnt给出
输入数据
-------------------------------------------------
labels    |label1|label2|label3|...|class
dataSet   |1     |1     |1     |...|是
          ...
          ...
          
"""

from math import log
import operator

def calcShannonEnt(dataSet):
    '''
    计算香农信息熵
    需要dataSet是list，元素是list形式的样本
    样本内容为特征加最后一个的分类
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0)+1
    shannonEnt = 0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt
    
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no'],]
    labels = ['no surfacing','flippers']
    return dataSet,labels
#myDat,labels = createDataSet()
def splitDataSet(dataSet, axis, value):
    retData = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedVec = featVec[:axis]+featVec[axis+1:]
            retData.append(reducedVec)
    return retData
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeature):
        featureList = [example[i] for example in dataSet]
        uniqueFeat = set(featureList)
        newEntropy = 0
        for feature in uniqueFeat:
            subDataSet = splitDataSet(dataSet,i,feature)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
    
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote,0)+1
        sortedClassCount = sorted(classCount.items(),
                                  key = operator.itemgetter(1),reverse = True)
        return sortedClassCount[0][0]
def createTree(dataSet,labels):
    #取出dataSet最后一列作为分类
    classList = [example[-1] for example in dataSet]
    #如果所有样本同属一类，返回分类
    if len(set(classList)) == 1:
        return classList[0]
    #如果所有特征都用完了，分类还没分完，返回最多的那类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeature])
    featVals = set(map(operator.itemgetter(bestFeature),dataSet))
    for value in featVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return myTree
    
#myTree = createTree(myDat,labels)