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
    
    