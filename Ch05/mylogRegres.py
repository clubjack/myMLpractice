# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:20:31 2016

@author: chen.jiankai
"""

from pylab import *


def loadDataSet():
    dataMat = [];labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1/(1+exp(-inX))
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights
def plotBestFit(wei):
    weights = wei.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1,ycord1,xcord2,ycord2=[],[],[],[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    xlabel('X1')
    ylabel('X2')
    show()