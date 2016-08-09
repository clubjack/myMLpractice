# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 21:20:11 2016

@author: Club_
"""

from pylab import *
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,
                            arrowprops=arrow_args)
def createPlot():
    fig = figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    show()