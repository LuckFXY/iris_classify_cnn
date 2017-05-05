# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:35:06 2017

@author: rain
"""

import numpy as np
import pandas as pd
import pickle
import pylab as pl

DATESZIE=200


#Generate random 3d data sets
import os
#print(os.getcwd())
#os.chdir("D:\GitHub\python")

def printdata(data):
    if len(data[0])==2:
        print('\tLabel\tTrue')
        for i in range(len(data)):
            
            print(
                    ("\t%d\t%.2f")
                    %(int(data[i][0]),data[i][1])
            )

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def getinfo(data):
    y=data[:,0]
    prob=data[:,1:]
    y_pred=np.argmax(prob,axis=1)

    cm=confusion_matrix(y,y_pred)
    print (cm)
    for i in range(3):
        TP=cm[i,i]
        pred_T=np.sum(cm[:,i])
        p=TP*1.0 /pred_T
        real_T=np.sum(cm[i,:])
        r=TP*1.0/real_T
        print('class %d 查准率 = %0.2f 查全率 = %0.2f'%
              (i,p,r)
              )
    return cm[2,2]*1.0/np.sum(cm[2,:])
        
def roc2d(filename):

    with open(filename,"rb") as f:
        data_3c=pickle.load(f)
   
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    r2=getinfo(data_3c)
    if r2<0.8: return
    for i in range(3):
        y=np.zeros(data_3c.shape[0])
        y[data_3c[:,0]==i]=1
        score=data_3c[:,i+1]
        #printdata(np.c_[y,score])
        fpr[i], tpr[i], _ = roc_curve(y,score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-'+str(i))
        plt.legend(loc="lower right")
        plt.show()


    
class myIteration():
    def __init__(self,contents):
        self.contents=contents
        self.LCLengths=[len(i) for i in contents]
        self.Length=len(contents)
        self.LC_indexs=[0]*self.Length
    
    def isEmpty(self):
        if self.LC_indexs[0]>=self.LCLengths[0]:
            return True
        return False
    def inc(self):
        if self.isEmpty()==True:
            return False
        i=self.Length-1
        self.LC_indexs[i]+=1
        while self.LC_indexs[i]==self.LCLengths[i]:
            if(i-1!=-1):
                self.LC_indexs[i-1]+=1
                self.LC_indexs[i]=0
                i-=1
            else:
                return False
            
        return True
    
    def getvalue(self):
        ret=[]
        for i in range(self.Length):
            j=self.LC_indexs[i]
            ret.append(self.contents[i][j])
        return ret

if __name__=='__main__':

    
    print '....................program begining .............................'
    count=0
    result=pd.read_csv('sorted_result.csv')
    print(result.shape)
    for params in result['params']:#test!
        filename='data_roc/'+str(params)+'_roc_data.pickle'
        if os.path.exists(filename)==False:
            print(filename," no exist")

        else:    
            print('-----------------------------------')
            print(params)
            roc2d(filename)
            count+=1
        
        #---------------------------------------
    print("count=%d"%(count))

    
