# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:35:06 2017

@author: rain
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import confusion_matrix
DATESZIE=200

#Generate random 3d data sets
import os
#print(os.getcwd())
#os.chdir("D:\GitHub\python")


def printdata(data):
    if len(data[0])==4:
        print('\tLabel\tc0\tc1\tc2')
        for i in range(len(data)):
            
            print(
                    ("\t%d\t%.2f\t%.2f\t%.2f")
                    %(int(data[i][0]),data[i][1],data[i][2],data[i][3])
            )
    elif len(data[0])==3:
        print('\tc0\tc1\tc2')
        for i in range(len(data)):
            
            print(
                    ("\t%.2f\t%.2f\t%.2f")
                    %(data[i][0],data[i][1],data[i][2])
            )
               
#print (data)

def getpredtags(prob_y,index_prob_real_y,TD=0):
    try_prob_y=prob_y.copy()
    for i in range(len(prob_y)):
        try_prob_y[i,index_prob_real_y[i]]-=TD
    #printdata(try_prob_y) #-------------test!!!!!!!!!!!
    return np.argmax(try_prob_y,axis=1) 


def __calculate_right_rate(cmatrix,index=2):
    Other=[[1,2],[0,2],[0,1]]
    s=np.sum(cmatrix[index,:])
    if s!=0:
        TPR=cmatrix[index,index]*1.0 /s
    else:
        TPR=0
    j=Other[index][0]
    s=np.sum(cmatrix[j,:])
    if s!=0:
        FPR1=(cmatrix[j,index])*1.0/s
    else :
        FPR1=0
    j=Other[index][1]
    s=np.sum(cmatrix[j,:])
    if s!=0:
        FPR2=(cmatrix[j,index])*1.0/s
    else :
        FPR2=0
    #print(index,FPR1,FPR2,TPR)
    return (FPR1,FPR2,TPR)
def calculate_rocxy(prob_y,real_y,index_prob_real_y,TD,MainClass):
    #notice index_prob_real_y is a list
    
    pred_y=getpredtags(prob_y,index_prob_real_y,TD)
    #print(pred_y)
    cmatrix=confusion_matrix(real_y,pred_y) 
    #print(cmatrix)
    FPR1,FPR2,TPR=__calculate_right_rate(cmatrix,MainClass)
    
    return (FPR1,FPR2,TPR)


from mpl_toolkits.mplot3d import axes3d
def mydraw(x,y,z):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_wireframe(x,y,z, rstride=5, cstride=5)
    
    plt.show()

'''
c1=np.random.rand(DATESZIE)
c2=np.random.rand(DATESZIE)*(1-c1)
c3=1-c1-c1
tags=np.random.randint(0,3,size=DATESZIE,dtype=int)
data=np.array([tags,c1,c2,(1-c1-c2)]).T
#data=data[np.argsort(data[:,-1])[::-1]]
with open("draw_row_test2_data.pickle","wb") as f:
    pickle.dump(data,f)
'''

filename='data/[[3, 5, 7], [20, 40, 60], 40, 0.0008, 10]_roc_data.pickle'
#filename="draw_row_test2_data.pickle"

def roc3d(filename):
    with open(filename,"rb") as f:
        data=pickle.load(f)
    
    #data=data[:5]#------------test!!!!!!!!!!!
    #printdata(data)
    
    temp_pred_y=np.argmax(data[:,1:],axis=1)+1 #data[0] is real y
    
    temp_prob_y=data[range(len(data)),temp_pred_y]
    data=data[np.argsort(temp_prob_y)[::-1]]
    
    real_y=data[:,0]
    prob_y=data[:,1:]
    #printdata(data)
    index_prob_real_y=np.argmax(data[:,1:],axis=1)
    
    threshold_list=prob_y[range(len(prob_y)),index_prob_real_y]
    z=np.zeros(1)
    threshold_list=np.concatenate((threshold_list,z))
    
    
    #printdata(data)
    #print(threshold_list)
    
    x=np.zeros(len(threshold_list))
    y=np.zeros(len(threshold_list))
    z=np.zeros(len(threshold_list))
    from math import sqrt
    def AUC(x,y,z):
        length=0
        area=0
        last_x=0
        last_y=0
        last_z=0
    
        for i in range(len(x)):
            dx=x[i]-last_x
            last_x=x[i]
            dy=y[i]-last_y
            last_y=y[i]
            #print(dx,dy)
            ds=sqrt(dx**2+dy**2)
            length+=ds           
            area+=ds*(z[i]+last_z)/2
            last_z=z[i]
        return area*1.0/length,length
            
    for mainclass in [2]:
        print("class %d is mainclass")%(mainclass)
        for i in range(len(threshold_list)):
            '''
            data has been sorted according to the probability of the predicted values in each sample
            prob_y is a matrix of the probability of output
            real_y is a list of a real class/y of each sample
            index_prob_real_y is a list of the index of the probiblity of the predicted vlaues in prob_y
            threshold_list is a list of thresholds arranged accroding to the predicted probability
            '''
            tx,ty,tz=calculate_rocxy(prob_y,real_y,index_prob_real_y,threshold_list[i],mainclass)#(FPR,TPR)
            #print(X[i],Y[i],Z[i])  
            x[i]+=tx
            y[i]+=ty
            z[i]+=tz
        mydraw(x,y,z)
        score,length=AUC(x,y,z)
        print("score = %.3f length = %.3f")%(score,length)
        
from read_and_write import myIteration
import pandas as pd

if __name__=='__main__':
    sizekerns_list=[
    [3,5,7],[3,3,4],[5,4,3],[7,5,4]
    ]
    nkerns=[20,40,60]
    sizebatch_list=[40,45,55,70,85]
    learning_rate_list=[0.0004,0.0005,0.0007,0.001]
    n_epochs=65 #------------test!!!!!!!!!
    contents=[sizekerns_list,sizebatch_list,learning_rate_list]
    myiter=myIteration(contents)
    
    print '....................program begining .............................'
    count=0
    result=pd.read_csv('sorted_result.csv')
    print(result.shape)
    for params in result['params'][:20]:
        filename='data_roc/'+str(params)+'_roc_data.pickle'
        if os.path.exists(filename)==False:
            print(filename," no exist")

        else:    
            print('-----------------------------------')
            print(params)
            roc3d(filename)
            count+=1
        
        #---------------------------------------
        myiter.inc()
    print("count=%d"%(count))

    
