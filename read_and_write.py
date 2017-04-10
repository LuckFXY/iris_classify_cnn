# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 17:09:52 2017

@author: rain
"""

import cPickle as pickle
import numpy as np
import os
import pylab as pl
from pandas import Series,DataFrame
def mydraw(rd_iter,rd_error,rd_error2):
    pl.plot(rd_iter,rd_error,label='train')
    pl.plot(rd_iter,rd_error2,label='valid')
    pl.title('iter-error')
    pl.xlabel('iter')
    pl.ylabel('error')
    pl.legend(loc='upper right')
    #pl.xlim(0,100)
    #pl.ylim(0,0.8)
    pl.show() 
    
def mywrite(data,filename):
    with open(filename,'a+b') as f:
        pickle.dump(data,f)
def myread(filename):
    columns=('params','error', 'loss', 'confusion_matrix','tend')
    df=DataFrame(columns=columns)
    with open(filename,'rb') as f_handler:
        f_handler.seek(0,os.SEEK_END)
        f_EOR=f_handler.tell()
        f_handler.seek(0,os.SEEK_SET)
        while(f_handler.tell()!= f_EOR):
            r=pickle.load(f_handler)
            s=Series(r,index=columns)
            df=df.append(s,ignore_index=True)
        #----------------------------------------
        for i in range(len(df)/10):
            params=df['params'][i]
            error=np.mean(df['error'][i:(i+1)*10])
            loss=np.mean(df['loss'][i:(i+1)*10])
            print("params ",params)
            print(("error:%f loss:%f")%(error,loss))
        #print("confusiong matrix")
        #print(confusion_matrix)
        #mydraw(rd_iter,error1,error2)
            
                
'''        
data=[[40,[5,4,3]],np.ones(4),np.zeros((2,2))]
mywrite(data,'test.pickle')
myread('test.pickle')
from pandas import Series,DataFrame
columns=('batch', 'nkerns', 'tend')
df=DataFrame(columns=columns)
column=('batch', 'nkerns', 'tend')
s=Series(data,index=column)
df.append(s,ignore_index=True)
'''
filename='[1, (0.0005,), 40, [20, 40, 60], [5, 4, 3]]_result.pickle'
myread(filename)

    

