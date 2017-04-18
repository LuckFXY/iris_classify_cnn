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
    error,loss,params=(None,None,None)
    with open(filename,'rb') as f_handler:
        f_handler.seek(0,os.SEEK_END)
        f_EOR=f_handler.tell()
        f_handler.seek(0,os.SEEK_SET)
        while(f_handler.tell()!= f_EOR):
            r=pickle.load(f_handler)
            s=Series(r,index=columns)
            df=df.append(s,ignore_index=True)
        #---------------------20170418-------------------
        #every file only have 10 itmes so len(df) == 1 
        for i in range(len(df)/10):
            params=df['params'][i]
            error=np.mean(df['error'][i:(i+1)*10])
            loss=np.mean(df['loss'][i:(i+1)*10])
            #print("params ",params)
            #print(("error:%f loss:%f")%(error,loss))
    
    return (error,loss,params)
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


filename='[1, (0.0005,), 40, [20, 40, 60], [5, 4, 3]]_result.pickle'


if __name__=='__main__':
    sizekerns_list=[
    [3,5,7],[3,3,4],[5,4,3],[7,5,4],[9,6,5]
    ]
    nkerns=[20,40,60]
    sizebatch_list=[40,50,60,70]
    learning_rate_list=[0.0008,0.0007,0.0006,0.0005,0.0004]
    n_epochs_list=[65]
    contents=[sizekerns_list,sizebatch_list,learning_rate_list,n_epochs_list]
    myiter=myIteration(contents)
    
    print '....................program begining .............................'
    count=0
    columns=('error', 'loss','params')
    df=DataFrame(columns=columns)
    while (myiter.isEmpty()==False):
        v=myiter.getvalue()
        sizekerns=v[0]
        sizebatch=v[1]       
        learning_rate=v[2]
        n_epochs=v[3]
        #------------------20170410-------------
        #这是保存的顺序，与v不一致。当时写错了
        vv=[n_epochs,learning_rate,sizebatch,nkerns,sizekerns]
        #---------------------------------------
        filename='data\\'+str(vv)+'_result.pickle'
        if os.path.exists(filename)==False:
            print(str(vv)," no exist")
            count+=1
        '''
        else:    
            r=myread(filename)
            if r[0]==None:
                print(filename," error")
            else:
                s=Series(r,index=columns)
                df=df.append(s,ignore_index=True)
        '''
        #---------------------------------------
        myiter.inc()
    print("count=%d"%(count))
    result=None
    if (df.size!=0):
        result=df.sort_index(by=['error','loss'])
        print(result)
        result.to_csv("sorted_result.csv")
