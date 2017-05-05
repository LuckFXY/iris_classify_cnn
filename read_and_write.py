# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 17:09:52 2017

@author: rain
"""

import cPickle as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
def draw_tend(rd_iter,rd_error,rd_error2,pName):

    plt.plot(rd_iter,rd_error, color='blue', label='train(%)')
    plt.plot(rd_iter,rd_error2, color='darkorange', label='test(%)')
    plt.xlabel('Iters')
    plt.ylabel('Error')
    plt.title('Iters-Error-'+pName)
    plt.legend(loc="lower right")
    plt.show()
def get_tend_mean(dataframe):
    num,rd_iter,rd_error,rd_error2=dataframe[0]
    rd_iter=np.array(rd_iter)
    rd_error=np.array(rd_error)
    rd_error2=np.array(rd_error2)
    length=len(dataframe)
    for tend in dataframe[1:]:
        num,rd_iter,rd_error,rd_error2=tend
        rd_error+=np.array(rd_error)
        rd_error2+=np.array(rd_error2)
        #print(rd_iter,rd_error,rd_error2)
    rd_error=rd_error/length
    rd_error2=rd_error2/length
    #print(rd_iter,rd_error,rd_error2)
    #------------------------------------notice-------------20170429
    return (rd_iter[5:],rd_error[5:],rd_error2[5:])
def mywrite(data,filename):
    with open(filename,'a+b') as f:
        pickle.dump(data,f)
def myread(filename,showpic=False):
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

        
        #for i in range(len(df)/10):
        params=df['params'][0]
        error=np.mean(df['error'])
        loss=np.mean(df['loss'])
        if showpic:
            print(params)
            print(("error:%f loss:%f")%(error,loss))
            print(df['confusion_matrix'][0])
            rd_iter,rd_error,rd_error2=get_tend_mean(df['tend'])
            draw_tend(rd_iter,rd_error,rd_error2,str(params))  #--test!!!
            
            #mydraw(rd_iter,rd_error,rd_error2)
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
    [7,5,4]
    ]
    #[3,5,7],[3,3,4],[5,4,3],
    nkerns=[20,40,60]
    sizebatch_list=[40,45,55,70,85]#,
    learning_rate_list=[0.001,]#0.0004,0.0005,0.0007,
    n_epochs=65 #------------test!!!!!!!!!
    contents=[sizekerns_list,sizebatch_list,learning_rate_list]
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
        v.insert(1,nkerns)
        v.append(n_epochs)
        v=[[3, 3, 4], [20, 40, 60], 40, 0.001, 65]#test!
        filename='data/'+str(v)+'_result.pickle'
        if os.path.exists(filename)==False:
            print(filename," no exist")
            count+=1
        else:    
            r=myread(filename)
            
            if r[0]==None:
                print(filename," error")
            else:
                s=Series(r,index=columns)
                df=df.append(s,ignore_index=True)
        
        #---------------------------------------
        myiter.inc()
        break#test!

    print("count=%d"%(count))
    result=None
    if (df.size!=0):
        result=df.sort_values(by=['error','loss'])
        print('top 5')
        for i in range(1):#teset!
            v=result['params'][i]
            filename='data/'+str(v)+'_result.pickle'
            myread(filename,True)
        print(result)
        result.to_csv("sorted_result.csv")

