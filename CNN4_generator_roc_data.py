# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:34:00 2016

@author: Administrator
"""
# encoding=utf-8
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


import cPickle as pickle
import pandas as pd
#import pylab as pl
import os
import time

import numpy as np

import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


TYPENUMBER=3
srng=RandomStreams()
#theano.config.compute_test_value='off'

def linear_rectified(x):
    y=T.maximum(T.cast(0.,theano.config.floatX),x)
    return (y)
        
def dropout(X,prob=0.1):
    #return X
    mask=srng.binomial(X.shape,p=1-prob,dtype=theano.config.floatX)
    return X*T.cast(mask, theano.config.floatX)
#==================Convoluation & pooling layer====================================
class LeNetConvPoolLayer(object):
    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2),W=None,b=None):
        #image_shape[1]和filter_shape[1]都是num input feature maps它们必须一样
        assert image_shape[1]==filter_shape[1] #输入特征图的个数
        self.input=input
        
        #每个隐层神经元（像素）与上一层的连接数为（卷积输入）
        #prod返回各元素的乘积
        fan_in=np.prod(filter_shape[1:])
        
        #lower layer上每个神经元获得的梯度来自（卷积输出）
        #num output feature maps* filter height * filter width/pooling size
        fan_out=(filter_shape[0]*np.prod(filter_shape[2:])/
                np.prod(poolsize))
        #以上求得fan_in,fan_out,将他们呢带入公式，从此来随机初始化W，W就是线性卷积核
    
        W_bound=np.sqrt(6./(fan_in+fan_out))
        if W is None:
            W=np.asarray(
                rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                dtype=theano.config.floatX
            )
        self.W=theano.shared(value=W,borrow=True)
        
        #偏置是一维向量，每一个特征图对应一个偏置
        #输出的特征图的个数由filter个数决定，因此用filter_shape[0]即number of filters初始化
        if b is None:
            b=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=b,borrow=True)
        
        #将输入图像与filter卷积，conv.conv2d
        #卷积完没有加b再通过sigmoid,这里简化了
        conv_out=conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        #maxpooling,最大子采样过程
        '''
        pooled_out=downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        '''
        pooled_out=pool.pool_2d(
            input=conv_out,
            mode='max',
            ws=poolsize,
            ignore_border=True
        )
        #加偏置，再通过tanh映射，得到卷积+子采样层的最终输出
        #因为b是一位向量，这里用维度转化函数dimshuffle将其reshape.比如b是（10,),
        #则b.dimshuffle('x',0,'x','x')将其变为(1,10,1,1)
        self.output=linear_rectified(pooled_out+self.b.dimshuffle('x',0,'x','x'))
        #卷积+采样层的参数
        self.params=[self.W,self.b]
 


#=============================DropoutHiddenLayer============================

'''    
def rescale_weights(params, incoming_max):
    incoming_max = np.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * np.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)   
'''        
class DropoutHiddenLayer(object):

    def __init__(self,rng,is_train,input,n_in,n_out,W=None,b=None,\
                activation=linear_rectified,p=0.5):

        self.input=input #类HiddenLayer的input即所传递进来的input

        if W is None:
            W = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6./(n_in+n_out)),
                    high=np.sqrt(6./(n_in+n_out)),
                    size=(n_in,n_out)
                ),
                dtype=theano.config.floatX
            )
        #if activation==theano.tensor.nnet.sigmoid:
        #W*=4
                
        W=theano.shared(value=W,name='W',borrow=True)
            
        if b is None:
            b=np.zeros((n_out,),dtype=theano.config.floatX)
            
        b=theano.shared(value=b,name='b',borrow=True)
#隐层内W，b 
        self.W=W
        self.b=b
        self.n_out=n_out
#隐层的输出
        lin_output=T.dot(input,self.W)+self.b
        
        output=activation(lin_output)
        
        train_output=dropout(T.cast((1./(1-p)), theano.config.floatX) * output)
        
        self.output=T.switch(
            T.neq(is_train,0),train_output,output
        )
#隐层的参数
        self.params=[self.W,self.b]
        
#===========================LogisticRegression=========================

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out,W=None,b=None):

        if W is None:
            W=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(value=W,name='W',borrow=True)
        
        if b is None:
            b=np.zeros(  
                (n_out,),  
                dtype=theano.config.floatX  
            )
        self.b = theano.shared(value=b,  name='b',  borrow=True ) 
        
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)
        
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
#params,模型的参数
        self.params=[self.W,self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
            
    def errors(self,y):
        #首先检查y与y_pred的维度是否一样，即是否包含相同的样本数
        if y.ndim!=self.y_pred.ndim:
            raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y',y.type,'y_pred',self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            
            return T.mean(T.neq(self.y_pred,y)) #妙！
        else:
            return NotImplementedError()         
    #def getPred(self):
    #    return self.y_pred
    def getPred_p(self):
        return self.p_y_given_x

        
         
#=======================loading dataset=======================
               
MAX_NUM_FILE=600

#================global verities=====================
_ReadImg_filelist=[]
_ReadImg_filenames=[]
_ReadImg_filenum=[0,]
#====================================================

def load_params(filename):
    print ('loading params')
    exist_params=False
    if os.path.isfile(filename):
        read_file=open(filename,'rb')
        try:
            params=pickle.load(read_file)
            exist_params=True
            print (filename+' loaded!')
        finally:
                read_file.close()   
    else:
        
        params=[]
        for i in range(12):
            params.append(None)
        print ('-.-cannont find '+filename)  
    return [exist_params,params]
def inc_i(clear=False,c=[-1,]):
    if(clear==True):
        c[0]=-1
        return -1
    c[0]=c[0]+1
    return c[0]    
def load_train_data(train_data,skip_training,N,time):
    
    if time==0:
        inc_i(clear=True)
        
    def shared_dataset(data_x2y,borrow=True):
        data_x,data_x2,data_y=data_x2y
        shared_x=theano.shared(\
        np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_x2=theano.shared(\
        np.asarray(data_x2,dtype=theano.config.floatX),borrow=borrow)
        shared_y=theano.shared(\
        np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x,shared_x2,T.cast(shared_y,'int32')
        
    if skip_training:
        valid_set_x,valid_set_x2,valid_set_y=shared_dataset(([np.zeros([2,2]),np.zeros([2,2]),[0,0]]))
        train_set_x,train_set_x2,train_set_y=shared_dataset(([np.zeros([2,2]),np.zeros([2,2]),[0,0]]))
        rval=[(train_set_x,train_set_x2,train_set_y),(valid_set_x,valid_set_x2,valid_set_y)]
        return rval 
    	   
    #print ('create cross_validation set')
    '''
    train_set_x,valid_set_x,train_set_y,valid_set_y=\
        cross_validation.train_test_split(\
        train_data[:,1:],train_data[:,0],\
        test_size=TZ,random_state=2016)
    '''
    N=min(N,len(train_data))
    step=len(train_data)/N
    
    starti=int(inc_i()*step)
    endi=starti+step
    valid_set_x=train_data[starti:endi,1:]
    valid_set_y=train_data[starti:endi,0]
    train_set_x=np.concatenate((train_data[:starti,1:],train_data[endi:,1:]),axis=0)
    train_set_y=np.concatenate((train_data[:starti,0],train_data[endi:,0]),axis=0)
#将数据设置成shared variables，主要是为了GPU加速，只有shared variables
#才能存到GPU memory中，只能为float类型，data_y是类别，所以最后又转换成int
    temp=valid_set_x.T
    valid_set_x2=temp[:255].T
    valid_set_x=temp[255:].T
    
    temp=train_set_x.T
    train_set_x2=temp[:255].T
    train_set_x=temp[255:].T
    
    valid_set_x,valid_set_x2,valid_set_y=shared_dataset((valid_set_x,valid_set_x2,valid_set_y))
    train_set_x,train_set_x2,train_set_y=shared_dataset((train_set_x,train_set_x2,train_set_y))

    
    rval=[(train_set_x,train_set_x2,train_set_y),(valid_set_x,valid_set_x2,valid_set_y)]
    return rval   

NUM_OF_PARAMS=12    #NOTICE!!!!!  (5+1)*2

def giveme_param(params,n=[NUM_OF_PARAMS,],init=False):
    if init:
        n[0]=NUM_OF_PARAMS
    n[0]=n[0]-1
    assert n!=-1
    return params[n[0]] 

def giveme_sizeMap(filter_size,image_size,pooling_size=2):
    image_size=(image_size-filter_size+1)/pooling_size
    return image_size

#IMAGE_SIZE=50 is defined in reader.py
'''
出问题了，内存泄漏了！！！！！
'''
def mywrite(data,filename):
    with open(filename,'a+b') as f:
        pickle.dump(data,f)
#===========================LeNet-5 demo================================
def evaluate_lenet(datasets,params,
                    n_epochs,learning_rate,sizebatch,
                    nkerns,sizekerns):#500
    '''
    n_epochs训练步数，每一次都会遍历所有batch，  即所有样本
    sizebatch这里设置500,每遍历500个样本，才计算梯度
    nkerns=[20,50],每一个LeNetConvPoolLayer卷积核的个数，第一层20个核，第二层50
    '''
    train_params=[
            sizekerns,nkerns,sizebatch,learning_rate,n_epochs
    ]
    #print(n_epochs,learning_rate,sizebatch,nkerns,sizekerns)

    train_set_x, train_set_x2,train_set_y = datasets[0]  
    valid_set_x, valid_set_x2,valid_set_y = datasets[1]  
    
    #calculate batch的个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]  
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]  

    sizebatch=min(n_train_batches,n_valid_batches,sizebatch)
    
    #print (' train_size=%d\n valid_size=%d\n sizebatch=%d')%\
    #    (n_train_batches,n_valid_batches,sizebatch)
      
    n_train_batches /= sizebatch  
    n_valid_batches /= sizebatch  #需要修改if data_size<sizebatch batch=size

    #print (' n_valid_batches=%d')%(n_valid_batches)
    
    #定义几个变量，index表示batch下标，x编制输入的训练数据，y是对应的标签
    index=T.lscalar()
    hist=T.matrix('hist')
    x=T.matrix('x')
    y=T.ivector('y')
    is_train=T.iscalar('is_train')

    rng=np.random.RandomState(20161024)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    #print ('...building the model')
    #我们加载数据是(sizebatch,220*220),但是LeNetConvPoolLayer输入是四维的
    
    CP0_input=x.reshape((sizebatch,1,50,50))
    '''
    layer0即第一个LeNetConvPoolLayer层
    输入的单张图片(220,220)经过conv得到200-21+1=200
    经过maxpooling得到200/2=100
    因为每一个batch都有sizebatch张图，第一个LeNetConvPoolLayer层有nkerns[0]卷积核
    故layer0输出为（sizebatch,nkerns[0],50,50)
    '''
    image_size=50
    CP0=LeNetConvPoolLayer(        
        input=CP0_input,
        rng=rng,
        image_shape=(sizebatch,1,image_size,image_size),
        filter_shape=(nkerns[0],1,sizekerns[0],sizekerns[0]),
        poolsize=(2,2),
        b=giveme_param(params,init=True),
        W=giveme_param(params)
    )
    '''
    layer1即第二层
    输入时layer0的输出，每张特征图为(100,100),经过conv得到（100-21+1，100-21+1）=（80，80）
    经过maxpooling得到(80/2,80/2)=(40,40)
    因为每个batch有sizebatch，第二层有nkerns[1]个卷积核
    故layer1输出为(sizebatch,nkerns[1],34,34)
    '''
    image_size=giveme_sizeMap(sizekerns[0],image_size)
    CP1=LeNetConvPoolLayer(
        rng,
        input=CP0.output,
        #输入nkerns[0]张特征图，即layer0输出
        image_shape=(sizebatch,nkerns[0],image_size,image_size),
        filter_shape=(nkerns[1],nkerns[0],sizekerns[1],sizekerns[1]),
        poolsize=(2,2),
        b=giveme_param(params),
        W=giveme_param(params)
        
    )
    '''
    layer1即第二层
    输入时layer0的输出，每张特征图为(40,40),经过conv得到40-21+1
    经过maxpooling得到20/4=5
    因为每个batch有sizebatch，第三层有nkerns[2]个卷积核
    故layer1输出为(sizebatch,nkerns[2],10,10)
    '''
    image_size=giveme_sizeMap(sizekerns[1],image_size)
    CP2=LeNetConvPoolLayer(
        rng,
        input=CP1.output,
        #输入nkerns[0]张特征图，即layer0输出
        image_shape=(sizebatch,nkerns[1],image_size,image_size),
        filter_shape=(nkerns[2],nkerns[1],sizekerns[2],sizekerns[2]),
        poolsize=(2,2),
        b=giveme_param(params),
        W=giveme_param(params)
        
    )
    
    '''
    定义好前二层layer0,layer1的卷积层，然后开始定义全连接层layer2
    用HiddenLayer老初始化layer2,layer2输入二维(sizebatch,num_pixels)
    故要将将上一层通过以图像全部（卷积层输出）特征图合并为一维向量
    将layer2的输出(sizebatch,nerkns[2],5,5)flatten为
    (sizebatch,nkerns[2]*6*6)=(120,1800)
    表示500个样本，每一行代表一个样本。layer3输出大小(sizebatch,n_out)=(500.500)
    '''
    DFC1_input=T.concatenate((CP2.output.flatten(2),hist),axis=1)
    #DFC1_input=CP2.output.flatten(2)
    image_size=giveme_sizeMap(sizekerns[2],image_size)
#========================================================20161024
    DFC1=DropoutHiddenLayer(
        rng=rng,
        is_train=is_train,
        input=DFC1_input,
        #n_in=nkerns[2]*4*4,
        n_in=nkerns[2]*image_size*image_size+255,
        n_out=1000,
        b=giveme_param(params),
        W=giveme_param(params)
    )
    DFC2 = DropoutHiddenLayer(
            rng=rng,
            is_train=is_train,
            input=DFC1.output,
            n_in=DFC1.n_out,
            n_out=200,
            b=giveme_param(params),
            W=giveme_param(params)
    )
    LR=LogisticRegression(
        input=DFC2.output,
        n_in=DFC2.n_out,
        n_out=TYPENUMBER,
        b=giveme_param(params),
        W=giveme_param(params)
    )

       
#========================================================20161024
  
    #代价函数NLL
   
    #params=LR.params+DFC2.params+DFC1.params+CP2.params+CP1.params+CP0.params

    #由于参数太多，在update里写很麻烦，所有用for in 

    #validate_model，验证模型，分析同上。  

    analyse_model=theano.function(  
        on_unused_input='ignore',
        inputs=[index],  
        outputs=[y,LR.getPred_p()],
        givens={  
            hist:valid_set_x2[index * sizebatch: (index + 1) * sizebatch],
            x: valid_set_x[index * sizebatch: (index + 1) * sizebatch],  
            y: valid_set_y[index * sizebatch: (index + 1) * sizebatch],                                    
            is_train: np.cast['int32'](0)
        } 
    )

 

    sum_y,sum_Pred_p=analyse_model(0)

    for minibatch_index in range(1,n_valid_batches):
        y,Pred_p=analyse_model(minibatch_index)
        sum_y=np.concatenate([sum_y,y])   
        sum_Pred_p=np.concatenate([sum_Pred_p,Pred_p])
   
    filename='data_roc/'+str(train_params)+'_roc_data.pickle'
    #print(sum_y)
    #print(sum_Pred_p)
    output=np.insert(sum_Pred_p,0,values=sum_y,axis=1)
    #print(output)
    #print(output.shape)
    mywrite(output,filename)

                

    #print ( 'ran for %.2fm ' %((end_time-start_time)/60.))

       

def LENET_pred(sizekerns,nkerns,sizebatch,learning_rate,n_epochs):
    #the filename of params is accroding to the rule as follow
    train_params=[
            sizekerns,nkerns,sizebatch,learning_rate,n_epochs
    ]      
    parameter_filename='data/'+str(train_params)+'_params.py'
    #I will give you trained parameters so I will skip the training process
    
    exist_params,params=load_params(parameter_filename)
    if exist_params == False:
        print("fail to load parameter_filename")
        return
    train_name='test_'+str(50)+'.csv'
       

    #print 'loading',train_name
    if os.path.exists(train_name)==False:
        raw_input('cannot find file,over!')
        return 
    train_csv=pd.read_csv(train_name)
    train_data=train_csv.values
    skip_training=False #just for load the dataset!! 20170429
    Num=10
    print ('date:'+time.strftime('%H:%M:%S'))
    for i in range(Num):#Num
        print i,
        datasets=load_train_data(train_data,skip_training,Num,i) #获取交叉验证的数据集 
        try:
            evaluate_lenet(
                    datasets,params,
    				n_epochs,learning_rate,sizebatch,nkerns,sizekerns
            )
        except NotImplementedError as e:
            print "kernel must have the same type,input"
            break
            
    

    print("predicting process is over")
    
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
    sizekerns_list=[
    [3,5,7],[3,3,4],[5,4,3],[7,5,4]
    ]
    nkerns=[20,40,60]
    sizebatch_list=[40,45,55,70,85]
    learning_rate_list=[0.0004,0.0005,0.0007,0.001]
    n_epochs=65 
    contents=[sizekerns_list,sizebatch_list,learning_rate_list]
    myiter=myIteration(contents)
    
    print ('....................program begining .............................')
    index=0
    while (myiter.isEmpty()==False):
        v=myiter.getvalue()
        sizekerns=v[0]
        sizebatch=v[1]       
        learning_rate=v[2]
        v.insert(1,nkerns)
        v.append(n_epochs)
        filename='data_roc/'+str(v)+'_roc_data.pickle'
        if os.path.exists(filename)==True:
            print(filename,"exist")          
        else:    
            LENET_pred(sizekerns,nkerns,sizebatch,learning_rate,n_epochs)   
            index+=1
            if index==5:
				break
        #---------------------------------------
        myiter.inc()
		
       
        
    print ('....................Program finished...........................')
    
    
