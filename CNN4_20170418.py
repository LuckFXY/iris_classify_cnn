# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:34:00 2016

@author: Administrator
"""
# encoding=utf-8
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


import cPickle as pickle
import pandas as pd
import pylab as pl
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
from sklearn.metrics import confusion_matrix
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
    def getPred(self):
        return self.y_pred

        
         
#=======================loading dataset=======================
               
MAX_NUM_FILE=600
import reader_1031 as MR
reload(MR)
#================global verities=====================
_ReadImg_filelist=[]
_ReadImg_filenames=[]
_ReadImg_filenum=[0,]
#====================================================
def ReadImg(folder_name,START_POS=[0,],reset=False):
    
    global _ReadImg_filelist,_ReadImg_filenames,_ReadImg_filenum    
    if reset==True:
        _ReadImg_filelist=[]
        _ReadImg_filenames=[]
        _ReadImg_filenum=[0,]
        START_POS=[0,]
    out_data=out_data2=None  
    EOF=True
    
    if _ReadImg_filenum[0]==0: #inside static var
        folder_name=os.getcwd()+'\\'+folder_name
        _ReadImg_filelist,_ReadImg_filenames=MR.scan_folder(folder_name)
        _ReadImg_filenum[0]=len(_ReadImg_filenames)
        
    filelist=_ReadImg_filelist
    filenames=_ReadImg_filenames  
    filenum=_ReadImg_filenum
    
    start_pos=START_POS[0]
    print(('filenum=%d start_pos=%d')%(filenum[0],start_pos))
    num=min(filenum[0]-start_pos,MAX_NUM_FILE)
    assert num>=0
    end_pos=start_pos+num 
    
    if num!=0:
        out_data=np.zeros((num,MR.IMAGE_SIZE*MR.IMAGE_SIZE),dtype=int)
        out_data2=np.zeros((num,255),dtype=int)
    for i in range(num):

        img=MR.cv2.imread(filelist[i+start_pos],0)#gray
        img,hist=MR.CutImg(img,MR.IMAGE_SIZE)
        #img=MR.CutImg(img,MR.IMAGE_SIZE+MR.LBP_R)
        #img=MR.LBP(img,MR.LBP_R)
        new_raw=img.flatten()
        out_data[i]=new_raw
        out_data2[i]=hist
    if end_pos==0:
        print ('cannot find folder_name')
    elif end_pos==filenum[0]:
        print ('read finished')        
    else:
        print ('-----read: %d / %d ------\r' %(end_pos,filenum[0]))
        EOF=False
    START_POS[0]=end_pos  
    return (out_data,out_data2,filelist[start_pos:end_pos],filenames[start_pos:end_pos],EOF)

def load_apply_data(file_name,EOF=False,reset=False):  
    
    if file_name==None:
        return (None,None,[],[],True)
    print ('loading %s'%(file_name))
    data_x,data_x2,filelist,filenames,EOF=ReadImg(file_name,reset=reset)
    #data_x=pd.DataFrame(data_csv)
    if len(filenames)!=0:
        data_x=theano.shared(\
        np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        data_x2=theano.shared(\
        np.asarray(data_x2,dtype=theano.config.floatX),borrow=True)
    return (data_x,data_x2,filelist,filenames,EOF)

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
    	   
    print ('create cross_validation set')
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
from read_and_write import mywrite 
#===========================LeNet-5 demo================================
def evaluate_lenet(datasets,params,skip_training,apply_set_namelist,
                    sizekerns,nkerns,sizebatch,learning_rate,n_epochs,isSaveParam=False):#500
    '''
    n_epochs训练步数，每一次都会遍历所有batch，  即所有样本
    sizebatch这里设置500,每遍历500个样本，才计算梯度
    nkerns=[20,50],每一个LeNetConvPoolLayer卷积核的个数，第一层20个核，第二层50
    '''
    #print(type(learning_rate))
    #print(n_epochs,learning_rate,sizebatch,nkerns,sizekerns)

    train_set_x, train_set_x2,train_set_y = datasets[0]  
    valid_set_x, valid_set_x2,valid_set_y = datasets[1]  

    #calculate batch的个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]  
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]  
    if skip_training==False:
        sizebatch=min(n_train_batches,n_valid_batches,sizebatch)
        print (' train_size=%d\n valid_size=%d\n sizebatch=%d')%\
            (n_train_batches,n_valid_batches,sizebatch)
        n_train_batches /= sizebatch  
        n_valid_batches /= sizebatch  #需要修改if data_size<sizebatch batch=size

    
    
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
    print ('...building the model')
    #我们加载数据是(sizebatch,220*220),但是LeNetConvPoolLayer输入是四维的
    
    CP0_input=x.reshape((sizebatch,1,MR.IMAGE_SIZE,MR.IMAGE_SIZE))
    '''
    layer0即第一个LeNetConvPoolLayer层
    输入的单张图片(220,220)经过conv得到200-21+1=200
    经过maxpooling得到200/2=100
    因为每一个batch都有sizebatch张图，第一个LeNetConvPoolLayer层有nkerns[0]卷积核
    故layer0输出为（sizebatch,nkerns[0],50,50)
    '''
    image_size=MR.IMAGE_SIZE
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
    cost=LR.negative_log_likelihood(y)
   
    params=LR.params+DFC2.params+DFC1.params+CP2.params+CP1.params+CP0.params
    
    #对各个参数的梯度
    grads=T.grad(cost,params)
    #由于参数太多，在update里写很麻烦，所有用for in 
    updates=[
        (param_i,param_i-learning_rate*grad_i)
        for param_i,grad_i in zip(params,grads)
    ]
    #下面是train_model，优化设计到SGD,需要计算梯度，更新参数
    #参数集合
    train_model = theano.function(  
        #mode='DebugMode',
        on_unused_input='ignore',
        inputs=[index],  
        outputs=(cost,LR.errors(y)),  
        updates=updates,  
        givens={  
        
            is_train: np.cast['int32'](1),
            x: train_set_x[index * sizebatch: (index + 1) * sizebatch],  
            y: train_set_y[index * sizebatch: (index + 1) * sizebatch],  
                                          
            hist:train_set_x2[index * sizebatch: (index + 1) * sizebatch],
            #is_train: T.cast(1, 'int32')
        }  
    )  
    #validate_model，验证模型，分析同上。  
    validate_model = theano.function(  
        on_unused_input='ignore',
        inputs=[index],  
        outputs=LR.errors(y),  
        givens={  
            hist:valid_set_x2[index * sizebatch: (index + 1) * sizebatch],
            x: valid_set_x[index * sizebatch: (index + 1) * sizebatch],  
            y: valid_set_y[index * sizebatch: (index + 1) * sizebatch],                                    
            is_train: np.cast['int32'](0)
        }  
    )
    analyse_model=theano.function(  
        on_unused_input='ignore',
        inputs=[index],  
        outputs=[LR.errors(y), cost,LR.getPred(),y],
        givens={  
            hist:valid_set_x2[index * sizebatch: (index + 1) * sizebatch],
            x: valid_set_x[index * sizebatch: (index + 1) * sizebatch],  
            y: valid_set_y[index * sizebatch: (index + 1) * sizebatch],                                    
            is_train: np.cast['int32'](0)
        } 
    )

    ret_model_params=theano.function(
        inputs=[],
        outputs=params
    )
    stroage_best_params=ret_model_params()
    best_params=params   
    ################
    #   开始训练   #
    ################
    print ('...training '+time.strftime('%H:%M:%S'))
    patience = 5000.0    
    patience_increase = 2    
    improvement_threshold = 0.995   
                                      
  
    best_validation_loss = np.inf   #最好的验证集上的loss，最好即最小  
    this_validation_loss=np.inf
    this_cost=np.inf
    min_cost=np.inf  
    
    #best_iter = 0                      #最好的迭代次数，以batch为单位。比如best_iter=10000，说明在训练完第10000个batch时，达到best_validation_loss  
    this_cost = 100.  
    start_time = time.clock()  
  
    epoch = 0  
    done_looping = False  

    
    rd_iter=np.ones(1000,dtype='int')
    rd_error=np.ones(1000)    
    rd_error2=np.ones(1000)  
    rd_index=0   
    
    if skip_training:
        done_looping=True;
    while(epoch<n_epochs) and (not done_looping):
        
        epoch=epoch+1
        for minibatch_index in xrange(n_train_batches):
            iter=(epoch-1)*n_train_batches+minibatch_index
            
            this_cost,this_error=train_model(minibatch_index)#!!!
            
            if min_cost*0.97 > this_cost:
                min_cost=this_cost
                rd_iter[rd_index]=iter
                
                rd_error[rd_index]=this_error
                rd_index+=1
                validation_losses=[validate_model(i) for i
                                    in xrange(n_valid_batches)]
                this_validation_loss=np.mean(validation_losses)
                
                if this_validation_loss < best_validation_loss:
                    
                    if this_validation_loss < best_validation_loss \
                        * improvement_threshold : 
                        temp=iter*patience_increase
                        if patience<temp:
                            patience=temp
                        
                    best_validation_loss=this_validation_loss
                    stroage_best_params=ret_model_params()
                    best_params=params
                    #best_iter=iter
                    
                rd_error2[rd_index]=this_validation_loss   
                print(  
                    ('epoch %i, minibatch %i/%i, train cost %f,train error %f %%')%  
                    (  
                        epoch,  
                        minibatch_index + 1,  
                        n_train_batches,  
                        this_cost,                    
                        this_error * 100.  
                    )  
                ) 
                print ('validation error %f %%')% (this_validation_loss*100.)
                end_time=time.clock()
                print (time.strftime('%H:%M:%S')+' ran for %.2fm' %((end_time-start_time)/60.))
            #save temp param for one epoch
            #write_file = open(No+'params_temp.py', 'wb')    
            #pickle.dump(stroage_best_params, write_file, -1)    
            #write_file.close()  
            if patience<=iter or this_cost<=0.0001:
                done_looping = True
                break
    
        
   
    end_time=time.clock()
    print ('train completed')
    print (
        'Best validation score of %f %% with test cost %f '%
        (best_validation_loss*100.,min_cost)
    )
    print  ('The code of file '+\
              os.path.split(__file__)[1]+\
              ' ran for %.2fm' %((end_time-start_time)/60.)
            )
    
    '''
    if skip_training==False:       
        #params=ret_model_params()
        save_name='params.py'
        print 'saving '+save_name
        write_file = open(save_name, 'wb')    
        pickle.dump(stroage_best_params, write_file, -1)    
        write_file.close()  
        print save_name+' saved!'
    
        rd_iter=rd_iter[1:rd_index]
        rd_error=rd_error[1:rd_index]*100.0
        rd_error2=rd_error2[1:rd_index]*100.0
        
        
        pl.plot(rd_iter,rd_error,label='train')
        pl.plot(rd_iter,rd_error2,label='valid')
            pl.title('iter-error')
        pl.xlabel('iter')
        pl.ylabel('error')
        pl.legend(loc='upper right')
        #pl.xlim(0,100)
        #pl.ylim(0,0.8)
        pl.show() 
    '''
    #--------------------------------------------   
    '''
    train_params=[
        n_epochs,learning_rate,sizebatch,nkerns,sizekerns
    ]    
    '''
    train_params=[
        sizekerns,nkerns,sizebatch,learning_rate,n_epochs
    ]
    params_filename='data\\'+str(train_params)+'_params.py'
    mywrite(stroage_best_params,params_filename)
    print ('saving '+params_filename)
    #------------------------------
    best_params=params
    all_errors=[]
    all_cost=[]
    myCM=np.zeros((3,3),dtype=int)
    
    for i in xrange(n_valid_batches):
        errors,cost,y_pred,y=analyse_model(i)
        all_errors.append(errors)
        all_cost.append(cost)
        myCM+=confusion_matrix(y,y_pred)
    errors=np.mean(all_errors)
    cost=np.mean(all_cost)
    #------------------------------
    rd_iter=rd_iter[1:rd_index]
    rd_error=rd_error[1:rd_index]*100.0
    rd_error2=rd_error2[1:rd_index]*100.0
    tend=[rd_index,rd_iter,rd_error,rd_error2]
    print (rd_index)
    if isSaveParam:
        output_data=[train_params,errors,cost,myCM,tend]
        outfilename='data\\'+str(train_params)+'_result.pickle'
        mywrite(output_data,outfilename)
        print ('saving '+outfilename)
    del rd_iter,rd_error,rd_error2
    del all_errors,all_cost
    '''
    train_set_x, train_set_x2,train_set_y = datasets[0]  
    valid_set_x, valid_set_x2,valid_set_y = datasets[1] 
    '''
    del train_set_x, train_set_x2,train_set_y
    del valid_set_x, valid_set_x2,valid_set_y
    #------------------------------------------- 
    
    
    if len(apply_set_namelist)==0: 
        return
    print 'apply model'
    params=best_params#best one
    start_time=time.clock()    
        
    num=0
    filelist=[]
    if os.path.exists('result.csv')==True:
		os.remove('result.csv')
    #str_time=time.strftime('%H:%M:%S')
    boss_df=pd.DataFrame()
    ireset=True
    for apply_set_name in apply_set_namelist:
        EOF=False
        ireset=True
        while(ireset!=True):
            ireset=True
        while EOF==False:  
            print 'read img...'
            apply_data,apply_data2,filelist,filenames,EOF=load_apply_data(apply_set_name,reset=ireset)
            ireset=False
            if len(filenames)==0:
                break
            apply_size=apply_data.get_value(borrow=True).shape[0]  
            n_apply_batches = apply_size
            if n_apply_batches==0:
                break
    #        if n_apply_batches<sizebatch:
    #            sizebatch=n_apply_batches#适应数据比批处理数量小
            n_apply_batches /= sizebatch
            print('n_apply_batches %d'%(n_apply_batches))
            print('sizebatch %d'%(sizebatch))
            apply_model=theano.function(
                on_unused_input='ignore',
                inputs=[index],
                outputs=LR.getPred(),
                givens={
                    hist:apply_data2[index * sizebatch: (index + 1) * sizebatch],
                    x:   apply_data [index * sizebatch: (index + 1) * sizebatch], 
                    is_train:np.cast['int32'](0)
                    }
            )
            print 'apply .....'    
            prediction=np.zeros(len(filelist))
        
            for minibatch_index in xrange(n_apply_batches):
                pred=apply_model(minibatch_index)
                prediction[minibatch_index*sizebatch:(minibatch_index+1)*sizebatch]=pred
                #print '%d / %d \r'%((minibatch_index+1)*sizebatch,n_apply_batches),
                
            print 'apply_size:',apply_size,' sum:',
            apply_set_type=np.ones(len(prediction),dtype=int)*int(apply_set_name)
            df=pd.DataFrame([apply_set_type,prediction,filenames,filelist])
            df=df.T
            boss_df=boss_df.append(df)
            num=num+apply_size
            print num
    
    with open('result.csv','w') as output:               
        boss_df.to_csv(output,index=False)
                
    end_time=time.clock()
    print ( 'ran for %.2fm ' %((end_time-start_time)/60.))

    print 'forecast finished'   

       

def LENET(sizekerns,nkerns,sizebatch,learning_rate,n_epochs,skip_training=False):
    
    Num=10
    exist_params,params=load_params('params.py')
    
    train_name='train_'+str(MR.IMAGE_SIZE)+'.csv'
       
    if skip_training==False:                  
        #train_name=raw_input('input filename of train set:')
        print 'loading',train_name
        if os.path.exists(train_name)==False:
            raw_input('cannot find file,over!')
            sys.exit(1)
        train_csv=pd.read_csv(train_name)
        train_data=train_csv.values
    else:
        datasets=load_train_data(None,True,0,0) 
        Num=0
    rand_num=np.random.randint(0,Num)
    rand_num=1 #----------------test!!!!!!!!!!!!!!!!
    for i in range(Num):#Num
        datasets=load_train_data(train_data,skip_training,Num,i) #获取交叉验证的数据集 
        flag= False if (rand_num!=i) else True
        evaluate_lenet(
                datasets,params,False,[],
				sizekerns,nkerns,sizebatch,learning_rate,n_epochs,
                isSaveParam = flag
        )
        if i==2:
            break#---------------test!!!!!!!!!!!!!!!!!!
        #exist_params,params=load_params('params.py')  
    #evaluate_lenet(datasets,params,True,['0','1','2'],n_epochs=0,sizebatch=1)

    print 'Program finished'
    
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
    [3,5,7],[3,3,4],[5,4,3],[7,5,4],[9,6,5]
    ]
    nkerns=[20,40,60]
    sizebatch_list=[40,60,80]
    learning_rate_list=[0.0008,0.0006,0.0004]
    n_epochs=65 #[5,10,30,50,70]
    contents=[sizekerns_list,sizebatch_list,learning_rate_list]
    myiter=myIteration(contents)
    
    print '....................program begining .............................'
    while (myiter.isEmpty()==False):
        v=myiter.getvalue()
        sizekerns=v[0]
        sizebatch=v[1]       
        learning_rate=v[2]
        
         ##just for test20170418!!!!!!!!!!!!!!!!!!!
        n_epochs=1#v[3]  #-------------test!!!!!!!!!!!
        
        v.insert(1,nkerns)
        v.append(n_epochs)
        #print v #-------------test!!!!!!!!!!!
        #------------------20170410-------------
        #这是保存的顺序，与v不一致。当时写错了
        
        #vv=[n_epochs,learning_rate,sizebatch,nkerns,sizekerns]
        outfilename='data\\'+str(v)+'_result.pickle'
        if os.path.exists(outfilename)==True:
            print v,' existed'
    #---------------------------------------
        else:
            print("start:",v)
            LENET(sizekerns,nkerns,sizebatch,learning_rate,n_epochs)        
        myiter.inc()
        
    print '....................Program finished...........................'
    
    