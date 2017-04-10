# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 16:12:24 2016

@author: Administrator
"""
#'导入一个图片并且灰度化'
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import cv2
import numpy as np
import os
import csv
import math
IMAGE_SIZE=50
LBP_R=3
def getdata(no,index,src,size):
    assert index!=size
    return src[(no+index)%size]
    
def minBinaryM(strbin):  
    size=len(strbin)
    aim=range(size)
    index=0
    while(index!=size):
        reaim=[]
        min='1'
        for no in aim:
            bit=getdata(no,index,strbin,size)
            if bit==min:
                reaim.append(no)
            elif bit<min:
                reaim=[no,]
                min='0'
        aim=reaim
        if len(aim)==1:
            break
        index=index+1
    no=aim[0]
    re=strbin[no:]+strbin[:no]       
    return int(re,base=2)     
            
def LBP(img,R=2,SC=8):   
    H,W=img.shape
    step=2*math.pi/SC
    reimg=np.zeros((W-R,H-R),dtype=int)
    for y in xrange(R,H-R):
        for x in xrange(R,W-R):
            pixel=int(img[y,x])
            LBP=np.zeros(SC,dtype=int)
            for i in xrange(SC):
                xp=int(x+R*math.cos(math.pi/2 - step*i))
                yp=int(y-R*math.sin(math.pi/2 - step*i))
                if img[yp,xp]>pixel:
                   LBP[i]=1
            t=''.join(str(i) for i in LBP)            
            m=minBinaryM(t)
            if m>255:
                m=255
            reimg[y-R,x-R]=m
    return reimg
    
def hist(img):
    #img = Image.open(filename).convert("L") 
    #img = np.array(img)
    pixel_n,bins = np.histogram(img.flatten(),255)
    choose=pixel_n > pixel_n.mean()*3
    pixel_n[choose]=0
    return pixel_n
    
def CutImg(img,img_size):
    #path_in=os.getcwd()+'\\'+filename
    #path_out=os.getcwd()+'\\new-'+filename
    #img=cv2.imread(path_in,0)
    center_y=img.shape[0]/2;
    center_x=img.shape[1]/2;
    m=min(img.shape)/2
    left_y=center_y-m
    left_x=center_x-m
    right_y=center_y+m
    right_x=center_x+m
    img=img[left_y:right_y,left_x:right_x]        
    img=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)  
    img_hist=hist(img)
    #img2=LBP(img2,LBP_R,8)
    #cv2.imshow('test',img2)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    return img,img_hist #/255.0  #标准化

def scan_folder(file_dir,file_type=None):
    #acquire the list of filename by scaning specified folder
    filelist=[]
    filenames=[]
    for root, dirs, files in os.walk(file_dir):  
        #print(root) #当前目录路径  
        #print(dirs) #当前路径下所有子目录  
        #print(files) #当前路径下所有非目录子文件  
        if file_type!=None:
            for file in files:
                if os.path.splitext(file)[1]=='.'+file_type:
                    filelist.append(os.path.join(root,file))
                    filenames.append(file)
        else:
            for file in files:
                postfix=os.path.splitext(file)[1]
                if postfix in {'.bmp','.png','.jpg'} :
                    filelist.append(os.path.join(root,file))
                    filenames.append(file)
    return filelist,filenames
#'格式化输入并且添加标签'

    
def ConvToCsv(folder_name,img_label,file_type=None):
    folder_name=os.getcwd()+'\\'+folder_name
    filelist,filenames=scan_folder(folder_name,file_type)
    num=len(filelist)
    n=1
    print num, ' '
    out_data=np.zeros((len(filelist),IMAGE_SIZE*IMAGE_SIZE+255+1))
    for i in xrange(len(filelist)):
        img=cv2.imread(filelist[i],0)#gray
        img,hist=CutImg(img,IMAGE_SIZE)
        #img=LBP(img,LBP_R)
        #print len(hist)
        out_data[i,1:256]=hist#/255.0
        out_data[i,256:]=img.flatten()#/255.0
        out_data[i,0]=img_label
        #print '.',
        n=n+1  
    print 'read finished'
    return out_data
    

    
def GenerateSet(folder_list,data_types,out_name):
    '''
    folder_list:
        type : list
        contents : names of folder
    data_types:
        type : list
        contents: types of folder
    out_name:
        type : string
        contents: the name of output(set)
    '''
    num=len(folder_list)
    data=ConvToCsv(folder_list[0],data_types[0])
    for i in range(1,num):
        d=ConvToCsv(folder_list[i],data_types[i])
        data=np.concatenate((d,data))
    print 'shuffle......'
    np.random.shuffle(data)
    
    #out_name='train-micro.csv'
    #out_name=raw_input('input out file name:')
    csvfile=file(out_name,'wb')
    writer=csv.writer(csvfile)
    print 'saving as ',out_name
    writer.writerows(data)
    csvfile.close()
    
if __name__=='__main__':
    
    #FL=['0people-interval','1people-smalleye','2people-verygood','3lion']
    FL=["people0","people1","lions2"]
    #FL=["w0","w1","w2"]
    #for i in range(len(FL)-1):
    #    L=[ FL[i],FL[-1] ]
    sym=[0,1,2]
    #outname=str(i)+'train.csv'
    outname='train_'+str(IMAGE_SIZE)+'.csv'
    GenerateSet(FL,sym,outname)
    print 'program over'