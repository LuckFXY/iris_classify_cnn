#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:38:10 2017

@author: rain
"""
from PIL import Image
import os
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


if __name__=="__main__":
    for i in range(2):
        filelist,filenames=scan_folder(str(i))
        for fn in filelist:
            print(fn)
            img=Image.open(fn)
            for d in range(0,180,18):
                print d,
                img2=img.rotate(d)
                f1=os.path.splitext(fn)[0]
                f2=os.path.splitext(fn)[1]
                fname=f1+'_'+str(d)+f2
                print(fname)
                img2.save(fname)