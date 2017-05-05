# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:49:14 2017

@author: rain
"""

import time

t0=time.clock()
t1=t0+1200
while(t1-t0>30):
    t0=time.clock()
    print("execfile again")
    
    execfile('CNN4_generator_roc_data.py')
    t1=time.clock()
    print("time: %d"%(t1-t0))

    
