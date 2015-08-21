#!/usr/bin/env python
# encoding=utf-8

'''
put into file <python_tool_yr> tobe called
'''

#####__docformat__='restructedtext en'

from PIL import Image
import numpy as np
import os,cPickle,theano
import pickle
import pylab 

 



from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng=RandomStreams()

def gaussian_noise(input_arr,input_shape):
	mask=_srng.normal(input_shape,avg=0.0,std=1.0,dtype=theano.config.floatX); 
	return input_arr*mask.eval()
def binomial_noise(input_arr,input_shape):
	mask=_srng.binomial(input_shape,p=0.7,dtype=theano.config.floatX); 
	return input_arr*mask.eval()
	

def resize_chop(arr):
	from skimage.transform import resize
	arr_rsz=resize(arr,(32,32)); 
	arr_rsz_dic={1:arr_rsz[:28,:28],\
			2:arr_rsz[:28,4:],\
			3:arr_rsz[4:,:28],\
			4:arr_rsz[4:,4:]}
	from random import sample
	i=sample([1,2,3,4],1)[0]
	return arr_rsz_dic[i]

def rotate_r(arr):
	from skimage.transform import rotate
	from random import sample
	angle=sample([-10,10],1)[0]
	arr_rotate=rotate(arr,angle,resize=False); 
	return arr_rotate
		
 
		
 
	
	 
	
	
	 


	

	
	
		

			
			

		


