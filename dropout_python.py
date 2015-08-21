#!/usr/bin/env python
# encoding=utf-8


__docformat__='restructedtext en'



from PIL import Image
import numpy as np
import pylab

x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0,1,1,0]]).T
print 'x',x #4x3
print 'y',y  #4
alpha,hidden_dim,dropout_percent,do_dropout=(0.5,4,0.2,True)
synapse_0=2*np.random.random((3,hidden_dim))-1;print '0',synapse_0.shape #3x4
synapse_1=2*np.random.random((hidden_dim,1))-1;print '1',synapse_1.shape #4x1
for j in xrange(2):
	layer_1=(1/(1+np.exp(-(np.dot(x,synapse_0)))));print 'l1',layer_1.shape  #4x3  x  3x4   4x4
	if do_dropout:
		layer_1*=np.random.binomial([np.ones((len(x),hidden_dim))],1-dropout_percent)[0]\
				*(1.0/(1-dropout_percent))  #binomial mask 
		layer_2=1/(1+np.exp(-(np.dot(layer_1,synapse_1)))) #4x4 x 4x1 4x1
		layer_2_delta=(layer_2-y)*(layer_2*(1-layer_2))  #4x1
		layer_1_delta=layer_2_delta.dot(synapse_1.T)*(layer_1*(1-layer_1)) #4x1  x  1x4  x  4x4
		synapse_1-=(alpha*layer_1.T.dot(layer_2_delta)) #4x4  x  4x1
		synapse_0-=(alpha*x.T.dot(layer_1_delta))  #3x4   x  4x4













