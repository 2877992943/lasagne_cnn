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

data_path='/home/yr/theanoExercise/mnist/mnist.pkl.gz'



from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng=RandomStreams()

def gaussian_noise(input_arr,input_shape):
	mask=_srng.normal(input_shape,avg=0.0,std=1.0,dtype=theano.config.floatX);print mask.eval().shape
	return input_arr*mask.eval()
def binomial_noise(input_arr,input_shape):
	mask=_srng.binomial(input_shape,p=0.7,dtype=theano.config.floatX);print mask.eval().shape
	return input_arr*mask.eval()
	

def resize_chop(arr):
	from skimage.transform import resize
	arr_rsz=resize(arr,(32,32));print 'rsz',arr_rsz.shape
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
	arr_rotate=rotate(arr,angle,resize=False);print 'rotate',arr_rotate.shape #resize=true size 29x29 instead of 28x28
	return arr_rotate
		

def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if \
		f.endswith('.jpg')]


def imresize(imlist):
	for imname in imlist:
		im=Image.open(imname).convert('L')
		im1=im.resize((80,80))
		im1.save('/home/yr/computer_vision/data_resize/'+imname[31:])

 

def arr2img(arr,count):
	num=arr.shape[0]
	for i in range(num)[:]:
		im=arr[i,0,:,:]*255.
		im=Image.fromarray(np.uint8(im))
		im.save('/home/yr/Lasagne_exer/image/'+str(count)+'_'+str(i)+'.jpg')
		
	
def load_data():
	import gzip
	with gzip.open(data_path,'rb') as f:
		data=cPickle.load(f)
	x_train,y_train=data[0]
	x_val,y_val=data[1]
	x_test,y_test=data[2]
	
	x_train=x_train.reshape((-1,1,28,28))
	x_val=x_val.reshape((-1,1,28,28))
	x_test=x_test.reshape((-1,1,28,28))

	y_train=y_train.astype(np.uint8)
	y_val=y_val.astype(np.uint8)
	y_test=y_test.astype(np.uint8)
	return x_train,x_val,x_test,y_train,y_val,y_test
		

if __name__=='__main__':
	'''
	##load arr 10000
	x1,x2,x3,y1,y2,y3=load_data()
	arr=x1[0,0,:,:]
	
	with open('/home/yr/Lasagne_exer/1.pkl','wb')as f:
		pickle.dump(arr,f)
	'''
	##load im 1
	with open('/home/yr/Lasagne_exer/1.pkl','rb')as f:
		arr=pickle.load(f)
	pylab.figure();pylab.gray()
	pylab.subplot(3,3,1);pylab.imshow(arr*255);pylab.title('original')
	##noise
	arr_noise=gaussian_noise(arr,(28,28))
	pylab.subplot(3,3,2);pylab.imshow(np.uint8(arr_noise*255));pylab.title('gaussian noise')

	arr_noise_1=binomial_noise(arr,(28,28))
	pylab.subplot(3,3,3);pylab.imshow(np.uint8(arr_noise_1*255));pylab.title('binomial noise')
	### resize+transform  [28x28]->[32,32]->[28x28]
	arr_rsz=resize_chop(arr)
	pylab.subplot(3,3,4);pylab.imshow(np.uint8(arr_rsz*255));pylab.title('chop')
	####rotate
	arr_rotate=rotate_r(arr)
	
	pylab.subplot(3,3,5);pylab.imshow(np.uint8(arr_rotate*255));pylab.title('rotate')


	pylab.show()
	 
	
	
	
	 
	
	
	 


	

	
	
		

			
			

		


