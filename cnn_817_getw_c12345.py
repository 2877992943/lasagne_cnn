#!/usr/bin/env python
# encoding=utf-8

'''
svd initial w
train to optimize w b
get w of all conv_layer
'''

#####__docformat__='restructedtext en'

from PIL import Image
import numpy as np
import os,cPickle,time,theano,lasagne
import theano.tensor as T
import pylab

data_path='/home/yr/theanoExercise/mnist/mnist.pkl.gz'



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


def build_cnn(input_var=None):
	network=lasagne.layers.InputLayer(shape=(None,1,28,28),
					input_var=input_var)
	network=lasagne.layers.Conv2DLayer(
			network,num_filters=9,filter_size=(5,5),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.Orthogonal())
	network=lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
	network=lasagne.layers.Conv2DLayer(
			network,num_filters=16,filter_size=(5,5),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify)
	network=lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
	network=lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network,p=0.5),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify)
	network=lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network,p=0.5),
			num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)
	return network


			
def iterate_minibatches(inputs,targets,batchsize,shuffle=False):#inputs [-1,1,40,40]
	assert len(inputs)==len(targets)
	if shuffle:
		indices=np.arange(len(inputs))#[0,1,2,...99]
		np.random.shuffle(indices)
	for start_idx in range(0,len(inputs)-batchsize+1,batchsize): #range[0,100-10+1,10] [0,10,20,...90]
		if shuffle:
			excerpt=indices[start_idx:start_idx+batchsize] #[0:10] 10 excluded
		else:
			excerpt=slice(start_idx,start_idx+batchsize)#[0,10] 10 excluded
		yield inputs[excerpt],targets[excerpt] # everytime when call this function not call funtion,but generate an iterator,
			

def main(model='cnn',num_epochs=2):
	#load data
	x_train,x_val,x_test,y_train,y_val,y_test=load_data()#[50000,1,28,28]
	# symbolic variable
	input_var=T.tensor4('inputs')
	target_var=T.ivector('targets')
	#make graph expression
	network=build_cnn(input_var)###network=build_cnn(input_var)
	prediction=lasagne.layers.get_output(network)
	loss=lasagne.objectives.categorical_crossentropy(prediction,target_var)
	loss=loss.mean()

	params=lasagne.layers.get_all_params(network,trainable=True)
	updates=lasagne.updates.nesterov_momentum(
				loss,params,learning_rate=0.01,momentum=0.9)
	test_prediction=lasagne.layers.get_output(network,deterministic=True)
	test_loss=lasagne.objectives.categorical_crossentropy(test_prediction,
								target_var)
	test_loss=test_loss.mean()
	test_acc=T.mean(T.eq(T.argmax(test_prediction,axis=1),target_var),   #100x10class
			dtype=theano.config.floatX)

	#compile
	train_fn=theano.function([input_var,target_var],loss,updates=updates)#loss symbolic, no data in it
	val_fn=theano.function([input_var,target_var],[test_loss,test_acc])#loss acc symbolic, no data in it
	#train
	print 'start training...'
	for epoch in range(num_epochs):
		#trainset
		train_err=0
		train_batches=0
		start_time=time.time()
		for batch in iterate_minibatches(x_train,y_train,500,shuffle=True):#[500,1,28,28] batchsize=100
			inputs,targets=batch#data augmentation
			train_err+=train_fn(inputs,targets)
			train_batches+=1
		#valid set
		val_err=0
		val_acc=0
		val_batches=0
		for batch in iterate_minibatches(x_val,y_val,500,shuffle=False):
			inputs,targets=batch
			err,acc=val_fn(inputs,targets)
			val_err+=err
			val_acc+=acc
			val_batches+=1
		#print
		print("epoch {} of {} took {:.3f}s".format(
			epoch+1,num_epochs,time.time()-start_time))
		print("train loss:\t\t{:.6f}".format(train_err/train_batches))
		print("valid loss:\t\t{:.6f}".format(val_err/val_batches))
		print("valid acc:\t\t{:.2f}%".format(val_acc/val_batches*100))
		
	#test
	test_err=0
	test_acc=0
	test_batches=0
	for batch in iterate_minibatches(x_test,y_test,500,shuffle=False):
		inputs,targets=batch
		err,acc=val_fn(inputs,targets)
		test_err+=err
		test_acc+=acc
		test_batches+=1
	print ("test loss:\t\t\t{:.6f}".format(test_err/test_batches))
	print("test acc:\t\t{:.2f}%".format(test_acc/test_batches*100))
	
	###########################get w from c1 net
	para_value_c=lasagne.layers.get_all_param_values(network)
	save_para(para_value_c)		

		
#################################################################### about params			
def save_para(para_value_c1):
	f=open('/home/yr/Lasagne_exer/mnist/wb','wb')
	cPickle.dump(para_value_c1,f,-1)
	f.close()
		
			
def load_para():
	f=open('/home/yr/Lasagne_exer/mnist/wb','rb')
	para_c1=cPickle.load(f);print 'para len',para_c1.__len__() 
	##w[9,1,5,5] [16,9,5,5][784,256][256,10]
	for i in range(len(para_c1)):
		print para_c1[i].shape
	f.close()
	return para_c1[0],para_c1[2],para_c1[4]
 

def gaussian_noise(input_shape):
	from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
	_srng=RandomStreams()
	mask=_srng.normal(input_shape,avg=0.0,std=1.0,dtype=theano.config.floatX);#variable
	mask=mask.eval()#get value from variable
	noise=mask.reshape((-1,1,28,28))
	return noise	
		

if __name__=='__main__':
	###generate and save wb
	#main()

	 
	###load wb from pickle
	w1,w2,w3=load_para()
	
	######################################
	#draw featureMap face
	#########################################
	#load data
	x_train,x_val,x_test,y_train,y_val,y_test=load_data()#[50000,1,28,28]
	imarr=x_train[1006:1007]#x_train[0] dim=3 not 4
	# symbolic variable
	input_var=T.tensor4('inputs')
	#make graph expression
	network=lasagne.layers.InputLayer(shape=(None,1,28,28),
					input_var=input_var)
	network_1=lasagne.layers.Conv2DLayer(
			network,num_filters=9,filter_size=(5,5),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=w1)
	network=lasagne.layers.MaxPool2DLayer(network_1,pool_size=(2,2))
	network_2=lasagne.layers.Conv2DLayer(
			network,num_filters=16,filter_size=(5,5),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=w2)
	network=lasagne.layers.MaxPool2DLayer(network_2,pool_size=(2,2))
	network_3=lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network,p=0.5),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify,
			W=w3)
	featMap_1=lasagne.layers.get_output(network_1,deterministic=True)#theano expression
	featMap_2=lasagne.layers.get_output(network_2,deterministic=True)#theano expression
	featMap_3=lasagne.layers.get_output(network_3,deterministic=True)#theano expression
	#compile
	fn_1=theano.function([input_var],featMap_1)# symbolic, no data in it
	fn_2=theano.function([input_var],featMap_2)   
	fn_3=theano.function([input_var],featMap_3)
	############apply to imarr1 or noise
	x=imarr
	#x=gaussian_noise([28,28])#[-1,1,28,28]
	feat_1=fn_1(x);print 'feature map',feat_1.shape #[1,9,28,28]conv layer,  [1,256]dense layer
	feat_2=fn_2(x);print 'feature map',feat_2.shape #[1,16,14,14]
	feat_3=fn_3(x);print 'feature map',feat_3.shape#[1,256]
	
	 
	####draw feature map
	 
	pylab.figure();pylab.gray();pylab.title('c1 output')
	for i in range(9):
		pylab.subplot(3,3,i+1)
		pylab.imshow(feat_1[0,i,:,:])
	 
	pylab.figure();pylab.gray()
	for i in range(16):
		pylab.subplot(4,4,i+1)
		pylab.imshow(feat_2[0,i,:,:])
	pylab.figure();pylab.gray();pylab.imshow(feat_3.reshape((16,16)));
	  
	
	pylab.show()
	
	 	
	
	 
	
	
	
	 
	
	
	 


	

	
	
		

			
			

		


