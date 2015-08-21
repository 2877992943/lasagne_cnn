#!/usr/bin/env python
# encoding=utf-8

'''
svd initial w 
use concatenate layers
 
'''

#####__docformat__='restructedtext en'

from PIL import Image
import numpy as np
import os,cPickle,time,theano,lasagne
import theano.tensor as T

data_path='/home/yr/theanoExercise/mnist/mnist.pkl.gz'
out_path='home/yr/Lasagne_exer/image/'



def data_augm():
	return 0	
	
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
	network_c1=lasagne.layers.Conv2DLayer(
			network,num_filters=5,filter_size=(2,2),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.Orthogonal())
	network_c2=lasagne.layers.Conv2DLayer(
			network,num_filters=5,filter_size=(3,3),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.Orthogonal())
	network_c3=lasagne.layers.Conv2DLayer(
			network,num_filters=5,filter_size=(5,5),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.Orthogonal())

	network_c40=lasagne.layers.PadLayer(network_c40,1)#[none,1,28,28]->[none,1,30,30] pad before maxpool
	network_c40=lasagne.layers.MaxPool2DLayer(network_c40,pool_size=(3,3),stride=(1,1))#[none,1,30,30]->[none,1,30-3+1,28]
	#network_c40=lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2),stride=(1,1))#[none,1,28,28]->[none,1,28-2+1,27]then pad->29
	network_c4=lasagne.layers.Conv2DLayer(
			network_c41,num_filters=5,filter_size=(5,5),
			stride=(1,1),border_mode='same',
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.Orthogonal())
	

	network=lasagne.layers.ConcatLayer((network_c1,network_c2,network_c3,network_c4),axis=1)

	network=lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
	
	network=lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network,p=0.5),
			num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)
	return network 


			
def iterate_minibatches(inputs,targets,batchsize,shuffle=False):
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
	network=build_cnn(input_var)
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
	for epoch in range(num_epochs)[:]:
		#trainset
		train_err=0
		train_batches=0
		start_time=time.time()
		for batch in iterate_minibatches(x_train,y_train,500,shuffle=True):#[500,1,28,28]
			inputs,targets=batch#data augmentation
			train_err+=train_fn(inputs,targets)
			train_batches+=1; 
		#valid set
		val_err=0
		val_acc=0
		val_batches=0
		for batch in iterate_minibatches(x_val,y_val,500,shuffle=False):
			inputs,targets=batch
			err,acc=val_fn(inputs,targets)
			val_err+=err
			val_acc+=acc
			val_batches+=1; 
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
		test_batches+=1; 
	print ("test loss:\t\t\t{:.6f}".format(test_err/test_batches))
	print("test acc:\t\t{:.2f}%".format(test_acc/test_batches*100))
	
	 
 

if __name__=='__main__':
	main()
	
	 
	
	
	
	 
	
	
	 


	

	
	
		

			
			

		


