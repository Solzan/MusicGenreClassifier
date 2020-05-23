import numpy as np
import os
import sklearn.utils

def load_data(split_into_k_equal_parts = False, k = 3):
	with np.load(f'training.npz') as data:
		trainX = data['spectrograms']
	with np.load(f'training_label.npz') as data:
		trainY = data['labels']
	with np.load(f'validation.npz') as data:
		validationX = data['spectrograms']
	with np.load(f'validation_label.npz') as data:
		validationY = data['labels']
	with np.load(f'test.npz') as data:
		testX = data['spectrograms']
	with np.load(f'test_label.npz') as data:
		testY = data['labels']
	#GTZAN
	with np.load('gtzan.npz') as data:
		gtzan = data['spectrograms']
	with np.load('gtzan_label.npz') as data:
		gtzan_labels = data['labels']


	if split_into_k_equal_parts:
		#we trim the spectrogram so its width is divisible by k in order for np.split function to work
		max_spect_index = trainX.shape[2] - trainX.shape[2]%k
		trainX = np.vstack(np.split(trainX[:,:,:max_spect_index], k, axis=2))
		trainY = np.vstack([trainY for i in range(k)])

		validationX = np.vstack(np.split(validationX[:,:,:max_spect_index], k, axis=2))
		validationY = np.vstack([validationY for i in range(k)])

		testX = np.vstack(np.split(testX[:,:,:max_spect_index], k, axis=2))
		testY = np.vstack([testY for i in range(k)])
		#GTZAN
		gtzan = np.vstack(np.split(gtzan[:,:,:max_spect_index], k, axis=2))
		gtzan_labels = np.vstack([gtzan_labels for i in range(k)])

	#shyffle gtzan so all genres are in each of train/val/test set
	gtzan, gtzan_labels = sklearn.utils.shuffle(gtzan, gtzan_labels, random_state=2)

	#split gtzan into 80/10/10 train/val/test
	gtzan_split = np.split(gtzan, [int(0.8*gtzan.shape[0]),int(0.9*gtzan.shape[0]),gtzan.shape[0]])
	gtzan_labels_split = np.split(gtzan_labels, [int(0.8*gtzan.shape[0]),int(0.9*gtzan.shape[0]),gtzan.shape[0]])
	gtzan_train, gtzan_val, gtzan_test = gtzan_split[0], gtzan_split[1], gtzan_split[2]
	gtzan_lab_train, gtzan_lab_val, gtzan_lab_test = gtzan_labels_split[0], gtzan_labels_split[1], gtzan_labels_split[2]


	#append GTZAN to train, validation and test set data
	trainX = np.vstack((trainX, gtzan_train))
	trainY = np.vstack((trainY, gtzan_lab_train))
	validationX = np.vstack((validationX, gtzan_val))
	validationY = np.vstack((validationY, gtzan_lab_val))	
	testX = np.vstack((testX, gtzan_test))
	testY = np.vstack((testY, gtzan_lab_test))

	#shuffle 
	trainX, trainY = sklearn.utils.shuffle(trainX, trainY, random_state=1)
	validationX, validationY = sklearn.utils.shuffle(validationX, validationY, random_state=3)
	testX, testY = sklearn.utils.shuffle(testX, testY, random_state=7)

	return (trainX, trainY, validationX, validationY, testX, testY)
