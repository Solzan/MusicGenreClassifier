import numpy as np
import os
import sklearn.utils

def load_data():
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

	trainX, trainY = sklearn.utils.shuffle(trainX, trainY, random_state=0)
	validationX, validationY = sklearn.utils.shuffle(validationX, validationY, random_state=0)
	testX, testY = sklearn.utils.shuffle(testX, testY, random_state=0)

	return (trainX, trainY, validationX, validationY, testX, testY)

