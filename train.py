import tensorflow as tf
import tensorflow.keras as keras
import librosa
import librosa.feature
import glob
import numpy as np
from utils import load_data
from models import simple_cnn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

trainX, trainY, validationX, validationY, testX, testY = load_data(split_into_k_equal_parts=True, k=7)

Y_True = []
for i in testY:
    Y_True.append(list(i).index(1))

#Reshaping into (batch_size, height, width, channels)
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], trainX.shape[2], 1))
validationX = validationX.reshape((validationX.shape[0], validationX.shape[1], validationX.shape[2], 1))
testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2], 1))

print(trainX.shape)
print(trainY.shape)
print(validationX.shape)
print(validationY.shape)

indim_x = trainX.shape[1]
indim_y = trainX.shape[2]

model = simple_cnn((indim_x,indim_y), conv_layers=[16, 32, 48])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(trainX, trainY, batch_size = 128, validation_data=(validationX, validationY), epochs=25)

model.evaluate(testX, testY, verbose=2)



# predict probabilities for test set
Y_PredictedProbabilities = model.predict(testX, verbose=0)
# predict classes for test set
Y_Predicted = model.predict_classes(testX, verbose=0)
print(Y_PredictedProbabilities)
print(Y_Predicted)
print(Y_True)

target_names = ['Classical', 'Electronic', 'Pop', 'Rock', 'Hip-Hop']
print(classification_report(Y_True, Y_Predicted,target_names=target_names))
print(confusion_matrix(Y_True, Y_Predicted))



