import numpy as np
from keras.models import Input, Model,Sequential
from keras.layers import Dense,Dropout,merge,Concatenate,Input,CuDNNLSTM
import tensorflow as tf
import math

##LOADING
embeddings = np.load("Sentence_Embeddings.npy",allow_pickle = True)
sent_len = np.load("Sent_Len.npy",allow_pickle = True)
y_labels = [math.floor((sent_len[i]-1)/9) for i in range(len(sent_len))]

##SPLITTING
test_x = np.array(embeddings[:10002])
test_y = np.array(y_labels[:10002])
training_x = np.array(embeddings[10002:])
training_y = np.array(y_labels[10002:])

##PROBING MODEL
Embeddings = Input(shape=(300,))
x = Dense(128,activation="relu",use_bias=True)(Embeddings)
x = Dense(9,activation="softmax",use_bias=True)(x)
model = Model(inputs=Embeddings,outputs=x)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics="accuracy")
#model.summary()
model.fit(
	x = training_x, y = training_y,
	epochs = 10,
	batch_size = 64,
	validation_data=(test_x,test_y)
	)