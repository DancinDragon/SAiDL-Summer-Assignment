from tqdm import tqdm
import numpy as np
from keras.models import Input, Model,Sequential
from keras.layers import Dense,Dropout,Concatenate,Input
import tensorflow as tf

##labels = ["[0,neutral]","[1,entailment]",,"[2,contradiction]"]
embeddings = np.load("Sentence_Embeddings.npy",allow_pickle = True)
encoded_labels = np.load("encoded_labels.npy",allow_pickle = True)
print("Here")

test_premise = [embeddings[i] for i in range(9843*2) if i%2==0]
test_premise = np.array(test_premise)

test_hypothesis =[embeddings[i] for i in range(9843*2) if i%2==1]
test_hypothesis = np.array(test_hypothesis)

test_labels = encoded_labels[:9843]
print("Here")
train_premise = [embeddings[i] for i in range(9843*2,len(embeddings)) if i%2==0]
train_premise = np.array(train_premise)

train_hypothesis =[embeddings[i] for i in range(9843*2,len(embeddings)) if i%2==1]
train_hypothesis = np.array(train_hypothesis)

train_labels = encoded_labels[9843:len(embeddings)]
print("Here")

Premise = Input(shape=(300,))
Hypothesis = Input(shape=(300,))
Merged = Concatenate(axis=1)([Premise,Hypothesis])
x = Dense(300,activation="tanh",use_bias=True)(Merged)
for i in range(2):
	x = Dense(300,activation="tanh",use_bias=True ,kernel_regularizer='l2')(x)
	Dropout(0.3)

x = Dense(3,activation="softmax",use_bias=True)(x)
model = Model(inputs=[Premise,Hypothesis],outputs=x)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics="accuracy")
#model.summary()
model.fit(
	x = [train_premise,train_hypothesis],
	y = train_labels,
	batch_size=64,
	epochs=30,
	validation_data=([test_premise,test_hypothesis],test_labels)
	)
model.save("My_model")
