import numpy as np
from numpy import array
import random
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array

from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, Concatenate, Dropout, BatchNormalization, Input
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model

def mergeModel(embedding_size,vocab_len,max_len):
    image_model = Sequential()
    image_model.add(Input(shape=(2048,)))
    image_model.add(BatchNormalization(axis=-1))
    image_model.add(Dropout(0.4))
    image_model.add(Dense(embedding_size, activation='relu'))
    image_model.add(RepeatVector(max_len))
    image_model.summary()


    language_model = Sequential()
    language_model.add(Embedding(input_dim=vocab_len, output_dim=embedding_size, input_length=max_len,mask_zero=True))
    language_model.add(Dropout(0.4))
    language_model.add(LSTM(256,return_sequences=True,dropout=0.35,recurrent_dropout=0.35))
    language_model.add(TimeDistributed(Dense(embedding_size)))
    language_model.summary()


    conca = Concatenate()([image_model.output,language_model.output])
    x = BatchNormalization(axis=-1)(conca)

    x = Bidirectional(LSTM(256, return_sequences=False,dropout=0.35,recurrent_dropout=0.35))(x)

    out = Dense(vocab_len,activation='softmax')(x)


    model = Model(inputs=[image_model.input,language_model.input], outputs = out)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def NIC(embedding_size,vocab_len,max_len):
    image_model = Sequential()
    image_model.add(Input(shape=(2048,)))
    image_model.add(BatchNormalization(axis=-1))
    image_model.add(Dropout(0.4))
    image_model.add(Dense(embedding_size, activation='relu'))
    image_model.add(RepeatVector(1))
    image_model.summary()

    language_model = Sequential()
    language_model.add(Input(shape=(max_len,)))
    language_model.add(Embedding(input_dim=vocab_len, output_dim=embedding_size,mask_zero=True, input_length=max_len))
    language_model.add(Dropout(0.4))
    language_model.summary()
    inp = image_model.output

    a0 = Input(shape=(256,))
    c0 = Input(shape=(256,))

    b1 = BatchNormalization()(inp)
    b2 = BatchNormalization()(language_model.output)
    LSTMlayer= LSTM(256,return_sequences=False,return_state=True,dropout=0.35,recurrent_dropout=0.35)

    _ , a ,c = LSTMlayer(b1,initial_state=[a0,c0])

    A, _ ,_ = LSTMlayer(b2,initial_state = [a,c]) 
    sequence_output = Dense(units=vocab_len,activation='softmax')(A)

    model = Model(inputs=[image_model.input, language_model.input,a0,c0],
                  outputs=sequence_output)
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = NIC(300, 2530,34)
    model = mergeModel(300, 2530,34)
