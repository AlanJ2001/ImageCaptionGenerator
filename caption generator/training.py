#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Embedding,
                          Input, Lambda)

# In[2]:


# model
def create_model():
    vocab_size = 5185+1
    max_length = 40
    unit_size = 512

    # image feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)
    fe4 = Lambda(lambda x : K.expand_dims(x, axis=1))(fe3)

    # partial caption sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)  

    LSTMLayer = LSTM(512, return_state = True, dropout=0.5)

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, b, c = LSTMLayer(fe4, initial_state = [a0, c0])

    A,_,_ = LSTMLayer(se2, initial_state=[b,c])

    outputs = Dense(vocab_size, activation='softmax')(A)

    # merge the two input models
    model = Model(inputs=[inputs1, inputs2, a0, c0], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[3]:


def gen(X_path, y_in_path, y_out_path, batch_size):
    for i in range(0, 2022727, batch_size):
        y_out = np.load(y_out_path, mmap_mode="r")[i:i+batch_size]
        X = np.load(X_path, mmap_mode="r")[i:i+batch_size]
        y_in = np.load(y_in_path, mmap_mode="r")[i:i+batch_size]
        
        X = np.array(X)
        y_in = np.array(y_in)
        y_out = np.array(y_out)
        
        yield [X, y_in, np.zeros(shape=(1024,512)), np.zeros(shape=(1024,512))], y_out
        del y_out
        del X
        del y_in


# In[4]:


def train(model, X_path, y_in_path, y_out_path, batch_size, start_epoch, end_epoch):
    total_num_of_samples = len(np.load(X_path, mmap_mode="r"))
    for i in range(start_epoch, end_epoch+1):
        g = gen(X_path, y_in_path, y_out_path, batch_size)
        print(f"epoch_{i}")
        model.fit(g, epochs=1, steps_per_epoch=total_num_of_samples//batch_size)
        model.save(f"image_caption_gen_epoch_{i}.h5")


# In[ ]:


if __name__ == "__main__":
    model = create_model()
    X_path = "/kaggle/input/30k-dataset/train/X_train.npy"
    y_in_path = "/kaggle/input/30k-dataset/train/y_in_train.npy"
    y_out_path = "/kaggle/input/30k-dataset/train/y_out_train.npy"
    batch_size = 1024
    start_epoch = 1
    end_epoch = 20
    train(model, X_path, y_in_path, y_out_path, batch_size, start_epoch, end_epoch )

