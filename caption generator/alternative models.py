
# alternative model architectures and configurations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Embedding,
                          Input, Lambda, add)
from keras import regularizers

# merge architecture 
def create_model():
    vocab_size = 5185+1
    max_length = 40
    unit_size = 512
    embedding_dim = 512

    # process image
    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.5)(inputs1)
    img2 = Dense(unit_size, activation='relu')(img1)

    # process caption
    inputs2 = Input(shape=(max_length,))
    text1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    text2 = Dropout(0.5)(text1)
    text3 = LSTM(unit_size)(text2)

    # merge outputs
    decoder1 = add([img2, text3])
    decoder2 = Dense(unit_size, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# inject architecture with L2 regularisation 
def create_model():
    vocab_size = 5185+1
    max_length = 40
    unit_size = 512
    reg = 0.0001

    # image feature extractor model
    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.6)(inputs1)
    img2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg))(img1)
    img3 = BatchNormalization()(img2)
    img4 = Lambda(lambda x : K.expand_dims(x, axis=1))(img3)

    # partial caption sequence model
    inputs2 = Input(shape=(max_length,))
    text1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
    text2 = Dropout(0.6)(text1)  

    LSTMLayer = LSTM(512, return_state = True, dropout=0.5)

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, b, c = LSTMLayer(img4, initial_state = [a0, c0])

    A,_,_ = LSTMLayer(text2, initial_state=[b,c])

    outputs = Dense(vocab_size, activation='softmax', kernel_regularizer = regularizers.l2(reg), bias_regularizer = regularizers.l2(reg))(A)

    # merge the two input models
    model = Model(inputs=[inputs1, inputs2, a0, c0], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])