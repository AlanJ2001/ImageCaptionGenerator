#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import cv2 
import os
from glob import glob
from keras.applications import ResNet50
import copy
import sys
from keras_preprocessing.sequence import pad_sequences
import time
from tensorflow.keras.utils import to_categorical
import itertools
import pandas as pd
import string
from keras.models import Model


# In[13]:


from keras.applications import ResNet50

def resnet():
    resnet_model = ResNet50(include_top=True)
    resnet_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    return resnet_model


# In[3]:


# preprocess the images
# generate a dictionary of image filename -> feature vector
# start_index is inclusive

def generate_feature_vectors(num_of_images, images_path, model):
    img_feature_vectors = {}
    images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f)) and f.endswith('.jpg')]
    
    count = 0
    for item in images[:num_of_images]:
        img = cv2.imread(images_path+item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.reshape(1, 224, 224, 3)

        feature_vector = model.predict(img, verbose=0).reshape(2048,)

        img_feature_vectors[item] = feature_vector

        count += 1

        if (count%50==0):
            print(count)

    return img_feature_vectors


# In[4]:


# makes a string lowercase, prepends it with the string 'sos' and appends with 'eos'
def process_string(s):
    s = s.lower()
    s = 'sos ' + s + ' eos'
    return s


# In[5]:



def generate_captions_dict(captions_path, img_feature_vectors):
    df = pd.read_csv(captions_path, delimiter='|')
    df.columns = ['image_name', 'comment_number', 'comment']
    del df['comment_number']
    df['comment'][19999] = ' A dog runs across the grass .'
    captions_dict = {}
    
    for index, row in df.iterrows():
        filename = row['image_name']
        caption = process_string(row['comment'])
        if filename in img_feature_vectors:
            if filename not in captions_dict:
                captions_dict[filename] = [caption]
            else:
                captions_dict[filename].append(caption)
    return captions_dict


# In[6]:


def create_vocab(captions_dict, n, train):
    vocab_freq = {}
    vocab_dict = {}
    for key, value in captions_dict.items():
        if key not in train:
            continue
        for caption in value:
            caption_as_list = caption.split()
            for word in caption_as_list:
                if word not in vocab_freq:
                    vocab_freq[word] = 1
                else:
                    vocab_freq[word] = vocab_freq[word]+1
    words_to_keep = [w for w in vocab_freq if vocab_freq[w] >= n]
    
    count = 1
    for word in words_to_keep:
        if word not in vocab_dict:
            vocab_dict[word] = count
            count+=1
    return vocab_dict


# In[7]:


def encode_string(s, vocab):
    s_list = s.split()
    encoded_string = []
    for word in s_list:
        if word in vocab:
            encoded_string.append(vocab[word])
    return encoded_string


# In[8]:


def encode_captions_dict(captions_dict, vocab):
    captions_dict_encoded = copy.deepcopy(captions_dict)

    for filename, captions in captions_dict_encoded.items():
        for i, caption in enumerate(captions):
            captions[i] = encode_string(caption, vocab)
    return captions_dict_encoded


# In[ ]:


if __name__ == "__main__":
    images_path = 'Flickr_Data/Images/'
    captions_path = "results.csv"
    max_len = 40
    
    resnet_model = resnet()
    img_feature_vectors = generate_feature_vectors(8091, images_path, resnet_model)
    images = list(img_feature_vectors.keys())
    train = set(images[:28605])
    test = set(images[28605:31783])
    captions_dict = generate_captions_dict(captions_path, img_feature_vectors)
    vocab = create_vocab(captions_dict, 10, train)
    captions_dict_encoded = encode_captions_dict(captions_dict, vocab)
    
    np.save('data/img_feature_vectors.npy', img_feature_vectors)
    np.save('data/captions_dict.npy', captions_dict)
    np.save('data/captions_dict_encoded.npy', captions_dict_encoded)
    np.save('data/vocab.npy', vocab)
    np.save("data/train.npy", train)
    np.save("data/test.npy", test)


# In[ ]:




