#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras_preprocessing.sequence import pad_sequences
from npy_append_array import NpyAppendArray
from tensorflow.keras.utils import to_categorical

# In[2]:


# batch size -> number of images will data be generated for at once

def generate_training_data(img_feature_vectors, images_set, captions_dict_encoded, max_len, vocab_len, batch_size, id_):
    
    img_feature_vectors = {k: v for k, v in img_feature_vectors.items() if k in images_set}
    
    X_file = f'X_{id_}.npy'
    y_in_file = f'y_in_{id_}.npy'
    y_out_file = f'y_out_{id_}.npy'
    filenames_file = f'filenames_{id_}.npy'
    
    X = []
    y_in = []
    y_out = []
    filenames = set()
    
    n = 0
    for filename, fv in img_feature_vectors.items():
        n+=1
        for caption in captions_dict_encoded[filename]:
            i = 0
            for word in caption:
                if i==0:
                    i+=1
                    continue
                y_in_item = [caption[:i]]
                y_in_item = pad_sequences(y_in_item, maxlen=max_len, truncating='post')[0]
                y_in.append(y_in_item)

                y_out_item = to_categorical([word], num_classes=vocab_len+1)[0]
                y_out.append(y_out_item)

                X.append(fv)
                
                filenames.add(filename)
                i+=1
        if n%batch_size==0 or n==len(img_feature_vectors):
            print(n)
            with NpyAppendArray(X_file) as npaa:
                npaa.append(np.array(X))
            with NpyAppendArray(y_in_file) as npaa:
                npaa.append(np.array(y_in))
            with NpyAppendArray(y_out_file) as npaa:
                npaa.append(np.array(y_out))
            X = []
            y_in = []
            y_out = []
            if n == len(img_feature_vectors):
                break
    np.save(filenames_file, filenames)


# In[8]:


if __name__ == "__main__":
    img_feature_vectors = np.load("img_feature_vectors.npy", allow_pickle=True).item()
    captions_dict_encoded = np.load("captions_dict_encoded.npy", allow_pickle=True).item()
    vocab = np.load("vocab.npy", allow_pickle=True).item()
    train = np.load("train.npy", allow_pickle=True).item()
    test = np.load("test.npy", allow_pickle=True).item()
    
    
    vocab_len = len(vocab)
    max_len = 40
    batch_size = 1000
    generate_training_data(img_feature_vectors, train, captions_dict_encoded, max_len, vocab_len, batch_size, "train")
    generate_training_data(img_feature_vectors, test, captions_dict_encoded, max_len, vocab_len, batch_size, "test")


# In[ ]:




