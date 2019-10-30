from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import re

"""
data preparation and processing
"""
path = './datasets/nottingham_database/nottingham_parsed.txt'

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    
def tracks_to_array(path):
    """
    returns tracks from file into array
    """
    f = open(path, 'r')
    track_list = f.readlines()
    track_list = ''.join(track_list)
    # replace triple or more new lines by double new lines for separation:
    track_list = re.sub('(\n){4,}', '\n\n', track_list)
    track_list = track_list.split('\n\n')
    for i in range(len(track_list)):
        track_list[i] = list(track_list[i])
        track_list[i][0] = track_list[i][0].replace('\n', '')
        track_list[i] = ''.join(track_list[i])

    return track_list


def get_track_data(path, tokenized=True):
    """
    returns tracks in array of tokens
    tokenized set to True if file data is already separated into tokens
    """
    tracks = tracks_to_array(path)
    track_data = []
    print("Getting tokenized track data...")
    for t in tracks:
        if (tokenized):
            data = t.split()
        else:
            data = list(t)
        data.append('</s>')
        track_data.extend(array(data))
    return track_data


def get_encoded_data(data):
    """
    returns data in one hot encoding
    """
    print("One-Hot encoding data...")

    values = array(data)
    
    # integer encode 
    integer_encoded = label_encoder.fit_transform(values)

    # binary encode
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # return encoded data as well as vocab size
    return onehot_encoded
    #, len(onehot_encoded[0])
    
def integer_encode(data):
    """
    returns dataset encoded into integers
    """
    values = array(data)
    
    integer_encoded = label_encoder.fit_transform(values)
    return integer_encoded
    
def one_hot_from_vocab(data, vocab_count):
    onehot_encoded = np.zeros((np.multiply(*data.shape), vocab_count), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    onehot_encoded[np.arange(onehot_encoded.shape[0]), data.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    onehot_encoded = onehot_encoded.reshape((*data.shape, vocab_count))
    
    return onehot_encoded
    
    
def get_text_from_onehot(onehot_arr):
    """
    returns onehot encoding as text
    """
    inverted = onehot_encoder.inverse_transform(onehot_arr)
    inverted = inverted.ravel()
    return label_encoder.inverse_transform(inverted) 
    
def get_vocab(data):
    """
    returns vocabulary used in text
    """
    return tuple(set(data))

# onehot = get_encoded_data(track_data)

def file_to_onehot(path, tokenized=True):
    track_data = get_track_data(path, tokenized)
    onehot = get_encoded_data(track_data)
    return torch.from_numpy(onehot)