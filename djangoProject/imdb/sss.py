import os


import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.saving.save import load_model
from tensorflow import keras
from keras.layers import Dense

import keras.datasets.imdb
import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def ddd():
    model = load_model(r"C:\Users\최민호\PycharmProjects\django-react-AIA\djangoProject\imdb\best-embedding-model.h5")
    (train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)
    print(test_input)
if __name__ == '__main__':
    ddd()