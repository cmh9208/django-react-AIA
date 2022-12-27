import os


import pandas as pd
import tensorflow as tf
from keras import Sequential
from tensorflow import keras
from keras.layers import Dense

import keras.datasets.imdb
import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class ImdbModel(object):
    def __init__(self):
        global train_input, val_input, train_target, val_target, train_seq, val_seq, train_oh, val_oh
        pass

    def create_model(self):
        (train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)

        print(train_input.shape, test_input.shape)
        print(len(train_input[0]))
        print(len(train_input[1]))
        print(train_input[0])

        print(train_target[:20])

        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=0.2, random_state=42
        )

        lengths = np.array([len(x) for x in train_input])
        print(np.mean(lengths), np.median(lengths))

        plt.hist(lengths)
        plt.xlabel('length')
        plt.ylabel('frequency')
        plt.show()

        train_seq = pad_sequences(train_input, maxlen=100)
        print(train_seq.shape)
        print(train_seq[0])
        print(train_input[0][-10:])
        print(train_seq[5])

        val_seq = pad_sequences(val_input, maxlen=100)

        ##########################################################################
        model = keras.Sequential()
        model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        train_oh = keras.utils.to_categorical(train_seq)
        print(train_oh.shape)
        print(train_oh[0][0][:12])
        print(np.sum(train_oh[0][0]))

        val_oh = keras.utils.to_categorical(val_seq)
        print(val_oh.shape)

        model.summary()

        # 순환 신경망 만들기
        model = keras.Sequential()
        model.add(keras.layers.SimpleRNN(8, input_shape=(
        100, 500)))  # 위에서 500개의 단어지정 = 훈련 데이터에 포함될 수 있는 정수값 0(패딩)~499 = 원핫코딩으로 표현하려면 배열 길이가 500이어야함
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # 원-핫코딩
        train_oh = keras.utils.to_categorical(train_seq)

        print(train_oh.shape)

        print(train_oh[0][0][:12])  # 첫 번째 샘플의 첫 번째 토큰 10이 잘 코딩 되었는지 확인
        print(np.sum(train_oh[0][0]))  # sum() 함수로 모든 원소의 값을 더해봄 = 1이나옴(500개중 한개만 1이라는 뜻)

        val_oh = keras.utils.to_categorical(val_seq)  # 검증 데이터도 원핫 인코딩

        model.summary()

        # 순환 신경망 훈련
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        history = model.fit(train_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()

        # Epoch 48 / 100
        # 313 / 313[ == == == == == == == == == == == == == == ==] - 33s 104ms / step - loss: 0.3968 - accuracy: 0.8292 - val_loss: 0.4545 - val_accuracy: 0.7886
        # file_name = os.path.join(os.path.abspath("save"), "imdb_model.h5")
        # print(f"저장경로: {file_name}")
        # history.save(file_name)

        print(train_seq.nbytes, train_oh.nbytes)  # nbytes속성으로 크기확인(원핫 코딩은 토큰마다 500으로 늘림=데이터 500배커짐=비효율)

        # 단어 임베딩 (원핫코딩 단점 보완)
        model2 = keras.Sequential()
        model2.add(
            keras.layers.Embedding(500, 16, input_length=100))  # Embedding(어휘사전 크기(500), 임베딩 벡터 크기(100), 샘플길이(100))
        model2.add(keras.layers.SimpleRNN(8))
        model2.add(keras.layers.Dense(1, activation='sigmoid'))

        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5', save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        history = model2.fit(train_seq, train_target, epochs=100, batch_size=64, validation_data=(val_seq, val_target),
                             callbacks=[checkpoint_cb, early_stopping_cb])

        # Epoch 25 / 100
        # 313 / 313[ == == == == == == == == == == == == == == ==] - 38s 120ms / step - loss: 0.4076 - accuracy: 0.8210 - val_loss: 0.4611 - val_accuracy: 0.7828



        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()








iris_menu = ["Exit",  # 0
               "Learning",  # 1
               ]

iris_lambda = {
    "1": lambda x: x.create_model(),
    }

if __name__ == '__main__':
    fs = ImdbModel()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](fs)
            except:
                print("Didn't catch error message")