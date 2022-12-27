import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.saving.save import load_model
from matplotlib import pyplot as plt

# from sentence_transformers.models import tokenizer

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import os
import csv
import os.path

import pandas as pd
from selenium import webdriver

from webcrawler.models import ScrapVO
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
import csv
import time
from os import path

### embedding ###
class ImdbService(object):
    def __init__(self):
        pass
    def embedding_service_model(self) -> '':
        # i = 53
        model = load_model(r"C:\Users\최민호\PycharmProjects\django-react-AIA\djangoProject\imdb\best-embedding-model.h5")
        (train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)
        test_seq = pad_sequences(test_input, maxlen=100) # test_input을 임베딩 방식 fit에 쓰인 train_input과 같이 만들어줌
        predictions = model.predict(test_seq)

        # result = np.argmax(predictions[i]) # 소수점으로 나오는 답을 정수로
        j = 0
        for i in range(20):
            result = predictions[j]
            print(f'{j}번 리뷰는')
            j += 1
            if result > 0.5:
                print(f'{result * 100} 확률로 긍정 리뷰 입니다.')
            else:
                print(f'{(1 - result) * 100} 확률로 부정 리뷰 입니다.')

    ### simplernn ###
    def simplernn_service_model(self) -> '':
        # i = 53
        model = load_model(
            r"C:\Users\최민호\PycharmProjects\django-react-AIA\djangoProject\imdb\best-simplernn-model.h5")
        (train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)
        test_seq = pad_sequences(test_input, maxlen=100)  # test_input을 임베딩 방식 fit에 쓰인 train_input과 같이 만들어줌
        test_oh = keras.utils.to_categorical(test_seq)
        predictions = model.predict(test_oh)

        # result = np.argmax(predictions[i]) # 소수점으로 나오는 답을 정수로
        j = 0
        for i in range(20):
            result = predictions[j]
            print(f'{j}번 리뷰는')
            j += 1
            if result > 0.5:
                print(f'{result * 100} 확률로 긍정 리뷰 입니다.')
            else:
                print(f'{(1 - result) * 100} 확률로 부정 리뷰 입니다.')


        # print(f"예측한 답 : {result}")
        #
        # if result == 0:
        #     resp = '부정'
        # elif result == 1:
        #     resp = '긍정'
        # print(f"해당 리뷰는 '{resp}'적인 리뷰 입니다.")
        # return resp

##########################################################################################################################
    # def service_model2(self) -> '':
    #     model = load_model(r"C:\Users\최민호\PycharmProjects\django-react-AIA\djangoProject\imdb\best-embedding-model.h5")
    #
    #
    #     test_sentence = '우와.. 진짜 완전 노잼이다'
    #     test_sentence = test_sentence.split(' ')
    #     test_sentences = []
    #     now_sentence = []
    #     for word in test_sentence:
    #         now_sentence.append(word)
    #         test_sentences.append(now_sentence[:])
    #
    #     test_X_1 = tokenizer.texts_to_sequences(test_sentences)
    #     test_X_1 = pad_sequences(test_X_1, padding='post', maxlen=25)
    #     predictions = model.predict(test_X_1)
    #     for idx, sentence in enumerate(test_sentences):
    #         print(sentence)
    #         print(predictions[idx])
################################################################################

class NaverMovieService(object):
    def __init__(self):
        global url, driver, file_name, encoding
        url = 'https://movie.naver.com/movie/point/af/list.naver?&page='
        driver = webdriver.Chrome(r'C:\Users\최민호\PycharmProjects\django-react-AIA\djangoProject\webcrawler\chromedriver.exe')
        file_name = r'C:\Users\최민호\PycharmProjects\django-react-AIA\djangoProject\imdb\naver_movie_review_corpus.csv'
        encoding = "UTF-8"

    def crawling(self):
        if not path.exists(file_name):
            review_data = []
            for page in range(1, 2):
                driver.get(url + str(page))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                all_tds = soup.find_all('td', attrs={'class', 'title'})
                for review in all_tds:
                    need_reviews_cnt = 1000
                    sentence = review.find("a", {"class": "report"}).get("onclick").split("', '")[2]
                    if sentence != "":  # 리뷰 내용이 비어있다면 데이터를 사용하지 않음
                        score = review.find("em").get_text()
                        review_data.append([sentence, int(score)])
            time.sleep(1)  # 다음 페이지를 조회하기 전 1초 시간 차를 두기
            with open(file_name, 'w', newline='', encoding=encoding) as f:
                wr = csv.writer(f)
                wr.writerows(review_data)
            driver.close()

        data = pd.read_csv(file_name, header=None)
        data.columns = ['review', 'score']
        result = [print(f"{i + 1}. {data['score'][i]}\n{data['review'][i]}\n") for i in range(len(data))]
        return result

imdb_menu = ["Exit",  # 0
               "embedding",  # 1
               "simplernn",  # 2
               ]

imdb_lambda = {
    "1": lambda x: x.embedding_service_model(),
    "2": lambda x: x.simplernn_service_model(),
    }

if __name__ == '__main__':
    imdb = ImdbService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(imdb_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                imdb_lambda[menu](imdb)
            except:
                print("Didn't catch error message")