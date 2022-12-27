import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.saving.save import load_model
from matplotlib import pyplot as plt
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



import csv
import time
from os import path

class ImdbService(object):
    def __init__(self):
        pass

    def asas(self) -> '':
        result = 1
        if result == 0:
            resp = '부정'
        elif result == 1:
            resp = '긍정'
        print(f"이 리뷰는 긍정? 부정?: {resp}")
        return resp

    def service_model(self) -> '':
        i = 3
        model = load_model(r"C:\Users\AIA\PycharmProjects\django-react-AIA\djangoProject\imdb\best-embedding-model.h5")
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        predictions = model.predict(test_images)
        predictions_array, true_label, img = predictions[i], test_labels[i], test_images[i]

        result = np.argmax(predictions_array)
        print(f"예측한 답 : {result}")

        if result == 0:
            resp = '부정'
        elif result == 1:
            resp = '긍정'
        print(f"이 리뷰는 긍정? 부정?: {resp}")
        return resp

        # predict_class = model.predict_classes(test_images)
        # print(predict_class[0:10].T)

    @staticmethod
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = \
            predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10),
                           predictions_array,
                           color='#777777')
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

#
# if __name__ == '__main__':
#     imdb = ImdbService()
#     imdb.asas()
# ###############################################################################





class NaverMovieService(object):
    def __init__(self):
        global url, driver, file_name, encoding
        url = 'https://movie.naver.com/movie/point/af/list.naver?&page='
        driver = webdriver.Chrome(r'C:\Users\AIA\PycharmProjects\django-react-AIA\djangoProject\webcrawler\chromedriver.exe')
        file_name = r'C:\Users\AIA\PycharmProjects\django-react-AIA\djangoProject\imdb\naver_movie_review_corpus.csv'
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






iris_menu = ["Exit",  # 0
               "Learning",  # 1
               ]

iris_lambda = {
    "1": lambda x: x.crawling(),
    }

if __name__ == '__main__':
    ir = NaverMovieService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](ir)
            except:

                print("Didn't catch error message")

