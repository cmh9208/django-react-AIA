import random
import string
import pandas as pd
from sqlalchemy import create_engine
class BuserService(object):
    def __init__(self):
        pass
    def create_users(self):
        df = self.frame_create()
        engine = create_engine(
            "mysql+pymysql://root:root@localhost:3306/mydb",
            encoding='utf-8')
        df.to_sql(name='blog_busers',
                  if_exists='append',
                  con = engine,
                  index=False)
    def frame_create(self):
        n = 5
        email = ''
        golbang = '@google.com'
        password = 1
        nickname = 'NICK'
        data = []
        #email_list = list()
        for i in range(100):
            for i in range(n):
                email +=str(random.choice(string.ascii_letters))
            email += golbang
            data.append([email,nickname,password])
            email = ''
        df = pd.DataFrame(data,columns=['email','nickname','password'])
        #중복체크
        #df2 = pd.DataFrame([[1],[2],[1]] ,columns=['email'])
        #print(df2)
        #print(df2.duplicated(['email']))
        print(df.duplicated(['email']))
        return df
if __name__ == '__main__':
    s =BuserService()
    s.create_users() # 만듬 이미 돌리지마