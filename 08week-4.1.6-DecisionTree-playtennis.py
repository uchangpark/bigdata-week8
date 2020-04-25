import os
print(os.getcwd())


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from IPython.display import Image

import pandas as pd
import numpy as np
import pydotplus
# ModuleNotFoundError: No module named 'pydotplus'
# (오류수정) (1) anaconda prompt를 관리자 권한으로 실행 -> 
#           (2) conda install pydotplus
import os

tennis_data = pd.read_csv('playtennis.csv')
tennis_data

# 범주형 변수 변경
tennis_data.Outlook = tennis_data.Outlook.replace('Sunny', 0)
tennis_data.Outlook = tennis_data.Outlook.replace('Overcast', 1)
tennis_data.Outlook = tennis_data.Outlook.replace('Rain', 2)
tennis_data.Temperature = tennis_data.Temperature.replace('Hot', 3)
tennis_data.Temperature = tennis_data.Temperature.replace('Mild', 4)
tennis_data.Temperature = tennis_data.Temperature.replace('Cool', 5)
tennis_data.Humidity = tennis_data.Humidity.replace('High', 6)
tennis_data.Humidity = tennis_data.Humidity.replace('Normal', 7)
tennis_data.Wind = tennis_data.Wind.replace('Weak', 8)
tennis_data.Wind = tennis_data.Wind.replace('Strong', 9)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('No', 10)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('Yes', 11)
tennis_data

# 입력변수, 목표변수 설정 'PlayTennis'
X = np.array(pd.DataFrame(tennis_data, columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']))
y = np.array(pd.DataFrame(tennis_data, columns = ['PlayTennis']))
print(X)
print(y)

# train, test 데이터 생성
X_train, X_test, y_train, y_test = train_test_split(X, y)
# train, test 비율은 default=0.25 로 랜덤하게 선택됨 => 아래 링크 참조.
# http://blog.naver.com/PostView.nhn?blogId=siniphia&logNo=221396370872
print(X_train)
print(X_test)
# 분석 시작
dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train) # 데이터 학습
dt_prediction = dt_clf.predict(X_test) # 예측값 계산
dt_prediction
print(confusion_matrix(y_test, dt_prediction)) # 분류평가표, 오차행렬
print(classification_report(y_test, dt_prediction)) # 결과
# check accuracy ? 

feature_names = tennis_data.columns.tolist() # ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
feature_names = feature_names[0:4] # ['Outlook', 'Temperature', 'Humidity', 'Wind']
target_name = np.array(['Play No', 'Play Yes'])
os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dt_dot_data = tree.export_graphviz(dt_clf, out_file = None, 
                                   feature_names = feature_names,
                                   class_names = target_name,
                                   filled = True, rounded = True,
                                   special_characters = True)
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
Image(dt_graph.create_png())
# InvocationException: GraphViz's executables not found
# (오류수정) http://blog.naver.com/PostView.nhn?blogId=resumet&logNo=221450970851
# (1) Download and install graphviz-2.38.msi (use the newest version) 
#       from https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# (2) Set the path variable
#       (a) Control Panel > System and Security > System > Advanced System Settings > Environment Variables > Path > Edit
#    or (b) add 'C:\Program Files (x86)\Graphviz2.38\bin'

