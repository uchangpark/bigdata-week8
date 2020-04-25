import os
print(os.getcwd())


from sklearn.datasets import load_iris
#from sklearn import tree
#import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
#loading the iris dataset
iris = load_iris()

# to excel... by Uchang
df = pd.DataFrame(data=iris['data'], columns = iris['feature_names'])
df['target']=iris.target
df.to_excel('iris.xlsx', index=False)

#############################################################
### train/test data 80%:20% at data => 정확도 0.76? ##########
#############################################################
x_train = iris.data[:-30]
y_train = iris.target[:-30]
#test data 설정
x_test = iris.data[-30:] # test feature data  
y_test = iris.target[-30:] # test target data
print(y_train)
print(y_test)

#RandomForestClassifier libary를 import
from sklearn.ensemble import RandomForestClassifier # RandomForest
#tree 의 개수 Random Forest 분류 모듈 생성
rfc = RandomForestClassifier(n_estimators=10) 
rfc
rfc.fit(x_train, y_train)
#Test data를 입력해 target data를 예측 
prediction = rfc.predict(x_test)
#예측 결과 precision과 실제 test data의 target 을 비교 
print (prediction==y_test)
#Random forest 정확도 측정
rfc.score(x_test, y_test) # 정확도 ...

#from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print ("Accuracy is : ",accuracy_score(prediction, y_test))
print ("=======================================================")
print (classification_report(prediction, y_test))

# 각 속성의 중요도 출력 petal length?
for feature, imp in zip(iris.feature_names, rfc.feature_importances_):
    print(feature, imp)

#############################################################
### train/test data random split => 정확도 0.93? #############
#############################################################
from sklearn.model_selection import train_test_split
x = iris.data
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print (X_test)
print (Y_test)

clf = RandomForestClassifier(n_estimators=10) # Random Forest
clf.fit(X_train, Y_train)
prediction_1 = rfc.predict(X_test)
#print (prediction_1 == Y_test)
print ("Accuracy is : ",accuracy_score(prediction_1, Y_test))
print ("=======================================================")
print (classification_report(prediction_1, Y_test))
# 각 속성의 중요도 출력 petal length?
for feature, imp in zip(iris.feature_names, clf.feature_importances_):
    print(feature, imp)

#############################################################
### 트리 개수를 n_estimators=200, oob => 정확도 1.0? #########
#############################################################
# Initialize the model
clf_2 = RandomForestClassifier(n_estimators=200, # Number of trees
                               max_features=4,    # Num features considered
                               oob_score=True)    # Use OOB scoring*
clf_2.fit(X_train, Y_train)
prediction_2 = clf_2.predict(X_test)
print (prediction_2 == Y_test)
print ("Accuracy is : ",accuracy_score(prediction_2, Y_test))
print ("=======================================================")
print (classification_report(prediction_2, Y_test))

# 각 속성의 중요도 출력 petal length?
for feature, imp in zip(iris.feature_names, clf_2.feature_importances_):
    print(feature, imp)
    
