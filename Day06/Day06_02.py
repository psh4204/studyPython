from sklearn.neighbors import KNeighborsClassifier # KNN 알고리즘 임포트
from sklearn import metrics, utils
import pandas as pd

## 기학습 모델 불러오기 ## (pre-trained model)
import joblib
clf = joblib.load("iris_knn3_150.dmp")

# 활용
# --. 데이터 예측
myIris = [4.1, 3.3, 1.5, 0.2]
result = clf.predict([myIris])
print('이 꽃은 -->', result, '입니다.')