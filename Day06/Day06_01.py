## 붓꽃 구별하기. 머신러닝 프로젝트
from sklearn.neighbors import KNeighborsClassifier # KNN 알고리즘 임포트
from sklearn import metrics, utils
import pandas as pd

# === 데이터준비 ===
# 0. 데이터를 준비한다. --> 통 150건_학습용과 테스트용
df = pd.read_csv('iris.csv')
df = utils.shuffle(df)
# 1. 학습용 80%, 테스트용 20%
dataLen = df.shape[0] # 데이터 개수
trainSize = int(dataLen * 0.8)
trainSize = dataLen - trainSize
train_data = df.iloc[0:trainSize, 0:-1] # 문제수 0~ 80%, 답지 0~ 전부
train_label = df.iloc[0:trainSize, [-1]] # 문제수 0~ 80%, 답은 1개
test_data = df.iloc[trainSize:, 0:-1] # 문제수 train~ 전부
test_label = df.iloc[trainSize:, [-1]] # 문제수 train~ 전부
# 2. 머신러닝 알고리즘을 선택 (KNN, SVN ... 딥러닝 알고리즘)
clf = KNeighborsClassifier(n_neighbors=3)
# 훈련시키기
# 3. 학습(훈련) 시키기 --> 오래걸림 (실제로는 AWS나 클라우드 써야 됨) --> 모델(Model)이 완성
clf.fit(train_data, train_label)
# 4. 모델의 정답률 구하기 (모의고사 시험)
results = clf.predict(test_data)
score = metrics.accuracy_score(results, test_label)
print("정답률 : %5.2f(score)" %(score))

# *** 모델 저장하기
import joblib
## 덤프파일 만들기.
joblib.dump(clf, "iris_knn3_150.dmp")

# 활용
# --. 데이터 예측
myIris = [4.1, 3.3, 1.5, 0.2]
result = clf.predict([myIris])
print('이 꽃은 -->', result, '입니다. 단,', score * 100, '%의 확률입니다.')