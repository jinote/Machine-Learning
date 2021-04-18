**Decision Tree**

- overfitting될 가능성이 높음.
- 가지치기를 통해 트리의 최대 높이를 설정해 줄 수 있지만 이로써는 overfitting을 충분히 해결해줄 수 없음.
- 따라서 더 일반화된 트리 방법인 Ramdom Forest를 사용함.

**Random Forest**

- Ensemble(앙상블) 머신러닝 모델.
- 여러 개의 decision tree를 형성하고 새로운 데이터 포인트를 각 트리에 통과시켜 각 트리가 분류한 결과에서 투표를 실시해 최다 득표를 가진 최종 분류 결과를 선택함
- Random Forest 또한 overfitting이 될 수 있지만, 여러 개의 트리를 생성하여 overfitting이 큰 영향을 미치지 못하도록 예방함

[https://eunsukimme.github.io/ml/2019/11/26/Random-Forest/](https://eunsukimme.github.io/ml/2019/11/26/Random-Forest/)

**[Datacamp] Random Forest  (RF) 정리**

- Bagging
    - Base estimator: Decision Tree, Logistic Regression, Neural Net, etc.
    - Each estimator is trained on a distinct bootstrap sample of the training set
    - Estimators use all features for training and prediction
- Further Diversity with Random Forests
    - Base estimator: Decision Tree
    - Each estimator is trained on a different bootstrap sample having the same size as the training set
    - RF - introduces further randomization in the training of individual trees
    - d features are sampled at each node without replacement (d < total number of features)

**Random Forests**

1. Classification
- Aggregates predictions by majority voting
- RandomForestClassifier in scikit-learn

2. Regression

- Aggregates predictions through averaging
- RamdomForestRegressor in scikit-learn

```python
# basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# set seed for reproducibility
SEED = 1

# split dataset into 70% train / 30% valid
trainx, validx, trainy, validy = train_test_split(X,y, test_size = 0.3, random_state = SEED)

# instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators = 400, min_samples_leaf = 0.12, random_state = SEED)

# fit 'rf' to the training set
rf.fit(trainx, trainy)
predy = rf.predict(validx)

# evaluate the test set RMSE
rmse_test = MSE(testy, predy) ** (1/2)

# print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```

**Feature importance**

- Tree-based methods: enable measuring the importance of each feature in prediction
- In sklearn:
    - how much the tree nodes use a particular feature (weighted average) to reduce impurity
    - accessed using the attribute feature_importance_

```python
import pandas as pd
import matplotlib.pyplot as plt

importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

sorted_importances_rf = importances_rf.sort_values()

sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()
```

**Train an RF regressor**

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 25, random_state = 2) # rf consisting of 25 trees

# fit rf to the training set
rf.fit(X_train, y_train)
```

```python
from sklearn.metrics import mean_squared_error as MSE

y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)

print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```

```python
importances = pd.Series
(data=rf.feature_importances_, index = X_train.columns)

importances_sorted=importances.sort_values()

importances_sorted.plot(kind = 'barh',
color = 'lightgreen')
plt.title('Features Importances')
plt.show()
```
