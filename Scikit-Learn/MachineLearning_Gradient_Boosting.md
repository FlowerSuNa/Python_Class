
# Gradient Boosting

참고 : 파이썬 라이브러리를 활용한 머신러닝

> 여러 개의 결정 트리를 묶어 강력한 모델을 만드는 또 다른 앙상블 방법이다. <br>
> 이전 트리의 오차를 보완하는 방식으로 순차적으로 트리를 만든다. <br>
> 무작위성이 없다. <br>
> 보통 하나에서 다섯 정도의 깊지 않은 트리를 사용하므로 메모리를 적데 사용하고 예측도 빠르다. <br>
> 각각의 트리는 데이터의 일부에 대해서만 예측을 잘 수행할 수 있어 트리가 많이 추가될수록 성능이 좋아진다. <br>


### parameter
* learning_rate
* n_estimators
* max_depth
* max_leaf_nodes

## 예제


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
```


```python
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
```


```python
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print('train score : {}'.format(gbrt.score(X_train, y_train)))
print('test score : {}'.format(gbrt.score(X_test, y_test)))
```

    train score : 1.0
    test score : 0.958041958041958
    


```python
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print('train score : {}'.format(gbrt.score(X_train, y_train)))
print('test score : {}'.format(gbrt.score(X_test, y_test)))
```

    train score : 0.9953051643192489
    test score : 0.965034965034965
    


```python
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print('train score : {}'.format(gbrt.score(X_train, y_train)))
print('test score : {}'.format(gbrt.score(X_test, y_test)))
```

    train score : 0.9953051643192489
    test score : 0.9440559440559441
    


```python
print('feature importances : \n{}'.format(gbrt.feature_importances_))
```

    feature importances : 
    [3.19210187e-04 1.31157083e-02 4.09707105e-04 1.96211563e-04
     2.01632631e-04 0.00000000e+00 2.38317511e-04 1.23686732e-01
     1.41300556e-03 2.04207575e-04 2.41017087e-03 9.30407338e-03
     3.27208269e-04 1.00820932e-02 7.17734012e-04 8.09452247e-04
     9.86421529e-04 7.80688104e-03 0.00000000e+00 3.87634095e-04
     1.94274524e-01 3.52613526e-02 3.89890961e-01 1.27953206e-02
     5.62850638e-03 6.85201393e-04 1.72381988e-02 1.64531824e-01
     7.07771004e-03 0.00000000e+00]
    
