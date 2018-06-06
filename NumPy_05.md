
# NumPy

참고 : 파이썬 데이터 사이언스 핸드북 89p-98p


```python
import numpy as np
```

### 팬시 인덱싱

> 인덱스로 배열에 접근


```python
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
x
```




    array([51, 92, 14, 71, 60, 20, 82, 86, 74, 74])




```python
[x[3], x[7], x[2]]
```




    [71, 86, 14]




```python
ind = [3,7,4]
x[ind]
```




    array([71, 86, 60])




```python
ind = np.array([[3,7],[4,5]])
x[ind]
```




    array([[71, 86],
           [60, 20]])




```python
X = np.arange(12).reshape((3,4))
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
row = np.array([0,1,2])
col = np.array([2,1,3])
X[row, col]
```




    array([ 2,  5, 11])




```python
X[row[:,np.newaxis], col]
```




    array([[ 2,  1,  3],
           [ 6,  5,  7],
           [10,  9, 11]])




```python
row[:, np.newaxis] * col
```




    array([[0, 0, 0],
           [2, 1, 3],
           [4, 2, 6]])



### 결합 인덱싱


```python
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
X[2, [2,0,1]]
```




    array([10,  8,  9])




```python
X[1:, [2,0,1]]
```




    array([[ 6,  4,  5],
           [10,  8,  9]])




```python
mask = np.array([1,0,1,0], dtype=bool)
X[row[:, np.newaxis], mask]
```




    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]])



### 예제 : 임의의 점 선택하기


```python
mean = [0,0]
cov = [[1,2],[2,5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape
```




    (100, 2)




```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:,0], X[:, 1])
```




    <matplotlib.collections.PathCollection at 0x26e69cf2710>




![png](output_20_1.png)



```python
indices = np.random.choice(X.shape[0], 20, replace=False)
indices
```




    array([79, 65, 17, 32,  9, 58,  8, 35, 40, 59,  2, 94,  5, 75, 20, 16, 46,
           28, 86, 69])




```python
selection = X[indices]
selection.shape
```




    (20, 2)




```python
plt.scatter(X[:,0], X[:,1], alpha=.3)
plt.scatter(selection[:,0], selection[:,1], facecolor='none', edgecolor='blue', s=200)
```




    <matplotlib.collections.PathCollection at 0x26e69d96d68>




![png](output_23_1.png)


### 팬시 인덱싱으로 값 변경하기


```python
x = np.arange(10)
i = np.array([2,1,8,4])
x[i] = 99
x
```




    array([ 0, 99, 99,  3, 99,  5,  6,  7, 99,  9])




```python
x[i] -= 10
x
```




    array([ 0, 89, 89,  3, 89,  5,  6,  7, 89,  9])




```python
x = np.zeros(10)
x[[0,0]] = [4,6]
x
# x[0]에 4가 할당된 후, 6이 다시 할당된다.
```




    array([6., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
i = [2,3,3,4,4,4]
x[i] += 1
x
# x[3]=2, x[4]=3 이 나오지 않는다.
```




    array([6., 0., 1., 1., 1., 0., 0., 0., 0., 0.])




```python
x = np.zeros(10)
np.add.at(x, i, 1)
x
```




    array([0., 0., 1., 2., 3., 0., 0., 0., 0., 0.])



### 예졔 : 데이터 구간화


```python
np.random.seed(42)
x = np.random.randn(100)

# 히스토그램 계산
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# 각 x에 대한 적절한 구간 찾기
i = np.searchsorted(bins, x)

np.add.at(counts, i, 1)

plt.plot(bins, counts, linestyle='steps')
```




    [<matplotlib.lines.Line2D at 0x26e69e0f0b8>]




![png](output_31_1.png)



```python
plt.hist(x, bins, histtype='step') # 더 간편하고 빠르다.
```




    (array([ 0.,  0.,  0.,  0.,  1.,  3.,  7.,  9., 23., 22., 17., 10.,  7.,
             1.,  0.,  0.,  0.,  0.,  0.]),
     array([-5.        , -4.47368421, -3.94736842, -3.42105263, -2.89473684,
            -2.36842105, -1.84210526, -1.31578947, -0.78947368, -0.26315789,
             0.26315789,  0.78947368,  1.31578947,  1.84210526,  2.36842105,
             2.89473684,  3.42105263,  3.94736842,  4.47368421,  5.        ]),
     <a list of 1 Patch objects>)




![png](output_32_1.png)

