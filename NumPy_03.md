
# NumPy

참고 : 파이썬 데이터 사이언스 핸드북 73p-80p


```python
import numpy as np
```

### 브로드캐스팅
> 다른 크기의 배열에 이항 유니버설 함수를 적용하기 위한 규칙의 집합


```python
a = np.array([0,1,2])
b = np.array([5,5,5])
print(a + b)
print(a + 5)
```

    [5 6 7]
    [5 6 7]
    


```python
M = np.ones((3,3))
print(M)
print(M + a)
```

    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    [[1. 2. 3.]
     [1. 2. 3.]
     [1. 2. 3.]]
    


```python
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a + b
```




    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])



### 브로드캐스팅 규칙
> 규칙1 : 두 배열의 차원 수가 다르면 더 작은 수의 차원을 가진 배열 형상의 앞쪽을 1로 채운다. <br>
> 규칙2 : 두 배열의 형상이 어떤 차원에서도 일치하지 않는다면 해당 차원의 형상이 1인 배열이 다른 형상과 일치하도록 늘어난다. <br>
> 규칙3 : 임의의 차원에서 크기가 일치하지 않고 1도 아니라면 오류가 발생한다.


```python
M = np.ones((2,3))
a = np.arange(3)
M + a
```




    array([[1., 2., 3.],
           [1., 2., 3.]])




```python
a = np.arange(3).reshape((3,1))
b = np.arange(3)
a + b
```




    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])




```python
M = np.ones((3,2))
a = np.arange(3)
M + a   # 오류
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-8-08757f1f0bf2> in <module>()
          1 M = np.ones((3,2))
          2 a = np.arange(3)
    ----> 3 M + a
    

    ValueError: operands could not be broadcast together with shapes (3,2) (3,) 



```python
M + a[:,np.newaxis]
```




    array([[1., 1.],
           [2., 2.],
           [3., 3.]])



### 실전 브로드캐스팅

배열을 중앙 정렬하기


```python
X = np.random.random((10,3))
Xmean = X.mean(axis=0)
Xmean
```




    array([0.44854406, 0.60067361, 0.48764059])




```python
X_centered = X - Xmean
X_centered.mean(axis=0)
```




    array([-1.11022302e-17,  1.11022302e-17, -9.99200722e-17])



2차원 함수 플로팅하기


```python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0,5,0,5], cmap='viridis')
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x1f8a9d7ee48>




![png](output_18_1.png)

