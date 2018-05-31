
# NumPy

참고 : 파이썬 데이터 사이언스 핸드북 80p-


```python
import numpy as np
```

### 비교 연산자


```python
x = np.array([1,2,3,4,5])
```


```python
x < 3
```




    array([ True,  True, False, False, False])




```python
x > 3
```




    array([False, False, False,  True,  True])




```python
x <= 3
```




    array([ True,  True,  True, False, False])




```python
x >= 3
```




    array([False, False,  True,  True,  True])




```python
x != 3
```




    array([ True,  True, False,  True,  True])




```python
(2 * x) == (x ** 2)
```




    array([False,  True, False, False, False])




```python
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3,4))
x
```




    array([[5, 0, 3, 3],
           [7, 9, 3, 5],
           [2, 4, 7, 6]])




```python
x < 6
```




    array([[ True,  True,  True,  True],
           [False, False,  True,  True],
           [ True,  True, False, False]])



### 부울 배열 작업

요소 개수 세기

> .count_nonzero() <br>
> .sum() <br>
> .any() <br>
> .all()


```python
np.count_nonzero(x < 6)
```




    8




```python
np.sum(x < 6)
```




    8




```python
## 각 행에 6보다 작은 값의 개수
np.sum(x < 6, axis=1)
```




    array([4, 2, 2])




```python
## 8보다 큰 값이 하나라도 있는지
np.any(x > 8)
```




    True




```python
## 모든 값이 10보다 작은지
np.all(x < 10)
```




    True




```python
## 각 행의 모든 값이 8보다 작은지
np.all(x < 8, axis=1)
```




    array([ True, False,  True])



부울 연산자

> & <br>
> | <br>
> ^ <br>
> ~ <br>


```python
np.sum((x > 3) & (x < 8))
```




    6




```python
np.sum((x < 3) | (x > 8))
```




    3



### 마스킹 연산

> 배열에서 조건에 맞는 값들을 선택할 때 부울 배열을 인덱스로 사용하는 것


```python
x < 5
```




    array([[False,  True,  True,  True],
           [False, False,  True, False],
           [ True,  True, False, False]])




```python
x[x < 5]
```




    array([0, 3, 3, 3, 2, 4])


