
# NumPy

참고 : 파이싼 데이터 사이언스 핸드북 38p-57p


```python
import numpy as np
```

### 배열만들기

> .array() <br>
> .zeros() <br>
> .ones() <br>
> .full() <br>
> .arange() <br>
> .linspace() <br>
> .random.random() <br>
> .random.normal() <br>
> .random.randint() <br>
> .eye() <br>
> .empty()


```python
np.array([1,2,3,4])
```




    array([1, 2, 3, 4])




```python
np.array([1,2,3,4], dtype='float32')
```




    array([1., 2., 3., 4.], dtype=float32)




```python
np.array([1,2,3.14,4])
```




    array([1.  , 2.  , 3.14, 4.  ])



다차원 배열


```python
np.array([range(i,i+3) for i in [2,4,6]])
```




    array([[2, 3, 4],
           [4, 5, 6],
           [6, 7, 8]])



0으로 채운 배열


```python
np.zeros(10, dtype=int)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



1로 채운 배열


```python
np.ones((3,5), dtype=float)
```




    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])



특정 값으로 채운 배열


```python
np.full((3,5), 3.14)
```




    array([[3.14, 3.14, 3.14, 3.14, 3.14],
           [3.14, 3.14, 3.14, 3.14, 3.14],
           [3.14, 3.14, 3.14, 3.14, 3.14]])



수열


```python
np.arange(0, 20, 2)
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])



0과 1사이 배열


```python
np.linspace(0, 1, 5)
```




    array([0.  , 0.25, 0.5 , 0.75, 1.  ])



0과 1사이 균등분포 배열


```python
np.random.random((3,3))
```




    array([[0.40899909, 0.70339937, 0.65975747],
           [0.77274556, 0.74018765, 0.78226009],
           [0.89331806, 0.36330785, 0.07512614]])



0과 1사이 정규분포 배열


```python
np.random.normal(0, 1, (3,3))
```




    array([[-1.35158696,  0.13502301, -0.01776949],
           [ 1.1638169 , -0.85403413, -0.9586457 ],
           [-0.21852499, -1.04592286, -0.73799129]])



정수 난수 배열


```python
np.random.randint(0, 10, (3,3))
```




    array([[7, 1, 3],
           [2, 6, 7],
           [9, 1, 5]])



단위 행렬


```python
np.eye(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])



초기화되지 않은 배열


```python
np.empty(3)
```




    array([1., 1., 1.])



### 배열 속성

> .ndim <br>
> .shape <br>
> .size <br>
> .dtype <br>
> .itemsize <br>
> .nbytes


```python
x = np.random.randint(10, size=(3,4,5))
```


```python
print('ndim : ', x.ndim)
print('shape : ', x.shape)
print('size : ', x.size)
```

    ndim :  3
    shape :  (3, 4, 5)
    size :  60
    


```python
print('dtype : ', x.dtype)
print('itemsize : ', x.itemsize, 'bytes')
print('nbytes : ', x.nbytes, 'bytes')
```

    dtype :  int32
    itemsize :  4 bytes
    nbytes :  240 bytes
    

### 배열 인덱싱


```python
x = np.random.randint(10, size=6)
x
```




    array([2, 1, 9, 9, 1, 9])




```python
x[0]
```




    2




```python
x[4]
```




    1




```python
x[-1]
```




    9




```python
x[-2]
```




    1




```python
x = np.random.randint(10, size=(3,4))
x
```




    array([[7, 7, 9, 8],
           [7, 2, 5, 2],
           [2, 0, 2, 6]])




```python
x[0,0]
```




    7




```python
x[0]
```




    array([7, 7, 9, 8])



### 배열 슬라이싱

> x[start:stop:step]


```python
x = np.arange(10)
x
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
x[:5]
```




    array([0, 1, 2, 3, 4])




```python
x[4:7]
```




    array([4, 5, 6])




```python
x[::2]
```




    array([0, 2, 4, 6, 8])




```python
x[1::2]
```




    array([1, 3, 5, 7, 9])




```python
x[::-1]
```




    array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])




```python
x[5::-2]
```




    array([5, 3, 1])




```python
x = np.random.randint(10, size=(3,4))
x
```




    array([[7, 5, 8, 6],
           [0, 9, 4, 1],
           [1, 0, 7, 1]])




```python
x[:2,:3]
```




    array([[7, 5, 8],
           [0, 9, 4]])




```python
x[:3,::2]
```




    array([[7, 8],
           [0, 4],
           [1, 7]])




```python
x[::-1,::-1]
```




    array([[1, 7, 0, 1],
           [1, 4, 9, 0],
           [6, 8, 5, 7]])




```python
x[:,0]
```




    array([7, 0, 1])




```python
x[0,:]
```




    array([7, 5, 8, 6])



### 배열 재구조화

> .reshape() <br>
> .newaxis


```python
grid = np.arange(1,10).reshape((3,3))
grid
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
x = np.array([1,2,3])
```


```python
x.reshape((1,3))
```




    array([[1, 2, 3]])




```python
x.reshape((3,1))
```




    array([[1],
           [2],
           [3]])




```python
x[np.newaxis,:]
```




    array([[1, 2, 3]])




```python
x[:,np.newaxis]
```




    array([[1],
           [2],
           [3]])



### 배열 연결

> .concatenate() <br>
> .vstack() <br>
> .hstack()


```python
x = np.array([1,2,3])
y = np.array([3,2,1])
z = np.array([99,99,99])
```


```python
np.concatenate([x,y])
```




    array([1, 2, 3, 3, 2, 1])




```python
np.concatenate([x,y,z])
```




    array([ 1,  2,  3,  3,  2,  1, 99, 99, 99])




```python
grid = np.arange(1,7).reshape((2,3))
```


```python
np.concatenate([grid,grid])
```




    array([[1, 2, 3],
           [4, 5, 6],
           [1, 2, 3],
           [4, 5, 6]])




```python
np.concatenate([grid,grid], axis=1)
```




    array([[1, 2, 3, 1, 2, 3],
           [4, 5, 6, 4, 5, 6]])




```python
x = np.array([9,9,9])
np.vstack([x,grid])
```




    array([[9, 9, 9],
           [1, 2, 3],
           [4, 5, 6]])




```python
y = np.array([[9],[9]])
np.hstack([y,grid])
```




    array([[9, 1, 2, 3],
           [9, 4, 5, 6]])



### 배열 분할

> .split() <br>
> .vsplit() <br>
> .hsplit()


```python
x = [1,2,3,99,99,3,2,1,]
x1, x2, x3 = np.split(x, [3,5])
print(x1, x2, x3)
```

    [1 2 3] [99 99] [3 2 1]
    


```python
grid = np.arange(16).reshape((4,4))
grid
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
```

    [[0 1 2 3]
     [4 5 6 7]]
    [[ 8  9 10 11]
     [12 13 14 15]]
    


```python
left, right = np.hsplit(grid, [1])
print(left)
print(right)
```

    [[ 0]
     [ 4]
     [ 8]
     [12]]
    [[ 1  2  3]
     [ 5  6  7]
     [ 9 10 11]
     [13 14 15]]
    
