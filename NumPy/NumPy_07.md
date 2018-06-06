
# NumPy

##### 참고 : 파이썬 데이터 사이언스 핸드북 106p-110p

```python
import numpy as np
```

<br>

### 구조화된 배열

##### In
```python
name = ['Alice','Bob','Cathy','Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data = np.zeros(4, dtype={'names':('name','age','weight'),
                          'formats':('U10','i4','f8')})
# U10 : 최대길이 10을 갖는 유니코드 문자열
# i4 : 4바이트(32비트) 정수
# f8 : 8바이트(64비트) 부동 소수점
data.dtype
```
##### Out
    dtype([('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])

<br>

##### In
```python
data['name'] = name
data['age'] = age
data['weight'] = weight
data
```
##### Out
    array([('Alice', 25, 55. ), ('Bob', 45, 85.5), ('Cathy', 37, 68. ),
           ('Doug', 19, 61.5)],
          dtype=[('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])

<br>

##### In
```python
data['name']
```
##### Out
    array(['Alice', 'Bob', 'Cathy', 'Doug'], dtype='<U10')

<br>

##### In
```python
data[0]
```
##### Out
    ('Alice', 25, 55.)

<br>

##### In
```python
data[-1]['name']
```
##### Out
    'Doug'

<br>

##### In
```python
data[data['age'] < 30]['name']
```
##### Out
    array(['Alice', 'Doug'], dtype='<U10')

<br>

### 구조화된 배열 만들기

##### In
```python
np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})
```
##### Out
    dtype([('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])

<br>

##### In
```python
np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_,10), int, np.float32)})
```
##### Out
    dtype([('name', '<U10'), ('age', '<i4'), ('weight', '<f4')])

<br>

##### In
```python
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
```
##### Out
    dtype([('name', 'S10'), ('age', '<i4'), ('weight', '<f8')])

<br>

##### In
```python
np.dtype('S10,i4,f8')
```
##### Out
    dtype([('f0', 'S10'), ('f1', '<i4'), ('f2', '<f8')])

<br>

<table>
    <tr> <td>문자</td> <td>설명</td> <td>예제</td> </tr>
    <tr> <td>b</td> <td>바이트</td> <td>np.dtype('b')</td> </tr>
    <tr> <td>i</td> <td>부호 있는 정수</td> <td>np.dtype('i4') == np.int32</td> </tr>
    <tr> <td>u</td> <td>부호 없는 정수</td> <td>np.dtype('u1') == np.unit8</td> </tr>
    <tr> <td>f</td> <td>부동 소수점</td> <td>np.dtype('f8') == np.float64</td> </tr>
    <tr> <td>c</td> <td>복소수 부동 소수점</td> <td>np.dtype('c16') == np.complex128</td> </tr>
    <tr> <td>S , a</td> <td>문자열</td> <td>np.dtype('S5')</td> </tr>
    <tr> <td>U</td> <td>유니코드 문자열</td> <td>np.dtype('U') == np.str_</td> </tr>
    <tr> <td>V</td> <td>원시 데이터</td> <td>np.dtype('V') == np.void</td> </tr>
</table>

<br>

### 고급 복합 타입

##### In
```python
tp = np.dtype([('id','i8'), ('mat', 'f8', (3,3))])
X = np.zeros(1, dtype=tp)
X
```
##### Out
    array([(0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])],
          dtype=[('id', '<i8'), ('mat', '<f8', (3, 3))])

<br>

##### In
```python
X[0]
```
##### Out
    (0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

<br>

##### In
```python
X['mat'][0]
```
##### Out
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

<br>

### 레코드 배열 : 트위스트를 가진 구조화된 배열

##### In
```python
data['age']
```
##### Out
    array([25, 45, 37, 19])

<br>

##### In
```python
data_rec = data.view(np.recarray)
data_rec.age
```
##### Out
    array([25, 45, 37, 19])

<br>

##### In
```python
%timeit data['age']
%timeit data_rec['age']
%timeit data_rec.age
```
##### Out
    81.6 ns ± 0.0886 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
    2.33 µs ± 10.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    3.04 µs ± 14.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    
