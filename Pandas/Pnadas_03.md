
# Pandas

##### 참고 : 파이썬 데이터 사이언스 핸드북 137p-146p


```python
import numpy as np
import pandas as pd
```

<br>

### 누락된 데이터

##### In
```python
vals = np.array([1, np.nan, 3, 4])
vals.dtype
```
##### Out
    dtype('float64')

<br>

##### In
```python
1 + np.nan
```
##### Out
    nan

<br>

##### In
```python
0 * np.nan
```
##### Out
    nan

<br>

##### In
```python
vals.sum(), vals.min(), vals.max()
```
##### Out
    (nan, nan, nan)

<br>

##### In
```python
np.nansum(vals), np.nanmin(vals), np.nanmax(vals)
```
##### Out
    (8.0, 1.0, 4.0)

<br>

Nan과 None

##### In
```python
pd.Series([1, np.nan, 2, None])
```
##### Out
    0    1.0
    1    NaN
    2    2.0
    3    NaN
    dtype: float64

<br>

##### In
```python
x = pd.Series(range(2),dtype=int)
x
```
##### Out
    0    0
    1    1
    dtype: int32

<br>

##### In
```python
x[0] = None
x
```
##### Out
    0    NaN
    1    1.0
    dtype: float64

<br>

### 널 값 탐지

> .isnull() <br>
> .notnull()

<br>

##### In
```python
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
```
##### Out
    0    False
    1     True
    2    False
    3     True
    dtype: bool

<br>

##### In
```python
data[data.notnull()]
```
##### Out
    0        1
    2    hello
    dtype: object

<br>

### 널 값 제거하기

> .dropna()

<br>

##### In
```python
df = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
df
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6</td>
    </tr>
</table>

<br>

##### In
```python
df.dropna()
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
</table>

<br>

##### In
```python
# axis : 축
df.dropna(axis='columns')
```
##### Out

<table>
    <tr>
      <th></th>
      <th>2</th>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
    </tr>
</table>

<br>

##### In
```python
df[3] = np.nan
df
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
</table>

<br>

##### In
```python
# how
# 'all' : 모두 널인 행이나 열 삭제
# 'any' : 널 값을 포함하는 행이나 열 삭제
df.dropna(axis='columns', how='all') 
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6</td>
    </tr>
</table>

<br>

##### In
```python
# thresh : 널이 아닌 값이 최소 몇 개 있어야 하는지 지정
df.dropna(axis='rows', thresh=3)
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
</table>

<br>

### 널 값 채우기

> .fillna()

<br>

##### In
```python
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
```
##### Out
    a    1.0
    b    NaN
    c    2.0
    d    NaN
    e    3.0
    dtype: float64

<br>

##### In
```python
data.fillna(0)
```
##### Out
    a    1.0
    b    0.0
    c    2.0
    d    0.0
    e    3.0
    dtype: float64

<br>

##### In
```python
# ffill : 이전 값으로 채우기
data.fillna(method='ffill')
```
##### Out
    a    1.0
    b    1.0
    c    2.0
    d    2.0
    e    3.0
    dtype: float64

<br>

##### In
```python
# bfill : 다음에 오는 값으로 채우기
data.fillna(method='bfill')
```
##### Out
    a    1.0
    b    2.0
    c    2.0
    d    3.0
    e    3.0
    dtype: float64

<br>

##### In
```python
df
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
</table>

<br>

##### In
```python
df.fillna(method='ffill', axis=1) # 행
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
</table>

<br>

##### In
```python
df.fillna(method='ffill', axis=0) # 열
```
##### Out

<table>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
</table>


