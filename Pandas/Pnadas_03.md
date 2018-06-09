
# Pandas

참고 : 파이썬 데이터 사이언스 핸드북 137p-146p


```python
import numpy as np
import pandas as pd
```

### 누락된 데이터


```python
vals = np.array([1, np.nan, 3, 4])
vals.dtype
```




    dtype('float64')




```python
1 + np.nan
```




    nan




```python
0 * np.nan
```




    nan




```python
vals.sum(), vals.min(), vals.max()
```

    C:\Users\GIGABYTE\Anaconda3\lib\site-packages\numpy\core\_methods.py:29: RuntimeWarning: invalid value encountered in reduce
      return umr_minimum(a, axis, None, out, keepdims)
    C:\Users\GIGABYTE\Anaconda3\lib\site-packages\numpy\core\_methods.py:26: RuntimeWarning: invalid value encountered in reduce
      return umr_maximum(a, axis, None, out, keepdims)
    




    (nan, nan, nan)




```python
np.nansum(vals), np.nanmin(vals), np.nanmax(vals)
```




    (8.0, 1.0, 4.0)



Nan과 None


```python
pd.Series([1, np.nan, 2, None])
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    dtype: float64




```python
x = pd.Series(range(2),dtype=int)
x
```




    0    0
    1    1
    dtype: int32




```python
x[0] = None
x
```




    0    NaN
    1    1.0
    dtype: float64



### 널 값 탐지

> .isnull() <br>
> .notnull()


```python
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
```




    0    False
    1     True
    2    False
    3     True
    dtype: bool




```python
data[data.notnull()]
```




    0        1
    2    hello
    dtype: object



### 널 값 제거하기

> .dropna()


```python
df = pd.DataFrame([[1, np.nan, 2],[2, 3, 5], [np.nan, 4, 6]])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>




```python
df.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# axis : 축
df.dropna(axis='columns')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>




```python
df[3] = np.nan
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>




```python
# how
# 'all' : 모두 널인 행이나 열 삭제
# 'any' : 널 값을 포함하는 행이나 열 삭제
df.dropna(axis='columns', how='all') 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>




```python
# thresh : 널이 아닌 값이 최소 몇 개 있어야 하는지 지정
df.dropna(axis='rows', thresh=3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 널 값 채우기

> .fillna()


```python
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
```




    a    1.0
    b    NaN
    c    2.0
    d    NaN
    e    3.0
    dtype: float64




```python
data.fillna(0)
```




    a    1.0
    b    0.0
    c    2.0
    d    0.0
    e    3.0
    dtype: float64




```python
# ffill : 이전 값으로 채우기
data.fillna(method='ffill')
```




    a    1.0
    b    1.0
    c    2.0
    d    2.0
    e    3.0
    dtype: float64




```python
# bfill : 다음에 오는 값으로 채우기
data.fillna(method='bfill')
```




    a    1.0
    b    2.0
    c    2.0
    d    3.0
    e    3.0
    dtype: float64




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill', axis=1) # 행
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill', axis=0) # 열
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
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
  </tbody>
</table>
</div>


