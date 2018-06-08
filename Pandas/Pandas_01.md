
# Pandas

참고 : 파이썬 데이터 사이언스 핸드북 111p-


```python
import numpy as np
import pandas as pd
```

### Series


```python
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
```




    0    0.25
    1    0.50
    2    0.75
    3    1.00
    dtype: float64




```python
data.values
```




    array([0.25, 0.5 , 0.75, 1.  ])




```python
data.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
data[1]
```




    0.5




```python
data[1:3]
```




    1    0.50
    2    0.75
    dtype: float64




```python
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data
```




    a    0.25
    b    0.50
    c    0.75
    d    1.00
    dtype: float64




```python
data['b']
```




    0.5




```python
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
data
```




    2    0.25
    5    0.50
    3    0.75
    7    1.00
    dtype: float64




```python
data[5]
```




    0.5




```python
population_dict = {'California':38332521,
                   'Texas':26448193,
                   'New York':19651127,
                   'Florida':19552860,
                   'Illinois':12882135}
population = pd.Series(population_dict)
population
```




    California    38332521
    Florida       19552860
    Illinois      12882135
    New York      19651127
    Texas         26448193
    dtype: int64




```python
population['California']
```




    38332521




```python
population['California':'Illinois']
```




    California    38332521
    Florida       19552860
    Illinois      12882135
    dtype: int64




```python
pd.Series([2, 4, 6])
```




    0    2
    1    4
    2    6
    dtype: int64




```python
pd.Series(5, index=[100, 200, 300])
```




    100    5
    200    5
    300    5
    dtype: int64




```python
pd.Series({2:'a', 1:'b', 3:'c'})
```




    1    b
    2    a
    3    c
    dtype: object




```python
pd.Series({2:'a', 3:'c'}, index=[3,2])
```




    3    c
    2    a
    dtype: object



### DataFrame


```python
area_dict = {'California':423967, 'Texas':695662, 'New York':141297, 'Florida':170312, 'Illinois':149995}
area = pd.Series(area_dict)
area
```




    California    423967
    Florida       170312
    Illinois      149995
    New York      141297
    Texas         695662
    dtype: int64




```python
states = pd.DataFrame({'population':population, 'area':area})
states
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
      <th>area</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>




```python
states.index
```




    Index(['California', 'Florida', 'Illinois', 'New York', 'Texas'], dtype='object')




```python
states.columns
```




    Index(['area', 'population'], dtype='object')




```python
states['area']
```




    California    423967
    Florida       170312
    Illinois      149995
    New York      141297
    Texas         695662
    Name: area, dtype: int64




```python
pd.DataFrame(population, columns=['population'])
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
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12882135</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = [{'a':i, 'b':2*i} for i in range(3)]
pd.DataFrame(data)
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
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(np.random.rand(3,2), columns=['foo', 'bar'], index=['a', 'b', 'c'])
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
      <th>foo</th>
      <th>bar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.921707</td>
      <td>0.037186</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.744721</td>
      <td>0.763437</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.881099</td>
      <td>0.759329</td>
    </tr>
  </tbody>
</table>
</div>




```python
A = np.zeros(3, dtype=[('A', 'i8'),('B', 'f8')])
A
```




    array([(0, 0.), (0, 0.), (0, 0.)], dtype=[('A', '<i8'), ('B', '<f8')])




```python
pd.DataFrame(A)
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Index


```python
ind = pd.Index([2, 3, 5, 7, 11])
ind
```




    Int64Index([2, 3, 5, 7, 11], dtype='int64')




```python
ind[1]
```




    3




```python
ind[::2]
```




    Int64Index([2, 5, 11], dtype='int64')




```python
print(ind.size, ind.shape, ind.ndim, ind.dtype)
```

    5 (5,) 1 int64
    


```python
ind[1] = 0
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-32-906a9fa1424c> in <module>()
    ----> 1 ind[1] = 0
    

    ~\Anaconda3\lib\site-packages\pandas\core\indexes\base.py in __setitem__(self, key, value)
       1722 
       1723     def __setitem__(self, key, value):
    -> 1724         raise TypeError("Index does not support mutable operations")
       1725 
       1726     def __getitem__(self, key):
    

    TypeError: Index does not support mutable operations



```python
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
```


```python
# 교집합
indA & indB
```




    Int64Index([3, 5, 7], dtype='int64')




```python
# 합집합
indA | indB
```




    Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')




```python
# 대칭 차
indA ^ indB
```




    Int64Index([1, 2, 9, 11], dtype='int64')


