
# Pandas

참고 : 파이썬 데이터 사이언스 핸드북


```python
import numpy as np
import pandas as pd
```

### 계층적 인덱싱


```python
index = [('California', 2000), ('California', 2010), 
         ('New York', 2000), ('New York', 2010), 
         ('Texas', 2000), ('Texas', 2010)]
index = pd.MultiIndex.from_tuples(index)
index
```




    MultiIndex(levels=[['California', 'New York', 'Texas'], [2000, 2010]],
               labels=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])




```python
populations = [38714648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop = pd.Series(populations, index=index)
pop
```




    California  2000    38714648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64




```python
pop[:,2010]
```




    California    37253956
    New York      19378102
    Texas         25145561
    dtype: int64




```python
pop['California']
```




    2000    38714648
    2010    37253956
    dtype: int64




```python
pop_df = pop.unstack()
pop_df
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
      <th>2000</th>
      <th>2010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38714648</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>20851820</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop_df.stack()
```




    California  2000    38714648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64




```python
under18 = [9267089, 9284094, 4687374, 4318033, 5906301, 6879014]
pop_df = pd.DataFrame({'total': pop, 'under18': under18})
pop_df
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
      <th></th>
      <th>total</th>
      <th>under18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>38714648</td>
      <td>9267089</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
      <td>9284094</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
      <td>4687374</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>19378102</td>
      <td>4318033</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Texas</th>
      <th>2000</th>
      <td>20851820</td>
      <td>5906301</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>25145561</td>
      <td>6879014</td>
    </tr>
  </tbody>
</table>
</div>




```python
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()
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
      <th>2000</th>
      <th>2010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>0.239369</td>
      <td>0.249211</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>0.247010</td>
      <td>0.222831</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>0.283251</td>
      <td>0.273568</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame(np.random.rand(4, 2), 
                 index=[['a','a','b','b'],[1,2,1,2]],
                 columns=['data1', 'data2'])
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
      <th></th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0.455970</td>
      <td>0.736968</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.319934</td>
      <td>0.638126</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>0.405061</td>
      <td>0.101964</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.492238</td>
      <td>0.891392</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)
```




    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64



### 명시적 MultiIndex 생성자


```python
pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])
```




    MultiIndex(levels=[['a', 'b'], [1, 2]],
               labels=[[0, 0, 1, 1], [0, 1, 0, 1]])




```python
pd.MultiIndex.from_tuples([('a',1), ('a',2), ('b',1),('b',2)])
```




    MultiIndex(levels=[['a', 'b'], [1, 2]],
               labels=[[0, 0, 1, 1], [0, 1, 0, 1]])




```python
pd.MultiIndex.from_product([['a','b'], [1,2]])
```




    MultiIndex(levels=[['a', 'b'], [1, 2]],
               labels=[[0, 0, 1, 1], [0, 1, 0, 1]])




```python
pd.MultiIndex(levels=[['a','b'], [1,2]],
              labels=[[0,0,1,1,], [0,1,0,1]])
```




    MultiIndex(levels=[['a', 'b'], [1, 2]],
               labels=[[0, 0, 1, 1], [0, 1, 0, 1]])



### MultiIndex 레벨 이름


```python
pop.index.names = ['states', 'year']
pop
```




    states      year
    California  2000    38714648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64



### 열의 MultiIndex


```python
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['subject', 'type'])

data = np.round(np.random.randn(4, 6), 1)
data[:,::2] *= 10
data += 37

health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>43.0</td>
      <td>36.6</td>
      <td>23.0</td>
      <td>36.4</td>
      <td>29.0</td>
      <td>37.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>37.2</td>
      <td>40.0</td>
      <td>37.1</td>
      <td>47.0</td>
      <td>36.7</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>41.0</td>
      <td>36.0</td>
      <td>47.0</td>
      <td>37.1</td>
      <td>23.0</td>
      <td>35.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46.0</td>
      <td>35.0</td>
      <td>50.0</td>
      <td>36.6</td>
      <td>28.0</td>
      <td>37.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data['Guido']
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
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>23.0</td>
      <td>36.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.0</td>
      <td>37.1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>47.0</td>
      <td>37.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50.0</td>
      <td>36.6</td>
    </tr>
  </tbody>
</table>
</div>



### MultiIndex 인덱싱 및 슬라이싱


```python
pop['California', 2000]
```




    38714648




```python
pop['California']
```




    year
    2000    38714648
    2010    37253956
    dtype: int64




```python
pop.loc['California':'New York']
```




    states      year
    California  2000    38714648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    dtype: int64




```python
pop[:, 2000]
```




    states
    California    38714648
    New York      18976457
    Texas         20851820
    dtype: int64




```python
pop[pop > 22000000]
```




    states      year
    California  2000    38714648
                2010    37253956
    Texas       2010    25145561
    dtype: int64




```python
pop[['California', 'Texas']]
```




    states      year
    California  2000    38714648
                2010    37253956
    Texas       2000    20851820
                2010    25145561
    dtype: int64




```python
health_data['Guido', 'HR']
```




    year  visit
    2013  1        23.0
          2        40.0
    2014  1        47.0
          2        50.0
    Name: (Guido, HR), dtype: float64




```python
health_data.iloc[:2,:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>43.0</td>
      <td>36.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>37.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data.loc[:, ('Bob','HR')]
```




    year  visit
    2013  1        43.0
          2        55.0
    2014  1        41.0
          2        46.0
    Name: (Bob, HR), dtype: float64




```python
health_data.loc[(:,1),(:,'HR')]
```


      File "<ipython-input-28-2d8e6b30864e>", line 1
        health_data.loc[(:,1),(:,'HR')]
                         ^
    SyntaxError: invalid syntax
    



```python
idx = pd.IndexSlice
health_data.loc[idx[:,1], idx[:,'HR']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th>Bob</th>
      <th>Guido</th>
      <th>Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>HR</th>
      <th>HR</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013</th>
      <th>1</th>
      <td>43.0</td>
      <td>23.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>2014</th>
      <th>1</th>
      <td>41.0</td>
      <td>47.0</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
</div>



### 정렬된 인덱스와 정렬되지 않은 인덱스


```python
index = pd.MultiIndex.from_product([['a','c','b'], [1,2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
```




    char  int
    a     1      0.119240
          2      0.377645
    c     1      0.907389
          2      0.025945
    b     1      0.833898
          2      0.496226
    dtype: float64




```python
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
```

    <class 'pandas.errors.UnsortedIndexError'>
    'Key length (1) was greater than MultiIndex lexsort depth (0)'
    


```python
data = data.sort_index()
data
```




    char  int
    a     1      0.119240
          2      0.377645
    b     1      0.833898
          2      0.496226
    c     1      0.907389
          2      0.025945
    dtype: float64




```python
data['a':'b']
```




    char  int
    a     1      0.119240
          2      0.377645
    b     1      0.833898
          2      0.496226
    dtype: float64



### 인덱스 스태킹 및 언스태킹


```python
pop.unstack(level=0)
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
      <th>states</th>
      <th>California</th>
      <th>New York</th>
      <th>Texas</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
      <td>38714648</td>
      <td>18976457</td>
      <td>20851820</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
      <td>19378102</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop.unstack(level=1)
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
      <th>year</th>
      <th>2000</th>
      <th>2010</th>
    </tr>
    <tr>
      <th>states</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38714648</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>20851820</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop.unstack().stack()
```




    states      year
    California  2000    38714648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64



### 인덱스 설정 및 재설정


```python
pop_flat = pop.reset_index(name='population')
pop_flat
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
      <th>states</th>
      <th>year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>California</td>
      <td>2000</td>
      <td>38714648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>California</td>
      <td>2010</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York</td>
      <td>2000</td>
      <td>18976457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>2010</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Texas</td>
      <td>2000</td>
      <td>20851820</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Texas</td>
      <td>2010</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop_flat.set_index(['states', 'year'])
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
      <th></th>
      <th>population</th>
    </tr>
    <tr>
      <th>states</th>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>38714648</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>19378102</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Texas</th>
      <th>2000</th>
      <td>20851820</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>



### 다중 인덱스에서 데이터 집계


```python
health_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>43.0</td>
      <td>36.6</td>
      <td>23.0</td>
      <td>36.4</td>
      <td>29.0</td>
      <td>37.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>37.2</td>
      <td>40.0</td>
      <td>37.1</td>
      <td>47.0</td>
      <td>36.7</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>41.0</td>
      <td>36.0</td>
      <td>47.0</td>
      <td>37.1</td>
      <td>23.0</td>
      <td>35.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46.0</td>
      <td>35.0</td>
      <td>50.0</td>
      <td>36.6</td>
      <td>28.0</td>
      <td>37.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_mean = health_data.mean(level='year')
data_mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013</th>
      <td>49.0</td>
      <td>36.9</td>
      <td>31.5</td>
      <td>36.75</td>
      <td>38.0</td>
      <td>37.00</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>43.5</td>
      <td>35.5</td>
      <td>48.5</td>
      <td>36.85</td>
      <td>25.5</td>
      <td>36.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_mean.mean(axis=1, level='type')
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
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013</th>
      <td>39.500000</td>
      <td>36.883333</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>39.166667</td>
      <td>36.233333</td>
    </tr>
  </tbody>
</table>
</div>


