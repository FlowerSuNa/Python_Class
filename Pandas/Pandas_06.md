
# Pandas

##### 참고 : 파이썬 데이터 사이언스 핸드북


```python
import numpy as np
import pandas as pd
```

<br>

### 예제 : 미국 주 데이터

##### In
```python
# 데이터 로드
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv
```
##### Out
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    100 57935  100 57935    0     0   129k      0 --:--:-- --:--:-- --:--:--  129k
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    100   835  100   835    0     0   1978      0 --:--:-- --:--:-- --:--:--  1978
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    100   872  100   872    0     0   2325      0 --:--:-- --:--:-- --:--:--  2325
    
<br>

##### In
```python
pop = pd.read_csv('state-population.csv')
areas = pd.read_csv('state-areas.csv')
abbrevs = pd.read_csv('state-abbrevs.csv')
```


```python
pop.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
    </tr>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
    </tr>
</table>

<br>

##### In
```python
areas.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>52423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>656425</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>114006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>53182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>163707</td>
    </tr>
</table>

<br>

##### In
```python
abbrevs.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state</th>
      <th>abbreviation</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>CA</td>
    </tr>
</table>

<br>

##### In
```python
# 다대다 병합
merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1)
merged.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
    </tr>
</table>

<br>

##### In
```python
merged.isnull().any() # 불일치하는 항목이 있는지 확인
```
##### Out
    state/region    False
    ages            False
    year            False
    population       True
    state            True
    dtype: bool

<br>

##### In
```python
merged[merged['population'].isnull()].head() # population 정보에서 null 값 항목 확인
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
    <tr>
      <th>2448</th>
      <td>PR</td>
      <td>under18</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2449</th>
      <td>PR</td>
      <td>total</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2450</th>
      <td>PR</td>
      <td>total</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2451</th>
      <td>PR</td>
      <td>under18</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2452</th>
      <td>PR</td>
      <td>total</td>
      <td>1993</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
</table>

<br>

##### In
```python
merged.loc[merged['state'].isnull(), 'state/region'].unique()
```
##### Out
    array(['PR', 'USA'], dtype=object)

<br>

##### In
```python
merged.loc[merged['state/region'] == 'PR', 'state'] == 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] == 'United States'
merged.isnull().any()
```
##### Out
    state/region    False
    ages            False
    year            False
    population       True
    state            True
    dtype: bool

<br>

##### In
```python
# 병합
final = pd.merge(merged, areas, on='state', how='left')
final.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
</table>

<br>

##### In
```python
final.isnull().any() # null 값 여부 확인
```
##### Out
    state/region     False
    ages             False
    year             False
    population        True
    state             True
    area (sq. mi)     True
    dtype: bool

<br>

##### In
```python
final['state'][final['area (sq. mi)'].isnull()].unique()
```
##### Out
    array([nan], dtype=object)

<br>

##### In
```python
final.dropna(inplace=True)
final.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
</table>

<br>

##### In
```python
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
```
##### Out
<table>
    <tr>
      <th></th>
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>AK</td>
      <td>total</td>
      <td>2010</td>
      <td>713868.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>AZ</td>
      <td>total</td>
      <td>2010</td>
      <td>6408790.0</td>
      <td>Arizona</td>
      <td>114006.0</td>
    </tr>
    <tr>
      <th>189</th>
      <td>AR</td>
      <td>total</td>
      <td>2010</td>
      <td>2922280.0</td>
      <td>Arkansas</td>
      <td>53182.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>CA</td>
      <td>total</td>
      <td>2010</td>
      <td>37333601.0</td>
      <td>California</td>
      <td>163707.0</td>
    </tr>
</table>

<br>

##### In
```python
# 인구 밀도 구하기
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(ascending=False, inplace=True)
density.head()
```
##### Out
    state
    District of Columbia    8898.897059
    New Jersey              1009.253268
    Rhode Island             681.339159
    Connecticut              645.600649
    Massachusetts            621.815538
    dtype: float64

<br>

##### In
```python
density.tail()
```
##### Out
    state
    South Dakota    10.583512
    North Dakota     9.537565
    Montana          6.736171
    Wyoming          5.768079
    Alaska           1.087509
    dtype: float64


