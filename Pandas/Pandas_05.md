
# Pandas

##### 참고 : 파이썬 데이터 사이언스 핸드북


```python
import numpy as np
import pandas as pd
```

<br>

```python
def make_df(cols, ind):
    data = {c:[str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)
```

<br>

##### In
```python
make_df('ABC', range(3))
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
    </tr>
</table>

<br>

### Concat

##### In
```python
ser1 = pd.Series(['A','B','C'], index=[1,2,3])
ser2 = pd.Series(['D','E','F'], index=[4,5,6])
pd.concat([ser1,ser2])
```
##### Out
    1    A
    2    B
    3    C
    4    D
    5    E
    6    F
    dtype: object

<br>

##### In
```python
df1 = make_df('AB', [1,2])
df2 = make_df('AB', [3,4])
pd.concat([df1,df2])
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
    </tr>
</table>

<br>

##### In
```python
df3 = make_df('AB', [0,1])
df4 = make_df('CD', [0,1])
pd.concat([df3, df4], axis=1)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
</table>

<br>

### 인덱스 복제

##### In
```python
x = make_df('AB', [0,1])
y = make_df('AB', [2,3])
y.index = x.index # 복제 인덱스 생성
pd.concat([x,y])
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
</table>

<br>

##### In
```python
try:
    pd.concat([x,y], verify_integrity=True) # 인덱스가 겹지치 않게 검증
except ValueError as e:
    print('ValueError : ', e)
```
##### Out
    ValueError :  Indexes have overlapping values: [0, 1]
    
<br>

##### In
```python
pd.concat([x,y], ignore_index=True) # 인덱스 자체가 중요하지 않은 경우, 인덱스 무시
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
</table>

<br>

##### In
```python
pd.concat([x,y], keys=['x','y']) # 계층적 인덱싱
```
##### Out
<table>
    <tr>
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th rowspan="2" valign="top">x</th>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">y</th>
      <th>0</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
</table>

<br>

### 조인을 이용한 연결

##### In
```python
df5 = make_df('ABC', [1,2])
df6 = make_df('BCD', [3,4])
pd.concat([df5, df6])
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
</table>

<br>

##### In
```python
pd.concat([df5, df6], join='inner')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>1</th>
      <td>B1</td>
      <td>C1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B2</td>
      <td>C2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B3</td>
      <td>C3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B4</td>
      <td>C4</td>
    </tr>
</table>

<br>

##### In
```python
pd.concat([df5, df6], join_axes=[df5.columns])
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>B3</td>
      <td>C3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>B4</td>
      <td>C4</td>
    </tr>
</table>

<br>

### Append

##### In
```python
df1.append(df2)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
    </tr>
</table>

<br>

### Merge

##### In
```python
# 일대일 조인
df1 = pd.DataFrame({'employee': ['Bob','Jake','Lisa','Sue'],
                    'group': ['Accounting','Engineering','Engineering','HR']})
df2 = pd.DataFrame({'employee': ['Lisa','Bob','Jake','Sue'],
                    'hire_date': [2004,2008,2012,2014]})
df3 = pd.merge(df1, df2)
df3
```
##### Out
<table>
    <tr>
      <th></th>
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
</table>

<br>

##### In
```python
# 다대일 조인
df4 = pd.DataFrame({'group': ['Accounting','Engineering','HR'],
                    'supervisor': ['Carly','Guido','Steve']})
pd.merge(df3, df4)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
      <th>supervisor</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
      <td>Carly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
      <td>Steve</td>
    </tr>
</table>

<br>

##### In
```python
# 다대다 조인
df5 = pd.DataFrame({'group': ['Accounting','Accounting','Engineering','Engineering','HR','HR'],
                    'skills': ['math','spreadsheets','coding','linux','spreadsheets','organization']})
pd.merge(df1, df5)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>employee</th>
      <th>group</th>
      <th>skills</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>math</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>coding</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>linux</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>coding</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>linux</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sue</td>
      <td>HR</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sue</td>
      <td>HR</td>
      <td>organization</td>
    </tr>
</table>

<br>

### 병합 키 지정

##### In
```python
pd.merge(df1, df2, on='employee')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
</table>

<br>

##### In
```python
df3 = pd.DataFrame({'name': ['Bob','Jake','Lisa','Sue'],
                    'salary': [70000,80000,120000,90000]})
pd.merge(df1, df3, left_on='employee', right_on='name')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>employee</th>
      <th>group</th>
      <th>name</th>
      <th>salary</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>Sue</td>
      <td>90000</td>
    </tr>
</table>

<br>

##### In
```python
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
pd.merge(df1a, df2a, left_index=True, right_index=True)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
      <td>2014</td>
    </tr>
</table>

<br>

##### In
```python
df1a.join(df2a)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
      <td>2014</td>
    </tr>
</table>

<br>

##### In
```python
pd.merge(df1a, df3, left_index=True, right_on='name')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>group</th>
      <th>name</th>
      <th>salary</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Accounting</td>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Engineering</td>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Engineering</td>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HR</td>
      <td>Sue</td>
      <td>90000</td>
    </tr>
</table>

<br>

### 조인을 위한 집합 연산 지정하기

##### In
```python
df6 = pd.DataFrame({'name': ['Peter','Paul','Mary'],
                    'food': ['fish','beans','bread']},
                   columns=['name','food'])
df7 = pd.DataFrame({'name': ['Mary','Joseph'],
                    'drink': ['wine','beer']},
                   columns=['name','drink'])
pd.merge(df6, df7)
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
</table>

<br>

##### In
```python
pd.merge(df6, df7, how='inner')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
</table>

<br>

##### In
```python
pd.merge(df6, df7, how='outer')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joseph</td>
      <td>NaN</td>
      <td>beer</td>
    </tr>
</table>

<br>

##### In
```python
pd.merge(df6, df7, how='left')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
</table>

<br>

##### In
```python
pd.merge(df6, df7, how='right')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joseph</td>
      <td>NaN</td>
      <td>beer</td>
    </tr>
</table>

<br>

### 열 이름이 겹치는 경우 : suffixes

##### In
```python
df8 = pd.DataFrame({'name': ['Bob','Jake','Lisa','Sue'],
                    'rank': [1,2,3,4]})
df9 = pd.DataFrame({'name': ['Bob','Jake','Lisa','Sue'],
                    'rank': [3,1,4,2]})
pd.merge(df8, df9, on='name')
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>rank_x</th>
      <th>rank_y</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
      <td>2</td>
    </tr>
</table>

<br>

##### In
```python
pd.merge(df8, df9, on='name', suffixes=['_L','_R'])
```
##### Out
<table>
    <tr>
      <th></th>
      <th>name</th>
      <th>rank_L</th>
      <th>rank_R</th>
    </tr>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
      <td>2</td>
    </tr>
</table>


