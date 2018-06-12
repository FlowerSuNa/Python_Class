
# TensorFlow

##### 참고 : TensorFlow Machine Learning Cookbook


```python
import tensorflow as tf
sess = tf.Session()
```

```python
import numpy as np
```

<br>

### 활성화 함수 구현

> 활성화 함수의 목표는 가중치와 평향치를 조절하는 것이다. <br>
> 텐서플로에서 활성화 함수는 텐서에 적용되는 비선형 연산이다. <br>

<br>

##### 선형 유닛 함수
> min (max (0 , x))

<br>

##### In
```python
print(sess.run(tf.nn.relu([-3.,3.,10.])))
```
##### Out
    [ 0.  3. 10.]
    
<br>

##### In
```python
print(sess.run(tf.nn.relu6([-3.,3.,10.])))
```
##### Out
    [0. 3. 6.]
    
<br>    

##### 시그모이드 함수 (로지스틱 함수)
> 1 / (1 + exp(-x))

<br>

##### In
```python
print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))
```
##### Out
    [0.26894143 0.5        0.7310586 ]

<br>

##### 하이퍼볼릭 탄젠트 함수
> (exp(x) - exp(-x)) / (exp(x) + exp(-x))

<br>

##### In
```python
print(sess.run(tf.nn.tanh([-1.,0.,1.])))
```
##### Out
    [-0.7615942  0.         0.7615942]
    
<br>

##### softsign 함수
> x / (abs(x) + 1)

<br>

##### In
```python
print(sess.run(tf.nn.softsign([-1.,0.,1.])))
```
##### Out
    [-0.5  0.   0.5]
    
<br>

##### softplus 함수
> log(exp(x) + 1)

<br>

##### In
```python
print(sess.run(tf.nn.softplus([-1.,0.,1.])))
```
##### Out
    [0.31326166 0.6931472  1.3132616 ]
    
<br>

##### 지수 선형 유닛 함수
> if x < 0 : (exp(x) + 1) <br>
> else : x

<br>

##### In
```python
print(sess.run(tf.nn.elu([-1.,0.,1.])))
```
##### Out
    [-0.63212055  0.          1.        ]
    
