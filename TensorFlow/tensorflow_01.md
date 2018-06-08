
# TensorFlow

##### 참고 : TensorFlow Machine Learning Cookbook 23p-50p


```python
import numpy as np
import tensorflow as tf
```

```python
sess = tf.Session()
```

<br>

### 텐서 생성

고정 텐서

> tf.zeros() <br>
> tf.ones() <br>
> tf.fill() <br>
> tf.constant() <br>
> tf.zeros_like() <br>
> tf.ones_like() <br>
> tf.linspace() <br>
> tf.range() <br>
> tf.random_uniform() <br>
> tf.random_normal() <br>
> tf.truncated_normal() <br>
> tf.random_shuffle() <br>
> tf.random_crop()

<br>

##### In
```python
# 0 값으로 채워진 텐서
zeros_tsr = tf.zeros([10,10])
print(zeros_tsr)

x = tf.Variable(zeros_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("zeros:0", shape=(10, 10), dtype=float32)
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    
<br>

##### In
```python
# 1 값으로 채워진 텐서
ones_tsr = tf.ones([10,10])
print(ones_tsr)

x = tf.Variable(ones_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("ones:0", shape=(10, 10), dtype=float32)
    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    
<br>

##### In
```python
# 동일한 상수 값으로 채워진 텐서
filled_tsr = tf.fill([10,10], 42)
print(filled_tsr)

x = tf.Variable(filled_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("Fill:0", shape=(10, 10), dtype=int32)
    [[42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]
     [42 42 42 42 42 42 42 42 42 42]]
    
<br>

##### In
```python
# 기존 상수를 이용해 텐서 생성
constant_tsr = tf.constant([1,2,3])
print(constant_tsr)

x = tf.Variable(constant_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("Const:0", shape=(3,), dtype=int32)
    [1 2 3]
 
<br> 

형태가 비슷한 텐서

##### In
```python
zeros_similar = tf.zeros_like(constant_tsr)
print(zeros_similar)

x = tf.Variable(zeros_similar)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("zeros_like:0", shape=(3,), dtype=int32)
    [0 0 0]
    
<br>

##### In
```python
ones_similar = tf.ones_like(constant_tsr)
print(ones_similar)

x = tf.Variable(ones_similar)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("ones_like:0", shape=(3,), dtype=int32)
    [1 1 1]
    
<br>

순열 텐서

##### In
```python
linear_tsr = tf.linspace(start=0.0, stop=1, num=3)
print(linear_tsr)

x = tf.Variable(linear_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("LinSpace:0", shape=(3,), dtype=float32)
    [0.  0.5 1. ]
    
<br>

##### In
```python
integer_seq_tsr = tf.range(start=6, limit=15, delta=3)
print(integer_seq_tsr)

x = tf.Variable(integer_seq_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("range:0", shape=(3,), dtype=int32)
    [ 6  9 12]
    
<br>

랜덤 텐서

##### In
```python
# 균등 분포를 따르는 난수
randunif = tf.random_uniform([5,5], minval=0, maxval=1)
print(randunif)

x = tf.Variable(randunif)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("random_uniform:0", shape=(5, 5), dtype=float32)
    [[0.53705573 0.7601783  0.866655   0.30148745 0.15259385]
     [0.4500171  0.61647916 0.7329619  0.87853837 0.03211832]
     [0.8108158  0.4579972  0.9278934  0.19503927 0.6620234 ]
     [0.9642333  0.03080726 0.36011708 0.10903025 0.4574163 ]
     [0.57827103 0.85613894 0.2165693  0.5530503  0.23494089]]
    
<br>

##### In
```python
# 정규 분포를 따르는 난수
randnorm = tf.random_normal([5,5], mean=0.0, stddev=1.0)
print(randnorm)

x = tf.Variable(randnorm)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("random_normal:0", shape=(5, 5), dtype=float32)
    [[-0.03070956  0.8399563   1.2052271  -0.5691961   0.07614964]
     [ 0.31681892  0.79478663  1.4148313  -1.5531563   0.11914554]
     [-0.8745117   1.0723083   0.8671376   0.12219578  0.99337655]
     [ 0.29660514  0.32552716 -0.11049543 -0.4223709  -0.39336452]
     [-0.54777056 -0.5253128  -0.45183325  0.35211897  0.09361728]]
    
<br>

##### In
```python
# 지정한 평균에서 표준편차 2배 이내에 속하는 정규 분포를 따르는 난수
runcnorm = tf.truncated_normal([5,5], mean=0.0, stddev=1.0)
print(runcnorm)

x = tf.Variable(runcnorm)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("truncated_normal:0", shape=(5, 5), dtype=float32)
    [[-1.4562662  -0.47551468  1.2035059  -0.564646   -0.62308925]
     [-0.12554401  0.15331154  1.0928133  -1.0212829  -0.03858028]
     [ 1.0943677  -1.8667918   1.924241    0.8978248  -1.0742313 ]
     [-0.45971468  0.13436101 -0.15776904  0.43173233  1.4060277 ]
     [ 0.8241787   1.9650444   0.14469278 -1.0749733  -0.90338767]]
    
<br>

##### In
```python
# 배열 항목을 임의로 뒤섞기
shuffled_output = tf.random_shuffle(x)
print(shuffled_output)

x = tf.Variable(shuffled_output)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("RandomShuffle:0", shape=(5, 5), dtype=float32)
    [[-1.0403867   0.2877181  -1.2973605  -0.03639733  1.0742434 ]
     [ 0.71414655  0.9373392   0.8505623   1.4090732  -0.0473349 ]
     [-0.04125489  0.50696284  1.0660099  -0.99220026  1.5822377 ]
     [-1.5710882  -0.70723367 -0.81574523 -0.42347383  1.1757109 ]
     [ 1.0786228  -0.49548015 -0.46400046 -1.5117693   1.1338954 ]]
    
<br>

##### In
```python
# 배열 항목을 임의로 섞고 형태 변환 
cropped_output = tf.random_crop(x,[3,5])
print(cropped_output)

x = tf.Variable(cropped_output)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```
##### Out
    Tensor("random_crop:0", shape=(3, 5), dtype=float32)
    [[-0.04125489  0.50696284  1.0660099  -0.99220026  1.5822377 ]
     [-1.5710882  -0.70723367 -0.81574523 -0.42347383  1.1757109 ]
     [-1.0403867   0.2877181  -1.2973605  -0.03639733  1.0742434 ]]

<br>

> tf.convert_to_tensort() : 어떤 numpy 배열이든 파이썬 리스트로 변환 <br>
> tf.global_variables_initializer() : 생성한 모든 변수 초기화 <br>
> 변수.initializer : 원하는 변수 초기화

<br>

### 행렬 다루기

```python
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3], 5.0)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-8.,-2.],[0.,5.,-2.]]))
```

<br>

##### In
```python
print(sess.run(identity_matrix))
```
##### Out
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

<br>

##### In
```python
print(sess.run(A))
```
##### Out
    [[-0.44267464 -0.4619699  -0.5434141 ]
     [-0.53997105  0.15313886  1.6242875 ]]
    
<br>

##### In
```python
print(sess.run(B))
```
##### Out
    [[5. 5. 5.]
     [5. 5. 5.]]
    
<br>

##### In
```python
print(sess.run(C))
```
##### Out
    [[0.7476628  0.6735562 ]
     [0.180071   0.22884429]
     [0.02259767 0.9689163 ]]
    
<br>

##### In
```python
print(sess.run(D))
```
##### Out
    [[ 1.  2.  3.]
     [-3. -8. -2.]
     [ 0.  5. -2.]]
    
<br>

##### In
```python
# 행렬의 합
print(sess.run(A+B))
```
##### Out
    [[5.2632475 5.6125937 3.439587 ]
     [5.278621  5.239634  3.9350107]]
    
<br>

##### In
```python
# 행렬의 차
print(sess.run(B-B))
```
##### Out
    [[0. 0. 0.]
     [0. 0. 0.]]
    
<br>

##### In
```python
# 행렬의 곱
print(sess.run(tf.matmul(B, identity_matrix)))
```
##### Out
    [[5. 5. 5.]
     [5. 5. 5.]]
    
<br>

##### In
```python
# 행렬 전치
print(sess.run(tf.transpose(C)))
```
##### Out
    [[0.55012035 0.08327782 0.77408063]
     [0.7599211  0.2402817  0.88874686]]
    
<br>

##### In
```python
print(sess.run(tf.matrix_determinant(D)))
```
##### Out
    -31.0
    
<br>

##### In
```python
# 역행렬
print(sess.run(tf.matrix_inverse(D)))
```
##### Out
    [[-0.83870968 -0.61290323 -0.64516129]
     [ 0.19354839  0.06451613  0.22580645]
     [ 0.48387097  0.16129032  0.06451613]]
    
<br>

##### In
```python
# 숄레스키 분해
print(sess.run(tf.cholesky(identity_matrix)))
```
##### Out
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    
<br>

##### In
```python
# 행렬의 고윳값과 고유 벡터 (고유 분해)
print(sess.run(tf.self_adjoint_eig(D)))
```
##### Out
    (array([-11.38910525,  -0.29446882,   2.68357407]), array([[ 0.20901067,  0.59907787, -0.77292965],
           [ 0.86315172,  0.25849588,  0.43376144],
           [-0.45965601,  0.75781633,  0.4630667 ]]))
    
<br>

### 연산

##### In
```python
print(sess.run(tf.div(3,4)))
```
##### Out
    0
    
<br>

##### In
```python
# 정수를 나누기 전에 소수로 변환해 항상 소수인 계산 결과를 반환
print(sess.run(tf.truediv(3,4)))
```
##### Out
    0.75
    
<br>

##### In
```python
# 소수를 대상으로 정수 나눗셈
print(sess.run(tf.floordiv(3.,4.)))
```
##### Out
    0.0
    
<br>

##### In
```python
print(sess.run(tf.mod(22.,5.)))
```
##### Out
    2.0
    
<br>

##### In
```python
# 외적
print(sess.run(tf.cross([1.,0.,0.,],[0.,1.,0.])))
```
##### Out
    [0. 0. 1.]
   
<br>   

<table>
    <tr> <td>tf.abs()</td> <td>절대값</td> </tr>
    <tr> <td>tf.ceil()</td> <td>상한 값</td> </tr>
    <tr> <td>tf.cos()</td> <td>cos 값</td> </tr>
    <tr> <td>tf.exp()</td> <td>밑이 e인 지수 값</td> </tr>
    <tr> <td>tf.floor()</td> <td>하한 값</td> </tr>
    <tr> <td>tf.inv()</td> <td>역수 값</td> </tr>
    <tr> <td>tf.log()</td> <td>자연 로그 값</td> </tr>
    <tr> <td>tf.maximum()</td> <td>최댓값</td> </tr>
    <tr> <td>tf.minimum()</td> <td>최솟값</td> </tr>
    <tr> <td>tf.neg()</td> <td>부호 반전 값</td> </tr>
    <tr> <td>tf.pow()</td> <td>거듭제곱한 값</td> </tr>
    <tr> <td>tf.round()</td> <td>반올림 값</td> </tr>
    <tr> <td>tf.resqrt()</td> <td>제곱근의 역수 값</td> </tr>
    <tr> <td>tf.sign()</td> <td>부호 반환</td> </tr>
    <tr> <td>tf.sin()</td> <td>sin 값</td> </tr>
    <tr> <td>tf.sqrt()</td> <td>제곱근 값</td> </tr>
    <tr> <td>tf.square()</td> <td>제곱 값</td> </tr>
</table>

<table>
    <tr> <td>tf.digamma()</td> <td>igamma() 함수의 도함수인 프사이 함수</td> </tr>
    <tr> <td>tf.erf()</td> <td>가우스 오차 함수</td> </tr>
    <tr> <td>tf.ertc()</td> <td>여오차 함수</td> </tr>
    <tr> <td>tf.igamma()</td> <td>하부 정규화 불완전 감마 함수</td> </tr>
    <tr> <td>tf.igammac()</td> <td>상부 정규화 불완전 감마 함수</td> </tr>
    <tr> <td>tf.lbeta()</td> <td>베타 함수 절대값의 자연로그 값</td> </tr>
    <tr> <td>tf.lgamma()</td> <td>감마 함수 절대값의 자연로그 값</td> </tr>
    <tr> <td>tf.squared_difference()</td> <td>차의 제곱 값</td> </tr>
</table>

<br>

##### In
```python
def custom_polynomial(value):
    return (tf.subtract(3 * tf.square(value), value) + 10)
print(sess.run(custom_polynomial(11)))
```
##### Out
    362
    
<br>

### 활성화 함수 구현

> 활성화 함수의 목표는 가중치와 평향치를 조절하는 것이다. <br>
> 텐서플로에서 활성화 함수는 텐서에 적용되는 비선형 연산이다. <br>

<br>

선형 유닛 함수
> min (max (0 , x))


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

시그모이드 함수 (로지스틱 함수)
> 1 / (1 + exp(-x))

<br>

##### In
```python
print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))
```
##### Out
    [0.26894143 0.5        0.7310586 ]
    
<br>

하이퍼볼릭 탄젠트 함수
> (exp(x) - exp(-x)) / (exp(x) + exp(-x))

<br>

##### In
```python
print(sess.run(tf.nn.tanh([-1.,0.,1.])))
```
##### Out
    [-0.7615942  0.         0.7615942]

<br>

softsign 함수
> x / (abs(x) + 1)

<br>

##### In
```python
print(sess.run(tf.nn.softsign([-1.,0.,1.])))
```
##### Out
    [-0.5  0.   0.5]
    
<br>

softplus 함수
> log(exp(x) + 1)

<br>

##### In
```python
print(sess.run(tf.nn.softplus([-1.,0.,1.])))
```
##### Out
    [0.31326166 0.6931472  1.3132616 ]
   
<br>

지수 선형 유닛 함수
> if x < 0 : (exp(x) + 1) <br>
> else : x

<br>

##### In
```python
print(sess.run(tf.nn.elu([-1.,0.,1.])))
```
##### Out
    [-0.63212055  0.          1.        ]
    
