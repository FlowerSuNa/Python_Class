
# TensorFlow

참고 : TensorFlow Machine Learning Cookbook


```python
import tensorflow as tf
sess = tf.Session()
```

    C:\Users\GIGABYTE\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    


```python
import numpy as np
```

### 텐서 생성

##### 고정 텐서

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


```python
# 0 값으로 채워진 텐서
zeros_tsr = tf.zeros([10,10])
print(zeros_tsr)

x = tf.Variable(zeros_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

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
    


```python
# 1 값으로 채워진 텐서
ones_tsr = tf.ones([10,10])
print(ones_tsr)

x = tf.Variable(ones_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

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
    


```python
# 동일한 상수 값으로 채워진 텐서
filled_tsr = tf.fill([10,10], 42)
print(filled_tsr)

x = tf.Variable(filled_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

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
    


```python
# 기존 상수를 이용해 텐서 생성
constant_tsr = tf.constant([1,2,3])
print(constant_tsr)

x = tf.Variable(constant_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("Const:0", shape=(3,), dtype=int32)
    [1 2 3]
    

##### 형태가 비슷한 텐서


```python
zeros_similar = tf.zeros_like(constant_tsr)
print(zeros_similar)

x = tf.Variable(zeros_similar)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("zeros_like:0", shape=(3,), dtype=int32)
    [0 0 0]
    


```python
ones_similar = tf.ones_like(constant_tsr)
print(ones_similar)

x = tf.Variable(ones_similar)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("ones_like:0", shape=(3,), dtype=int32)
    [1 1 1]
    

##### 순열 텐서


```python
linear_tsr = tf.linspace(start=0.0, stop=1, num=3)
print(linear_tsr)

x = tf.Variable(linear_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("LinSpace:0", shape=(3,), dtype=float32)
    [0.  0.5 1. ]
    


```python
integer_seq_tsr = tf.range(start=6, limit=15, delta=3)
print(integer_seq_tsr)

x = tf.Variable(integer_seq_tsr)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("range:0", shape=(3,), dtype=int32)
    [ 6  9 12]
    

##### 랜덤 텐서


```python
# 균등 분포를 따르는 난수
randunif = tf.random_uniform([5,5], minval=0, maxval=1)
print(randunif)

x = tf.Variable(randunif)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("random_uniform:0", shape=(5, 5), dtype=float32)
    [[0.47994602 0.27949786 0.47412622 0.0462718  0.8612536 ]
     [0.36950815 0.7915094  0.712957   0.9372518  0.5246682 ]
     [0.01603019 0.384122   0.72070074 0.9169071  0.19344926]
     [0.1104511  0.2980758  0.29821336 0.0358659  0.0964452 ]
     [0.24499059 0.8546803  0.05407107 0.3529098  0.36996067]]
    


```python
# 정규 분포를 따르는 난수
randnorm = tf.random_normal([5,5], mean=0.0, stddev=1.0)
print(randnorm)

x = tf.Variable(randnorm)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("random_normal:0", shape=(5, 5), dtype=float32)
    [[ 0.58746004 -0.20157363 -1.8584481   0.34749243  0.9767901 ]
     [ 0.9899406   1.0308627   1.3199621  -0.67974395  1.7002864 ]
     [-0.9548243  -0.6181818   1.539062    1.0252948   1.4210128 ]
     [-1.0111387  -0.34878066  0.07782032 -1.0935448  -1.0925817 ]
     [-1.0126168  -0.8538167   0.48519436  0.21645649 -0.5057047 ]]
    


```python
# 지정한 평균에서 표준편차 2배 이내에 속하는 정규 분포를 따르는 난수
runcnorm = tf.truncated_normal([5,5], mean=0.0, stddev=1.0)
print(runcnorm)

x = tf.Variable(runcnorm)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("truncated_normal:0", shape=(5, 5), dtype=float32)
    [[ 0.13184944  0.9158358   1.4713159  -0.862025   -1.4972615 ]
     [-0.05550429  0.06317545 -0.4878063   0.46186748 -1.0651603 ]
     [-0.02870526  0.2860767  -1.1595843  -1.0563946  -0.78925246]
     [ 0.00990442 -0.39756858 -1.02692    -0.31451374  0.08980581]
     [ 0.43964928 -0.65537506 -0.22230221 -0.82207286  1.3520684 ]]
    


```python
# 배열 항목을 임의로 뒤섞기
shuffled_output = tf.random_shuffle(x)
print(shuffled_output)

x = tf.Variable(shuffled_output)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("RandomShuffle:0", shape=(5, 5), dtype=float32)
    [[ 0.89549124 -0.08373728 -0.72742015  0.26157135 -1.6027105 ]
     [ 1.408906    1.5596704   0.19840299 -0.09341444  0.8897242 ]
     [-0.6443429   0.00210681  0.624688    0.9025737   1.3810036 ]
     [ 0.7633238   0.91395867 -0.63057286  0.85223305 -0.25467086]
     [-0.81976277 -0.29858184  0.8681306   0.00596275  0.6005683 ]]
    


```python
# 배열 항목을 임의로 섞고 형태 변환 
cropped_output = tf.random_crop(x,[3,5])
print(cropped_output)

x = tf.Variable(cropped_output)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
print(sess.run(x))
```

    Tensor("random_crop:0", shape=(3, 5), dtype=float32)
    [[-0.81976277 -0.29858184  0.8681306   0.00596275  0.6005683 ]
     [ 1.408906    1.5596704   0.19840299 -0.09341444  0.8897242 ]
     [ 0.89549124 -0.08373728 -0.72742015  0.26157135 -1.6027105 ]]
    

> tf.convert_to_tensort() : 어떤 numpy 배열이든 파이썬 리스트로 변환 <br>
> tf.global_variables_initializer() : 생성한 모든 변수 초기화 <br>
> 변수.initializer : 원하는 변수 초기화

### 행렬 다루기


```python
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3], 5.0)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-8.,-2.],[0.,5.,-2.]]))
```


```python
print(sess.run(identity_matrix))
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    


```python
print(sess.run(A))
```

    [[-0.07530292 -0.2598322  -1.8144697 ]
     [-0.31242022 -0.03449913  0.721739  ]]
    


```python
print(sess.run(B))
```

    [[5. 5. 5.]
     [5. 5. 5.]]
    


```python
print(sess.run(C))
```

    [[0.7825252  0.36602592]
     [0.4789884  0.7631116 ]
     [0.8019793  0.587728  ]]
    


```python
print(sess.run(D))
```

    [[ 1.  2.  3.]
     [-3. -8. -2.]
     [ 0.  5. -2.]]
    


```python
# 행렬의 합
print(sess.run(A+B))
```

    [[4.618255  3.9893596 5.345642 ]
     [6.139927  3.611445  4.943446 ]]
    


```python
# 행렬의 차
print(sess.run(B-B))
```

    [[0. 0. 0.]
     [0. 0. 0.]]
    


```python
# 행렬의 곱
print(sess.run(tf.matmul(B, identity_matrix)))
```

    [[5. 5. 5.]
     [5. 5. 5.]]
    


```python
# 행렬 전치
print(sess.run(tf.transpose(C)))
```

    [[0.8205328  0.50053453 0.517951  ]
     [0.98720133 0.94836295 0.72501945]]
    


```python
print(sess.run(tf.matrix_determinant(D)))
```

    -31.0
    


```python
# 역행렬
print(sess.run(tf.matrix_inverse(D)))
```

    [[-0.83870968 -0.61290323 -0.64516129]
     [ 0.19354839  0.06451613  0.22580645]
     [ 0.48387097  0.16129032  0.06451613]]
    


```python
# 숄레스키 분해
print(sess.run(tf.cholesky(identity_matrix)))
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    


```python
# 행렬의 고윳값과 고유 벡터 (고유 분해)
print(sess.run(tf.self_adjoint_eig(D)))
```

    (array([-11.38910525,  -0.29446882,   2.68357407]), array([[ 0.20901067,  0.59907787, -0.77292965],
           [ 0.86315172,  0.25849588,  0.43376144],
           [-0.45965601,  0.75781633,  0.4630667 ]]))
    

### 연산


```python
print(sess.run(tf.div(3,4)))
```

    0
    


```python
# 정수를 나누기 전에 소수로 변환해 항상 소수인 계산 결과를 반환
print(sess.run(tf.truediv(3,4)))
```

    0.75
    


```python
# 소수를 대상으로 정수 나눗셈
print(sess.run(tf.floordiv(3.,4.)))
```

    0.0
    


```python
print(sess.run(tf.mod(22.,5.)))
```

    2.0
    


```python
# 외적
print(sess.run(tf.cross([1.,0.,0.,],[0.,1.,0.])))
```

    [0. 0. 1.]
    

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


```python
def custom_polynomial(value):
    return (tf.subtract(3 * tf.square(value), value) + 10)
print(sess.run(custom_polynomial(11)))
```

    362
    
