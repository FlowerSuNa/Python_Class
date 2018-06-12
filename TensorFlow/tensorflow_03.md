
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

### 활성화 함수 구현

> 활성화 함수의 목표는 가중치와 평향치를 조절하는 것이다. <br>
> 텐서플로에서 활성화 함수는 텐서에 적용되는 비선형 연산이다. <br>

##### 선형 유닛 함수
> min (max (0 , x))


```python
print(sess.run(tf.nn.relu([-3.,3.,10.])))
```

    [ 0.  3. 10.]
    


```python
print(sess.run(tf.nn.relu6([-3.,3.,10.])))
```

    [0. 3. 6.]
    

##### 시그모이드 함수 (로지스틱 함수)
> 1 / (1 + exp(-x))


```python
print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))
```

    [0.26894143 0.5        0.7310586 ]
    

##### 하이퍼볼릭 탄젠트 함수
> (exp(x) - exp(-x)) / (exp(x) + exp(-x))


```python
print(sess.run(tf.nn.tanh([-1.,0.,1.])))
```

    [-0.7615942  0.         0.7615942]
    

##### softsign 함수
> x / (abs(x) + 1)


```python
print(sess.run(tf.nn.softsign([-1.,0.,1.])))
```

    [-0.5  0.   0.5]
    

##### softplus 함수
> log(exp(x) + 1)


```python
print(sess.run(tf.nn.softplus([-1.,0.,1.])))
```

    [0.31326166 0.6931472  1.3132616 ]
    

##### 지수 선형 유닛 함수
> if x < 0 : (exp(x) + 1) <br>
> else : x


```python
print(sess.run(tf.nn.elu([-1.,0.,1.])))
```

    [-0.63212055  0.          1.        ]
    
