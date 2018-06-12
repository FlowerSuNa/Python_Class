
# TensorFlow

##### 참고 : TensorFlow Machine Learning Cookbook


```python
import tensorflow as tf
sess = tf.Session()
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

<br>

### 계산 그래프의 연산

##### In
```python
# 투입할 데이터와 플레이스홀더 생성
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)

# 상수 생성
m_const = tf.constant(3.)

# 연산 선언
my_product = tf.multiply(x_data, m_const)

# 그래프에 데이터 투입
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data:x_val}))
```
##### Out
    3.0
    9.0
    15.0
    21.0
    27.0
 
<br>

### 다중 연산 중첩

##### In
```python
# 투입할 데이터와 플레이스홀더 생성
my_array = np.array([[1., 3., 5., 7., 9.], 
                     [-2., 0., 2., 4., 6.], 
                     [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3,5))

# 상수 생성
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 연산 선언
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

# 그래프에 데이터 투입
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data:x_val}))
```
##### Out
    [[102.]
     [ 66.]
     [ 58.]]
    [[114.]
     [ 78.]
     [ 70.]]
    
<br>

### 다층 처리

##### In
```python
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)
x_data = tf.placeholder(tf.float32, shape=x_shape)

my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]

mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')
```


```python
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b)
    return(tf.sigmoid(temp))
```


```python
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)
```


```python
print(sess.run(custom_layer1, feed_dict={x_data:x_val}))
```
##### Out
    [[0.938902   0.94110966]
     [0.93825805 0.90098923]]
    
