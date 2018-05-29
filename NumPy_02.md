
# NumPy

참고 : 파이 데이터 사이언스 핸드북 58p-72p


```python
import numpy as np
```

### 유니버설 함수 (UFuncs)


```python
x = np.arange(4)
print('x = ', x)
print('x + 5 = ', x+5)
print('x - 5 = ', x-5)
print('x * 2 = ', x*2)
print('x / 2 = ', x/2)
print('x // 2 = ', x//2) # 몫
print('x % 2 = ', x%2)   # 나머지
print('x ** 2 = ', x**2) # 지수
print('-x = ', -x)
```

    x =  [0 1 2 3]
    x + 5 =  [5 6 7 8]
    x - 5 =  [-5 -4 -3 -2]
    x * 2 =  [0 2 4 6]
    x / 2 =  [0.  0.5 1.  1.5]
    x // 2 =  [0 0 1 1]
    x % 2 =  [0 1 0 1]
    x ** 2 =  [0 1 4 9]
    -x =  [ 0 -1 -2 -3]
    

산술 연산 함수
> .add() <br>
> .subtract() <br>
> .multiply() <br>
> .divide() <br>
> .floor_divide() <br>
> .mod() <br>
> .power() <br>
> .negative()


```python
print('x = ', x)
print('x + 5 = ', np.add(x,5))
print('x - 5 = ', np.subtract(x,5))
print('x * 2 = ', np.multiply(x,2))
print('x / 2 = ', np.divide(x,2))
print('x // 2 = ', np.floor_divide(x,2))
print('x % 2 = ', np.mod(x,2))
print('x ** 2 = ', np.power(x,2))
print('-x = ', np.negative(x))
```

    x =  [0 1 2 3]
    x + 5 =  [5 6 7 8]
    x - 5 =  [-5 -4 -3 -2]
    x * 2 =  [0 2 4 6]
    x / 2 =  [0.  0.5 1.  1.5]
    x // 2 =  [0 0 1 1]
    x % 2 =  [0 1 0 1]
    x ** 2 =  [0 1 4 9]
    -x =  [ 0 -1 -2 -3]
    

절대값 함수
> abs() <br>
> .absolute() <br>
> .abs()


```python
x = np.array([-2,-1,0,1,2])
print(abs(x))
print(np.absolute(x))
print(np.abs(x))
```

    [2 1 0 1 2]
    [2 1 0 1 2]
    [2 1 0 1 2]
    

삼각함수
> .sin() <br>
> .cos() <br>
> .tan() <br>
> .arcsin() <br>
> .arccos() <br>
> .arctan()


```python
theta = np.linspace(0, np.pi, 3)
print('theta = ', theta)
print('sin(theta) = ', np.sin(theta))
print('cos(theta) = ', np.cos(theta))
print('tan(theta) = ', np.tan(theta))
```

    theta =  [0.         1.57079633 3.14159265]
    sin(theta) =  [0.0000000e+00 1.0000000e+00 1.2246468e-16]
    cos(theta) =  [ 1.000000e+00  6.123234e-17 -1.000000e+00]
    tan(theta) =  [ 0.00000000e+00  1.63312394e+16 -1.22464680e-16]
    


```python
x = [-1,0,1]
print('x = ', x)
print('arcsin(x) = ', np.arcsin(x))
print('arccos(x) = ', np.arccos(x))
print('arctan(x) = ', np.arctan(x))
```

    x =  [-1, 0, 1]
    arcsin(x) =  [-1.57079633  0.          1.57079633]
    arccos(x) =  [3.14159265 1.57079633 0.        ]
    arctan(x) =  [-0.78539816  0.          0.78539816]
    

지수와 로그
> .exp() <br>
> .exp2() <br>
> .power()


```python
x = [1,2,3]
print('x = ', x)
print('e^x = ', np.exp(x))
print('2^x = ', np.exp2(x))
print('3^x = ', np.power(3,x))
```

    x =  [1, 2, 3]
    e^x =  [ 2.71828183  7.3890561  20.08553692]
    2^x =  [2. 4. 8.]
    3^x =  [ 3  9 27]
    


```python
x = [1,2,4,10]
print('x = ', x)
print('ln x = ', np.log(x))
print('log2 x = ', np.log2(x))
print('log10 x = ', np.log10(x))
```

    x =  [1, 2, 4, 10]
    ln x =  [0.         0.69314718 1.38629436 2.30258509]
    log2 x =  [0.         1.         2.         3.32192809]
    log10 x =  [0.         0.30103    0.60205999 1.        ]
    


```python
## x값이 작을 때 더 정확한 값이다.
x = [0, 0.001, 0.01, 0.1]
print('exp(x) - 1 = ', np.expm1(x))
print('log (1 + x) = ', np.log1p(x))
```

    exp(x) - 1 =  [0.         0.0010005  0.01005017 0.10517092]
    log (1 + x) =  [0.         0.0009995  0.00995033 0.09531018]
    

특수 함수


```python
from scipy import special
```


```python
x = [1,5,10]
print('gamma(x) = ', special.gamma(x))
print('gammaln(x) = ', special.gammaln(x))
print('beta(x,2) = ', special.beta(x,2))
```

    gamma(x) =  [1.0000e+00 2.4000e+01 3.6288e+05]
    gammaln(x) =  [ 0.          3.17805383 12.80182748]
    beta(x,2) =  [0.5        0.03333333 0.00909091]
    


```python
x = np.array([0, 0.3, 0.7, 1.0])
print('erf(x) = ', special.erf(x))
print('erfc(x) = ', special.erfc(x))
print('erfinv(x) = ', special.erfinv(x))
```

    erf(x) =  [0.         0.32862676 0.67780119 0.84270079]
    erfc(x) =  [1.         0.67137324 0.32219881 0.15729921]
    erfinv(x) =  [0.         0.27246271 0.73286908        inf]
    

### UFncs 고급 기능

출력 지정
> 대단히 큰 규모의 배열에서는 out 인수를 신중하게 사용함으로써 절약되는 메모리가 상당히 크다.


```python
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)
```

    [ 0. 10. 20. 30. 40.]
    


```python
y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)
```

    [ 1.  0.  2.  0.  4.  0.  8.  0. 16.  0.]
    

집계


```python
x = np.arange(1,6)
```


```python
## 결과가 하나만 남을 때까지 연산
print(np.add.reduce(x))
print(np.multiply.reduce(x))
```

    15
    120
    


```python
## 중간 결과 저장
print(np.add.accumulate(x))
print(np.multiply.accumulate(x))
```

    [ 1  3  6 10 15]
    [  1   2   6  24 120]
    

외적


```python
x = np.arange(1,6)
np.multiply.outer(x,x)
```




    array([[ 1,  2,  3,  4,  5],
           [ 2,  4,  6,  8, 10],
           [ 3,  6,  9, 12, 15],
           [ 4,  8, 12, 16, 20],
           [ 5, 10, 15, 20, 25]])



### 집계

> .sum() <br>
> .prod() <br>
> .mean() <br>
> .std() <br>
> .var() <br>
> .median() <br>
> .percentile() <br>
> .min() <br>
> .max() <br>
> .argmin() <br>
> .argmax() <br>
> .any() <br>
> .all()

합


```python
L = np.random.random(100)
print(sum(L))
print(np.sum(L))
```

    47.2573461465544
    47.25734614655439
    


```python
big = np.random.rand(10000000)
%timeit sum(big)
%timeit np.sum(big)
```

    1.16 s ± 19 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    10.5 ms ± 310 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    

최솟값, 최댓값


```python
%timeit min(big)
%timeit max(big)
%timeit np.min(big)
%timeit np.max(big)
```

    445 ms ± 4.67 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    445 ms ± 954 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    3.96 ms ± 60.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    3.88 ms ± 65.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    

다차원 집계


```python
M = np.random.random((3,4))
M
```




    array([[0.34791622, 0.23643475, 0.37276258, 0.27689493],
           [0.62915986, 0.59773876, 0.41217007, 0.23194053],
           [0.97118358, 0.96498814, 0.46033423, 0.59360769]])




```python
M.sum()
```




    6.095131333562109




```python
M.min(axis=0)
```




    array([0.34791622, 0.23643475, 0.37276258, 0.23194053])




```python
M.max(axis=1)
```




    array([0.37276258, 0.62915986, 0.97118358])


