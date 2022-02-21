# 2. RNN



### Recurrent Neural Networks
RNN은 자기자신을 향하는 weight를 이용해 데이터간의 시간관계를 학습할 수 있다. 이러한 문제들을 시계열 학습이라고 부르며, 기존에 널리 쓰이던 Hidden Markov Model을 뉴럴넷을 이용해 구현했다고 볼 수 있다.



## 필요 라이브러리


```python
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
 
```

## 데이터 예시 세팅
x　　y<br>
123　4<br>
234　5<br>
345　6<br>
456　7<br>
다음 수를 예측할 수 있는 모델을 목표로 하는 데이터 세팅


```python
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])
y = array([4,5,6,7,8])

print('x shape : ', x.shape) # (4,3)
print('y shape : ', y.shape) # (4,)

```

RNN 모델에 맞게 Reshape


```python
x = x.reshape((x.shape[0], x.shape[1], 1)) # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
print('x shape : ', x.shape)
print(x)
```

## 모델 구성


```python
model = Sequential()
model.add(SimpleRNN(10, activation = 'relu', input_shape=(3,1)))
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
model.add(Dense(5))
model.add(Dense(1))
 
model.summary()
```

## 모델 학습 시키기


```python
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)
 
```




    <keras.callbacks.History at 0x7f34b0467310>



## TEST


```python

x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

```

출처: https://ebbnflow.tistory.com/135 [Dev Log : 삶은 확률의 구름]


```python

```
