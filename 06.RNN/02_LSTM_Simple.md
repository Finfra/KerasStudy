# LSTM
기존의 RNN이 데이터의 연속성을 기억할 수 없다는 단점을 보완하여 장/단기 기억을 가능하게 설계한 신경망의 구조를 말합니다. 주로 시계열 처리나, 자연어 처리에 사용됩니다.

## 0. 최소 필요 라이브러리


```python
from numpy import array 
from keras.models import Sequential
from keras.layers import Dense, LSTM
 
```

## 1. 데이터

- 3 개의 값을 통해 다음 값을 유추해내는 것을 목표로 데이터를 구성했습니다.


```python
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])
y = array([4,5,6,7,8])
 
print('x shape : ', x.shape)
print('y shape : ', y.shape)
```

- LSTM 은 그냥 데이터 값 외에 연속성에 대한 관계도 저장되어야 하기 때문에 차원이 하나 더 필요합니다.


```python
print('-------x reshape-----------')
x = x.reshape((x.shape[0], x.shape[1], 1)) # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
print('x shape : ', x.shape)
print(x)
```

## 2. 모델 구성


```python
model = Sequential()
# 들어온 데이터보다 많은 레이어를 잡아줘야 값이 잘 나오는 것 같음.
model.add(LSTM(10, activation = 'relu', input_shape=(x.shape[1],x.shape[2])))  # timestep, feature
# 예전 Tensorflow 에서는 LSTM 에서 연결되게 작업을 해주는게 있었으나 이제 그냥 연결 됨.
model.add(Dense(1))

model.summary()
```


```python
model.compile(optimizer='adam', loss='mse')
```

## 3. 모델 학습


```python
model.fit(x, y, epochs=400, batch_size=1)
```




    <keras.callbacks.History at 0x7f3948f17e50>



## 4. 예측


```python
x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat) # 예상 값 9
```

## 5. 모델 평가
R2score 로 평가해준다. 1에 가까울 수록 좋은 모델이다.


```python
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
def PrintRegScore(y_true, y_pred):
    print('explained_variance_score: {}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_errors: {}'.format(mean_squared_error(y_true, y_pred)))
    print('r2_score: {}'.format(r2_score(y_true, y_pred)))

# 예측용 데이터
y_true = [8,9,12,14]
x2 = array([[5,6,7], [6,7,8], [9,10,11], [11,12,13]])
x_scaled = x2.reshape((x2.shape[0], x2.shape[1], 1))
y_pred = model.predict(x_scaled)
PrintRegScore(y_true, y_pred)
```


```python
y_true
```


```python
y_pred
```


```python

```
