# 다차원 LSTM

## 0. Library Import


```python
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
```

## 1. 데이터
1,2,3 을 예측 가능한 형태로시계열이 추가된 형태의 데이터로 구성하여 그 다음 값을 예측하게 구성


```python
# [[1,2,3],
# [2,3,4],
# [3,4,5],
# [4,5,6],
# [9,10,11]]
# 위의 각 값에 각 값이 유추 가능한 형태로 구성
x = array(
    [
      [[0,0,0,1],[0,0,1,2],[0,1,2,3]], 
      [[0,0,1,2],[0,1,2,3],[1,2,3,4]], 
      [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 
      [[1,2,3,4],[2,3,4,5],[3,4,5,6]],
      [[6,7,8,9],[7,8,9,10],[8,9,10,11]]
    ]
          )
y = array([4,5,6,7,12])
 
print('x shape : ', x.shape) # (3,4)
print('y shape : ', y.shape) # (5,)
```


```python
print('-------x reshape-----------')
# 이미 차원이 추가 되었기 때문에 reshape 이 필요 없다.
# x = x.reshape((x.shape[0], x.shape[1], 1)) # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
print('x shape : ', x.shape)
```


```python
print( x.shape[1], x.shape[2])
```

## 2. 모델 구성


```python
model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape=(x.shape[1],x.shape[2]))) # timestep, feature
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
# model.add(Dense(5))
model.add(Dense(1))
 
model.summary()

```


```python
model.compile(optimizer='adam', loss='mse')
```

## 3. 학습


```python
model.fit(x, y, epochs=300, batch_size=1)
```




    <keras.callbacks.History at 0x7f7d42f7a150>



## 4. 실행해보기


```python
x_input = array([[[0,0,0,1],[0,0,1,2],[0,1,2,3]]])
x_input.shape

yhaat = model.predict(x_input)
print(yhaat) # 예측 값 4
```

## 5. 모델 평가


```python
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
def PrintRegScore(y_true, y_pred):
    print('explained_variance_score: {}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_errors: {}'.format(mean_squared_error(y_true, y_pred)))
    print('r2_score: {}'.format(r2_score(y_true, y_pred)))

# 예측용 데이터
y_true = [12,14]
x_scaled = array([ [[6,7,8,9],[7,8,9,10],[8,9,10,11]], [[8,9,10,11],[9,10,11,12],[10,11,12,13]] ])
y_pred = model.predict(x_scaled)
PrintRegScore(y_true, y_pred)
```


```python
y_true
```


```python
y_pred
```
