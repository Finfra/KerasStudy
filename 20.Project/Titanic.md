# 출처. [캐글타이타닉](https://www.kaggle.com/munmun2004/titanic-for-begginers/comments#871092)


```
#!kaggle competitions download -c titanic
```


```
!ls
```

# 데이터 load


```
import pandas as pd
import numpy as np
```


```
test_df = pd.read_csv('./test.csv')
train_df = pd.read_csv('./train.csv')
submission_df = pd.read_csv('gender_submission.csv')
print ( test_df.shape, train_df.shape, submission_df.shape)
```


```
test_df.head()
```


```
train_df.head()
```


```
submission_df.head()
```

# 데이터 분석
test로 submission 의 값을 나오게 하는게 목표인듯 함. train에서는 Survived 를 빼서 target으로 만들어 줘야할거 같고, submission 에서는 불필요한 PassengerId 를 지우고 0,1 을 기반으로 원핫 인코딩을 해줘야 할 것 같음.


```
train_target = train_df['Survived']
print( train_target.shape, train_target.head())
```


```
submission_df = submission_df['Survived']
print(  submission_df.shape, submission_df.head())
```


```
from tensorflow import keras

train_y = keras.utils.to_categorical(train_target, 2)
submission_y = keras.utils.to_categorical(submission_df, 2)
print(train_y.shape, submission_y.shape)
```


```
train_x = train_df.drop(['PassengerId','Survived', 'Name', 'Ticket', 'Embarked', 'Cabin'], axis=1, inplace=False)
#train_x = train_x.to_numpy()
test_x = test_df.drop([ 'PassengerId','Name', 'Ticket', 'Embarked', 'Cabin'], axis=1, inplace=False)
#test_x = test_x.to_numpy()
print (train_x, test_x, train_x.shape)
```


```
#train_x [train_x.Sex == 'male']['Sex'] = 1
change_dict = {'male' : 1., 'female' : 0. }
train_x = train_x.replace({'Sex' : change_dict})
train_x.fillna(0, inplace=True)
test_x = test_x.replace({'Sex' : change_dict})
test_x.fillna(0, inplace=True)

```


```
print (test_x)
```


```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(6,)))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))

model.add(Dense(2, activation='softmax'))
model.summary()

```


```

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'])
```


```
model.fit(train_x.astype(float), train_y,
          batch_size=10,
          epochs=100,
          verbose=1
)
```




    <tensorflow.python.keras.callbacks.History at 0x7f91e24a0450>




```
score = model.evaluate(test_x.astype(float), submission_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```


```

```
