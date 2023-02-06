# Tensorflow Study Example
- 예전 버전 혹은 Tensorflow버전은 https://github.com/Finfra/TensorflowStudyExample 을 참고 하세요.


## Keras Study
```
본 문서는 Keras를 사용하여 Deep Learning을 구현하기 위한 기초적인 실습 자료.
```

The code and comments are written by NamJungGu <nowage@gmail.com> <br>
Maintained by  SungKukKim <nackchun@gmail.com> <br>


<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.




## Example Agenda
### [00.Background](https://github.com/Finfra/KerasStudy/tree/main/00.Background)
### [01.CoLab](https://github.com/Finfra/KerasStudy/tree/main/01.CoLab)
### [02.AnalyticProcess](https://github.com/Finfra/KerasStudy/tree/main/02.AnalyticProcess)
### [03.KerasIntro](https://github.com/Finfra/KerasStudy/tree/main/03.KerasIntro)
### [04.MLP](https://github.com/Finfra/KerasStudy/tree/main/04.MLP)
### [05.CNN](https://github.com/Finfra/KerasStudy/tree/main/05.CNN)
### [06.RNN](https://github.com/Finfra/KerasStudy/tree/main/06.RNN)
### [07.AutoEncoder](https://github.com/Finfra/KerasStudy/tree/main/07.AutoEncoder)
### [08.ReinforcemetLearning](https://github.com/Finfra/KerasStudy/tree/main/08.ReinforcemetLearning)
### [09.EtcExample](https://github.com/Finfra/KerasStudy/tree/main/09.EtcExample)
### [10.Project](https://github.com/Finfra/KerasStudy/tree/main/10.Project)

---


## info

### Version Info
* Python3.6
* Tensorflow1.15
* Jupyter5.x or new

### BUGS

Please report bugs to nackchun@gmail.com

### todo
- Support The Version

### CONTRIBUTING

The github repository is at https://github.com/Finfra/KerasStudy

### SEE ALSO

Some other stuff.

### AUTHOR

NamJungGu, <nowage[at]gmail.com>

### COPYRIGHT AND LICENSE

(c) Copyright 2005-2021 by finfra.com

### References
## Link
- Keras Home : https://keras.io/kr
- Keras(github) : https://github.com/keras-team/keras
- slideShare Keras 빨리 훑어볿기 https://www.slideshare.net/madvirus/keras-intro
- DeepBrick for Keras (케라스를 위한 딥브릭) : https://tykimos.github.io/DeepBrick/
- 케라스 이야기 : https://tykimos.github.io/2017/01/27/Keras_Talk/
- Keras Example : https://github.com/tgjeon/Keras-Tutorials.git
- Keras Tutorial (데이터 사이언스 스쿨): https://datascienceschool.net/view-notebook/995bf4ee6c0b49f8957320f0a2fea9f0/

# Snippets
## Training Snippet
* tensorflow.keras

### Model
#### Sequential
* keras.engine.sequential.Sequential
```python
from tensorflow.python.keras.models import Sequential
model = Sequential()
```

#### summary
```python
model.summary()
```
* image version
```python
## model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='vgg.png')
```

#### compile
```python
model.compile(loss='categorical_crossentropy',
optimizer=RMSprop(),
metrics=['accuracy'])
```

#### fit
```python
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
```
#### predict_classes
```python
from matplotlib import pyplot as plt
i=plt.imread("/content/DeepLearningStudy/data/MNIST_Simple/test/0/0_1907.png")
img=i[:,:,1:2].reshape(1,28,28,1)
print(model.predict_classes(img) )
```

#### save
* Model 저장.
```python
model.save("/content/gdrive/mnist.h5")
```

* 모델은 Json파일로 저장되고 Weigth파일은 hdf5 형태로 저장시
```python
model_json = model.to_json()
with open("/content/mnist_mlp_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/content/mnist_mlp_model.h5")
```

#### load
```python
from keras.models import load_model
model = load_model("/content/gdrive/mnist.h5")
```
* 모델은 Json파일로 저장되고 Weigth파일은 hdf5 형태로 저장시
```python
from tensorflow.python import keras
from keras.models import model_from_json
from tensorflow.python.keras.models import model_from_json
with open('/content/gdrive/mnist_mlp_model.json', 'r') as json_file:
  loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/content/gdrive/mnist_mlp_model.h5")
```

#### hdf5 파일에서 정보 읽기
* 저장
```python
model.add(layers.Conv2D(32, (3, 3), ,,,,  ,name='c1'))
            <<~ Omitted ~>>
model.save_weights("/content/gdrive/mnist_model.h5")
```
* 읽기
```python
import h5py
import numpy as np
filename = "/content/gdrive/mnist_model.h5"
h5 = h5py.File(filename,'r')
print(h5.keys())
b=h5['c1']['c1']['bias:0']
k=h5['c1']['c1']['kernel:0']
bb=np.array(b)
print(bb)
kk=np.array(k)
kk[:,:,:,0].reshape((3,3))
h5.close()
```



### Callback
* Usage of callbacks : https://keras.io/ko/callbacks/

#### CheckPoint
```python
epochs = 40
batch_size = 100
![ ! -d /content/ckpt ] &&mkdir /content/ckpt

from tensorflow.keras.callbacks import ModelCheckpoint
filename = f'/content/ckpt/checkpoint-epoch-{epochs}-batch-{batch_size}-trial-001.h5'
checkpoint = ModelCheckpoint(filename,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='auto'
                            )
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
hist=model.fit(train_images,
               train_labels,
               batch_size=batch_size,
               epochs=epochs,
               validation_data=(test_images,test_labels),
               callbacks=[checkpoint]
             )
```
* 확인
```
!ls /content/ckpt
```


#### EarlyStopping
* 계속 같은 값이 20개 나오면 멈추기
```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25)
model.fit(X, Y, callbacks=[early_stopping])
```

#### Tensorboard Callback

```
from tensorflow.keras.callbacks import TensorBoard
import datetime

![ ! -d /content/logs/my_board/ ]&& mkdir -p /content/logs/my_board/
log_dir = "/content/logs/my_board/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

%load_ext tensorboard
%tensorboard --logdir {log_dir}

history = model.fit(X,Y,callbacks=[tensorboard_callback]

## Source : 학습과정 표시하기 (텐서보드 포함) : https://tykimos.github.io/2017/07/09/Training_Monitoring/

```


#### LearningRateScheduler
```python
from tensorflow.keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    if epoch < 10:
      #print(0.001)
      return 0.001
    else:
      return 0.001 * tf.math.exp(0.1 * (10 - epoch))
learning_rate_scheduler = LearningRateScheduler(scheduler)

model.fit(dataset, epochs=100, callbacks=[learning_rate_scheduler])

## Source : Tensorflow Callback 사용하기 : https://jins-sw.tistory.com/27

```

#### LambdaCallback
```python
from keras.callbacks import LambdaCallback

print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print("\n",model.layers[3].get_weights()))
history = model.fit(X,Y,callbacks=[print_weights])


## Source : https://rarena.tistory.com/entry/keras트레이닝-된되고있는-weigth값-확인 [deep dev]
```

#### Custom callback
```python
import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.previous_loss = None

    def on_epoch_begin(self, epoch, logs=None):
        print('\nFrom {}: Epoch {} is starting'.format(self.name, epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        print('\nFrom {}: Epoch {} ended.'.format(self.name, epoch + 1))

        if epoch > 0:
            if (logs['loss'] < self.previous_loss):
                print('From {}: loss got better! {:.4f} -> {:.4f}'.format(self.name, self.previous_loss, logs['loss']))

        self.previous_loss = logs['loss']

    def on_train_batch_begin(self, batch, logs=None):
        print('\nFrom {}: Batch {} is starting.'.format(self.name, batch + 1))

    def on_train_batch_end(self, batch, logs=None):
        print('\nFrom {}: Batch {} ended'.format(self.name, batch + 1))

first_callback = MyCallback('1st callback')

history = model.fit(X,Y,callbacks=[first_callback])


## Source : Tensorflow Callback 사용하기 : https://jins-sw.tistory.com/27

```
## Algorithm  Snippet
### MLP
* Basic
```python
from tensorflow.keras.layers import Dense
model.add(Dense(4, activation='relu', input_shape=(4,)))
```
* softmax
```python
model.add(Dense(4, activation='softmax'))
```
### CNN
#### Convolution
```python
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1), kernel_regularizer='l2'))
```
#### Max Pooling
```python
model.add(layers.MaxPooling2D((2, 2)))
```

#### Flatten
```python
model.add(layers.Flatten())
```

### RNN
#### RNN Example string
1. hihello
```
hihello
```
2. 떳다떳다비행기
```
떳다떳다비행기날아라날아라높이높이날아라우리비행기내가만든비행기날아라날아라높이높이날아라우리비행기
```
3. 미래도래미미
```
미래도래미미래래래미미미미래도래미미미래래미래도미래도래미미래래래미미미미래도래미미미래래미래도
```
4. 주기도문
```
하늘에 계신 우리 아버지여,이름이 거룩히 여김을 받으시오며,나라이 임하옵시며,뜻이 하늘에서 이룬 것 같이땅에서도 이루어지이다. 오늘날 우리에게 일용한 양식을 주옵시고,우리가 우리에게 죄 지은자를 사하여 준 것 같이우리 죄를 사하여 주옵시고,우리를 시험에 들게 하지 마옵시고,다만 악에서 구하옵소서. 대개 나라와 권세와 영광이 아버지께영원히 있사옵 나이다. - 아멘 -
```
5. 반야심경
```
관자재보살 행심반야바라밀다시 조견오온개공 도일체고액 사리자! 색불이공 공불이색 색즉시공 공즉시색 수상행식 역부여시 사리자! 시제법공상 불생불멸 불구부정 부증불감 시고공중무색 무수상행식 무안이비설신의 무색성향미촉법 무안계 내지무의 식계 무무명 역무무명진 내지무로사 역무로사진 무고집멸도 무지역무득. 이무소득고 보리살타의 반야바라밀다고 심무가애 무가애고 무유공포 원리전 도몽상 구경열반. 삼세제불의 반야바라밀다고 득아뇩다라삼먁삼보리. 고지 반야바라밀다 시대신주 시대명주 시무상주 시무등등주 능제일체고 진실불허 고설반야바라밀다주 즉설주왈 아제 아제 바라아제 바라승아제 모지 사바하 아제 아제 바라아제 바라승아제 모지 사바하 아제 아제 바라아제 바라승아제 모지 사바하.
```
#### SimpleRnn
```python
from tensorflow.keras.layers import SimpleRNN
model.add(SimpleRNN(10, activation = 'relu', input_shape=(input_w,1)))
```

#### LSTM
```python
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(input_w,1)))
## DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
model.add(Dense(5))
model.add(Dense(1))
```

#### n-depth RNN
```python
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(input_w, activation = 'relu',
               batch_input_shape=(1,input_w,dict_size),
               stateful=True,
               return_sequences = True
               )
)
model.add(LSTM(input_w, activation = 'relu',
               stateful=True ,
               return_sequences = False
               )
)
model.add(Dense(dict_size) )
model.summary()
```

#### Bidirectional RNN
```python
from keras.layers import Bidirectional
model = Sequential()
model.add(Bidirectional(LSTM(input_w, activation = 'relu',return_sequences=True),
               input_shape=(input_w,dict_size)
            )
)
model.add(Bidirectional(LSTM(input_w)))

model.add(Dense(dict_size) )
model.add(Activation('softmax'))

```

#### GRU
```python
from tensorflow.keras.layers import GRU
model = Sequential()
model.add(GRU(input_w, activation = 'relu', \
               batch_input_shape=(1,input_w,dict_size),stateful=True ) )
model.add(Dense(dict_size) )
model.summary()

```

### AutoEncoder

### DQN

## Job  Snippet
### CV
#### Object Dection
* yolo_v4 → Torch 사용

#### Image Segmentation


### NLP
#### Text Encoding
```python
from tensorflow.keras.preprocessing.text import Tokenizer
text="떳다떳다비행기날아라날아라높이높이날아라우리비행기내가만든비행기날아라날아라높이높이날아라우리비행기"
t = Tokenizer()
t.fit_on_texts(text)
print(t.word_index)
sub_text="높이높이날아라"
encoded=t.texts_to_sequences(sub_text)
print(encoded)
```
#### EDA를 위한 NLP시각화
* https://medium.com/plotly/nlp-visualisations-for-clear-immediate-insights-into-text-data-and-outputs-9ebfab168d5b
* https://statkclee.github.io/nlp2/nlp-text-viz.html
* https://kanoki.org/2019/03/17/text-data-visualization-in-python/

#### Tree Map
#### Lexical dispersion plot
#### Frequency distribution plot
#### Word length distribution plot
#### N-gram frequency distribution plot




## etc  Snippet
### History Graph
```python
## hist=model.fit(x_train, t_train, epochs=40,validation_data=(x_test, y_test))
def hist_view(hist):
  print('## training loss and acc ##')
  fig, loss_ax = plt.subplots()
  acc_ax = loss_ax.twinx()

  loss_ax.plot(hist.history['loss'], 'y', label='train loss')
  loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

  loss_ax.set_xlabel('epoch')
  loss_ax.set_ylabel('loss')
  loss_ax.legend(loc='center')

  acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
  acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
  acc_ax.set_ylabel('accuracy')
  acc_ax.legend(loc='center right')

  plt.show()

hist_view(hist)
```

### Encoding for RNN
* for input_shape and ( stateful=false), and base encoding
```python
input_string="goodMorning".lower()
char_set=sorted(set(input_string))
char_set=[c for c in char_set if c not in (',',' ','!','\n')]
## char_set=[c for c in char_set if c != ' ' and c != ',' and c != '!']
encoder={k:v for v,k in enumerate(char_set)}
decoder={v:k for v,k in enumerate(char_set)}
print('# Encoder')
print(encoder)
encoded_string= [encoder[c] for c in input_string ]
print('# Encoded string')
print(encoded_string)
string_width=len(input_string)
input_w=4
output_w=string_width-input_w
x=[]
y=[]
for i in range(output_w):
  x.append( encoded_string[i:i+input_w]  )
  y.append( encoded_string[input_w+i]    )
x=array(x)
y=array(y)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x,y)
```
* for batch_input_shape ( stateful=true), and one hot encoding
```python
input_string="떳다떳다비행기날아라날아라높이높이날아라우리비행기내가만든비행기날아라날아라높이높이날아라우리비행기"
char_set=sorted(set(input_string))
dict_size=len(char_set)
encoder={k:v for v,k in enumerate(char_set)}
one_hot_encoder=eye(dict_size)
decoder={v:k for v,k in enumerate(char_set)}
encoded_string= [encoder[c] for c in input_string ]
one_hot_encoded_string=[one_hot_encoder[i] for i in encoded_string]
string_width=len(input_string)
output_w=string_width-input_w
x=[];y=[]
for i in range(output_w):
  x.append( one_hot_encoded_string[i:i+input_w]  )
  y.append( one_hot_encoded_string[input_w+i]    )
x=array(x)
y=array(y)
x.shape
x = x.reshape(( output_w, input_w, dict_size))
```

### CheckString for RNN
* Check String [one hot 하기 전]
```python
ok=0
for i in range(output_w):
  test_string=input_string[i:i+input_w]
  x_input = array([encoder[c] for c in test_string ] )
  x_input = x_input.reshape((1,input_w,1))
  # print(f"# test string\n {test_string}")
  yhat = model.predict(x_input)
  org=input_string[i+input_w:i+input_w+1]
  su=round(yhat[0][0])
  if su >= len(decoder):su=len(decoder)-1
  if su < 0            :su=0

  out=decoder[su]
  if org==out :
    ok+=1
  print(f"{org} {out}  {org==out}")
pct=int(ok/output_w * 100  *10000)/10000
print(f"{ok}/{output_w}  acc={pct}%")
```
* Check string for one hot encoded data
```python
def test_it(test_string,y,debug=True):
  x_input = array([encoder[c] for c in test_string ] )
  x_input=[one_hot_encoder[i] for i in x_input]
  x_input=array(x_input)
  x_input = x_input.reshape((1,input_w,dict_size))
  yhat = model.predict(x_input)
  out=decoder[argmax(yhat)]
  isOk= y==out
  if debug:
    print(f"# {test_string} →  {out}   {isOk}")
  return isOk
print(f'# InputString : {input_string}')
okCount=0
for s in range(output_w):
  if test_it(input_string[s:input_w+s],input_string[input_w+s:input_w+s+1],False):
    okCount+=1
okPct=okCount/output_w * 100
print(f' {okPct}% : {okCount} / {output_w}')
```
