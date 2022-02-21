# Load & Save, MSE, Confusion Matrix

The code and comments are by NamJungGu <nowage@gmail.com> 

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.






## 1. Save & Load

### Save


```python
import tensorflow as tf
import numpy as np


x=np.float32(np.random.rand(10,1))
np.savetxt('./x.csv',x,fmt='%f',delimiter=',')
print(x)
y=np.float32(np.random.rand(10,1))
np.savetxt('./y.csv',y,fmt='%f',delimiter=',')
print(y)
```

## Load


```python
x_load=np.loadtxt('./x.csv',delimiter=',')
y_load=np.loadtxt('./y.csv',delimiter=',')
arr=np.hstack([x,y])
arr
```

## 2. MSE(Mean Squared Error)


```python
y=np.array([1.,1.,1.,1.])
yTarget=np.array([0.8, 1.1,1.2,1.1])
np.square( (y-yTarget)**2  ).mean()
```

## 3. Confusion Matrix



```python
# Example of a confusion matrix in Python
from sklearn.metrics import confusion_matrix
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



expected = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
results = confusion_matrix(expected, predicted)
print(results)
```


```python
# fail나면 root계정에서  "pip install seaborn" 명령 실행할 것. 

import seaborn as sn

df_cm = pd.DataFrame(results, range(2),
                  range(2))
plt.figure(figsize = (10,7))

sn.heatmap(df_cm, annot=True)
```

## 4. Numpy Matrix Operation

### Matrix 만들기


```python
import numpy as np
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
```

### Column추출


```python
arr_data=arr[:,0:2]
arr_data
```


```python
arr_label=arr[:,2:3]
arr_label
```

### 합치기 
#### 열(column) 합치기


```python
np.hstack([arr_data,arr_label])
```

#### 행(row) 합치기


```python
arr1=np.array([[11,12],[13,14],[15,16]])
arr2=np.array([[21,22],[23,24],[25,26]])
np.vstack([arr1,arr2])

```


```python

```
