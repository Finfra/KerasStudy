# Iris Version
* Source https://yamalab.tistory.com/31


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
```


```python
iris = datasets.load_iris()
print(iris.DESCR)
```

# Data Exploration
* source : [데이터 사이언스 스쿨 알파(05.01 분류용 예제 데이터)](https://datascienceschool.net/view-notebook/577a01e24d4f456bb5060da6e47054e1/)
* DataFrame로 만들어서 작업





```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
sy = pd.Series(iris.target, dtype="category")
sy = sy.cat.rename_categories(iris.target_names)
df['species'] = sy
df.tail()
```

* 각 특징값의 분포와 상관관계를 히스토그램과 스캐터플롯으로 나타내면 다음과 같다.


```python
sns.pairplot(df, hue="species")
plt.show()
```

* 이 분포를 잘 살펴보면 꽃잎의 길이만으로도 세토사와 다른 종을 분류할 수 있다는 것을 알 수 있다.


```python
sns.distplot(df[df.species != "setosa"]["petal length (cm)"], hist=True, rug=True, label="setosa")
sns.distplot(df[df.species == "setosa"]["petal length (cm)"], hist=True, rug=True, label="others")
plt.legend()
plt.show()
```

# Data Preparation


```python
X = iris.data[:, 0:4]
y = iris.target

# 자동으로 데이터셋을 분리해주는 함수
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 데이터 표준화 작업
sc = StandardScaler()
sc.fit(X_train)

# 표준화된 데이터셋
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```


```python
iris.feature_names
```


```python
iris_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
iris_tree.fit(X_train, y_train)
```


```python
from sklearn.metrics import accuracy_score

y_pred_tr = iris_tree.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_tr))
```


```python
# Apt install for graphviz
!x=$(cat /etc/apt/sources.list|grep "http://kr.archive.ubuntu.com/ubuntu/");[ ${#x} -eq 0 ] && echo "deb http://kr.archive.ubuntu.com/ubuntu/ bionic universe" >> /etc/apt/sources.list
!apt update --upgrade
!apt install python-pydot python-pydot-ng graphviz
!pip install pydotplus
    
```


```python
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

dot_data = export_graphviz(iris_tree, out_file=None, feature_names=['sepal length', 'sepal width','petal length', 'petal width'],
                          class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
```


```python

```
