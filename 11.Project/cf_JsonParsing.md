# Json Parsing 
* 결론 : eval()을 하면 굳이 파싱할 필요 없음.


```python
x="""[ 
      {"id":1 , "name":"aaa"}, 
      {"id":2 , "name":"bbb"} 
 ] 
""" 
xx=eval(x)
print(xx[0]['name'])

```


```python
for i in xx:
    print("%d,%s"%(i["id"],i["name"]))
 
```
