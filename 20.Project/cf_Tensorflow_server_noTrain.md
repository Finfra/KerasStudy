```
!cp -r /content/drive/MyDrive/Lec_Capture/_data/saved_model .
!pwd
!ls
```


```
script_path=input('Input path !')
```


```
%cd {script_path}
!ls
```


```
!bash installTensorflowServing.sh
```


```
!saved_model_cli show --dir /content/saved_model/1 --all
```


```
!saved_model_cli run  --dir /content/saved_model/1/ \
--signature_def serving_default                      \
--tag_set=serve                                \
--input_expr "dense_input=[[1., 0., 0., 0.]]"

```


```
%%shell
#실행전 스크립트를 죽인다. 
pnum=$(ps -ef|grep -v gre|grep model|awk '{printf $2}')
kill -9  $pnum

# rest server실행
tensorflow_model_server  \
--port=9018              \
--rest_api_port=9020     \
--model_name=iris        \
--model_base_path=/content/saved_model &


```


```
!curl http://127.0.0.1:9020/v1/models/iris:predict
```


```
!python3 servingRequest.py
```


```
# unbuntu용 이고요........redHat계열을 다름.. 
```
