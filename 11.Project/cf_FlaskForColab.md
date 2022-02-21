```
!pip install flask_ngrok
```


```bash
%%bash
cd /content
echo '
from flask_ngrok import run_with_ngrok
from flask import Flask
app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/")
def home():
    return "<h1>Running Flask on Google Colab!</h1>"

@app.route("/post/<int:post_id>")
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return "Post %d" % post_id
 
app.run()
'>app.py
```


```
!pip install flask_ngrok 
!pip install flask 

```


```
!python /content/app.py
```


```

```
