from flask import Flask
from predictor import make_prediction
from flask import render_template
import random

app = Flask(__name__, static_folder="static")

def getRandomImage():
    rand_int = random.randint(0, 1)
    print(rand_int)
    if rand_int == 0:
        return "./static/data_sets/test_set/dogs/dog." + str(random.randint(4001,5000))+ ".jpg"
    else:
        return "./static/data_sets/test_set/cats/cat." + str(random.randint(4001,5000))+ ".jpg"


@app.route('/')
def index():
    img = getRandomImage()
    print(img)
    prediction = make_prediction(img)
    
    return """
<html>
    <head>
        <title>CatDog Identifier</title>
    </head>
    <body>    
        <h1>CatDog Identifier</h1>
        <img src=""" + img + """>
        <h3>""" + prediction + """</h3>
    </body>
</html>

"""

if __name__ == '__main__':
    app.run()
