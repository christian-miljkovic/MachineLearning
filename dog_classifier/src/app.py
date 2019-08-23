from flask import Flask
import sys
sys.path.insert(0, '../model')

from predictor import make_prediction

app = Flask(__name__)


@app.route('/')
def index():
    return make_prediction('../model/data_sets/test_set/cats/cat.4001.jpg')

if __name__ == '__main__':
    app.run()
