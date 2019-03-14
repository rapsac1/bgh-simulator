
# A very simple Flask app
from flask import Flask, redirect, render_template, request, url_for
from fastai.text import *
from pathlib import Path
import os

#init code

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
learner = load_learner(dir_path / 'model' ,fname='production_model.pkl')


app = Flask(__name__)
app.debug = True




@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    start_text = request.form["contents"]
    text = learner.predict(start_text, 100, temperature=1.3, min_p=0.001)
    text = text.replace('xxbos ', '')
    return render_template("index.html", urteil=text)
