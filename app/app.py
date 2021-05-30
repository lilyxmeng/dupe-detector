from flask import Flask, render_template, request
from PIL import Image
from pytesseract import pytesseract
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from random import randint

app = Flask(__name__)

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

real = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

fake['type'] = 0
real['type'] = 1

modify_text = [real['text'][i].replace('WASHINGTON (Reuters) - ', '').replace('BRUSSELS (Reuters) - ', '').replace('MINSK (Reuters) - ', '').replace('MOSCOW (Reuters) - ', '').replace('JAKARTA (Reuters) - ', '').replace(
    'LONDON (Reuters) - ', '').replace('(Reuters) - ', '').replace('LIMA (Reuters) - ', '').replace('SAN FRANCISCO (Reuters) - ', '').replace('MEXICO CITY (Reuters) - ', '') for i in range(len(real['text']))]
real['text'] = modify_text

df = pd.concat([fake, real], axis=0)
df = df.drop(columns=['subject', 'date'])
df = df.sample(frac=1)
df = df.reset_index(drop=True)

X = df['text']  # input set
y = df['type']  # output set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True)

vectorization = TfidfVectorizer(stop_words='english', max_df=0.7)
Xv_train = vectorization.fit_transform(X_train)
Xv_test = vectorization.transform(X_test)

model = DecisionTreeClassifier()
model.fit(Xv_train, y_train)
predictions = model.predict(Xv_test)



@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/text", methods=["GET", "POST"])
def text():
    result = "Invalid article"
    news = request.form.get("news")

    if request.method == "POST" and len(news) > 10:
        vector_test = vectorization.transform([news])
        prediction = model.predict(vector_test)
        result = ouiOuNon(prediction)
    return render_template("result.html", result=result, news=news)


@app.route("/img", methods=["GET", "POST"])
def img():
    result = "Invalid article"
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("result.html", result=result)

        file = request.files["file"]

        if file.filename == "":
            return render_template("result.html", result=result)

        if file:
            file_open = Image.open(file)
            news = pytesseract.image_to_string(file_open, config='--psm 11')

            if len(news) > 10:
                vector_test = vectorization.transform([news])
                prediction = model.predict(vector_test)
                result = ouiOuNon(prediction)

    return render_template("result.html", result=result, news=news)


def ouiOuNon(n):
    fake = [
        "Seems like fake news... ﾍ(･_|", "That article is lookin a lil sus Σ(°△°|||)︴"]
    real = ["(๑˃ᴗ˂)ﻭ This seems real to me!", "Looks real! <(￣︶￣)>"]
    if n == 0:
        return fake[randint(0, len(fake) - 1)]
    else:
        return real[randint(0, len(real) - 1)]


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
