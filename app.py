from flask_restful import Resource, Api
from flask_cors import CORS
from crypt import methods
from pandas import DataFrame
from flask import Flask, render_template, request, redirect
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd

# Inisialisasi object flask
app = Flask(__name__)

# Inisialisasi object flask_restful
#api = Api(app)

# Inisialisasi flask_cors
#CORS(app)

model = pickle.load(open('bernoulli_nb.pkl', 'rb'))
file = pickle.load(open('selected_features.pkl', 'rb'))
#file = np.array(file)
#file = np.reshape(1, -1)
dataframe = pd.read_excel("datasets/data_clean_no_duplicate.xlsx")

x = dataframe['clean_teks']
y = dataframe['label']
tfvect = TfidfVectorizer(ngram_range = (1,2))

#vec_tf_idf = TfidfVectorizer(ngram_range = (1,2))
#vec_tf_idf.fit(x)
#x_tf_idf = vec_tf_idf.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)

def fake_news(berita):
    tfidf_x_train = tfvect.fit_transform(x_train)
    tfidf_x_test = tfvect.transform(x_test)
    input_data =  [berita]
    #vectorized_input_data = tfvect.transform(input_data)
    vectorized_input_data = TfidfVectorizer(decode_error="replace", vocabulary=set(file))
    prediction = model.predict(vectorized_input_data.fit_transform(input_data))
    
    return prediction

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':

        berita = request.form['berita']
        #datas = np.array(berita)
        #berita = np.reshape(1, -1)   
        pred = fake_news(berita)
        print(pred)


        return render_template('hasil.html', prediction = pred)
    else:
        return "Unsupported Request Method"



if __name__ == '__main__':
    app.run(port=5000, debug=True)



