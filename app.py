import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from flask import Flask, request ,render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

port_stem = PorterStemmer()

nltk.download('stopwords')

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

vectorizer=pickle.load(open("vectorizer.pkl",'rb'))

model=pickle.load(open("final_model.pkl",'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/description")
def description():
    return render_template("description.html")

@app.route("/real")
def real():
    return render_template("real.html")

@app.route("/fake")
def fake():
    return render_template("fake.html")

@app.route("/intro")
def intro():
    return render_template("intro.html")

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method=="POST":
         news=request.form['author']+' '+request.form['title']
         news=stemming(news)
         predict=model.predict(vectorizer.transform([news]))
         print(predict)
         if predict==1:
              return render_template("fake.html")
         else:
             return render_template("real.html")             
    else:
        return render_template("prediction.html")

if __name__=='__main__':
    app.run()