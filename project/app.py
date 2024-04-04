import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import model

# Create flask app
flask_app = Flask(__name__)
model1 = pickle.load(open("model.pkl", "rb"))
vectorizer1 = pickle.load(open("vectorizer.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict(): 
    tweet_text = request.form['tweet']
    tweet_data = pd.Series(tweet_text)
    new_data = tweet_data.apply(model.clean_data)
    vect = vectorizer1.transform(new_data).toarray()  # Transform the input for the model
    prediction = model1.predict(vect)
    if prediction[0] == 0:
        prediction_text = "\nThe tweet has a positive impact."
    elif prediction[0] == 1:
        prediction_text = "\nThe tweet has a negative impact. Please consider revising and resubmitting."
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
