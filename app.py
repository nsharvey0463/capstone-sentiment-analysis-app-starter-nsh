import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

model = None
tokenizer = None

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# Load the model and tokenizer before the first request using app.before_first_request
with app.app_context():
    load_tokenizer()
    load_keras_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sid = SentimentIntensityAnalyzer()
        text = request.form.get("user_text")
        sentiment = sid.polarity_scores(text)
        # Use sid instead of analyzer 
        sentiment = sid.polarity_scores(text) # VADER Results
        sentiment['custom model positive'] = sentiment_analysis(text)

        return render_template("form.html", sentiment=sentiment)
    else:
        return render_template("form.html")           

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]),2)
    
if __name__ == "__main__":
   app.run(debug=True)