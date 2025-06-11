import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import Orthogonal
from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()
model = None
tokenizer = None

def load_keras_model():
    global model
    model = keras.models.load_model('model.h5', custom_objects={'Orthogonal': Orthogonal})

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# Load the model and tokenizer before the first request using app.before_first_request
#with app.app_context():
#    load_tokenizer()
#    load_keras_model()

@app.before_first_request
def before_first_request():
    load_keras_model()
    load_tokenizer()
    
def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]),2)
    
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text = ""

    if request.method == "POST":
        text = request.form["user_text"]
        sentiment = analyzer.polarity_scores(text)
        sentiment['custom model positive'] = sentiment_analysis(text)
        
    return render_template("form.html", sentiment=sentiment)

if __name__ == "__main__":
 app.run()
    
