import pickle
from pydoc import text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize global variables
analyzer = SentimentIntensityAnalyzer()
model = None
tokenizer = None

def load_keras_model():
    global model
    if model is None:
        model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

# Sentiment analysis with custom model
def sentiment_analysis(input_text):
    load_keras_model()  # Ensure model is loaded
    load_tokenizer()    # Ensure tokenizer is loaded
    user_sequences = tokenizer.texts_to_sequences([input_text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text = ""
    barChartData = None
    gauge_data = None

    if request.method == "POST":
        text = request.form.get("user_text", "")
        if text:
            # Use VADER for sentiment analysis
            vader_sentiment = analyzer.polarity_scores(text)
            vader_sentiment["custom_model_positive"] = sentiment_analysis(text)
            sentiment = vader_sentiment  # Combine results
            sentiment["custom model positive"] = sentiment_analysis(text)
            
            # Bar Chart Data
            barChartData = {
                'labels': ['Positive', 'Neutral', 'Negative'],
                'values': [vader_sentiment['pos'] * 100, vader_sentiment['neu'] * 100, vader_sentiment['neg'] * 100]
            }

            # Gauge Data
            gauge_data = {
                'value': sentiment["custom model positive"] * 100  # assuming the value is between 0 and 1
            }

    return render_template('form.html', text=text, sentiment=sentiment, barChartData=barChartData, gauge_data=gauge_data)




if __name__ == "__main__":
    app.run(debug=True)
