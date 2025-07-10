from flask import Flask, request, render_template
import joblib

Vector=joblib.load(open("vectorizer_ngram.pkl","rb"))
model=joblib.load(open("fake_news_model_ngram.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/how')
def how():
    return render_template('how.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=="POST":
        news = str(request.form['news']).strip()

        if not news:
            message = "Oops! It seems like you accidentally pressed analyze without entering a news article or headline. Please enter some text to analyze."
            return render_template('result.html', prediction_text=message)

        prediction = model.predict(Vector.transform([news]))[0]
        label = "Real" if prediction == 1 else "Fake"
        print(prediction)

        return render_template('result.html', prediction_text="The Given News or News Headline is {label}".format(label=label))
    else:
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
