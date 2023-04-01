from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)
# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    features = [float(x) for x in request.form.values()]
    x = [np.array(features)]

    # Use the model to make a prediction
    prediction = model.predict(x)[0]

    # Customize the prediction message
    if prediction == 1:
        prediction_text = 'Your loan will be approved, Thank You!!'
    else:
        prediction_text = 'Your loan will be rejected, Thank You.. Better luck next time!!'

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
