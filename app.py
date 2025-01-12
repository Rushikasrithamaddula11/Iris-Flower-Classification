from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Class labels for the iris dataset
class_labels = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

@app.route('/')
def home():
    result = ''
    return render_template('index.html', prediction=result, image='')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get form data
        sepal_length = float(request.form['SepalLength'])
        sepal_width = float(request.form['SepalWidth'])
        petal_length = float(request.form['PetalLength'])
        petal_width = float(request.form['PetalWidth'])
        
        # Make prediction using the model
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        
        # Map numerical prediction to class label
        prediction_label = class_labels.get(prediction, '')

        # Map the prediction to the corresponding image URL
        if prediction_label == 'Iris-setosa':
            image = 'https://tse4.mm.bing.net/th?id=OIP.sWFQQWcw-UeQmklPwINZXQHaFR&pid=Api&P=0&h=180'
        elif prediction_label == 'Iris-versicolor':
            image = 'https://tse3.mm.bing.net/th?id=OIP.66zT04N0pVmneCoRHEVkBAHaFj&pid=Api&P=0&h=180'
        elif prediction_label == 'Iris-virginica':
            image = 'https://tse3.mm.bing.net/th?id=OIP.j_cvNlsMUkkGtKn1651W4gHaHa&pid=Api&P=0&h=180'
        else:
            image = ''

        return render_template('index.html', prediction=prediction_label, image=image)
    return render_template('index.html', prediction='', image='')

if __name__ == '__main__':
    app.run(debug=True)
