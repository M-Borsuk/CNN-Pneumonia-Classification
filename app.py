import flask
from flask import Flask, jsonify, request,render_template
import json
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('savedmodel.h5')
@app.route('/')
def home_page():
 return render_template('index.html')

#@app.route('/predict', methods=['GET'])
#def predict():
 #response = json.dumps({'response': 'yahhhh!'})
 #return response, 200
if __name__ == '__main__':
 app.run(debug=True)