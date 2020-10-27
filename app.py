import flask
from flask import Flask, jsonify, request, render_template
import json
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model('savedmodel.h5')
#model._make_predict_function()
def make_prediction(model,img_path):
 img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
 resized_arr = cv2.resize(img_arr, (150,150))
 resized_arr = resized_arr / 255.
 x = resized_arr.reshape(-1,150,150,1)
 preds = model.predict(x)
 return preds


@app.route('/')
def home_page():
 return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
 response = json.dumps({'response': 'yahhhh!'})
 return response, 200




if __name__ == '__main__':
 app.run(debug=True)