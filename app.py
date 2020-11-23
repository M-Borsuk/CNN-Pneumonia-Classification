
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.types import ARRAY,FLOAT
from datetime import datetime

app = Flask(__name__)
load_dotenv()
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
db = SQLAlchemy(app)
migrate = Migrate(app, db)
model = load_model('savedmodel.h5')
os.environ.values()
class RTGModel(db.Model):
    __tablename__ = 'rtg_data'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime())
    rtg_arr = db.Column(ARRAY(FLOAT))
    pred = db.Column(db.Numeric())

    def __init__(self, date, rtg_arr, pred):
        self.date = date
        self.rtg_arr = rtg_arr
        self.pred = pred

    def __repr__(self):
        return f"<RTG {self.id}>"


#model._make_predict_function()
def make_prediction(model, f):
    img_arr = np.array(Image.open(f))
    resized_arr = cv2.resize(img_arr, (150, 150))
    resized_arr = resized_arr / 255.
    x = resized_arr.reshape(-1, 150, 150, 1)
    preds = (model.predict(x),model.predict_classes(x),x)
    return preds


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        result = make_prediction(model,f)
        new_entry = RTGModel(date=datetime.now(),rtg_arr=[float(x[0]) for x in result[2][0][0].tolist()],pred=int(result[1][0][0]))
        db.session.add(new_entry)
        db.session.commit()
    return render_template("predict.html", pred="Patient with this RTG does{}suffer from pneumonia".format(" NOT " if result[1][0][0] == 0 else " "), info = "Probability that a patient with this RTG photo suffers from pneumonia: {}%".format(np.round(result[0][0][0]*100,2)))


if __name__ == '__main__':
    app.run(debug=False)
