
from flask import Flask, request, render_template

from tensorflow.keras.models import load_model
import cv2
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')



app = Flask(__name__)
model = load_model('savedmodel.h5')


# model._make_predict_function()
def make_prediction(model, img_path):
    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized_arr = cv2.resize(img_arr, (150, 150))
    resized_arr = resized_arr / 255.
    x = resized_arr.reshape(-1, 150, 150, 1)
    preds = (model.predict(x),model.predict_classes(x))
    return preds


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('static\\img\\uploaded', f.filename)
        print(file_path)
        f.save(file_path)
        result = make_prediction(model, file_path)
        print([result[0][0][0]*100,100-(result[0][0][0]*100)])
        sns.barplot(x=["Pneumonia","Normal"],y=[result[0][0][0]*100,100-(result[0][0][0]*100)])
        plt.ylabel("Percent of the prediction for each class")
        plt.title("Prediction [%]")
        plot_path = os.path.join('static\\img\\uploaded', f.filename + "_plot.png")
        print(plot_path)
        plt.savefig(plot_path)
    return render_template("predict.html", pred="Patient with this RTG does{}suffer from pneumonia".format(" NOT " if result[1][0][0] == 0 else " "), plot = plot_path)


if __name__ == '__main__':
    app.run(debug=True)
