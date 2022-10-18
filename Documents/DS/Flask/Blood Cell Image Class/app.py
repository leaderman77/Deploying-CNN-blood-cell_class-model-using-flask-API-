from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.applications.Xception import preprocess_input
import os
import cv2
import pandas as pd
from tensorflow.keras.preprocessing import image
from load import *

global model

model = init()

app = Flask(__name__)
#model = load_model('model.h5')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function to load and prepare the image in right shape
def read_image(filename):
    #img = load_img(filename, target_size=(150, 150))

    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
#   #x = preprocess_input(x)

    w, h = 150, 150
    final_class = 8

    pred_image = []
    img = load_img(filename)
    x = image.img_to_array(img)
    img = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred_image.append(img)

    x = np.array(pred_image)
    x = x / 255
    return x


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)
            print("Class: ")
            print(classes_x)
            if classes_x == 0:
                cellType = "basophil"
                print("basophil")
            elif classes_x == 1:
                cellType = "eosinophil"
                print("eosinophil")
            elif classes_x == 2:
                cellType = "erythroblast"
                print("erythroblast")
            elif classes_x == 3:
                cellType = "ig"
                print("ig")
            elif classes_x == 4:
                cellType = "lymphocyte"
                print("lymphocyte")
            elif classes_x == 5:
                cellType = "monocyte"
                print("monocyte")
            elif classes_x == 6:
                cellType = "neutrophil"
                print("neutrophil")
            else:
                cellType = "platelet"
                print("platelet")
            return render_template('predict.html', cellType=cellType, prob=class_prediction, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)