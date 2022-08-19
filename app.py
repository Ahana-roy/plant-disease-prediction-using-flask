import sys
import os
import glob
import re
import numpy as np
import cx_Oracle
from cv2 import cv2
import psycopg2
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session,sessionmaker
from passlib.hash import sha256_crypt
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session,logging,redirect,flash
from werkzeug.utils import secure_filename
import urllib.request
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']='postgresql://postgres(user):password@localhost/plantai'


UPLOAD_FOLDER = 'path/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 440 * 440

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Model saved with Keras model.save()
MODEL_PATH = 'path/models/plantcnn.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model._make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    #update by ViPS
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(256,256))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 256,256, 3)


    preds = model.predict(new_arr)
    return preds


@app.route('/')
def home():
    # Main page
    return render_template('home.html',title='PLANT DISEASE PREDICTION')

@app.route('/login',methods=["GET", "POST"])
def login():
    # Main page
    if request.method=='POST':
        email = request.form.get("email")
        password = request.form.get("password")

        usernamedata = db.execute("SELECT email FROM PLANTAI where email=:email",{"email":email}).fetchone()
        passworddata = db.execute("SELECT password FROM PLANTAI where email=:email",{"email":email}).fetchone()

        if usernamedata is None:
            flash("No Username","danger")
            return render_template('login.html')
        
        else:
            for pdata in passworddata:
                if sha256_crypt.verify(password,pdata):
                    session["log"]=True
                    flash("You are now logged in!!","success")
                    return redirect(url_for('upload'))
                else:
                    flash("Incorrect Password","danger")
                    return render_template('login.html')


    return render_template('login.html',title='LOGIN!!')

@app.route('/register',methods=["GET", "POST"])
def register():
    # Main page
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        secure_password = sha256_crypt.encrypt(str(password))

        try:
            if password == confirm:
                db.execute("INSERT INTO PLANTAI(NAME,EMAIL,PASSWORD,PHONE) VALUES (:name,:email,:password,:phone)",
                            {"name":name,"email":email,"password":secure_password,"phone":phone})
                db.commit()
                flash("Account Created, Login Next!","success")
                return redirect(url_for('login'))
            else:
                flash("password doesn't match","danger")
                return render_template('register.html',title="REGISTER!!")
        except Exception as e:
            flash('Mail id already registered','danger')
            return render_template('register.html',title="REGISTER!!")

    return render_template('register.html',title="REGISTER!!")


@app.route("/logout")
def logout():
    session.clear()
    flash("YOU are now logged out","success")
    return redirect(url_for("home"))
    
@app.route('/upload')
def upload():
    # if not session.get('log'):
    #     return render_template('login.html')
    return render_template('upload.html')
 
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below','success')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/predict/<filename>')
def predict_image(filename):
    file_path= 'E:/pythonproject/PlantDiseasePrediction/static/uploads/' + filename
    

    preds = model_predict(file_path, model)

        # Process your result for human
    pred_class = preds.argmax()              # Simple argmax

    
    CATEGORIES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
        'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',
        'Tomato_Bacterial_spot' ,'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite' ,'Tomato__Target_Spot',
        'Tomato__YellowLeaf__Curl_Virus', 'Tomato_mosaic_virus',
        'Tomato_healthy']

    return render_template('upload.html',result=CATEGORIES[pred_class])

        #return CATEGORIES[pred_class]
    
@app.route('/remedy')
def remedy():
    # Main page
    return render_template('remedy.html')


@app.route('/care')
def care():
    return render_template('care.html')


@app.route('/variety')
def variety():
    return render_template('variety.html')

if __name__ == '__main__':
    app.secret_key="123456plantai"
    app.run(debug=True)

