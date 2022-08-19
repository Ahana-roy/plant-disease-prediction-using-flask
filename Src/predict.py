import keras
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
CATEGORIES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
 'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',
 'Tomato_Bacterial_spot' ,'Tomato_Early_blight', 'Tomato_Late_blight',
 'Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite' ,'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']


def image(path):
    img = cv2.imread(path)
    # new_arr = cv2.resize(img,(256, 256))
    new_arr =np.array(cv2.resize(img,(256, 256)))
    print(type(new_arr))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 256, 256, 3)
    return new_arr


model = keras.models.load_model('path/plantcnn.h5')
prediction = model.predict(image('path/inputs/PotatoHealthy.JPG'))

print(CATEGORIES[prediction.argmax()])
