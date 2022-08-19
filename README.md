# plant-disease-prediction-using-flask


dataset link----> https://www.kaggle.com/datasets/emmarex/plantdisease
To deploy our trained model for use via an API, we would do something similar to the following :

Load our trained model
Accept incoming data and preprocess it
Predict using our loaded model
Handling the prediction output.
  model_file = load_model(‘my_model.h5’)      "HAS TO BE COMPILED BY EXECUTING PLANTCNNCODE"

Make the model available for use via an API. Which is what I did for Plant AI. There are many ways in which we can make a model available via an API.


